"""Public full/rolling entry points and independently evaluated execution costs."""

from __future__ import annotations

import time

from intercity_delivery.models.gurobi_results import optional_model_float
from .config import RepositioningPolicy
from .cargo_validation import validate_cargo
from .ledger import evaluate_committed
from .optimizer import RepositioningOptimizer
from .rolling import solve_rolling


def _serialize_decisions(decisions):
    return {name: {str(tuple(key)): float(value) for key, value in values.items()}
            for name, values in decisions.items()}


def _result(config, data, raw, approach, full_objective=None, best_bound=None, mip_gap=None):
    complete = raw["completed"]
    last_status = raw["status"]
    # An optimal window does not certify completion of the rolling execution.
    # Preserve actual error/infeasibility statuses; only convert an unfinished
    # execution's last successful OPTIMAL status to TIME_LIMIT.
    execution_status = (
        9 if not complete and approach == "flexible_direct_empty_rolling"
        and last_status == 2 else last_status
    )
    accounting = evaluate_committed(
        config, data, raw["decisions"], raw["policy"], raw["current_time"], complete
    )
    cargo_validation = validate_cargo(
        config, data, raw["decisions"], raw["current_time"], complete,
        order_summaries=raw.get("model_order_summaries"),
    )
    demand = sum(order.quantity for order in data.all_orders.values())
    direct = accounting["direct_quantity"]
    served = accounting["served_quantity"]
    unserved = max(0.0, demand - served) if complete else None
    transshipment = max(0.0, served - direct)
    manual_usage = sum(sum(raw["decisions"].get(name, {}).values())
                       for name in ("x_manual", "w_direct", "empty_vehicles"))
    total = accounting["total_cost"] if complete else None
    accounting_gap = total - full_objective if total is not None and full_objective is not None else None
    return {
        "status": execution_status, "total_cost": total,
        "solve_time_sec": raw["solve_time_sec"], "completed": complete,
        "unserved_rate": (unserved / demand if demand else 0.0) if complete else None,
        "unserved_amount": unserved,
        "auto_usage": sum(raw["decisions"].get("y_auto", {}).values()),
        "manual_usage": manual_usage, "direct_volume": direct,
        "transshipment_volume": transshipment,
        "direct_ratio": direct / served if served else 0.0,
        "empty_vehicle_trips": sum(raw["decisions"].get("empty_vehicles", {}).values()),
        "best_bound": best_bound, "mip_gap": mip_gap,
        "detail": {
            "model_id": RepositioningOptimizer.model_id,
            "algorithm": {
                "approach": approach, "completed": complete,
                "last_window_status": last_status,
                "completed_horizon": raw["current_time"],
                "candidate_policy": "unpruned_physical_arcs_with_explicit_window_bounds",
                "boundary_policy": "new_tasks_finish_by_completion_end; committed_events_retained",
                "window_terminal_reset": False,
                "bound_scope": "full_horizon" if approach == "flexible_direct_empty_mip"
                               else "window_local_only; no_global_bound",
                "cost_scope": "complete_execution" if complete else "partial_full_committed_task_costs; not_elapsed_cost",
                "model_scope": "V0-F plus independent empty moves; not V1 wave MILP",
            },
            "policy": raw["policy"].metadata(), "windows": raw.get("windows", []),
            "execution_accounting": accounting,
            "cargo_validation": cargo_validation,
            "full_model_objective": full_objective,
            "accounting_gap": accounting_gap,
            "accounting_matches_objective": (
                abs(accounting_gap) <= 1e-5 * max(1.0, abs(full_objective))
                if accounting_gap is not None else None
            ),
            "solution": {
                "committed_decisions": _serialize_decisions(raw["decisions"]),
                "unserved_by_order": accounting["unserved_by_order"],
                "served_by_order": accounting["served_by_order"],
            },
        },
        "message": raw["message"],
    }


def run_full(config, data, time_limit, policy=None, output_flag=0):
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    started = time.monotonic()
    policy = (policy or RepositioningPolicy()).resolve(config)
    optimizer = RepositioningOptimizer(config, data, policy).build_model()
    try:
        optimizer.model.setParam("OutputFlag", output_flag)
        optimizer.model.setParam("Threads", 1)
        remaining = time_limit - (time.monotonic() - started)
        if remaining > 0:
            optimizer.model.setParam("TimeLimit", remaining)
            optimizer.model.optimize()
        feasible = optimizer.model.SolCount > 0
        raw = {
            "decisions": optimizer.extract_decisions() if feasible else {},
            "status": optimizer.model.Status, "policy": policy,
            "completed": feasible, "current_time": config.T if feasible else 0,
            "solve_time_sec": time.monotonic() - started,
            "model_order_summaries": {
                name: {str(key): float(var.X) for key, var in variables.items()}
                for name, variables in (
                    ("r_transshipment", optimizer.r_transshipment),
                    ("q_direct", optimizer.q_direct), ("z_unserved", optimizer.z_unserved)
                )
            } if feasible else {},
            "message": "Full-horizon incumbent evaluated independently." if feasible
                       else "No incumbent obtained within the budget.",
        }
        result = _result(
            config, data, raw, "flexible_direct_empty_mip",
            full_objective=float(optimizer.model.ObjVal) if feasible else None,
            best_bound=optional_model_float(optimizer.model, "ObjBound"),
            mip_gap=optional_model_float(optimizer.model, "MIPGap") if feasible else None,
        )
        result["detail"]["solution"]["model_order_summaries"] = raw["model_order_summaries"]
        return result
    finally:
        optimizer.model.dispose()


def run_rolling(config, data, algorithm_config, time_limit, policy=None, output_flag=0):
    raw = solve_rolling(config, data, algorithm_config, time_limit, policy, output_flag)
    return _result(config, data, raw, "flexible_direct_empty_rolling")
