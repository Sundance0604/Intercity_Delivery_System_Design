"""Absolute-time rolling controller for the opt-in empty repositioning model."""

from dataclasses import replace
import time

from intercity_delivery.data.loader import DataLoader
from intercity_delivery.models.gurobi_results import optional_model_float
from .config import RepositioningPolicy
from .optimizer import RepositioningOptimizer


def visible_data(config, data, current_time):
    """Retain all revealed orders, including expired orders and their cargo."""
    orders = {key: order for key, order in data.all_orders.items()
              if order.earliest_start <= current_time}
    pos = {key: order for key, order in orders.items() if order.flow == "+"}
    neg = {key: order for key, order in orders.items() if order.flow == "-"}
    return replace(
        data, all_orders=orders, pos_orders=pos, neg_orders=neg,
        epsilon_sets=DataLoader(config).generate_epsilon_sets(
            pos, neg, data.arcs_manual_1, data.arcs_manual_2),
    )


def commit_control_window(decisions, start, end):
    result = {}
    for name, values in decisions.items():
        result[name] = {}
        for key, value in values.items():
            if start <= key[0] < end:
                if name in RepositioningOptimizer.vehicle_groups:
                    if abs(value - round(value)) > 1e-6:
                        raise ValueError("Non-integral vehicle commitment")
                    value = float(round(value))
                if abs(value) > 1e-8:
                    result[name][tuple(key)] = float(value)
    return result


def solve_rolling(config, data, algorithm_config, time_limit, policy=None,
                  output_flag=0):
    """Return raw execution state, never treating the last MILP value as cost."""
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    algorithm_config.validate()
    policy = (policy or RepositioningPolicy()).resolve(config)
    if policy.terminal_target is not None:
        raise ValueError(
            "rolling terminal_target is unsupported until a real-T safe tail "
            "reachability check is implemented; full-horizon targets are supported"
        )
    started = time.monotonic()
    current = 0
    committed = {}
    windows = []
    last_status = None
    failure = None
    while current < config.T:
        remaining = time_limit - (time.monotonic() - started)
        if remaining <= 0:
            break
        start_end = min(config.T, current + algorithm_config.prediction_horizon)
        completion_end = min(config.T, start_end + algorithm_config.extension_horizon)
        control_end = min(config.T, current + algorithm_config.rolling_step)
        known_data = visible_data(config, data, current)
        optimizer = RepositioningOptimizer(config, known_data, policy).build_model()
        try:
            optimizer.configure_event_window(current, start_end, completion_end, committed)
            optimizer.model.setParam("OutputFlag", output_flag)
            optimizer.model.setParam("Threads", 1)
            remaining = time_limit - (time.monotonic() - started)
            if remaining <= 0:
                break
            optimizer.model.setParam("TimeLimit", remaining)
            optimizer.model.optimize()
            last_status = optimizer.model.Status
            feasible = optimizer.model.SolCount > 0
            record = {
                "window_start": current, "start_window_end": start_end,
                "completion_window_end": completion_end, "control_end": control_end,
                "known_order_ids": sorted(known_data.all_orders),
                "status": last_status, "feasible": feasible,
                "objective": float(optimizer.model.ObjVal) if feasible else None,
                "best_bound": optional_model_float(optimizer.model, "ObjBound"),
                "mip_gap": optional_model_float(optimizer.model, "MIPGap") if feasible else None,
                "bound_scope": "window_local_only",
            }
            windows.append(record)
            if not feasible:
                failure = "A window has no incumbent; execution is partial."
                break
            additions = commit_control_window(optimizer.extract_decisions(), current, control_end)
            for name, values in additions.items():
                committed.setdefault(name, {}).update(values)
            current = control_end
        finally:
            optimizer.model.dispose()
    return {
        "decisions": committed, "windows": windows, "current_time": current,
        "completed": current == config.T, "status": last_status,
        "solve_time_sec": time.monotonic() - started, "policy": policy,
        "message": failure or (
            "Rolling execution reached the real horizon."
            if current == config.T else "Runtime budget exhausted; execution is partial."
        ),
    }


def run_rolling(config, data, algorithm_config, time_limit, policy=None, output_flag=0):
    """Public registry entry point (import lazily to avoid a runner cycle)."""
    from .runner import run_rolling as run
    return run(config, data, algorithm_config, time_limit, policy, output_flag)
