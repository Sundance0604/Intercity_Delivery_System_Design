"""Explicit local-separate counterfactual; frozen V1 sources stay unchanged."""

import ast
import hashlib
import math
from pathlib import Path
import time

from intercity_delivery.v1.capacity import _integer, _real, local_capacity
from intercity_delivery.v1.flexible_configuration import flexible_profile, validate_f_instance
from intercity_delivery.v1.flexible_validation import validate_flexible_schedule

MODEL_ID = "v1_comparison_local_separate_v1"


def separate_capacity(duration, vehicles, a, b, rho, capacity):
    v = _integer("vehicles", vehicles)
    # Use the adopted single-vehicle kernel for construction. Replay below uses
    # the original work inequality independently, never this table/helper.
    one = local_capacity(duration, 1, a, b, rho, capacity)
    value = v * one
    if not math.isfinite(value):
        raise ValueError("separate capacity exceeds supported float range")
    return value


def configure_separate(optimizer):
    """Replace instance local K only, before any model construction."""
    if optimizer.model is not None:
        raise ValueError("capacity comparison must be configured before build")
    cfg = optimizer.config
    optimizer.capacities = {
        (i, j, k, v): separate_capacity((j-i)*cfg.t0_minutes, v,
                                       cfg.a[k], cfg.b[k], cfg.rho[k], cfg.M)
        for i, j, k, v in optimizer.capacities
    }
    return optimizer


def check_separate_local(config, schedule, tolerance=1e-6):
    """Independent original-f check: a*q+b*sqrt(v*q) <= v*B, q<=v*M.

    Called after the original full-domain validator, so tuple domains, integer
    vehicles, nonnegative cargo and causality have already been checked.
    """
    tol = _real("tolerance", tolerance, positive=True)
    if tol > 1e-3:
        raise ValueError("tolerance must be at most 1e-3")
    def group(name):
        return {(ast.literal_eval(k) if isinstance(k, str) else tuple(k)): float(v)
                for k, v in schedule[name].items()}
    vehicles, loads = group("x_manual"), {}
    for key, q in group("g_manual").items():
        loads[key[:4]] = loads.get(key[:4], 0.) + q
    maximum = {"separate_vehicle_capacity": 0., "separate_original_f_budget": 0.}
    count = 0
    for key in vehicles.keys() | loads.keys():
        i, j, city, _ = key
        v, q = round(vehicles.get(key, 0.)), loads.get(key, 0.)
        work = config.a[city]*q + config.b[city]*math.sqrt(v*q)
        bound = v*max((j-i)*config.t0_minutes-2*config.rho[city], 0.)
        for name, lhs, rhs in (("separate_vehicle_capacity", q, v*config.M),
                               ("separate_original_f_budget", work, bound)):
            residual = max(0., lhs-rhs)
            if not math.isfinite(residual) or lhs > rhs + tol:
                raise ValueError(f"{name} on {key}: {lhs} > {rhs}")
            maximum[name] = max(maximum[name], residual)
        count += 1
    return {"checked_local_waves": count, "maximum_violations": maximum,
            "units": {"separate_vehicle_capacity": "synthetic_job_equivalent",
                      "separate_original_f_budget": "aggregate minutes"},
            "method": "independent original f(q/v) inequality; no K-table lookup"}


def validate_separate(config, orders, raw):
    complete = bool(raw.get("completed"))
    evidence = validate_flexible_schedule(
        config, orders, flexible_profile(), raw["schedule"], completed=complete,
        completed_horizon=config.T if complete else raw.get("current_time", 0),
        reported_objective=raw.get("total_cost") if complete else None)
    evidence["local_separate"] = check_separate_local(config, raw["schedule"])
    evidence["validation_model"] = MODEL_ID
    return evidence


def capacity_curves():
    return [{"scenario": label, "b": b, "duration_minutes": 60, "vehicles": v,
             "pooled": local_capacity(60, v, 5, b, 5, 4),
             "separate": separate_capacity(60, v, 5, b, 5, 4)}
            for label, b in (("baseline", 20), ("load_saturated_control", 10))
            for v in (1, 2, 3, 5)]


def run_separate(config, orders, time_limit, output_flag=0):
    """Offline comparison runner; only local capacity changes from V1-F."""
    from intercity_delivery.v1.flexible_optimizer import V1FlexibleOptimizer
    from intercity_delivery.v1.flexible_runner import (
        _base, _optional_number, _replay_result, _validate_runtime)
    from intercity_delivery.v1.flexible_rolling import PhaseTimer, model_counts, mother_domain_sha256
    budget = _validate_runtime(time_limit, output_flag)
    started = time.monotonic()
    timer = PhaseTimer(started)
    profile = flexible_profile()
    orders = tuple(orders)
    validate_f_instance(config, orders, profile)
    result = _base(config, orders, profile, "full", budget)
    result.update(model_id=MODEL_ID, semantic_status="explicit_counterfactual_local_capacity_only",
                  bound_scope="full_horizon_local_separate_comparison_only")
    result["metadata"].update(model_scope="local-separate counterfactual to frozen V1-F",
        capacity_profile="local_separate", changed_component="local_capacity_only",
        direct_capacity="original_formula_unchanged",
        comparison_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    timer.values["setup_provenance"] = time.monotonic()-started
    optimizer = None
    try:
        with timer.measure("build"):
            optimizer = configure_separate(V1FlexibleOptimizer(config, orders, profile))
            optimizer.build_model()
        model = optimizer.model
        with timer.measure("configure"):
            model.Params.OutputFlag = output_flag
            model.Params.Threads = 1
            model.Params.Seed = 0
        remaining = budget-(time.monotonic()-started)
        result.update(build_elapsed_sec=time.monotonic()-started,
                      solver_time_limit_sec=max(0., remaining))
        if remaining > 0:
            model.Params.TimeLimit = remaining
            result["solver_optimize_attempts"] = 1
            with timer.measure("solve"):
                model.optimize()
            result["status"] = int(model.Status)
            result["status_name"] = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
                5: "UNBOUNDED", 9: "TIME_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC",
                13: "SUBOPTIMAL"}.get(int(model.Status), f"GUROBI_{model.Status}")
        else:
            result["status_name"] = "BUDGET_EXHAUSTED_DURING_BUILD"
        result.update(best_bound=_optional_number(model, "ObjBound"),
                      solver_runtime_sec=_optional_number(model, "Runtime"))
        result.update(model_counts(model))
        result.update(mother_domain_sha256(optimizer.groups))
        if model.SolCount > 0:
            with timer.measure("extract_commit"):
                schedule = optimizer.extract_schedule()
            with timer.measure("replay_accounting"):
                evidence = validate_flexible_schedule(config, orders, profile, schedule,
                                                       reported_objective=float(model.ObjVal))
                evidence["local_separate"] = check_separate_local(config, schedule)
                evidence["validation_model"] = MODEL_ID
                _replay_result(result, schedule, evidence)
            result.update(completed=True, has_incumbent=True, optimal=int(model.Status) == 2,
                          incumbent_objective=float(model.ObjVal), mip_gap=_optional_number(model, "MIPGap"))
        return result
    except ImportError as exc:
        result["status_name"] = "DEPENDENCY_UNAVAILABLE"
        result["warnings"].append(str(exc))
        return result
    except Exception as exc:
        if type(exc).__module__.startswith("gurobipy"):
            result.update(status_name="SOLVER_ERROR", solver_error_code=getattr(exc, "errno", None))
            result["warnings"].append(str(exc))
            return result
        raise
    finally:
        if optimizer is not None and optimizer.model is not None:
            optimizer.model.dispose()
        result["elapsed_sec"], result["phase_timing_sec"] = timer.finish()
        result["budget_exceeded"] = result["elapsed_sec"] > budget
