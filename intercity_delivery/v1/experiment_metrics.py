"""Descriptive finite-run metrics; no solver import or optimality inference.

Busy time uses dispatched vehicle-task durations, including direct and empty
relocations. Lead time is weighted by delivered continuous flow, measured from
the source order's release until its delivery arc ends. It is not a customer
level or order-average statistic. Partial schedules never receive a full cost.
"""

import ast
import math
import statistics


def weighted_quantile(samples, probability):
    """Left inverse of the empirical continuous-flow weighted CDF."""
    if not 0 <= probability <= 1:
        raise ValueError("probability outside [0,1]")
    items = sorted((float(x), float(w)) for x, w in samples if w > 0)
    total = sum(w for _, w in items)
    if not items or total <= 0:
        return None
    target, cumulative = probability * total, 0.
    for value, weight in items:
        cumulative += weight
        if cumulative >= target:
            return value
    return items[-1][0]


def _group(schedule, name):
    return {(ast.literal_eval(k) if isinstance(k, str) else tuple(k)): float(v)
            for k, v in schedule.get(name, {}).items()}


def summarize_result(result, config, orders):
    """Use the original-f validator's evidence, retaining partial denominators."""
    valid = bool(result.get("schedule_valid"))
    complete = valid and bool(result.get("completed"))
    evidence = result.get("validation") or {}
    schedule = result.get("schedule") or {}
    orders = {o["order_id"]: o for o in orders}
    total = sum(o["quantity"] for o in orders.values())
    served = evidence.get("served_quantity") if valid else None
    direct = evidence.get("direct_quantity") if valid else None
    manual = sum(v * (k[1]-k[0]) * config["t0_minutes"]
                 for name in ("x_manual", "w_direct", "empty_vehicles")
                 for k, v in _group(schedule, name).items()) if valid else None
    av = sum(v * (k[1]-k[0]) * config["t0_minutes"]
             for k, v in _group(schedule, "y_auto").items()) if valid else None
    human_fleet = sum(config["N_manual"].values())
    av_fleet = sum(config["N_auto"].values())
    horizon = config["T"] * config["t0_minutes"]
    direct_loads = {}
    for k, v in _group(schedule, "h_direct").items():
        direct_loads[k[:3]] = direct_loads.get(k[:3], 0.) + v
    # Zero means zero up to the disclosed reporting tolerance. A positive
    # aggregate wave does not identify individual vehicles' empty/nonempty state.
    tolerance = 1e-8
    zero_direct = sum(v for k, v in _group(schedule, "w_direct").items()
                      if direct_loads.get(k, 0.) <= tolerance) if valid else None
    explicit_empty = evidence.get("empty_trip_count") if valid else None
    vehicle_tasks = sum(v for name in ("x_manual", "w_direct", "empty_vehicles")
                        for v in _group(schedule, name).values()) if valid else None
    cut = evidence.get("completed_horizon", config["T"])
    samples = []
    if valid:
        for k, q in _group(schedule, "h_direct").items():
            order = orders[k[-1]]
            if q > 0 and k[1] <= cut:
                samples.append(((k[1]-order["earliest_start"])*config["t0_minutes"], q))
        for k, q in _group(schedule, "g_manual").items():
            order = orders[k[-1]]
            origin = 1 if order["flow"] == "+" else 2
            if q > 0 and k[2] != origin and k[1] <= cut:
                samples.append(((k[1]-order["earliest_start"])*config["t0_minutes"], q))
    weight = sum(w for _, w in samples)
    cost = result.get("total_cost") if complete else None
    return {
        "status": result.get("status_name"), "completed_feasible": complete,
        "validation_passed": valid, "demand_quantity": total,
        "total_cost": cost, "cost_components": evidence.get("costs") if valid else None,
        "served_quantity": served, "unserved_quantity": evidence.get("unserved_quantity") if valid else None,
        "unserved_scope": "final refusal" if complete else "residual, not necessarily final refusal",
        "served_rate": served/total if served is not None and total else None,
        "direct_share": direct/served if served else None,
        "explicit_empty_vehicle_tasks": explicit_empty,
        "zero_cargo_direct_vehicle_tasks": zero_direct,
        "zero_cargo_reporting_tolerance": tolerance,
        "manual_vehicle_tasks": vehicle_tasks,
        "explicit_empty_task_ratio": explicit_empty/vehicle_tasks if vehicle_tasks else None,
        "empty_including_zero_direct_task_ratio": (explicit_empty+zero_direct)/vehicle_tasks if vehicle_tasks else None,
        "empty_scope": "manual vehicle-task counts; aggregate wave cargo cannot identify each vehicle's load",
        "manual_busy_vehicle_minutes": manual, "av_busy_vehicle_minutes": av,
        "manual_busy_fraction": manual/(human_fleet*horizon) if valid and human_fleet*horizon else None,
        "av_busy_fraction": av/(av_fleet*horizon) if valid and av_fleet*horizon else None,
        "utilization_scope": "all committed task durations / initial total fleet times full real horizon; partial may include future commitments",
        "delivered_flow_weight": weight,
        "lead_time_mean_minutes": sum(x*w for x, w in samples)/weight if weight else None,
        "lead_time_p50_minutes": weighted_quantile(samples, .5),
        "lead_time_p95_minutes": weighted_quantile(samples, .95),
        "lead_time_scope": "continuous delivered-flow weighted arc end minus order release, conditional on delivery",
        "end_to_end_sec": result.get("elapsed_sec"),
        "phase_timing_sec": result.get("phase_timing_sec"),
        "window_count": len(result.get("windows", [])),
        "window_statistics": result.get("windows", []),
        "global_optimal_oracle": bool(complete and result.get("optimal") and
                                      result.get("metadata", {}).get("run_mode") == "full"),
        "full_horizon_best_bound": result.get("best_bound") if result.get("metadata", {}).get("run_mode") == "full" else None,
        "bound_scope": result.get("bound_scope"),
    }


def mean_ci95(values, expected_n=5):
    """t_4 interval only for a full predeclared five-seed set; no missing fill."""
    values = list(values)
    mean = statistics.mean(values) if values else None
    interval = None
    if len(values) == expected_n == 5:
        half_width = 2.7764451051977987 * statistics.stdev(values)/math.sqrt(5)
        interval = [mean-half_width, mean+half_width]
    return {"n": len(values), "planned_n": expected_n, "mean": mean,
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "ci95": interval, "ci_method": "Student t, df=4; only when all 5 seeds eligible; descriptive normality assumption"}
