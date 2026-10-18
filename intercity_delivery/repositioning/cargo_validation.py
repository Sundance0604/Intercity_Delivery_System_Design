"""Independent event replay of V0-F cargo, without solver expressions.

This checks cargo feasibility, not vehicle availability (see ``ledger.py``).
Local capacity deliberately follows V0-F's inverse service function without
adding the V1 stem, joint wave economy, or manual-vehicle capacity truncation.
"""

import ast
import math
from collections import defaultdict


def _number(value, label, tolerance):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a finite number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{label}: unsupported numeric range") from error
    if not math.isfinite(value) or value < -tolerance:
        raise ValueError(f"{label}: expected a finite nonnegative number")
    return max(0.0, value)


def _inverse(minutes, a, b):
    if not all(math.isfinite(v) for v in (minutes, a, b)) or minutes <= 0 or a <= 0 or b < 0:
        raise ValueError("service inverse requires finite positive duration/a and nonnegative b")
    # Rationalized root avoids cancellation when minutes or a is small.
    discriminant = b * b + 4 * a * minutes
    if not math.isfinite(discriminant) or not math.isfinite(2 * minutes):
        raise ValueError("service inverse overflow: unsupported numeric range")
    try:
        capacity = (2 * minutes / (math.sqrt(discriminant) + b)) ** 2
    except OverflowError as error:
        raise ValueError("service inverse overflow: unsupported numeric range") from error
    if not math.isfinite(capacity):
        raise ValueError("service inverse capacity must be finite")
    return capacity


def validate_cargo(config, data, decisions, completed_horizon=None,
                   completed=False, tolerance=1e-6, *, order_summaries=None):
    """Validate all initiated cargo commitments and return JSON-safe states.

    At a partial cut, started jobs retain their full future end events; only
    events through the cut enter current inventory. Final demand, split-variable
    equalities, and aggregate direct-ratio policy are deferred. A full replay
    requires all pickups, linehauls, and destination services to balance.
    ``order_summaries`` optionally checks exported full-model r/q/z values.
    Serialized tuple keys from the public registry are accepted via literal_eval.
    """
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or
            not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(completed, bool):
        raise ValueError("completed must be bool")
    for name in ("T", "travel_time_periods", "direct_travel_time_periods"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("t_0", "service_a_1", "service_a_2", "capacity_auto", "capacity_direct"):
        value = _number(getattr(config, name), name, 0.0)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("service_b_1", "service_b_2", "direct_ratio_min", "direct_ratio_max"):
        _number(getattr(config, name), name, 0.0)
    if not 0 <= config.direct_ratio_min <= config.direct_ratio_max <= 1:
        raise ValueError("direct ratio config must satisfy 0 <= min <= max <= 1")
    cut = config.T if completed_horizon is None else completed_horizon
    if isinstance(cut, bool) or not isinstance(cut, int) or not 0 <= cut <= config.T:
        raise ValueError("invalid completed_horizon")
    if completed and cut != config.T:
        raise ValueError("complete execution must reach real T")
    theta = config.transfer_time_periods
    if isinstance(theta, bool) or not isinstance(theta, int) or theta < 0:
        raise ValueError("transfer_time_periods must be a nonnegative integer")
    orders = data.all_orders
    for oid, order in orders.items():
        if isinstance(oid, bool) or order.flow not in ("+", "-"):
            raise ValueError("invalid order identifier or direction")
        _number(order.quantity, f"order {oid} demand", 0.0)
        for name in ("earliest_start", "latest_completion"):
            value = getattr(order, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"order {oid}: {name} must be finite")
            try:
                valid = math.isfinite(value)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"order {oid}: {name} must be finite")
        if order.earliest_start > order.latest_completion:
            raise ValueError(f"order {oid}: unordered service window")
    groups = {"x_manual": 4, "y_auto": 3, "w_direct": 3,
              "empty_vehicles": 3, "g_manual": 5, "g_auto": 4, "h_direct": 4}
    unknown = set(decisions) - set(groups)
    if unknown:
        raise ValueError(f"unknown decision groups: {sorted(unknown)}")
    parsed = {name: {} for name in groups}
    manual_arcs = {1: set(map(tuple, data.arcs_manual_1)),
                   2: set(map(tuple, data.arcs_manual_2))}
    auto_arcs = set(map(tuple, data.arcs_auto))
    for name, width in groups.items():
        for raw_key, raw_value in decisions.get(name, {}).items():
            try:
                key = ast.literal_eval(raw_key) if isinstance(raw_key, str) else tuple(raw_key)
                if not isinstance(key, tuple):
                    raise ValueError("decision key must be a tuple")
            except (SyntaxError, TypeError) as error:
                raise ValueError(f"{name}: invalid decision key") from error
            value = _number(raw_value, name, tolerance)
            if len(key) != width:
                raise ValueError(f"{name}: invalid key width")
            i, j = key[:2]
            if (isinstance(i, bool) or isinstance(j, bool) or
                    not isinstance(i, int) or not isinstance(j, int) or
                    not 0 <= i < j <= config.T):
                raise ValueError(f"{name}: invalid event interval")
            if value > tolerance and i >= cut:
                raise ValueError(f"{name}: uninitiated future commitment")
            flow = key[3] if name in ("x_manual", "g_manual") else key[2]
            if flow not in ("+", "-"):
                raise ValueError(f"{name}: invalid direction")
            if name in ("x_manual", "g_manual"):
                city = key[2]
                if isinstance(city, bool) or city not in manual_arcs or (i, j) not in manual_arcs[city]:
                    raise ValueError(f"{name}: unknown local arc")
            elif name in ("y_auto", "g_auto"):
                if (i, j) not in auto_arcs or j - i != config.travel_time_periods:
                    raise ValueError(f"{name}: unknown or invalid AV arc")
            elif name in ("w_direct", "h_direct"):
                if j - i <= config.direct_travel_time_periods:
                    raise ValueError(f"{name}: invalid direct arc")
            if name in ("x_manual", "y_auto", "w_direct", "empty_vehicles"):
                if abs(value - round(value)) > tolerance:
                    raise ValueError(f"{name}: nonintegral vehicle count")
            else:
                order_id = key[-1]
                if isinstance(order_id, bool) or order_id not in orders:
                    raise ValueError(f"{name}: unknown order")
                if flow != orders[order_id].flow:
                    raise ValueError(f"{name}: order direction mismatch")
            if key in parsed[name]:
                raise ValueError(f"{name}: duplicate normalized key")
            parsed[name][key] = value

    loads = {name: defaultdict(float) for name in ("g_manual", "g_auto", "h_direct")}
    stages = {oid: {s: [] for s in ("pickup", "auto", "delivery", "direct")}
              for oid in orders}
    for name in loads:
        for key, value in parsed[name].items():
            loads[name][key[:-1]] += value
            i, j, oid = key[0], key[1], key[-1]
            order = orders[oid]
            origin = 1 if order.flow == "+" else 2
            stage = ("auto" if name == "g_auto" else "direct" if name == "h_direct"
                     else "pickup" if key[2] == origin else "delivery")
            if value > tolerance:
                if stage in ("pickup", "direct") and i < order.earliest_start:
                    raise ValueError(f"order {oid}: departure before release")
                if stage in ("delivery", "direct") and j > order.latest_completion:
                    raise ValueError(f"order {oid}: delivery after deadline")
            stages[oid][stage].append((i, j, value))

    for key, amount in loads["g_manual"].items():
        i, j, city, _flow = key
        a, b = ((config.service_a_1, config.service_b_1) if city == 1 else
                (config.service_a_2, config.service_b_2))
        capacity = _inverse((j - i) * config.t_0, a, b)
        if amount > capacity * parsed["x_manual"].get(key, 0.0) + tolerance:
            raise ValueError(f"manual shared capacity exceeded: {key}")
    for key, amount in loads["g_auto"].items():
        if amount > config.capacity_auto * parsed["y_auto"].get(key, 0.0) + tolerance:
            raise ValueError(f"AV shared capacity exceeded: {key}")
    for key, amount in loads["h_direct"].items():
        i, j, _flow = key
        capacity = min(config.capacity_direct, _inverse(
            (j - i - config.direct_travel_time_periods) * config.t_0,
            config.service_a_1 + config.service_a_2,
            config.service_b_1 + config.service_b_2))
        if amount > capacity * parsed["w_direct"].get(key, 0.0) + tolerance:
            raise ValueError(f"direct shared capacity exceeded: {key}")

    output = {}
    total_direct = total_served = 0.0
    for oid, order in orders.items():
        demand = _number(order.quantity, f"order {oid} demand", tolerance)
        if order.flow not in ("+", "-"):
            raise ValueError(f"order {oid}: invalid flow")
        stage = stages[oid]
        totals = {name: sum(v for _i, _j, v in values) for name, values in stage.items()}
        if totals["pickup"] + totals["direct"] > demand + tolerance:
            raise ValueError(f"order {oid}: reserved demand exceeded")
        events = []
        # Replay future completions of initiated tasks as well as the current cut;
        # this catches insufficient upstream stock even when downstream is in flight.
        for t in range(config.T + 1):
            pickup_complete = sum(v for _i, j, v in stage["pickup"] if j <= t)
            pickup_ready = sum(v for _i, j, v in stage["pickup"] if j + theta <= t)
            av_departed = sum(v for i, _j, v in stage["auto"] if i <= t)
            av_arrived = sum(v for _i, j, v in stage["auto"] if j <= t)
            av_ready = sum(v for _i, j, v in stage["auto"] if j + theta <= t)
            delivery_departed = sum(v for i, _j, v in stage["delivery"] if i <= t)
            delivered = sum(v for _i, j, v in stage["delivery"] if j <= t)
            direct_delivered = sum(v for _i, j, v in stage["direct"] if j <= t)
            if av_departed > pickup_ready + tolerance:
                raise ValueError(f"order {oid}: origin cargo causality violated at {t}")
            if delivery_departed > av_ready + tolerance:
                raise ValueError(f"order {oid}: destination cargo causality violated at {t}")
            if t <= cut:
                events.append({"time": t,
                    "origin_hub": max(0.0, pickup_complete - av_departed),
                    "origin_ready": max(0.0, pickup_ready - av_departed),
                    "destination_hub": max(0.0, av_arrived - delivery_departed),
                    "destination_ready": max(0.0, av_ready - delivery_departed),
                    "pickup_in_progress": sum(v for i, j, v in stage["pickup"] if i <= t < j),
                    "av_in_transit": av_departed - av_arrived,
                    "delivery_in_progress": delivery_departed - delivered,
                    "delivered": delivered + direct_delivered})
        served = events[-1]["delivered"]
        if completed and (abs(totals["pickup"] - totals["auto"]) > tolerance or
                          abs(totals["auto"] - totals["delivery"]) > tolerance):
            raise ValueError(f"order {oid}: full-horizon transshipment imbalance")
        residual = max(0.0, demand - served)
        if completed and order_summaries is not None:
            for name, expected in (("r_transshipment", totals["pickup"]),
                                   ("q_direct", totals["direct"]),
                                   ("z_unserved", residual)):
                values = order_summaries.get(name, {})
                if oid not in values and str(oid) not in values:
                    raise ValueError(f"missing {name} summary for order {oid}")
                value = _number(values.get(oid, values.get(str(oid))), name, tolerance)
                if abs(value - expected) > tolerance:
                    raise ValueError(f"order {oid}: inconsistent {name} summary")
        total_direct += sum(v for _i, j, v in stage["direct"] if j <= cut)
        total_served += served
        committed_events = [
            {"stage": name, "start": i, "end": j, "quantity": value,
             "hub_ready_time": j + theta if name in ("pickup", "auto") else None}
            for name, values in stage.items() for i, j, value in values
        ]
        output[str(oid)] = {"demand": demand, "stage_totals": totals,
                            "origin_city": 1 if order.flow == "+" else 2,
                            "destination_city": 2 if order.flow == "+" else 1,
                            "committed_events": committed_events,
                            "served": served, "residual": residual,
                            "events": events, "state": events[-1]}
    if completed:
        if (total_direct + tolerance < config.direct_ratio_min * total_served or
                total_direct > config.direct_ratio_max * total_served + tolerance):
            raise ValueError("full-horizon aggregate direct ratio violated")
    return {"valid": True, "scope": "independent V0-F cargo replay; not V1",
            "completed": completed, "completed_horizon": cut,
            "checks": ["per_order_hub_causality", "shared_V0_F_capacity", "service_windows",
                       "demand_reservation", "full_stage_balance" if completed else "partial_state"],
            "deferred": ([] if completed else ["final_stage_balance", "r_q_z_equalities",
                                                "aggregate_direct_ratio"]),
            "order_summaries_checked": completed and order_summaries is not None,
            "excludes": ["vehicle_availability", "V1_wave_stem_capacity", "global_optimality"],
            "served_quantity": total_served, "direct_quantity": total_direct,
            "orders": output}
