"""Independent V1-F replay: original f inequalities, not optimizer/K tables.

Full schedules certify the adopted finite-horizon profile. Partial schedules
certify commitments and accounting only; future completion is not certified.
"""

import math
from collections.abc import Mapping

from .configuration import PROFILE_DECISION_ID
from .flexible_configuration import (empty_duration, empty_hourly_cost,
                                     validate_f_instance)
from .validation import _finite, _key, _node, _number, _sum


GROUPS = {"x_manual": 4, "y_auto": 3, "u_wave": 5, "g_manual": 5,
          "g_auto": 4, "z_unserved": 1, "w_direct": 3, "u_direct": 4,
          "empty_vehicles": 3, "h_direct": 4}
INTEGER_GROUPS = {"x_manual", "y_auto", "u_wave", "w_direct", "u_direct",
                  "empty_vehicles"}


def validate_flexible_schedule(config, orders, profile, schedule, *, completed=True,
                               completed_horizon=None, tolerance=1e-6,
                               reported_objective=None):
    """Check all domains and physical constraints; raise ValueError on failure.

    Missing tuple keys mean zero, but all ten group mappings are required. JSON
    string tuple keys are accepted. Vehicle counts/selectors are integer within
    absolute tolerance. Positive continuous loads are never rounded away.
    Partial commitments contain departures strictly before completed_horizon,
    retain their whole durations, and do not commit z/unserved decisions.
    """
    validate_f_instance(config, orders, profile)
    tol = _number(tolerance, "tolerance")
    if not 0 < tol <= 1e-3:
        raise ValueError("tolerance must lie in (0,1e-3]")
    if not isinstance(completed, bool):
        raise ValueError("completed must be bool")
    cut = config.T if completed_horizon is None else _node(completed_horizon)
    if not 0 <= cut <= config.T or completed and cut != config.T:
        raise ValueError("invalid completion horizon")
    if reported_objective is not None and not completed:
        raise ValueError("partial commitments have no full-horizon objective")
    by_id = {o.order_id: o for o in orders}
    fleet = sum(config.N_manual.values())
    f = lambda city, q: _finite(config.a[city]*q + config.b[city]*math.sqrt(q),
                                "original f(load)")
    limits = {k: math.ceil(_finite((2*config.rho[k]+f(k, config.M))/
                                    config.t0_minutes, "local arc bound"))
              for k in (1, 2)}
    direct_limit = config.tau_periods + math.ceil(_finite(
        (2*max(config.rho.values()) + f(1, config.M) + f(2, config.M))/
        config.t0_minutes, "direct arc bound"))
    if not isinstance(schedule, Mapping) or set(schedule) != set(GROUPS):
        raise ValueError("schedule must contain exactly ten V1-F groups")

    violations, comparisons = {}, {}
    check_units = {
        "one-hot": "dimensionless selector sum",
        "selector vehicle link": "vehicles",
        "manual city inventory": "vehicles",
        "AV city inventory": "vehicles",
        "real terminal fleet target": "vehicles",
        "mechanism disables independent empty": "vehicles",
        "mechanism disables direct": "mixed variable units: vehicles, dimensionless selectors, or service-equivalents",
        "local original f budget": "minutes of aggregate routing work",
        "direct original directional f budget": "minutes",
        "objective mismatch": "currency",
    }

    def leq(a, b, name):
        a, b = _finite(a, name), _finite(b, name)
        residual = max(0., _finite(a-b, f"{name} residual"))
        violations[name] = max(violations.get(name, 0.), residual)
        comparisons[name] = comparisons.get(name, 0)+1
        if a > b + tol:
            raise ValueError(f"{name}: {a} > {b}")

    def equal(a, b, name):
        leq(a, b, name)
        leq(b, a, name)

    groups = {}
    for name, size in GROUPS.items():
        if not isinstance(schedule[name], Mapping):
            raise ValueError(f"{name} must be a mapping")
        group = {}
        for raw, value in schedule[name].items():
            key = _key(raw, size)
            if key in group:
                raise ValueError("duplicate normalized schedule key")
            if name == "z_unserved":
                if not isinstance(key[0], str) or key[0] not in by_id:
                    raise ValueError("unserved references unknown order")
            else:
                i, j = _node(key[0]), _node(key[1])
                if not 0 <= i < j <= config.T:
                    raise ValueError("arc outside real horizon")
                local = name in ("x_manual", "u_wave", "g_manual")
                flow = key[3] if local else key[2]
                if not isinstance(flow, str) or flow not in ("+", "-"):
                    raise ValueError("invalid direction")
                if local:
                    city = _node(key[2])
                    if city not in (1, 2) or j-i > limits[city]:
                        raise ValueError("local mother arc domain")
                elif name in ("y_auto", "g_auto"):
                    if j-i != config.tau_periods:
                        raise ValueError("AV physical duration")
                elif name == "empty_vehicles":
                    if j-i != empty_duration(config):
                        raise ValueError("empty physical duration")
                elif not config.tau_periods <= j-i <= direct_limit:
                    raise ValueError("direct mother arc domain")
                if name in ("u_wave", "u_direct"):
                    if not 1 <= _node(key[-1]) <= fleet:
                        raise ValueError("selector outside global manual fleet")
                if name in ("g_manual", "g_auto", "h_direct"):
                    oid = key[-1]
                    if not isinstance(oid, str) or oid not in by_id:
                        raise ValueError("cargo references unknown order")
                    if by_id[oid].flow != flow:
                        raise ValueError("cargo flow differs from order")
            value = _number(value, name)
            if value < -tol:
                raise ValueError(f"negative variable {name}")
            value = max(0., value)
            if name in INTEGER_GROUPS:
                rounded = round(value)
                if abs(value-rounded) > min(tol, 1e-6):
                    raise ValueError("noninteger vehicle or selector")
                if name in ("u_wave", "u_direct") and rounded not in (0, 1):
                    raise ValueError("selector must be binary")
                value = rounded
            if not completed and value > 0:
                if name == "z_unserved":
                    raise ValueError("partial schedule cannot commit unserved decisions")
                if key[0] >= cut:
                    raise ValueError("partial schedule contains uninitiated event")
            group[key] = value
        groups[name] = group
    x, y, u, g, ga, z, w, ud, e, h = (groups[n] for n in GROUPS)
    if config.service_mechanism in ("T0", "T1"):
        for group in (w, ud, h):
            for value in group.values():
                leq(value, 0, "mechanism disables direct")
    if config.service_mechanism in ("T0", "T2"):
        for value in e.values():
            leq(value, 0, "mechanism disables independent empty")
    for vehicles, selectors, cargo, prefix, local in (
            (x, u, g, 4, True), (w, ud, h, 3, False)):
        arcs = set(vehicles) | {k[:prefix] for k in selectors} | {k[:prefix] for k in cargo}
        for arc in arcs:
            count = _sum((k[-1]*v for k, v in selectors.items() if k[:prefix] == arc),
                         "selected fleet")
            leq(_sum((v for k, v in selectors.items() if k[:prefix] == arc),
                     "one-hot sum"), 1, "one-hot")
            equal(vehicles.get(arc, 0), count, "selector vehicle link")
            load = _sum((v for k, v in cargo.items() if k[:prefix] == arc), "wave load")
            if not local:
                if count > 0 and config.min_direct_load_per_vehicle > 0 and load <= 0:
                    raise ValueError("minimum direct load requires strictly positive cargo")
                leq(_finite(count*config.min_direct_load_per_vehicle, "minimum direct load"),
                    load, "minimum direct load per vehicle")
            leq(load, _finite(count*config.M, "wave physical capacity"), "manual physical cap")
            i, j = arc[:2]
            duration = _finite((j-i)*config.t0_minutes, "task duration")
            if local:
                budget = _finite(count*max(duration-2*config.rho[arc[2]], 0), "wave budget")
                leq(f(arc[2], load), budget, "local original f budget")
            elif load > 0:
                if count == 0 or j-i == config.tau_periods:
                    leq(load, 0, "zero-capacity direct wave")
                else:
                    origin = 1 if arc[2] == "+" else 2
                    original = _finite(2*config.rho[origin]+f(origin, load)/count+
                        config.tau_periods*config.t0_minutes+f(3-origin, load/count),
                        "direct original duration")
                    leq(original, duration, "direct original directional f budget")
    for arc in set(y) | {k[:3] for k in ga}:
        leq(_sum((v for k, v in ga.items() if k[:3] == arc), "AV load"),
            _finite(config.M_auto*y.get(arc, 0), "AV capacity"), "shared AV capacity")

    min_manual = {str(k): float(config.N_manual[k]) for k in (1, 2)}
    min_auto = {str(k): float(config.N_auto[k]) for k in (1, 2)}
    for t in range(config.T+1):
        for city, flow in ((1, "+"), (2, "-")):
            busy = _sum((v for (i, j, k, d), v in x.items() if k == city and i <= t < j), "local busy")
            out = _sum((v for group in (w, e) for (i, j, d), v in group.items()
                        if d == flow and i <= t), "manual departed")
            inc = _sum((v for group in (w, e) for (i, j, d), v in group.items()
                        if d != flow and j <= t), "manual arrived")
            idle = _finite(config.N_manual[city]+inc-out-busy, "manual idle")
            leq(0, idle, "manual city inventory")
            min_manual[str(city)] = min(min_manual[str(city)], idle)
            outa = _sum((v for (i, j, d), v in y.items() if d == flow and i <= t), "AV departed")
            inca = _sum((v for (i, j, d), v in y.items() if d != flow and j <= t), "AV arrived")
            idlea = _finite(config.N_auto[city]+inca-outa, "AV idle")
            leq(0, idlea, "AV city inventory")
            min_auto[str(city)] = min(min_auto[str(city)], idlea)
            if completed and t == config.T and config.terminal_target is not None:
                equal(idle, config.terminal_target[city], "real terminal fleet target")

    evidence = {}
    theta = config.transfer_time_periods
    for order in orders:
        oid = order.order_id
        origin = 1 if order.flow == "+" else 2
        pickup = [(i, j, v) for (i, j, city, flow, item), v in g.items()
                  if item == oid and city == origin]
        delivery = [(i, j, v) for (i, j, city, flow, item), v in g.items()
                    if item == oid and city != origin]
        auto = [(i, j, v) for (i, j, flow, item), v in ga.items() if item == oid]
        direct = [(i, j, v) for (i, j, flow, item), v in h.items() if item == oid]
        for i, j, v in pickup + direct:
            if i < order.earliest_start:
                leq(v, 0, "release window")
        for i, j, v in delivery + direct:
            if j > order.latest_completion:
                leq(v, 0, "delivery deadline")
        for t in range(config.T+1):
            ready = _sum((v for i, j, v in pickup if j+theta <= t), "pickup ready")
            depart = _sum((v for i, j, v in auto if i <= t), "AV departure")
            arrived = _sum((v for i, j, v in auto if j+theta <= t), "AV ready")
            distribute = _sum((v for i, j, v in delivery if i <= t), "delivery departure")
            leq(depart, ready, "pickup before AV")
            leq(distribute, arrived, "AV before delivery")
        totals = {n: _sum((v for i, j, v in items), f"{n} total")
                  for n, items in (("pickup", pickup), ("linehaul", auto),
                                   ("delivery", delivery), ("direct", direct))}
        for n in ("pickup", "linehaul", "delivery"):
            leq(_sum((totals[n], totals["direct"]), "reserved quantity"), order.quantity,
                "demand reservation")
        delivered_transfer = _sum((v for i, j, v in delivery if j <= cut), "transfer delivered")
        delivered_direct = _sum((v for i, j, v in direct if j <= cut), "direct delivered")
        served = _sum((delivered_transfer, delivered_direct), "served")
        residual = max(0., _finite(order.quantity-served, "residual demand"))
        penalized = residual if completed or order.latest_completion <= cut else 0.
        if completed:
            for n in ("pickup", "linehaul"):
                equal(totals[n], totals["delivery"], "completed stage balance")
            equal(_sum((served, z.get((oid,), 0)), "demand balance"),
                  order.quantity, "completed demand balance")
        evidence[oid] = dict(totals, served=served, unserved=residual,
                            penalized_unserved=penalized,
                            delivered_transfer=delivered_transfer,
                            delivered_direct=delivered_direct)
    served = _sum((v["served"] for v in evidence.values()), "total served")
    direct = _sum((v["delivered_direct"] for v in evidence.values()), "total direct")
    if completed:
        leq(_finite(config.direct_ratio_min*served, "min direct ratio"), direct,
            "minimum direct ratio")
        leq(direct, _finite(config.direct_ratio_max*served, "max direct ratio"),
            "maximum direct ratio")

    def task_cost(group, rate):
        return _sum((_finite(v*((k[1]-k[0])*config.t0_minutes/60)*rate, "task cost")
                     for k, v in group.items()), "group cost")

    costs = {"manual": task_cost(x, config.c_manual_hour),
             "auto": task_cost(y, config.c_auto_hour),
             "direct": task_cost(w, config.c_direct_hour),
             "empty": task_cost(e, empty_hourly_cost(config)),
             "transfer": _sum((_finite(v["delivered_transfer"]*config.transfer_cost_per_unit,
                                        "transfer term") for v in evidence.values()), "transfer cost"),
             "penalty": _sum((_finite(evidence[o.order_id]["penalized_unserved"]*o.penalty,
                                       "penalty term") for o in orders), "penalty cost")}
    committed_cost = _sum(costs.values(), "committed cost")
    costs["committed_total"] = committed_cost
    costs["total"] = committed_cost if completed else None
    if reported_objective is not None:
        equal(_number(reported_objective, "reported objective"), committed_cost, "objective mismatch")
    return {"schedule_valid": True, "validator": "independent_original_f_v1_f",
            "max_constraint_violation": max(violations.values(), default=0.),
            "constraint_violation_by_check": violations,
            "constraint_comparison_count_by_check": comparisons,
            "constraint_violation_units_by_check": {name: check_units.get(name, "continuous service-equivalents")
                                                     for name in violations},
            "constraint_violation_scope": "leq/equal comparisons only; maximum is a mixed-units diagnostic, not a normalized certificate; do not sum across checks; domain, integrality, finite-value and strict-positive-load checks remain separate rejection checks",
            "profile_decision_id": PROFILE_DECISION_ID, "completed": completed,
            "completed_horizon": cut, "tolerance": tol, "comparison": "absolute",
            "costs": costs, "objective": costs["total"], "orders": evidence,
            "served_quantity": served,
            "unserved_quantity": _sum((v["unserved"] for v in evidence.values()), "total unserved"),
            "direct_quantity": direct,
            "transfer_quantity": _sum((v["delivered_transfer"] for v in evidence.values()), "total transfer"),
            "empty_trip_count": _sum(e.values(), "empty trips"),
            "direct_trip_count": _sum(w.values(), "direct trips"),
            "min_manual_inventory": min_manual, "min_AV_inventory": min_auto,
            "checks": ["domains", "one_hot", "global_fleet_selector_bound",
                       "original_f_local_and_directional_direct", "physical_cap",
                       "manual_migration_and_occupancy", "AV_inventory", "empty_duration",
                       "windows", "cargo_causality", "demand_reservations", "costs"],
            "deferred_checks": [] if completed else ["final_stage_balance", "final_direct_ratio",
                                                       "terminal_target", "future_completion_reachability"],
            "cost_scope": "initiated whole tasks charged once; transfer on actual delivery; "
                          "penalties final or expired; partial is not full-day objective"}
