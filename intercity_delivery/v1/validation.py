"""Independent full-horizon checks of main_v2_BHH.tex:82--137.

This module uses the original f(load), not the inverse/capacity kernel or MIP
expressions. It certifies only the explicit conditional V1-M profile.
"""

import ast
import math
from collections.abc import Mapping
from numbers import Integral, Real

from .configuration import PROFILE_DECISION_ID, PROFILE_SEMANTIC_STATUS, validate_instance


GROUPS = {"x_manual": 4, "y_auto": 3, "u_wave": 5,
          "g_manual": 5, "g_auto": 4, "z_unserved": 1}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be real, excluding bool")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is outside float range") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _finite(value, name):
    try:
        valid = math.isfinite(value)
    except (ValueError, OverflowError, TypeError) as exc:
        raise ValueError(f"{name} outside float range") from exc
    if not valid:
        raise ValueError(f"{name} overflow")
    return value


def _sum(values, name):
    try:
        return _finite(math.fsum(values), name)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} overflow") from exc


def _key(raw, size):
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (SyntaxError, ValueError, TypeError, RecursionError) as exc:
            raise ValueError("schedule key must encode a tuple") from exc
    if not isinstance(raw, tuple) or len(raw) != size:
        raise ValueError("schedule key has wrong tuple shape")
    try:
        hash(raw)
    except TypeError as exc:
        raise ValueError("schedule tuple key must be hashable") from exc
    return raw


def _node(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("nodes, cities and wave sizes must be integer, not bool")
    return int(value)


def validate_schedule(config, orders, profile, schedule, *, tolerance=1e-6,
                      reported_objective=None):
    """Raise ValueError on any failed domain/constraint/cost check.

    Missing keys within a group mean zero. All six groups must be present.
    Absolute tolerance lies in (0,1e-3]; near-integral vehicles/selectors are
    rounded only after checking the integer domain. Continuous positive values
    are retained regardless of their size. No relative tolerance scales with
    the demand, cost, fleet or horizon.
    """
    tolerance = _number(tolerance, "tolerance")
    if not 0 < tolerance <= 1e-3:
        raise ValueError("tolerance must lie in (0,1e-3]")
    validate_instance(config, orders, profile)
    orders = tuple(orders)
    by_id = {order.order_id: order for order in orders}
    if not isinstance(schedule, Mapping) or set(schedule) != set(GROUPS):
        raise ValueError("schedule must contain exactly the six V1-M groups")

    # Rebuild the source mother arc bounds directly from f(M), including arcs
    # with zero payload capacity. No import of network/capacity is involved.
    local_limits = {}
    for city in (1, 2):
        f_M = _finite(config.a[city] * config.M +
                      config.b[city] * math.sqrt(config.M), "f(M)")
        span = _finite((2.0 * config.rho[city] + f_M) /
                       config.t0_minutes, "local arc bound")
        local_limits[city] = math.ceil(span)

    groups = {}
    for name, size in GROUPS.items():
        raw_group = schedule[name]
        if not isinstance(raw_group, Mapping):
            raise ValueError(f"{name} must be a mapping")
        group = {}
        for raw, value in raw_group.items():
            key = _key(raw, size)
            if key in group:
                raise ValueError(f"duplicate normalized key in {name}")
            if name == "z_unserved":
                if not isinstance(key[0], str) or key[0] not in by_id:
                    raise ValueError("z references an unknown order")
            else:
                i, j = _node(key[0]), _node(key[1])
                if not 0 <= i < j <= config.T:
                    raise ValueError("arc outside the full horizon")
                if name in ("x_manual", "u_wave", "g_manual"):
                    city = _node(key[2])
                    flow = key[3]
                    if city not in (1, 2) or j - i > local_limits[city]:
                        raise ValueError("manual arc outside source mother domain")
                    if name == "u_wave":
                        v = _node(key[4])
                        if not 1 <= v <= config.N_manual[city]:
                            raise ValueError("wave size outside city fleet domain")
                    order_id = key[4] if name == "g_manual" else None
                else:
                    flow = key[2]
                    if j - i != config.tau_periods:
                        raise ValueError("AV arc has wrong transit duration")
                    order_id = key[3] if name == "g_auto" else None
                if not isinstance(flow, str) or flow not in ("+", "-"):
                    raise ValueError("unknown direction")
                if name in ("g_manual", "g_auto"):
                    if not isinstance(order_id, str) or order_id not in by_id:
                        raise ValueError("cargo references an unknown order")
                    if by_id[order_id].flow != flow:
                        raise ValueError("cargo direction disagrees with its order")
            value = _number(value, name)
            if value < -tolerance:
                raise ValueError(f"negative variable in {name}")
            value = max(0.0, value)
            if name in ("x_manual", "y_auto", "u_wave"):
                rounded = round(value)
                if abs(value - rounded) > min(tolerance, 1e-6):
                    raise ValueError(f"noninteger variable in {name}")
                if name == "u_wave" and rounded not in (0, 1):
                    raise ValueError("selector must be binary")
                value = rounded
            group[key] = value
        groups[name] = group

    x, y, u = (groups[name] for name in ("x_manual", "y_auto", "u_wave"))
    g, ga, z = (groups[name] for name in ("g_manual", "g_auto", "z_unserved"))

    def leq(left, right, name):
        left, right = _finite(left, name), _finite(right, name)
        if left > right + tolerance:
            raise ValueError(f"{name}: {left} > {right}")

    def equal(left, right, name):
        leq(left, right, name)
        leq(right, left, name)

    waves = set(x) | {key[:4] for key in u} | {key[:4] for key in g}
    for arc in waves:
        i, j, city, flow = arc
        selections = [(key[4], value) for key, value in u.items()
                      if key[:4] == arc]
        selected = _sum((value for _, value in selections), "one-hot sum")
        leq(selected, 1, "one-hot wave selection")
        count = _sum((v * value for v, value in selections), "wave count")
        equal(x.get(arc, 0), count, "wave vehicle link")
        load = _sum((value for key, value in g.items() if key[:4] == arc),
                    "wave load")
        leq(load, _finite(count * config.M, "wave physical capacity"),
            "manual shared physical capacity")
        duration = _finite((j - i) * config.t0_minutes, "manual duration")
        budget = _finite(count * max(duration - 2.0 * config.rho[city], 0.0),
                         "wave tour budget")
        f_load = _finite(config.a[city] * load +
                         config.b[city] * math.sqrt(load), "f(wave load)")
        leq(f_load, budget, "manual BHH tour budget")

    auto_arcs = set(y) | {key[:3] for key in ga}
    for arc in auto_arcs:
        load = _sum((value for key, value in ga.items() if key[:3] == arc),
                    "AV arc load")
        leq(load, _finite(config.M_auto * y.get(arc, 0), "AV capacity"),
            "shared AV capacity")

    peak_manual = {"1": 0.0, "2": 0.0}
    peak_auto = 0.0
    min_hub = {str(city): float(config.N_auto[city]) for city in (1, 2)}
    for time in range(config.T + 1):
        for city in (1, 2):
            active = _sum((value for (i, j, k, flow), value in x.items()
                           if k == city and i <= time < j), "manual occupancy")
            leq(active, config.N_manual[city], "manual city occupancy")
            peak_manual[str(city)] = max(peak_manual[str(city)], active)
        active = _sum((value for (i, j, flow), value in y.items()
                       if i <= time < j), "AV occupancy")
        leq(active, config.N_auto[1] + config.N_auto[2], "total AV occupancy")
        peak_auto = max(peak_auto, active)
        for city, outgoing in ((1, "+"), (2, "-")):
            departed = _sum((value for (i, j, flow), value in y.items()
                             if flow == outgoing and i <= time), "hub departures")
            arrived = _sum((value for (i, j, flow), value in y.items()
                           if flow != outgoing and j <= time), "hub arrivals")
            idle = _finite(config.N_auto[city] + arrived - departed, "hub idle")
            leq(0, idle, "AV hub inventory")
            min_hub[str(city)] = min(min_hub[str(city)], idle)

    order_evidence = {}
    for order in orders:
        oid = order.order_id
        origin = 1 if order.flow == "+" else 2
        destination = 3 - origin
        pickup = [(i, j, value) for (i, j, city, flow, item), value in g.items()
                  if item == oid and city == origin]
        delivery = [(i, j, value) for (i, j, city, flow, item), value in g.items()
                    if item == oid and city == destination]
        linehaul = [(i, j, value) for (i, j, flow, item), value in ga.items()
                    if item == oid]
        for i, j, value in pickup:
            if i < order.earliest_start:
                leq(value, 0, "pickup release window")
        for i, j, value in delivery:
            if j > order.latest_completion:
                leq(value, 0, "delivery deadline")
        for time in range(config.T + 1):
            picked = _sum((v for i, j, v in pickup if j <= time), "picked by t")
            loaded = _sum((v for i, j, v in linehaul if i <= time), "loaded by t")
            arrived = _sum((v for i, j, v in linehaul if j <= time), "arrived by t")
            dispatched = _sum((v for i, j, v in delivery if i <= time),
                              "distributed by t")
            leq(loaded, picked, "pickup before AV departure")
            leq(dispatched, arrived, "AV arrival before distribution")
        totals = {"pickup": _sum((v for i, j, v in pickup), "pickup total"),
                  "delivery": _sum((v for i, j, v in delivery), "delivery total"),
                  "linehaul": _sum((v for i, j, v in linehaul), "linehaul total")}
        unserved = z.get((oid,), 0.0)
        served = _finite(order.quantity - unserved, "served quantity")
        leq(0, served, "unserved exceeds demand")
        for leg, quantity in totals.items():
            equal(quantity, served, f"{leg} demand balance")
        totals.update({"served": served, "unserved": unserved})
        order_evidence[oid] = totals

    manual_cost = _sum((_finite(value * ((j - i) * config.t0_minutes / 60.0) *
                                 config.c_manual_hour, "manual cost term")
                        for (i, j, city, flow), value in x.items()), "manual cost")
    auto_cost = _sum((_finite(value * ((j - i) * config.t0_minutes / 60.0) *
                               config.c_auto_hour, "AV cost term")
                      for (i, j, flow), value in y.items()), "AV cost")
    penalty = _sum((_finite(z.get((order.order_id,), 0.0) * order.penalty,
                           "penalty term") for order in orders), "penalty cost")
    objective = _sum((manual_cost, auto_cost, penalty), "objective")
    if reported_objective is not None:
        equal(_number(reported_objective, "reported objective"), objective,
              "reported objective mismatch")
    return {"schedule_valid": True, "validator": "independent_original_f_v1_m",
            "semantic_status": PROFILE_SEMANTIC_STATUS,
            "profile_decision_id": PROFILE_DECISION_ID,
            "tolerance": tolerance, "comparison": "absolute",
            "integer_tolerance": min(tolerance, 1e-6),
            "costs": {"manual": manual_cost, "auto": auto_cost,
                      "penalty": penalty, "total": objective},
            "objective": objective, "orders": order_evidence,
            "peak_manual": peak_manual, "peak_auto": peak_auto,
            "min_AV_hub_inventory": min_hub,
            "checks": ["mother_arc_domains", "variable_domains", "one_hot",
                       "wave_vehicle_link", "original_f_joint_wave_capacity",
                       "shared_AV_capacity", "manual_occupancy", "AV_occupancy",
                       "AV_hub_inventory", "order_windows", "cargo_causality",
                       "all_three_leg_totals", "objective_hour_conversion"]}
