"""Conservative time-support pruning for V1-F cargo variables.

This is a necessary-condition filter, not a global optimality certificate.
Vehicle/selector arcs are retained by the optimizer. In particular, no city
availability test based only on historical inventory is used: vehicles may
arrive later on direct or independent empty movements.
"""

import math

from .capacity import local_capacity, _integer
from .network import av_arcs, local_arcs
from .flexible_capacity import direct_arcs, direct_capacity


CARGO_GROUPS = ("g_manual", "g_auto", "h_direct")


def prune_candidates(config, orders, current_time, start_end, completion_end,
                     committed):
    """Return sparse cargo keys and actual variable-removal counts.

    Future three-leg paths must have a pickup, AV and delivery with same-time
    handovers (plus the configured transfer delay). Previously committed
    pickups and AV trips seed downstream paths even when no new pickup fits.
    All nonzero historical cargo keys are retained for exact history fixing.
    Quantity and future vehicle availability are deliberately relaxed here.
    """
    current_time = _integer("current_time", current_time)
    start_end = _integer("start_end", start_end)
    completion_end = _integer("completion_end", completion_end)
    if not 0 <= current_time <= start_end <= completion_end <= config.T:
        raise ValueError("window bounds must be ordered within the real horizon")
    committed = committed or {}
    candidates = {name: set() for name in CARGO_GROUPS}
    manual = {city: local_arcs(
        config.T, config.t0_minutes, config.a[city], config.b[city],
        config.rho[city], config.M) for city in (1, 2)}
    auto = av_arcs(config.T, config.tau_periods)
    direct = direct_arcs(config)
    fleet = sum(config.N_manual.values())
    positive_manual = {
        city: [(i, j) for i, j in manual[city] if local_capacity(
            (j-i)*config.t0_minutes, fleet, config.a[city], config.b[city],
            config.rho[city], config.M) > 0]
        for city in (1, 2)
    }
    positive_direct = {
        flow: [(i, j) for i, j in direct
               if direct_capacity(config, i, j, flow, fleet) > 0]
        for flow in ("+", "-")
    }
    delay = config.transfer_time_periods
    raw = {"g_manual": len(orders)*sum(map(len, manual.values())),
           "g_auto": len(orders)*len(auto),
           "h_direct": len(orders)*len(direct)}
    for order in orders:
        oid, flow = order.order_id, order.flow
        origin = 1 if flow == "+" else 2
        destination = 3-origin
        # Positive cargo on historical tasks must survive every later window.
        historical_pickups = []
        historical_av = []
        for name in CARGO_GROUPS:
            for key, value in committed.get(name, {}).items():
                if key[-1] != oid or not value:
                    continue
                if not math.isfinite(value) or value < 0:
                    raise ValueError("historical cargo must be finite and nonnegative")
                candidates[name].add(tuple(key))
                if name == "g_manual" and key[2] == origin:
                    historical_pickups.append(key[1])
                elif name == "g_auto":
                    historical_av.append(key[1])
        if order.quantity <= 0:
            continue
        pickups = [(i, j) for i, j in positive_manual[origin]
                   if max(current_time, order.earliest_start) <= i < start_end
                   and j <= completion_end]
        deliveries = [(i, j) for i, j in positive_manual[destination]
                      if current_time <= i < completion_end
                      and j <= min(completion_end, order.latest_completion)]
        earliest_pickup = min([j for i, j in pickups]+historical_pickups,
                              default=math.inf)
        # An AV start is supported by a pickup and a possible later delivery.
        latest_delivery_start = max([i for i, j in deliveries], default=-math.inf)
        av_supported = [(i, j) for i, j in auto
                        if sum(config.N_auto.values()) > 0
                        and current_time <= i < completion_end
                        and j <= completion_end
                        and earliest_pickup+delay <= i
                        and j+delay <= latest_delivery_start]
        # Historical AV cargo can be delivered without a new pickup or AV trip.
        earliest_av_arrival = min([j for i, j in av_supported]+historical_av,
                                  default=math.inf)
        for i, j in deliveries:
            if earliest_av_arrival+delay <= i:
                candidates["g_manual"].add((i, j, destination, flow, oid))
        for i, j in av_supported:
            candidates["g_auto"].add((i, j, flow, oid))
        latest_supported_av = max([i for i, j in av_supported], default=-math.inf)
        for i, j in pickups:
            if j+delay <= latest_supported_av:
                candidates["g_manual"].add((i, j, origin, flow, oid))
        for i, j in positive_direct[flow]:
            if (max(current_time, order.earliest_start) <= i < start_end
                    and j <= min(completion_end, order.latest_completion)):
                candidates["h_direct"].add((i, j, flow, oid))
    kept = {name: len(values) for name, values in candidates.items()}
    removed = {name: raw[name]-kept[name] for name in CARGO_GROUPS}
    return {"candidates": candidates, "statistics": {
        "scope": "cargo_variables_only", "rule": "time_support_positive_capacity",
        "before": raw, "after": kept, "removed_by_group": removed,
        "removed": sum(removed.values()), "vehicle_arcs_retained": True,
        "selector_bound": "global_manual_fleet",
        "global_safety_proof": False,
    }}
