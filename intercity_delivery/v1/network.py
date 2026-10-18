"""Source mother arcs and half-open occupancy, main_v2_BHH.tex:82.

No order filtering, city/direction policy, fleet bound, wave grouping, or costs.
"""

import math

from .capacity import _finite, _integer, _real, bhh_time


def local_arcs(horizon, t0, a, b, rho, capacity):
    """Return sorted local (i,j) tuples in nodes 0..horizon.

    Bound: ceil((2*rho+f(capacity))/t0), strictly on the computed float ratio;
    no epsilon, round, or isclose correction. Zero-capacity arcs remain.
    horizon=0 is a tool-domain extension.
    """
    horizon = _integer("horizon", horizon)
    t0 = _real("t0", t0, positive=True)
    a = _real("a", a, positive=True)
    b = _real("b", b)
    rho = _real("rho", rho)
    capacity = _real("capacity", capacity, positive=True)
    stem = _finite("2*rho", 2.0 * rho)
    time_bound = _finite("local time bound", stem + bhh_time(capacity, a, b))
    ratio = _finite("local period bound", time_bound / t0)
    max_duration = min(horizon, math.ceil(ratio))
    return tuple((i, j) for i in range(horizon)
                 for j in range(i + 1, min(horizon, i + max_duration) + 1))


def av_arcs(horizon, tau_periods):
    """Return sorted (i,i+tau_periods) inside 0..horizon; tau is periods."""
    horizon = _integer("horizon", horizon)
    tau_periods = _integer("tau_periods", tau_periods, positive=True)
    return tuple((i, i + tau_periods)
                 for i in range(max(0, horizon - tau_periods + 1)))


def _arc(arc):
    try:
        start, end = arc
    except (TypeError, ValueError) as exc:
        raise ValueError("arc must contain exactly two integer nodes") from exc
    start = _integer("arc start", start)
    end = _integer("arc end", end)
    if end <= start:
        raise ValueError("arc end must be strictly after its start")
    return start, end


def is_active(arc, time):
    """Return i<=time<j; input nodes/time are nonnegative integers."""
    start, end = _arc(arc)
    time = _integer("time", time)
    return start <= time < end


def active_arcs(arcs, time):
    """Return active arcs as sorted unique tuples, validating every input arc."""
    time = _integer("time", time)
    normalized = {_arc(arc) for arc in arcs}
    return tuple(sorted(arc for arc in normalized if arc[0] <= time < arc[1]))
