"""V1-F direction-dependent direct wave capacity, main_v2_BHH.tex:141."""

import math

from .capacity import _finite, _integer, bhh_inverse, bhh_time
from .flexible_configuration import empty_duration
from .network import av_arcs


def direct_capacity(config, i, j, flow, vehicles):
    i = _integer("start", i)
    j = _integer("end", j)
    v = _integer("vehicles", vehicles)
    if flow not in ("+", "-") or j <= i:
        raise ValueError("direct arc must have a direction and end after start")
    if v == 0 or j-i <= config.tau_periods:
        return 0.0
    origin = 1 if flow == "+" else 2
    destination = 3-origin
    budget = (j-i-config.tau_periods)*config.t0_minutes-2*config.rho[origin]
    if budget <= 0:
        return 0.0
    a = _finite("direct a", (config.a[origin]+config.a[destination])/v)
    b = _finite("direct b", config.b[origin]/v+config.b[destination]/math.sqrt(v))
    return min(_finite("direct load cap", v*config.M), bhh_inverse(budget, a, b))


def direct_arcs(config):
    physical = _finite("direct physical bound", 2*max(config.rho.values())+
                       sum(bhh_time(config.M, config.a[k], config.b[k]) for k in (1, 2)))
    length = config.tau_periods+math.ceil(_finite("direct period bound", physical/config.t0_minutes))
    return tuple((i, j) for i in range(config.T)
                 for j in range(i+config.tau_periods, min(config.T, i+length)+1))


def empty_arcs(config):
    return av_arcs(config.T, empty_duration(config))
