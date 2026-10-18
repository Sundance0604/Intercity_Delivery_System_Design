"""V1 f/h/K from main_v2_BHH.tex:71,74,82; no configuration imports.

All time inputs must share a physical unit. Real inputs are converted to binary
floats. Finite input is necessary but not sufficient: an unrepresentable
intermediate/output raises ValueError. Underflow follows Python float rules.
"""

import math
from numbers import Integral, Real


def _real(name, value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, excluding bool")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is outside the supported float range") from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        relation = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {relation}")
    return result


def _integer(name, value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, excluding bool")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} is outside its permitted integer domain")
    return int(value)


def _finite(name, value):
    if not math.isfinite(value):
        raise ValueError(f"{name} exceeds the supported float range")
    return value


def bhh_time(q, a, b):
    """Return f(q)=a*q+b*sqrt(q), without stem; q,b>=0 and a>0."""
    q = _real("q", q)
    a = _real("a", a, positive=True)
    b = _real("b", b)
    handling = _finite("a*q", a * q)
    routing = _finite("b*sqrt(q)", b * math.sqrt(q))
    return _finite("f(q)", handling + routing)


def bhh_inverse(theta, a, b):
    """Return h(theta)=f^-1(theta) using a cancellation-resistant root.

    Rationalization gives sqrt(h)=theta/(sqrt(b*b+4*a*theta)/2+b/2).
    hypot avoids squaring b. b=0 uses h=theta/a, including theta=0.
    """
    theta = _real("theta", theta)
    a = _real("a", a, positive=True)
    b = _real("b", b)
    if theta == 0:
        return 0.0
    if b == 0:
        return _finite("theta/a", theta / a)
    cross = _finite("2*sqrt(a)*sqrt(theta)",
                    2.0 * (math.sqrt(a) * math.sqrt(theta)))
    discriminant_root = _finite("discriminant root", math.hypot(b, cross))
    denominator = discriminant_root / 2.0 + b / 2.0
    if denominator == 0:
        raise ValueError("inverse denominator underflowed to zero")
    root = _finite("sqrt(h)", theta / denominator)
    return _finite("h(theta)", root * root)


def local_capacity(duration, vehicles, a, b, rho, capacity):
    """Return min(v*M,h(v*max(duration-2*rho,0))) in stops.

    duration/rho/a/b share a physical time unit. vehicles is a nonnegative
    integer; K(0)=0 is a tool-domain extension. M must be strictly positive.
    The stem is subtracted once per vehicle's available tour duration.
    """
    duration = _real("duration", duration)
    vehicles = _integer("vehicles", vehicles)
    a = _real("a", a, positive=True)
    b = _real("b", b)
    rho = _real("rho", rho)
    capacity = _real("capacity", capacity, positive=True)
    if vehicles == 0:
        return 0.0
    v = _real("vehicles", vehicles)
    cap = _finite("vehicles*capacity", v * capacity)
    stem = _finite("2*rho", 2.0 * rho)
    tour_budget = max(duration - stem, 0.0)
    budget = _finite("vehicles*tour_budget", v * tour_budget)
    return min(cap, bhh_inverse(budget, a, b))
