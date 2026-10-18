"""Explicit inputs for the author-confirmed V1 mandatory benchmark.

The six profile choices remain required. The author adopted D1--D6 option A
on 2026-10-18; that decision does not amend the manuscript or cover F/RH/QVI.
All physical time coefficients use minutes; vehicle rates use currency/hour.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

from .capacity import _integer, _real


PROFILE_DECISION_ID = "V1M-D1-D6-A-2026-10-18"
PROFILE_SEMANTIC_STATUS = "author_confirmed_profile"


@dataclass(frozen=True)
class V1MProfile:
    information: str
    wave: str
    demand: str
    units: str
    service: str
    input_domain: str


@dataclass(frozen=True)
class V1MConfig:
    T: int
    t0_minutes: float
    tau_periods: int
    N_manual: dict
    N_auto: dict
    a: dict
    b: dict
    rho: dict
    M: float
    M_auto: float
    c_manual_hour: float
    c_auto_hour: float


@dataclass(frozen=True)
class V1Order:
    order_id: str
    flow: str
    quantity: float
    earliest_start: int
    latest_completion: int
    penalty: float


def _city_mapping(name, value):
    if (not isinstance(value, Mapping) or set(value) != {1, 2}
            or any(isinstance(k, bool) or not isinstance(k, Integral) for k in value)):
        raise ValueError(f"{name} must have exactly integer city keys 1 and 2")
    return value


def validate_instance(config, orders, profile):
    """Reject unsupported semantics or malformed inputs, without coercing data.

    Orders must be a list/tuple, so a one-shot iterable cannot be consumed by
    validation and disappear before optimization or independent checking.
    """
    if not isinstance(config, V1MConfig):
        raise ValueError("config must be a V1MConfig")
    if not isinstance(profile, V1MProfile):
        raise ValueError("an explicit V1MProfile is required")
    expected = {
        "information": "offline", "wave": "arc",
        "demand": "continuous_stops_zero_allowed", "units": "minutes_hourly",
        "service": "tour_end_same_time",
        "input_domain": "integer_windows_within_horizon",
    }
    for name, choice in expected.items():
        if getattr(profile, name) != choice:
            raise ValueError(f"unsupported conditional profile {name}: expected {choice!r}")
    _integer("T", config.T, positive=True)
    _integer("tau_periods", config.tau_periods, positive=True)
    for name in ("t0_minutes", "M", "M_auto"):
        _real(name, getattr(config, name), positive=True)
    for name in ("c_manual_hour", "c_auto_hour"):
        _real(name, getattr(config, name))
    for name in ("N_manual", "N_auto", "a", "b", "rho"):
        values = _city_mapping(name, getattr(config, name))
        for city, value in values.items():
            if name in ("N_manual", "N_auto"):
                _integer(f"{name}[{city}]", value)
            else:
                _real(f"{name}[{city}]", value, positive=name == "a")
    if not isinstance(orders, (list, tuple)):
        raise ValueError("orders must be a list or tuple of V1Order objects")
    seen = set()
    for order in orders:
        if not isinstance(order, V1Order):
            raise ValueError("each order must be a V1Order")
        if not isinstance(order.order_id, str) or not order.order_id.strip():
            raise ValueError("order_id must be a nonempty string")
        if order.order_id in seen:
            raise ValueError(f"duplicate order_id {order.order_id!r}")
        seen.add(order.order_id)
        if order.flow not in ("+", "-"):
            raise ValueError("order flow must be '+' or '-'")
        _real("order quantity", order.quantity)
        _real("order penalty", order.penalty)
        start = _integer("earliest_start", order.earliest_start)
        end = _integer("latest_completion", order.latest_completion)
        if not 0 <= start <= end <= config.T:
            raise ValueError("order windows must satisfy 0 <= s <= e <= T")
