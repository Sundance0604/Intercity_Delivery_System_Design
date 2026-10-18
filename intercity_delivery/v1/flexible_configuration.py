"""Explicit V1-F extensions to the adopted arc-wave profile (minutes/hour)."""

from dataclasses import dataclass, replace

from .capacity import _integer, _real
from .configuration import V1MConfig, V1MProfile, _city_mapping, validate_instance


@dataclass(frozen=True)
class V1FConfig(V1MConfig):
    c_direct_hour: float
    empty_tau_periods: int | None = None
    c_empty_hour: float | None = None
    terminal_target: dict | None = None
    transfer_time_periods: int = 0
    transfer_cost_per_unit: float = 0
    direct_ratio_min: float = 0
    direct_ratio_max: float = 1


def flexible_profile(information="offline"):
    return V1MProfile(information, "arc", "continuous_stops_zero_allowed",
                      "minutes_hourly", "tour_end_same_time",
                      "integer_windows_within_horizon")


def validate_f_instance(config, orders, profile):
    if not isinstance(config, V1FConfig):
        raise ValueError("config must be V1FConfig")
    if not isinstance(profile, V1MProfile) or profile.information not in (
            "offline", "release_revealed"):
        raise ValueError("V1-F information must be offline or release_revealed")
    validate_instance(config, orders, replace(profile, information="offline"))
    _real("c_direct_hour", config.c_direct_hour)
    if config.empty_tau_periods is not None:
        _integer("empty_tau_periods", config.empty_tau_periods, positive=True)
    if config.c_empty_hour is not None:
        _real("c_empty_hour", config.c_empty_hour)
    _integer("transfer_time_periods", config.transfer_time_periods)
    _real("transfer_cost_per_unit", config.transfer_cost_per_unit)
    low = _real("direct_ratio_min", config.direct_ratio_min)
    high = _real("direct_ratio_max", config.direct_ratio_max)
    if not 0 <= low <= high <= 1:
        raise ValueError("direct ratios must satisfy 0 <= min <= max <= 1")
    if config.terminal_target is not None:
        target = _city_mapping("terminal_target", config.terminal_target)
        for value in target.values():
            _integer("terminal_target vehicles", value)
        if sum(target.values()) != sum(config.N_manual.values()):
            raise ValueError("terminal_target must conserve the manual fleet")


def empty_duration(config):
    return config.tau_periods if config.empty_tau_periods is None else config.empty_tau_periods


def empty_hourly_cost(config):
    return config.c_direct_hour if config.c_empty_hour is None else config.c_empty_hour
