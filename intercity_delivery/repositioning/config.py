"""Explicit opt-in policy; defaults are disclosed compatibility assumptions."""

from dataclasses import dataclass, field
import math
from numbers import Real


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def fleet(value, name):
    if not isinstance(value, dict) or set(value) != {1, 2}:
        raise ValueError(f"{name} must map cities 1 and 2 to integer counts")
    # bool keys compare equal to integer keys, so check their actual type too.
    if any(type(key) is not int for key in value):
        raise ValueError(f"{name} city keys must be integers")
    return {key: integer(count, name) for key, count in value.items()}


@dataclass(frozen=True)
class RepositioningPolicy:
    empty_travel_time_periods: int | None = None
    empty_cost_per_hour: float | None = None
    terminal_target: dict | None = None
    _sources: tuple = field(default=(), repr=False, compare=False)

    def resolve(self, cfg):
        integer(cfg.T, "T", 1)
        fleet(cfg.N_manual, "N_manual")
        fleet(cfg.N_auto, "N_auto")
        for name in ("t_0", "cost_manual", "cost_auto", "cost_direct",
                     "transfer_cost_per_unit"):
            value = nonnegative(getattr(cfg, name), name)
            if name == "t_0" and value <= 0:
                raise ValueError("t_0 must be positive")
        tau = (cfg.direct_travel_time_periods if self.empty_travel_time_periods is None
               else self.empty_travel_time_periods)
        cost = (cfg.cost_direct if self.empty_cost_per_hour is None
                else self.empty_cost_per_hour)
        integer(tau, "empty_travel_time_periods", 1)
        cost = nonnegative(cost, "empty_cost_per_hour")
        target = None if self.terminal_target is None else fleet(self.terminal_target, "terminal_target")
        if target is not None and sum(target.values()) != sum(cfg.N_manual.values()):
            raise ValueError("terminal_target must conserve the manual fleet")
        sources = self._sources or (
            "cfg.direct_travel_time_periods" if self.empty_travel_time_periods is None else "explicit",
            "cfg.cost_direct" if self.empty_cost_per_hour is None else "explicit",
        )
        return RepositioningPolicy(tau, cost, target, sources)

    def metadata(self):
        return {
            "empty_travel_time_periods": self.empty_travel_time_periods,
            "empty_cost_per_hour": self.empty_cost_per_hour,
            "terminal_target": self.terminal_target,
            "empty_time_source": self._sources[0] if self._sources else "unresolved",
            "empty_cost_source": self._sources[1] if self._sources else "unresolved",
            "cost_units": "currency per vehicle hour; period length in minutes",
            "terminal_policy": "explicit real-horizon target" if self.terminal_target is not None else "no target specified",
            "completion_policy": "finite completion domain; retain absolute commitments",
        }
