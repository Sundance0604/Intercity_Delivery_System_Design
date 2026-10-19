"""Budgeted release-revealed V1-F execution with absolute commitments."""

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import math
import time

from .capacity import _integer
from .flexible_configuration import flexible_profile, validate_f_instance
from .flexible_optimizer import V1FlexibleOptimizer
from .pruning import prune_candidates


INTEGER_GROUPS = frozenset(("x_manual", "y_auto", "w_direct", "empty_vehicles",
                            "u_wave", "u_direct"))


PHASES = ("setup_provenance", "candidate", "build", "configure", "solve",
          "extract_commit", "replay_accounting", "other")


def mother_domain_sha256(groups):
    """Hash vehicle/selector tuple keys only, excluding bounds and history."""
    names = ("x_manual", "y_auto", "w_direct", "empty_vehicles", "u_wave", "u_direct")
    if any(name not in groups for name in names):
        return {"vehicle_selector_domain_sha256": None,
                "vehicle_selector_domain_counts": {name: len(groups[name]) if name in groups else None for name in names}}
    domains = {name: [list(key) for key in sorted(groups[name])] for name in names}
    canonical = json.dumps(domains, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"vehicle_selector_domain_sha256": hashlib.sha256(canonical).hexdigest(),
            "vehicle_selector_domain_counts": {name: len(domains[name]) for name in names}}


class PhaseTimer:
    """Exclusive wall-time intervals; callers must not nest measurements."""
    def __init__(self, started):
        self.started = started
        self.values = dict.fromkeys(PHASES, 0.)

    @contextmanager
    def measure(self, phase):
        begin = time.monotonic()
        try:
            yield
        finally:
            self.values[phase] += time.monotonic()-begin

    def finish(self):
        elapsed = time.monotonic()-self.started
        values = dict(self.values)
        values["other"] = max(0., elapsed-sum(v for k, v in values.items() if k != "other"))
        return elapsed, values


def model_counts(model):
    values = {}
    for key, attribute in (("variable_count", "NumVars"), ("binary_variable_count", "NumBinVars"),
                           ("integer_variable_count", "NumIntVars"), ("constraint_count", "NumConstrs")):
        value = _optional_float(model, attribute)
        values[key] = int(value) if value is not None else None
    values["continuous_variable_count"] = (values["variable_count"]-values["integer_variable_count"]
        if values["variable_count"] is not None and values["integer_variable_count"] is not None else None)
    return values


@dataclass(frozen=True)
class RollingOptions:
    prediction_horizon: int = 8
    rolling_step: int = 2
    extension_horizon: int = 6
    prune: bool = False

    def validate(self):
        _integer("prediction_horizon", self.prediction_horizon, positive=True)
        _integer("rolling_step", self.rolling_step, positive=True)
        _integer("extension_horizon", self.extension_horizon)
        if self.rolling_step > self.prediction_horizon:
            raise ValueError("rolling_step must not exceed prediction_horizon")
        if not isinstance(self.prune, bool):
            raise ValueError("prune must be bool")


def revealed_orders(orders, current_time):
    """Past deadlines do not erase orders or cargo obligations."""
    return tuple(order for order in orders if order.earliest_start <= current_time)


def commit_schedule(schedule, start, end):
    """Commit starts in the control interval; terminal z is never executed."""
    result = {}
    for name, values in schedule.items():
        if name == "z_unserved":
            continue
        result[name] = {}
        for key, raw in values.items():
            if not start <= key[0] < end:
                continue
            value = float(raw)
            if not math.isfinite(value) or value < -1e-8:
                raise ValueError("commitments must be finite and nonnegative")
            if name in INTEGER_GROUPS:
                rounded = round(value)
                if abs(value-rounded) > 1e-6:
                    raise ValueError("non-integral vehicle/selector commitment")
                value = float(rounded)
            # Preserve every strictly positive continuous shipment. A scale-
            # dependent 0.1 cutoff (or even 1e-8 cutoff) would lose small cargo.
            if value > 0:
                result[name][tuple(key)] = value
    return result


def _optional_float(model, attribute):
    try:
        value = float(getattr(model, attribute))
        return value if math.isfinite(value) else None
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None


def solve_flexible_rolling(config, orders, options, time_limit, profile=None,
                           output_flag=0):
    """Execute a single total-budget run; window bounds stay window-local."""
    started = time.monotonic()
    timer = PhaseTimer(started)
    setup_started = time.monotonic()
    if (isinstance(time_limit, bool) or not isinstance(time_limit, (int, float))
            or not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be finite and positive")
    if not isinstance(options, RollingOptions):
        raise ValueError("options must be RollingOptions")
    options.validate()
    profile = profile or flexible_profile(information="release_revealed")
    if profile.information != "release_revealed":
        raise ValueError("rolling requires the release_revealed information profile")
    validate_f_instance(config, orders, profile)
    if config.terminal_target is not None:
        raise ValueError("rolling terminal_target requires a safe real-T reachability tail")
    committed, windows = {}, []
    optimize_attempts = 0
    current = 0
    status = "PARTIAL_TIME_LIMIT"
    message = "Total runtime budget exhausted; execution is partial."
    timer.values["setup_provenance"] += time.monotonic()-setup_started
    while current < config.T:
        if time.monotonic()-started >= time_limit:
            break
        start_end = min(config.T, current+options.prediction_horizon)
        window_timer = PhaseTimer(time.monotonic())
        row = None
        completion_end = min(config.T, start_end+options.extension_horizon)
        control_end = min(config.T, current+options.rolling_step)
        known = revealed_orders(orders, current)
        candidate_started = time.monotonic()
        filtered = (prune_candidates(config, known, current, start_end,
                                     completion_end, committed)
                    if options.prune else None)
        duration = time.monotonic()-candidate_started
        timer.values["candidate"] += duration
        window_timer.values["candidate"] += duration
        # Building and pruning both consume the same run budget as optimize.
        if time.monotonic()-started >= time_limit:
            break
        with timer.measure("build"), window_timer.measure("build"):
            optimizer = V1FlexibleOptimizer(
                config, known, profile,
                candidates=filtered["candidates"] if filtered else None).build_model()
        try:
            with timer.measure("configure"), window_timer.measure("configure"):
                optimizer.configure_window(current, start_end, completion_end, committed)
                optimizer.model.setParam("OutputFlag", output_flag)
                optimizer.model.setParam("Threads", 1)
                optimizer.model.setParam("Seed", 0)
            remaining = time_limit-(time.monotonic()-started)
            if remaining <= 0:
                break
            optimizer.model.setParam("TimeLimit", remaining)
            optimize_attempts += 1
            with timer.measure("solve"), window_timer.measure("solve"):
                try:
                    optimizer.model.optimize()
                except Exception as exc:
                    exc.solver_optimize_attempts = optimize_attempts
                    raise
            feasible = optimizer.model.SolCount > 0
            row = {
                "window_start": current, "start_window_end": start_end,
                "completion_window_end": completion_end, "control_end": control_end,
                "known_order_ids": sorted(order.order_id for order in known),
                "status": optimizer.model.Status, "feasible": feasible,
                "objective": _optional_float(optimizer.model, "ObjVal") if feasible else None,
                "best_bound": _optional_float(optimizer.model, "ObjBound"),
                "mip_gap": _optional_float(optimizer.model, "MIPGap") if feasible else None,
                "bound_scope": "window_local_only",
                "pruning": filtered["statistics"] if filtered else None,
                "cargo_variable_count": sum(len(optimizer.groups[name]) for name in
                                            ("g_manual", "g_auto", "h_direct")),
                **model_counts(optimizer.model),
                **mother_domain_sha256(optimizer.groups),
            }
            windows.append(row)
            if not feasible:
                status = "PARTIAL_NO_INCUMBENT"
                message = "A window has no incumbent; execution is partial."
                break
            with timer.measure("extract_commit"), window_timer.measure("extract_commit"):
                additions = commit_schedule(optimizer.extract_schedule(), current, control_end)
                for name, values in additions.items():
                    committed.setdefault(name, {}).update(values)
            current = control_end
        finally:
            optimizer.model.dispose()
            if row is not None:
                row["elapsed_sec"], row["phase_timing_sec"] = window_timer.finish()
    if current == config.T:
        status = "COMPLETED"
        message = "Rolling execution reached the real horizon."
    elapsed, phases = timer.finish()
    return {"schedule": committed, "windows": windows, "current_time": current,
            "completed": current == config.T, "status": status,
            "elapsed_sec": elapsed, "phase_timing_sec": phases, "message": message,
            "solver_optimize_attempts": optimize_attempts,
            "profile": profile}
