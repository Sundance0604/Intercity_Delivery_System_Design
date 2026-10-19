"""Unified isolated V1-F full/RH runners with independent physical replay."""

from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
from numbers import Integral
from pathlib import Path
import platform
import subprocess
import sys
import time

from .capacity import _real
from .configuration import PROFILE_DECISION_ID
from .flexible_configuration import (empty_duration, empty_hourly_cost,
                                     flexible_profile, validate_f_instance)
from .flexible_validation import GROUPS, validate_flexible_schedule
from .runner import _optional_number, _plain
from .flexible_rolling import PhaseTimer, model_counts, mother_domain_sha256


MODEL_ID = "v1_flexible_arc_wave_empty_v1"


def _source_binding(root):
    """Verify enumerated bytes without asserting mathematical equivalence."""
    relative_manifest = "experiments/specs/v1_source_manifest.json"
    binding = {"manifest_path": relative_manifest, "manifest_sha256": None,
               "manifest": None, "current_sources_match": None, "verification_errors": [],
               "scope": "source hashes recorded only; PDF/source version equivalence is not certified"}
    path = root / relative_manifest
    if not path.is_file():
        binding["verification_errors"].append("source manifest not present")
        return binding
    errors = binding["verification_errors"]
    try:
        raw = path.read_bytes()
        binding["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        binding["manifest"] = manifest
        if not isinstance(manifest, dict) or manifest.get("schema_version") != "v1_p0_source_manifest/1":
            raise ValueError("unsupported source manifest schema")
        entries = manifest.get("files")
        if not isinstance(entries, dict) or not entries:
            raise ValueError("missing source entries")
        canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != manifest.get("content_sha256"):
            errors.append("manifest content fingerprint differs")
        for relative, expected in entries.items():
            source = Path(relative)
            resolved = (root / source).resolve()
            if source.is_absolute() or not resolved.is_relative_to(root.resolve()):
                errors.append(f"source outside project or not relative: {relative}")
                continue
            try:
                content = resolved.read_bytes()
            except OSError:
                errors.append(f"unreadable source: {relative}")
                continue
            actual = {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
            if actual != expected:
                errors.append(f"source bytes differ: {relative}")
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"source manifest verification failed: {exc}")
    binding["current_sources_match"] = not errors
    return binding


def _provenance(config, orders, profile):
    root = Path(__file__).resolve().parents[2]
    paths = ["research/paper/main_v2_BHH.tex", "coordination/V1_F_RUNTIME_CONTRACT.md",
             "coordination/V1_M_IMPLEMENTATION_PROFILE.md"]
    paths += [f"intercity_delivery/v1/{name}.py" for name in (
        "capacity", "network", "configuration", "flexible_configuration",
        "flexible_capacity", "flexible_optimizer", "flexible_validation",
        "flexible_runner", "flexible_rolling", "pruning", "adapter")]
    manuscript = root / "research/paper/final_manuscript"
    paths += sorted(p.relative_to(root).as_posix() for p in manuscript.rglob("*")
                    if p.is_file() and p.suffix.lower() in (".tex", ".bib", ".pdf"))
    hashes = {p: hashlib.sha256((root/p).read_bytes()).hexdigest()
              if (root/p).is_file() else None for p in paths}
    source_binding = _source_binding(root)
    inputs = _plain({"config": config, "orders": orders, "profile": profile})
    encoded = json.dumps(inputs, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                  capture_output=True, text=True, timeout=1,
                                  check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    try:
        version = package_metadata.version("gurobipy")
    except package_metadata.PackageNotFoundError:
        version = None
    return {"input": inputs, "input_sha256": hashlib.sha256(encoded).hexdigest(),
            "source_sha256": hashes, "source_binding": source_binding, "git_revision": revision,
            "git_revision_scope": "HEAD only; SHA256 includes uncommitted implementation",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "platform": platform.platform(),
            "gurobipy_version": version}


def _metadata(config, orders, profile, mode, options=None):
    decision_path = Path(__file__).resolve().parents[2] / "experiments/specs/v1_p0_decisions.json"
    endpoint = {"status": "UNAVAILABLE", "scope": "metadata only; does not certify physical routes"}
    if decision_path.is_file():
        raw_decisions = decision_path.read_bytes()
        endpoint.update(json.loads(raw_decisions).get("direct_endpoint", {}))
        endpoint["decision_file_sha256"] = hashlib.sha256(raw_decisions).hexdigest()
    return {"provenance": _provenance(config, orders, profile),
            "direct_endpoint_author_decision": endpoint,
            "phase_timing_scope": "exclusive wall-time partition within runner API; upstream input loading excluded; window timings are nested diagnostics, not additive to run phases",
            "vehicle_selector_domain_signature_scope": "all tuple keys of x_manual/y_auto/w_direct/empty_vehicles/u_wave/u_direct; excludes cargo, values, bounds, window restrictions and history; equal signatures alone do not certify identical subproblems",
            "model_scope": "adopted V1-F arc-wave + independent empty; excludes S/QVI",
            "profile_decision_id": PROFILE_DECISION_ID, "profile": _plain(profile),
            "information": profile.information, "run_mode": mode,
            "service_mechanism": config.service_mechanism,
            "min_direct_load_per_vehicle": config.min_direct_load_per_vehicle,
            "mechanism_policy": {"direct_enabled": config.service_mechanism in ("T2", "T3"),
                                 "independent_empty_enabled": config.service_mechanism in ("T1", "T3"),
                                 "minimum_load_scope": "aggregate cargo per direct arc >= minimum times dispatched vehicles; synthetic policy when positive",
                                 "zero_cargo_direct_allowed": config.service_mechanism in ("T2", "T3") and config.min_direct_load_per_vehicle == 0},
            "wave_grouping": "absolute (start,end,city,direction)",
            "selector_bound": "global_manual_fleet",
            "selector_bound_reason": "migrated vehicles can serve local waves",
            "quantity_units": "continuous service-equivalents; input contract identifies job/stop proxy and conversion",
            "time_units": "minutes", "vehicle_rate_units": "currency/hour",
            "cost_units": "currency; minutes divided by 60 exactly once",
            "window_terminal_reset": False, "terminal_target": _plain(config.terminal_target),
            "terminal_default": "none; no forced real-T city reset unless explicit full-run target",
            "empty_tau_periods": empty_duration(config),
            "empty_duration_source": "tau_periods fallback" if config.empty_tau_periods is None else "explicit",
            "c_empty_hour": empty_hourly_cost(config),
            "empty_cost_source": "c_direct_hour fallback" if config.c_empty_hour is None else "explicit",
            "compatibility_extensions": {"transfer_time_periods": config.transfer_time_periods,
                                         "transfer_cost_per_unit": config.transfer_cost_per_unit,
                                         "direct_ratio_min": config.direct_ratio_min,
                                         "direct_ratio_max": config.direct_ratio_max},
            "candidate_scope": "full mother domains" if options is None or not options.prune else
                               "conservative release/deadline/capacity/shortest-chain cargo filtering",
            "completion_policy": "new tasks finish within completion window; prior commitments persist",
            "rolling_options": _plain(options), "threads": 1, "solver_seed": 0,
            "validation_tolerance": 1e-6,
            "pruning_evidence_scope": "finite checks do not prove general pruning equivalence"}


def _base(config, orders, profile, mode, budget, options=None):
    return {"schema_version": "v1_flexible_result/1", "model_id": MODEL_ID,
            "semantic_status": "author_confirmed_profile_with_disclosed_vehicle_extensions",
            "status": None, "status_name": "NOT_RUN", "completed": False,
            "has_incumbent": False, "optimal": False, "schedule_valid": False,
            "total_cost": None, "best_bound": None, "mip_gap": None,
            "bound_scope": "full_horizon_adopted_V1_F_profile_only" if mode == "full" else "none_window_local_only",
            "elapsed_sec": 0., "requested_time_limit_sec": budget,
            "solver_optimize_attempts": 0,
            "schedule": None, "validation": None, "windows": [],
            "metadata": _metadata(config, orders, profile, mode, options),
            "served_quantity": 0., "unserved_quantity": None, "direct_quantity": 0.,
            "transshipment_quantity": 0., "empty_trip_count": 0.,
            "auto_usage": 0., "manual_usage": 0., "warnings": []}


def _replay_result(result, schedule, evidence):
    result.update({"schedule": {name: {str(tuple(k)): float(v) for k, v in values.items()}
                                for name, values in schedule.items()},
                   "validation": evidence, "schedule_valid": True,
                   "total_cost": evidence["objective"],
                   "cost_components": evidence["costs"],
                   "served_quantity": evidence["served_quantity"],
                   "unserved_quantity": evidence["unserved_quantity"],
                   "direct_quantity": evidence["direct_quantity"],
                   "transshipment_quantity": evidence["transfer_quantity"],
                   "empty_trip_count": evidence["empty_trip_count"],
                   "auto_usage": sum(schedule["y_auto"].values()),
                   "manual_usage": sum(sum(schedule[n].values()) for n in
                                       ("x_manual", "w_direct", "empty_vehicles"))})
    result["metadata"]["vehicle_usage_scope"] = "dispatched vehicle-task counts, not unique vehicles"


def _validate_runtime(time_limit, output_flag):
    budget = _real("time_limit", time_limit, positive=True)
    if isinstance(output_flag, bool) or not isinstance(output_flag, Integral) or output_flag not in (0, 1):
        raise ValueError("output_flag must be integer 0 or 1")
    return budget


def run_flexible(config, orders, profile, time_limit, output_flag=0):
    """One full-horizon run, with total build+solve budget and full replay."""
    from .flexible_optimizer import V1FlexibleOptimizer
    budget = _validate_runtime(time_limit, output_flag)
    started = time.monotonic()
    timer = PhaseTimer(started)
    validate_f_instance(config, orders, profile)
    if profile.information != "offline":
        raise ValueError("full runner requires offline all-orders-known profile")
    result = _base(config, orders, profile, "full", budget)
    timer.values["setup_provenance"] = time.monotonic()-started
    optimizer = None
    try:
        with timer.measure("build"):
            optimizer = V1FlexibleOptimizer(config, orders, profile)
            optimizer.build_model()
        model = optimizer.model
        with timer.measure("configure"):
            model.Params.OutputFlag = output_flag
            model.Params.Threads = 1
            model.Params.Seed = 0
        remaining = budget-(time.monotonic()-started)
        result["build_elapsed_sec"] = time.monotonic()-started
        result["solver_time_limit_sec"] = max(0., remaining)
        if remaining > 0:
            model.Params.TimeLimit = remaining
            result["solver_optimize_attempts"] = 1
            with timer.measure("solve"):
                model.optimize()
            result["status"] = int(model.Status)
            result["status_name"] = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
                5: "UNBOUNDED", 9: "TIME_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC",
                13: "SUBOPTIMAL"}.get(int(model.Status), f"GUROBI_{model.Status}")
        else:
            result["status_name"] = "BUDGET_EXHAUSTED_DURING_BUILD"
        result["best_bound"] = _optional_number(model, "ObjBound")
        result["solver_runtime_sec"] = _optional_number(model, "Runtime")
        result["variable_count"] = int(model.NumVars)
        result["constraint_count"] = int(model.NumConstrs)
        result.update(model_counts(model))
        if hasattr(optimizer, "groups"):
            result.update(mother_domain_sha256(optimizer.groups))
        if model.SolCount > 0:
            with timer.measure("extract_commit"):
                schedule = optimizer.extract_schedule()
            with timer.measure("replay_accounting"):
                evidence = validate_flexible_schedule(config, orders, profile, schedule,
                    reported_objective=float(model.ObjVal))
                _replay_result(result, schedule, evidence)
            result.update(completed=True, has_incumbent=True, optimal=int(model.Status) == 2,
                          incumbent_objective=float(model.ObjVal),
                          mip_gap=_optional_number(model, "MIPGap"))
        return result
    except ImportError as exc:
        result["solver_optimize_attempts"] = getattr(exc, "solver_optimize_attempts", result["solver_optimize_attempts"])
        result["status_name"] = "DEPENDENCY_UNAVAILABLE"
        result["warnings"].append(str(exc))
        return result
    except Exception as exc:
        result["solver_optimize_attempts"] = getattr(exc, "solver_optimize_attempts", result["solver_optimize_attempts"])
        if type(exc).__module__.startswith("gurobipy"):
            result["status_name"] = "SOLVER_ERROR"
            result["solver_error_code"] = getattr(exc, "errno", None)
            result["warnings"].append(str(exc))
            return result
        raise
    finally:
        if optimizer is not None and optimizer.model is not None:
            optimizer.model.dispose()
        result["elapsed_sec"], result["phase_timing_sec"] = timer.finish()
        result["budget_exceeded"] = result["elapsed_sec"] > budget


def run_flexible_rolling(config, orders, options, time_limit, profile=None, output_flag=0):
    """Replay the executed commitments; never copy the final window objective."""
    from .flexible_rolling import RollingOptions, solve_flexible_rolling
    budget = _validate_runtime(time_limit, output_flag)
    started = time.monotonic()
    timer = PhaseTimer(started)
    if not isinstance(options, RollingOptions):
        raise ValueError("options must be RollingOptions")
    options.validate()
    profile = profile or flexible_profile("release_revealed")
    validate_f_instance(config, orders, profile)
    if profile.information != "release_revealed":
        raise ValueError("rolling requires release_revealed profile")
    if config.terminal_target is not None:
        raise ValueError("rolling terminal target requires safe real-T reachability tail")
    result = _base(config, orders, profile, "rolling", budget, options)
    timer.values["setup_provenance"] = time.monotonic()-started
    try:
        remaining = budget-(time.monotonic()-started)
        if remaining <= 0:
            raw = {"schedule": {}, "windows": [], "current_time": 0, "completed": False,
                   "status": "PARTIAL_TIME_LIMIT", "message": "Budget exhausted before controller."}
        else:
            raw = solve_flexible_rolling(config, orders, options, remaining, profile, output_flag)
        # Controller phases already partition its wall time. Do not time the
        # controller call again or add its window diagnostic phases twice.
        for phase, duration in raw.get("phase_timing_sec", {}).items():
            if phase in timer.values and phase != "other":
                timer.values[phase] += duration
        result["solver_optimize_attempts"] = raw.get("solver_optimize_attempts", 0)
        # Controller commits vehicle/cargo/selector events only; z is a final
        # accounting variable, reconstructed after the independent partial replay.
        with timer.measure("replay_accounting"):
            schedule = {name: dict(raw["schedule"].get(name, {})) for name in GROUPS}
            cut = raw["current_time"]
            evidence = validate_flexible_schedule(config, orders, profile, schedule,
                                                  completed=False, completed_horizon=cut)
            if raw["completed"]:
                schedule["z_unserved"] = {(oid,): item["unserved"]
                                           for oid, item in evidence["orders"].items()}
                evidence = validate_flexible_schedule(config, orders, profile, schedule,
                                                      completed=True, completed_horizon=cut)
            _replay_result(result, schedule, evidence)
        result.update({"status": raw["status"], "status_name": raw["status"],
                       "completed": raw["completed"], "windows": raw["windows"],
                       "current_time": cut, "message": raw["message"],
                       "has_incumbent": any(w.get("feasible", False) for w in raw["windows"])})
        return result
    except ImportError as exc:
        result["solver_optimize_attempts"] = getattr(exc, "solver_optimize_attempts", result["solver_optimize_attempts"])
        result["status_name"] = "DEPENDENCY_UNAVAILABLE"
        result["warnings"].append(str(exc))
        return result
    except Exception as exc:
        result["solver_optimize_attempts"] = getattr(exc, "solver_optimize_attempts", result["solver_optimize_attempts"])
        if type(exc).__module__.startswith("gurobipy"):
            result["status_name"] = "SOLVER_ERROR"
            result["solver_error_code"] = getattr(exc, "errno", None)
            result["warnings"].append(str(exc))
            return result
        raise
    finally:
        result["elapsed_sec"], result["phase_timing_sec"] = timer.finish()
        result["budget_exceeded"] = result["elapsed_sec"] > budget
