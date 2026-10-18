"""Explicit author-confirmed V1-M entry point; keeps the V0 registry intact."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from numbers import Integral, Real
from pathlib import Path
import platform
import subprocess
import sys
import time

from .capacity import _real
from .configuration import PROFILE_DECISION_ID
from .mandatory_optimizer import V1MandatoryOptimizer


def _plain(value):
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    raise ValueError(f"unsupported JSON metadata type: {type(value).__name__}")


def _provenance(config, orders, profile):
    root = Path(__file__).resolve().parents[2]
    source_paths = (
        "research/paper/main_v2_BHH.tex", "coordination/V1_M_IMPLEMENTATION_PROFILE.md",
        "intercity_delivery/v1/capacity.py", "intercity_delivery/v1/network.py",
        "intercity_delivery/v1/configuration.py", "intercity_delivery/v1/mandatory_optimizer.py",
        "intercity_delivery/v1/runner.py", "intercity_delivery/v1/validation.py",
    )
    hashes = {}
    for relative in source_paths:
        path = root / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    inputs = _plain({"config": config, "orders": orders, "profile": profile})
    encoded = json.dumps(inputs, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, timeout=1, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    try:
        gurobi_version = metadata.version("gurobipy")
    except metadata.PackageNotFoundError:
        gurobi_version = None
    return {
        "input": inputs, "input_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_sha256": hashes, "git_revision": revision,
        "git_revision_scope": "HEAD only; source hashes identify uncommitted implementation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "gurobipy_version": gurobi_version,
    }


def _optional_number(model, attribute):
    try:
        value = float(getattr(model, attribute))
        return value if math.isfinite(value) else None
    except (AttributeError, RuntimeError):
        return None


def run_mandatory(config, orders, profile, time_limit, output_flag=0):
    """Build, solve and independently validate one explicitly profiled model.

    TimeLimit receives the remaining wall-clock budget after input checks,
    provenance and model construction. Final validation/export is measured too,
    but cooperative solver time limits are not a hard real-time deadline.
    No incumbent gives null cost/schedule, never a fabricated all-reject result.
    """
    budget = _real("time_limit", time_limit, positive=True)
    if (isinstance(output_flag, bool) or not isinstance(output_flag, Integral)
            or output_flag not in (0, 1)):
        raise ValueError("output_flag must be integer 0 or 1")
    started = time.monotonic()
    optimizer = V1MandatoryOptimizer(config, orders, profile)
    provenance = _provenance(config, orders, profile)
    result = {
        "schema_version": "v1_mandatory_result/1",
        "model_id": optimizer.model_id,
        "semantic_status": optimizer.semantic_status,
        "profile_decision_id": PROFILE_DECISION_ID,
        "profile": _plain(profile), "provenance": provenance,
        "status": None, "status_name": "NOT_RUN", "has_incumbent": False,
        "optimal": False, "schedule_valid": False, "total_cost": None,
        "incumbent_objective": None, "best_bound": None, "mip_gap": None,
        "bound_scope": "full_horizon_adopted_V1_M_profile_only",
        "schedule": None, "indices": None, "validation": None, "cost_components": None,
        "requested_time_limit_sec": budget, "solver_time_limit_sec": 0.0,
        "threads": 1, "solver_seed": 0, "feasibility_tolerance": 1e-8,
        "integrality_tolerance": 1e-8, "validation_tolerance": 1e-6,
        "cost_units": "currency; minute durations divided by 60 exactly once",
        "traceability": optimizer.traceability,
        "scope": "offline V1-M arc-wave adopted benchmark; excludes F/RH/S/QVI",
        "warnings": ["Author adoption applies to this benchmark; manuscript F/RH/S/QVI remain separate."],
    }
    try:
        optimizer.build_model()
        model = optimizer.model
        model.Params.OutputFlag = output_flag
        remaining = budget-(time.monotonic()-started)
        result["build_elapsed_sec"] = time.monotonic()-started
        if remaining > 0:
            model.Params.TimeLimit = remaining
            result["solver_time_limit_sec"] = remaining
            model.optimize()
            result["status"] = int(model.Status)
            names = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED",
                     9: "TIME_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL"}
            result["status_name"] = names.get(int(model.Status), f"GUROBI_{model.Status}")
        else:
            result["status"] = int(model.Status)
            result["status_name"] = "BUDGET_EXHAUSTED_DURING_BUILD"
        result["best_bound"] = _optional_number(model, "ObjBound")
        result["solver_runtime_sec"] = _optional_number(model, "Runtime")
        result["variable_count"] = int(model.NumVars)
        result["constraint_count"] = int(model.NumConstrs)
        result["indices"] = {name: [list(key) for key in group]
                             for name, group in optimizer.groups.items()}
        if model.SolCount > 0:
            from .validation import validate_schedule
            schedule = optimizer.extract_schedule()
            objective = float(model.ObjVal)
            evidence = validate_schedule(config, orders, profile, schedule,
                                         tolerance=1e-6, reported_objective=objective)
            result.update({
                "has_incumbent": True, "optimal": int(model.Status) == 2,
                "schedule_valid": True, "total_cost": objective,
                "incumbent_objective": objective,
                "mip_gap": _optional_number(model, "MIPGap"),
                "schedule": {name: {str(tuple(key)): float(value)
                                    for key, value in values.items()}
                             for name, values in schedule.items()},
                "validation": evidence,
                "cost_components": evidence["costs"],
            })
        return result
    except ImportError as exc:
        result["status_name"] = "DEPENDENCY_UNAVAILABLE"
        result["warnings"].append(str(exc))
        return result
    except Exception as exc:
        # Report an unavailable license explicitly; implementation/validation
        # errors remain exceptions and must never masquerade as feasible runs.
        if type(exc).__module__.startswith("gurobipy"):
            result["status_name"] = "SOLVER_ERROR"
            result["solver_error_code"] = getattr(exc, "errno", None)
            result["warnings"].append(str(exc))
            return result
        raise
    finally:
        if optimizer.model is not None:
            optimizer.model.dispose()
        result["elapsed_sec"] = time.monotonic()-started
        result["budget_exceeded"] = result["elapsed_sec"] > budget
