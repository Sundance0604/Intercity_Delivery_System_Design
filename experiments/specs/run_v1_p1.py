"""Immutable CFS pilot preparation and bounded, isolated V1-F runs.

Prepare never solves. Run consumes fixed JSON only and fails closed on source
drift or pending decisions. Every solver attempt gets its own process, raw
payload and logs; failures are retained and never silently retried/replaced.
Usage: --prepare --batch NAME; --run --batch NAME --stage all;
       --analyze --batch NAME. The coordinator authorizes execution separately.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.specs.v1_p0_preflight import check_p0, config_from_design
from intercity_delivery.v1.configuration import V1Order
from intercity_delivery.v1.experiment_metrics import mean_ci95, summarize_result

SEEDS = tuple(range(4101, 4106))
SIZES = (10, 20, 40)
SOLVE_SECONDS = 5.
PROCESS_SECONDS = 15.
MAX_TOTAL_SECONDS = 600.
BUDGET = {"total_wall_sec": 600., "pilot_per_run_sec": 5.,
          "mechanism_per_run_sec": 3., "sensitivity_per_run_sec": 3.,
          "hard_process_sec": 15., "maximum_attempts": 75}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    """Exclusive creation protects original instances and failed attempts."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def batch_path(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name):
        raise ValueError("batch must be a simple name, maximum 80 characters")
    return ROOT / "simulation/v1_runs" / name


def environment():
    versions = {}
    for package in ("gurobipy", "numpy", "scipy", "pytest"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    ram = None
    try:
        if os.name == "nt":
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (k, ctypes.c_ulonglong) for k in ("total_phys", "avail_phys", "total_page",
                    "avail_page", "total_virtual", "avail_virtual", "avail_extended")]
            info = MemoryStatus()
            info.length = ctypes.sizeof(info)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(info)):
                ram = info.total_phys
        else:
            ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        pass
    git = {}
    for name, arguments in (("revision", ["rev-parse", "HEAD"]),
                            ("status_porcelain", ["status", "--porcelain", "--untracked-files=normal"])):
        try:
            process = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True,
                                     text=True, timeout=3, check=True)
            git[name] = process.stdout.strip()
            git[name+"_warnings"] = process.stderr.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            git[name] = None
            git[name+"_warnings"] = str(exc)
    return {"timestamp_utc": now(), "python": sys.version,
            "executable": sys.executable, "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "physical_ram_bytes": ram, "package_versions": versions, "git": git}


def freeze_snapshot():
    directory = ROOT / "experiments/specs"
    manifest = read_json(directory / "v1_source_manifest.json")
    design = read_json(directory / "v1_p0_capacity_design.json")
    decisions = read_json(directory / "v1_p0_decisions.json")
    return {"source_content_sha256": manifest["content_sha256"],
            "source_manifest_sha256": hashlib.sha256((directory / "v1_source_manifest.json").read_bytes()).hexdigest(),
            "source_manifest": manifest, "design": design, "decisions": decisions,
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def require_preflight():
    gate = check_p0()
    if not gate["p0_checks_passed"]:
        raise ValueError("P0 preflight failed: " + "; ".join(gate["errors"]))
    return gate


def prepare(name, source):
    started = time.monotonic()
    gate = require_preflight()
    snapshot = freeze_snapshot()
    config = config_from_design(snapshot["design"], snapshot["decisions"])
    if config.tau_periods != 9:
        raise ValueError("this pilot freezes LA-SF tau=9; do not recalibrate by seed")
    source = Path(source).resolve()
    if source.suffix.lower() not in (".sqlite", ".sqlite3", ".db") or not source.is_file():
        raise ValueError("prepare requires the existing SQLite database")
    directory = batch_path(name)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "inputs").mkdir()
    from intercity_delivery.v1.experiment_inputs import load_cfs_job_orders
    instances = {}
    for size in SIZES:
        for seed in SEEDS:
            before = time.monotonic()
            orders, source_metadata = load_cfs_job_orders(source, num_orders=size, seed=seed,
                city_pair=("06-348", "06-488"), T=config.T,
                t0_minutes=config.t0_minutes, tau_periods=9,
                penalty=snapshot["design"]["baseline"]["penalty_per_unit"])
            identity = {"config": asdict(config), "orders": [asdict(o) for o in orders],
                        "source_metadata": source_metadata}
            identifier = f"n{size}_s{seed}"
            payload = {"schema_version": "v1_p1_instance/1", "id": identifier,
                       "n": size, "seed": seed, "identity": identity,
                       "input_sha256": fingerprint(identity),
                       "orders_sha256": fingerprint(identity["orders"]),
                       "config_sha256": fingerprint(identity["config"]),
                       "prepare_elapsed_sec": time.monotonic()-before,
                       "source_content_sha256": snapshot["source_content_sha256"]}
            path = directory / "inputs" / (identifier+".json")
            write_new(path, payload)
            instances[identifier] = {"path": path.relative_to(directory).as_posix(),
                                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "input_sha256": payload["input_sha256"],
                                    "orders_sha256": payload["orders_sha256"]}
    write_new(directory / "budget.json", BUDGET)
    write_new(directory / "planned_jobs.json", job_list("all"))
    manifest = {"schema_version": "v1_p1_batch/1", "created_utc": now(),
                "batch": name, **snapshot, "preflight": gate,
                "environment": environment(), "instances": instances,
                "sampling_seconds": time.monotonic()-started,
                "sampling_scope": "balanced, CFS weighted reservoir; source records and synthetic windows retained; no database-wide hash",
                "budget": BUDGET, "budget_sha256": fingerprint(BUDGET),
                "planned_jobs_sha256": fingerprint(job_list("all")),
                "solver": {"Threads": 1, "Seed": 0, "pilot_per_run_seconds": SOLVE_SECONDS,
                           "mechanism_sensitivity_per_run_seconds": 3.,
                           "hard_process_seconds": PROCESS_SECONDS},
                "limits": {"maximum_n": 40, "seeds": SEEDS,
                           "maximum_new_attempts_all_stages": 75,
                           "maximum_total_wall_seconds": MAX_TOTAL_SECONDS},
                "new_solver_runs": 0}
    write_new(directory / "manifest.json", manifest)
    return {"batch": str(directory), "instances": len(instances), "new_solver_runs": 0}


def load_batch(name):
    require_preflight()
    directory = batch_path(name)
    manifest = read_json(directory / "manifest.json")
    current = freeze_snapshot()
    for key in ("source_content_sha256", "source_manifest_sha256", "harness_sha256"):
        if current[key] != manifest[key]:
            raise ValueError("batch frozen source changed: " + key)
    budget = read_json(directory / "budget.json")
    if budget != BUDGET or fingerprint(budget) != manifest["budget_sha256"] or manifest["budget"] != budget:
        raise ValueError("batch approved budget changed")
    if read_json(directory / "planned_jobs.json") != job_list("all") or manifest["planned_jobs_sha256"] != fingerprint(job_list("all")):
        raise ValueError("predeclared job plan changed")
    return directory, manifest


def load_instance(directory, manifest, identifier):
    item = manifest["instances"][identifier]
    path = directory / item["path"]
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("instance path outside batch")
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["file_sha256"]:
        raise ValueError("instance file bytes changed")
    instance = read_json(path)
    if fingerprint(instance["identity"]) != instance["input_sha256"] or instance["input_sha256"] != item["input_sha256"]:
        raise ValueError("instance identity changed")
    if fingerprint(instance["identity"]["orders"]) != instance["orders_sha256"]:
        raise ValueError("instance orders changed")
    return instance


def job_list(stage):
    """Stage references share job IDs; full T3 reference is solved only once."""
    if stage == "all":
        jobs = {}
        for part in ("pilot", "mechanism", "sensitivity"):
            jobs.update((j["id"], j) for j in job_list(part))
        return list(jobs.values())
    jobs = []
    sizes = SIZES if stage == "pilot" else (10,)
    for size in sizes:
        for seed in SEEDS:
            choices = ([("T3", 0., algorithm) for algorithm in ("full", "dense", "pruned")]
                       if stage == "pilot" else
                       [("T0", 0., "full"), ("T1", 0., "full"), ("T2", .1, "full"), ("T3", 0., "full")]
                       if stage == "mechanism" else
                       [("T2", m, "full") for m in (.05, .1, .2)] + [("T3", .1, "full")])
            for mechanism, minimum, algorithm in choices:
                identifier = f"n{size}_s{seed}"
                jobs.append({"id": f"{identifier}_{mechanism}_m{minimum:g}_{algorithm}",
                             "instance": identifier, "n": size, "seed": seed,
                             "mechanism": mechanism, "minimum": minimum, "algorithm": algorithm})
    return jobs


def job_config(manifest, job):
    config = config_from_design(manifest["design"], manifest["decisions"], job["mechanism"])
    return replace(config, min_direct_load_per_vehicle=job["minimum"])


def job_budget(job, budget=BUDGET):
    # Shared T3 references belong to the original pilot, T2=.1 to mechanisms.
    stage = ("pilot" if job["mechanism"] == "T3" and job["minimum"] == 0 else
             "mechanism" if job["mechanism"] in ("T0", "T1") or
             job["mechanism"] == "T2" and job["minimum"] == .1 else "sensitivity")
    return budget[stage+"_per_run_sec"]


def require_micro(path, manifest):
    evidence = read_json(path)
    runs = evidence.get("runs", [])
    if evidence.get("all_passed") is not True or len(runs) != 5 or any(r.get("passed") is not True for r in runs):
        raise ValueError("five successful physical/solver micro checks are required before pilot runs")
    binding = evidence.get("source_binding", {})
    hashes = {"source_content_sha256": binding.get("manifest", {}).get("content_sha256"),
              "source_manifest_sha256": binding.get("manifest_sha256")}
    for key in ("source_content_sha256", "source_manifest_sha256"):
        if evidence.get(key, hashes[key]) != manifest[key]:
            raise ValueError("micro evidence frozen source mismatch: " + key)
    elapsed = evidence.get("elapsed_sec")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("micro evidence must record positive total elapsed_sec")
    return {"path": str(Path(path).resolve()),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "evidence": evidence}


def cumulative_wall(directory):
    return sum(read_json(path)["wall_sec"] for path in (directory / "sessions").glob("*.json"))


def worker(name, job_id):
    directory, manifest = load_batch(name)
    job = next((j for j in job_list("all") if j["id"] == job_id), None)
    if job is None:
        raise ValueError("job is outside predeclared pilot")
    instance = load_instance(directory, manifest, job["instance"])
    config = job_config(manifest, job)
    target = directory / "jobs" / job_id
    if (target / "raw.json").exists() or (target / "record.json").exists():
        raise ValueError("worker refuses an already completed attempt")
    request = read_json(target / "request.json")
    if request["job"] != job or request["input_sha256"] != instance["input_sha256"] or request["config_sha256"] != fingerprint(asdict(config)):
        raise ValueError("worker request differs from frozen job")
    if read_json(directory / "run.lock")["pid"] != request["parent_pid"]:
        raise ValueError("worker lacks active parent registration")
    micro = request["micro_evidence"]
    if hashlib.sha256(Path(micro["path"]).read_bytes()).hexdigest() != micro["sha256"]:
        raise ValueError("micro evidence changed")
    require_micro(micro["path"], manifest)
    seconds = job_budget(job, manifest["budget"])
    if request["per_run_budget_sec"] != seconds or request["source_content_sha256"] != manifest["source_content_sha256"]:
        raise ValueError("worker request budget/source changed")
    # Claim before imports/model creation: invoking hidden --worker again cannot
    # consume a second solve, even when an earlier attempt failed or timed out.
    write_new(target / "worker_claim.json", {"pid": os.getpid(), "claimed_utc": now()})
    orders = [V1Order(**item) for item in instance["identity"]["orders"]]
    from intercity_delivery.v1.flexible_configuration import flexible_profile
    from intercity_delivery.v1.flexible_runner import run_flexible, run_flexible_rolling
    if job["algorithm"] == "full":
        result = run_flexible(config, orders, flexible_profile(), seconds, output_flag=1)
    else:
        from intercity_delivery.v1.flexible_rolling import RollingOptions
        rh = manifest["design"]["candidate_rh"]
        options = RollingOptions(rh["H"], rh["h"], rh["E"], job["algorithm"] == "pruned")
        result = run_flexible_rolling(config, orders, options, seconds, output_flag=1)
    write_new(directory / "jobs" / job_id / "raw.json", result)
    return {"status": result["status_name"], "completed": result.get("completed")}


def execute_attempt(command, timeout, process_run=subprocess.run):
    """Return logs and exit status even when the isolated worker times out."""
    started = time.monotonic()
    def decoded(value):
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
    try:
        run = process_run(command, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, check=False)
        return {"worker_exit_status": run.returncode, "process_status": "EXITED",
                "stdout": decoded(run.stdout), "stderr": decoded(run.stderr),
                "parent_wall_sec": time.monotonic()-started}
    except subprocess.TimeoutExpired as exc:
        return {"worker_exit_status": None, "process_status": "TIMEOUT",
                "stdout": decoded(exc.stdout), "stderr": decoded(exc.stderr),
                "parent_wall_sec": time.monotonic()-started}
    except OSError as exc:
        return {"worker_exit_status": None, "process_status": "SPAWN_ERROR",
                "stdout": "", "stderr": str(exc),
                "parent_wall_sec": time.monotonic()-started}


def remaining_process_timeout(available, started):
    """Recheck after registration/environment work, reserving export slack."""
    return max(0., min(PROCESS_SECONDS,
                       available-(time.monotonic()-started)-.1))


def run_batch(name, stage, max_total_wall=MAX_TOTAL_SECONDS, micro_evidence=None):
    if not math.isfinite(max_total_wall) or not 0 < max_total_wall <= MAX_TOTAL_SECONDS:
        raise ValueError("total wall budget must be positive and at most 600 seconds")
    started = time.monotonic()
    directory, manifest = load_batch(name)
    if micro_evidence is None:
        raise ValueError("--micro-evidence is required; complete small physical/solver verification first")
    micro = require_micro(micro_evidence, manifest)
    spent_before = (cumulative_wall(directory)+manifest["sampling_seconds"]+
                    micro["evidence"]["elapsed_sec"])
    for path in (directory / "sessions").glob("*.json"):
        previous = read_json(path)
        if previous.get("micro_sha256") != micro["sha256"]:
            raise ValueError("cannot substitute a different micro report in this batch")
        if previous.get("stop_reason"):
            raise ValueError("batch stopped for a solver/correctness failure; coordinator review required")
    available = min(max_total_wall, BUDGET["total_wall_sec"]-spent_before)
    (directory / "jobs").mkdir(exist_ok=True)
    # Lock cannot be stolen; a failed controller leaves honest evidence and
    # requires coordinator inspection before any recovery, never an auto-retry.
    lock = directory / "run.lock"
    write_new(lock, {"created_utc": now(), "pid": os.getpid(), "stage": stage})
    attempted, reused, not_run, stop_reason = [], [], [], None
    try:
        for job in job_list(stage):
            target = directory / "jobs" / job["id"]
            if target.exists():
                # Existing incomplete/failed attempts are retained too.
                reused.append(job["id"])
                continue
            if available-(time.monotonic()-started) < PROCESS_SECONDS+1:
                not_run.append(job["id"])
                continue
            instance = load_instance(directory, manifest, job["instance"])
            config = asdict(job_config(manifest, job))
            if sum(p.is_dir() for p in (directory / "jobs").iterdir()) >= BUDGET["maximum_attempts"]:
                raise ValueError("75-attempt batch budget exhausted")
            target.mkdir(exist_ok=False)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", job["id"], "--batch", name]
            record = {"schema_version": "v1_p1_attempt/1", "started_utc": now(),
                      "job": job, "input_sha256": instance["input_sha256"],
                      "orders_sha256": instance["orders_sha256"],
                      "config": config, "config_sha256": fingerprint(config),
                      "source_content_sha256": manifest["source_content_sha256"],
                      "command": command, "per_run_budget_sec": job_budget(job, manifest["budget"]),
                      "hard_process_timeout_cap_sec": PROCESS_SECONDS,
                      "solver_parameters": {"Threads": 1, "Seed": 0},
                      "environment": environment(), "micro_evidence": micro,
                      "parent_pid": os.getpid()}
            write_new(target / "request.json", record)
            # Metadata capture and writing can consume the remaining budget.
            # Do not spawn/claim a worker unless a fresh check leaves time.
            actual_timeout = remaining_process_timeout(available, started)
            record["hard_process_timeout_sec"] = actual_timeout
            record["worker_spawned"] = actual_timeout > 0
            if actual_timeout > 0:
                outcome = execute_attempt(command, actual_timeout)
            else:
                outcome = {"worker_exit_status": None, "process_status": "NOT_RUN_BUDGET",
                           "stdout": "", "stderr": "Budget exhausted during attempt registration; worker not started.",
                           "parent_wall_sec": 0.}
            (target / "stdout.log").write_text(outcome.pop("stdout"), encoding="utf-8")
            (target / "stderr.log").write_text(outcome.pop("stderr"), encoding="utf-8")
            record.update(outcome, finished_utc=now())
            raw_path = target / "raw.json"
            if outcome["process_status"] == "EXITED" and outcome["worker_exit_status"] == 0 and raw_path.is_file():
                try:
                    result = read_json(raw_path)
                    record["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                    record["status"] = result["status_name"]
                    record["metrics"] = summarize_result(result, config, instance["identity"]["orders"])
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    record["status"] = "RESULT_PARSE_ERROR"
                    record["error"] = str(exc)
            else:
                record["status"] = outcome["process_status"] if outcome["process_status"] != "EXITED" else "WORKER_ERROR"
            if record["status"] in ("SOLVER_ERROR", "DEPENDENCY_UNAVAILABLE", "DEPENDENCY_ERROR", "WORKER_ERROR", "RESULT_PARSE_ERROR", "SPAWN_ERROR"):
                stop_reason = "solver/license/dependency/worker correctness failure: " + record["status"]
            metrics = record.get("metrics", {})
            if metrics.get("completed_feasible") and job["mechanism"] == "T3" and job["minimum"] == 0:
                candidates = [read_json(path) for path in (directory / "jobs").glob("*/record.json")]
                candidates.append(record)
                peers = [r for r in candidates if r["input_sha256"] == record["input_sha256"] and
                         r["config_sha256"] == record["config_sha256"] and
                         r.get("metrics", {}).get("completed_feasible")]
                full = next((r for r in peers if r["job"]["algorithm"] == "full" and r["metrics"].get("global_optimal_oracle")), None)
                if full and any(r["metrics"]["total_cost"] < full["metrics"]["total_cost"]-1e-6 for r in peers):
                    stop_reason = "CORRECTNESS_ERROR: full OPTIMAL exceeds a matched complete feasible schedule"
                    record["status"] = "CORRECTNESS_ERROR"
            record["stop_reason"] = stop_reason
            write_new(target / "record.json", record)
            attempted.append(job["id"])
            print(json.dumps({"job": job["id"], "status": record["status"],
                              "wall_sec": outcome["parent_wall_sec"]}), flush=True)
            if stop_reason:
                not_run.extend(j["id"] for j in job_list(stage)
                               if not (directory / "jobs" / j["id"]).exists())
                break
    finally:
        session = {"stage": stage, "finished_utc": now(), "attempted": attempted,
                   "existing_retained": reused, "budget_not_run": not_run,
                   "budget_sec": max_total_wall, "wall_sec": time.monotonic()-started,
                   "cumulative_wall_before_sec": spent_before,
                   "available_batch_wall_sec": available,
                   "micro_sha256": micro["sha256"], "stop_reason": stop_reason,
                   "cumulative_scope": "input preparation + five micro checks + controller sessions; excludes human waiting, analysis/replay reported separately",
                   "no_automatic_retries": True}
        (directory / "sessions").mkdir(exist_ok=True)
        write_new(directory / "sessions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json"), session)
        lock.unlink()
    return session


def replay(name):
    """Recompute physical validation from raw schedules without solving."""
    started = time.monotonic()
    directory = batch_path(name)
    manifest = read_json(directory / "manifest.json")
    from intercity_delivery.v1.flexible_configuration import flexible_profile
    from intercity_delivery.v1.flexible_validation import validate_flexible_schedule
    results = []
    for path in sorted((directory / "jobs").glob("*/record.json")):
        record = read_json(path)
        raw_path = path.parent / "raw.json"
        row = {"job": record["job"]["id"], "status": record["status"], "replay": "NO_SCHEDULE"}
        if "raw_sha256" in record:
            if hashlib.sha256(raw_path.read_bytes()).hexdigest() != record["raw_sha256"]:
                raise ValueError("raw output changed: " + str(raw_path))
            result = read_json(raw_path)
            if result.get("schedule") is not None:
                instance = load_instance(directory, manifest, record["job"]["instance"])
                config = job_config(manifest, record["job"])
                completed = bool(result.get("completed"))
                profile = flexible_profile("offline" if record["job"]["algorithm"] == "full" else "release_revealed")
                try:
                    evidence = validate_flexible_schedule(config,
                        [V1Order(**o) for o in instance["identity"]["orders"]],
                        profile, result["schedule"], completed=completed,
                        completed_horizon=config.T if completed else result.get("current_time", 0),
                        reported_objective=result.get("total_cost") if completed else None)
                    row.update(replay="PASSED", evidence=evidence)
                except ValueError as exc:
                    row.update(replay="FAILED", error=str(exc))
        results.append(row)
    output = {"generated_utc": now(), "elapsed_sec": time.monotonic()-started,
              "scope": "original-f independent physical replay; no solver invocation", "runs": results}
    (directory / "analyses").mkdir(exist_ok=True)
    target = directory / "analyses" / ("replay_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json")
    write_new(target, output)
    return {"replay": str(target), "runs": len(results), "failed": sum(r["replay"] == "FAILED" for r in results)}


def analyze(name):
    """Paired estimates require matching immutable orders and source versions."""
    started = time.monotonic()
    directory = batch_path(name)
    manifest = read_json(directory / "manifest.json")
    records = {}
    for path in sorted((directory / "jobs").glob("*/record.json")):
        record = read_json(path)
        if record["source_content_sha256"] != manifest["source_content_sha256"]:
            raise ValueError("analysis source mismatch")
        instance = load_instance(directory, manifest, record["job"]["instance"])
        if record["orders_sha256"] != instance["orders_sha256"]:
            raise ValueError("analysis order mismatch")
        if fingerprint(record["config"]) != record["config_sha256"]:
            raise ValueError("analysis config mismatch")
        if "raw_sha256" in record:
            raw_path = path.parent / "raw.json"
            if hashlib.sha256(raw_path.read_bytes()).hexdigest() != record["raw_sha256"]:
                raise ValueError("analysis raw result bytes changed")
            # Recompute descriptive metrics from saved raw output, rather than
            # trusting an edited summary. Physical replay is a separate command.
            record["metrics"] = summarize_result(read_json(raw_path), record["config"], instance["identity"]["orders"])
        if record.get("status") == "CORRECTNESS_ERROR":
            record.get("metrics", {})["completed_feasible"] = False
        records[record["job"]["id"]] = record
    for path in sorted((directory / "jobs").glob("*/request.json")):
        request = read_json(path)
        if request["job"]["id"] not in records:
            records[request["job"]["id"]] = {**request, "status": "CONTROLLER_INTERRUPTED_NO_FINAL_RECORD"}
    groups = {}
    for job in job_list("all"):
        key = f'n{job["n"]}_{job["mechanism"]}_m{job["minimum"]:g}_{job["algorithm"]}'
        record = records.get(job["id"], {})
        metrics = record.get("metrics", {})
        group = groups.setdefault(key, {"planned": 5, "attempted": 0, "completed_feasible": 0,
                                       "statuses": {}, "eligible_costs": []})
        status = record.get("status", "NOT_RUN")
        group["statuses"][status] = group["statuses"].get(status, 0)+1
        group["attempted"] += bool(record)
        if metrics.get("completed_feasible"):
            group["completed_feasible"] += 1
            group["eligible_costs"].append(metrics["total_cost"])
    for group in groups.values():
        group["cost_summary"] = mean_ci95(group.pop("eligible_costs"))
    pairs = []
    for size in SIZES:
        for left_algorithm, right_algorithm in (("full", "dense"), ("full", "pruned"), ("dense", "pruned")):
            deltas, speedups, details = [], [], []
            for seed in SEEDS:
                left_id = f"n{size}_s{seed}_T3_m0_{left_algorithm}"
                right_id = f"n{size}_s{seed}_T3_m0_{right_algorithm}"
                left, right = records.get(left_id), records.get(right_id)
                eligible = bool(left and right and left.get("metrics", {}).get("completed_feasible") and right.get("metrics", {}).get("completed_feasible"))
                detail = {"seed": seed, "eligible": eligible, "left": left_id, "right": right_id}
                if eligible:
                    if left["input_sha256"] != right["input_sha256"] or left["orders_sha256"] != right["orders_sha256"]:
                        raise ValueError("paired comparison input mismatch")
                    lm, rm = left["metrics"], right["metrics"]
                    delta = rm["total_cost"]-lm["total_cost"]
                    deltas.append(delta)
                    detail.update(cost_difference_right_minus_left=delta,
                                  left_global_optimal_oracle=lm["global_optimal_oracle"])
                    if left_algorithm == "dense" and right_algorithm == "pruned" and right["parent_wall_sec"] > 0:
                        speedup = left["parent_wall_sec"]/right["parent_wall_sec"]
                        detail["process_end_to_end_speedup_dense_over_pruned"] = speedup
                        speedups.append(speedup)
                details.append(detail)
            pairs.append({"n": size, "left": left_algorithm, "right": right_algorithm,
                          "cost_difference": mean_ci95(deltas), "dense_pruned_speedup": mean_ci95(speedups), "seeds": details})
    mechanism_pairs = []
    comparisons = [("mechanism_"+mechanism+"_minus_T0", ("T0", 0.), (mechanism, minimum))
                   for mechanism, minimum in (("T1", 0.), ("T2", .1), ("T3", 0.))]
    comparisons += [("matched_minimum_independent_empty_T3_minus_T2", ("T2", .1), ("T3", .1)),
                    ("T3_minimum_policy_m01_minus_m0", ("T3", 0.), ("T3", .1)),
                    ("T2_sensitivity_m005_minus_m01", ("T2", .1), ("T2", .05)),
                    ("T2_sensitivity_m02_minus_m01", ("T2", .1), ("T2", .2))]
    for label, (left_m, left_min), (right_m, right_min) in comparisons:
        differences, details = [], []
        for seed in SEEDS:
            left_id = f"n10_s{seed}_{left_m}_m{left_min:g}_full"
            right_id = f"n10_s{seed}_{right_m}_m{right_min:g}_full"
            left, right = records.get(left_id), records.get(right_id)
            eligible = bool(left and right and left.get("metrics", {}).get("completed_feasible") and right.get("metrics", {}).get("completed_feasible"))
            row = {"seed": seed, "eligible": eligible, "left": left_id, "right": right_id}
            if eligible:
                if left["orders_sha256"] != right["orders_sha256"] or left["input_sha256"] != right["input_sha256"]:
                    raise ValueError("mechanism paired inputs differ")
                difference = right["metrics"]["total_cost"]-left["metrics"]["total_cost"]
                row["cost_difference_right_minus_left"] = difference
                row["left_optimal"] = left["metrics"]["global_optimal_oracle"]
                row["right_optimal"] = right["metrics"]["global_optimal_oracle"]
                differences.append(difference)
            details.append(row)
        mechanism_pairs.append({"comparison": label, "cost_difference": mean_ci95(differences), "seeds": details})
    output = {"schema_version": "v1_p1_analysis/1", "generated_utc": now(),
              "elapsed_sec": time.monotonic()-started,
              "source_content_sha256": manifest["source_content_sha256"],
              "groups": groups, "pilot_pairs": pairs, "mechanism_sensitivity_pairs": mechanism_pairs,
              "selection_policy": "only independently validated completed schedules in cost summaries; all failures/partial/missing retained in planned denominators",
              "oracle_policy": "only full-horizon OPTIMAL is an optimum; time-limited full bound is separate; rolling window gaps never global",
              "ci_policy": "paired five-seed Student t intervals, descriptive only; missing members suppress CI; no simulation proves a general mathematical claim"}
    (directory / "analyses").mkdir(exist_ok=True)
    path = directory / "analyses" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json")
    write_new(path, output)
    return {"analysis": str(path), "record_count": len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    mode.add_argument("--replay", action="store_true")
    mode.add_argument("--worker", metavar="JOB_ID", help=argparse.SUPPRESS)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "data/cfs_2022_pums.sqlite")
    parser.add_argument("--stage", choices=("pilot", "mechanism", "sensitivity", "all"), default="pilot")
    parser.add_argument("--max-total-wall", type=float, default=MAX_TOTAL_SECONDS)
    parser.add_argument("--micro-evidence", type=Path)
    args = parser.parse_args()
    try:
        result = (prepare(args.batch, args.source) if args.prepare else
                  run_batch(args.batch, args.stage, args.max_total_wall, args.micro_evidence) if args.run else
                  analyze(args.batch) if args.analyze else
                  replay(args.batch) if args.replay else worker(args.batch, args.worker))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
