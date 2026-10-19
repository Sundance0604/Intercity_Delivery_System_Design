"""Fixed N40 mechanism follow-up. Prepare/analyze/replay never call a solver.

Execution requires a separate coordinator invocation of --run. No resampling,
retries, old output mutation, source-manifest refresh, or new model equations.
"""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.specs import run_v1_p1 as old
from intercity_delivery.v1.configuration import V1Order
from intercity_delivery.v1.experiment_metrics import mean_ci95, summarize_result
from intercity_delivery.v1.flexible_configuration import flexible_profile

SEEDS = tuple(range(4101, 4106))
SOURCE_BATCH = "P1_SMALL_20261018_01"
DEFAULT_BATCH = "P1_FOLLOWUP_20261019_01"
CHOICES = (("T0", 0.), ("T1", 0.), ("T2", .1), ("T3", .1), ("T2", .05), ("T2", .2))
BUDGET = {"maximum_attempts": 30, "total_wall_sec": 300., "logical_per_run_sec": 5.,
          "hard_process_sec": 15., "Threads": 1, "Seed": 0}
ADDITIONAL_SOURCES = ("experiments/specs/run_v1_followup_20261019.py",
                      "tests/test_followup_20261019.py",
                      "experiments/specs/V1_FOLLOWUP_20261019_PROTOCOL.md")
MICRO_GATE = ROOT / "simulation/v1_micro/P1_SMALL_20261018_01/gate_evidence.json"
COMPARISONS = (("T1_minus_T0", ("T0", 0.), ("T1", 0.)),
               ("T2_m01_minus_T0", ("T0", 0.), ("T2", .1)),
               ("T3_m0_minus_T0", ("T0", 0.), ("T3", 0.)),
               ("T3_m01_minus_T2_m01", ("T2", .1), ("T3", .1)),
               ("T3_m01_minus_T3_m0", ("T3", 0.), ("T3", .1)),
               ("T2_m005_minus_T2_m01", ("T2", .1), ("T2", .05)),
               ("T2_m02_minus_T2_m01", ("T2", .1), ("T2", .2)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jobs():
    return [{"id": f"n40_s{seed}_{mechanism}_m{minimum:g}_full", "instance": f"n40_s{seed}",
             "n": 40, "seed": seed, "mechanism": mechanism, "minimum": minimum, "algorithm": "full"}
            for seed in SEEDS for mechanism, minimum in CHOICES]


def reference_id(seed):
    return f"n40_s{seed}_T3_m0_full"


def config_for(manifest, job):
    return replace(old.config_from_design(manifest["design"], manifest["decisions"], job["mechanism"]),
                   min_direct_load_per_vehicle=job["minimum"])


def check_sources():
    old.require_preflight()
    return old.freeze_snapshot()


def additional_identities():
    return {p: {"sha256": sha(ROOT / p), "size_bytes": (ROOT / p).stat().st_size}
            for p in ADDITIONAL_SOURCES}


def prepare(name=DEFAULT_BATCH):
    started = time.monotonic()
    snapshot = check_sources()
    source, original = old.load_batch(SOURCE_BATCH)
    extra = additional_identities()
    micro = old.require_micro(MICRO_GATE, original)
    instances, references = {}, {}
    # Validate every original byte before creating an exclusive output folder.
    for seed in SEEDS:
        identifier = f"n40_s{seed}"
        old.load_instance(source, original, identifier)
        instances[identifier] = original["instances"][identifier]
        target = source / "jobs" / reference_id(seed)
        record = old.read_json(target / "record.json")
        if sha(target / "raw.json") != record["raw_sha256"]:
            raise ValueError("original reference raw changed")
        if record["config_sha256"] != old.fingerprint(asdict(config_for(original, {"mechanism": "T3", "minimum": 0.}))):
            raise ValueError("reference model configuration differs")
        references[reference_id(seed)] = {"record_path": (target / "record.json").relative_to(ROOT).as_posix(),
            "record_sha256": sha(target / "record.json"), "raw_sha256": record["raw_sha256"],
            "reuse": True, "additional_independent_sample": False}
    directory = old.batch_path(name)
    if directory == source:
        raise ValueError("new batch must differ from original")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "inputs").mkdir()
    for entry in instances.values():
        shutil.copyfile(source / entry["path"], directory / entry["path"])
        if sha(directory / entry["path"]) != entry["file_sha256"]:
            raise ValueError("copied immutable input bytes differ")
    shutil.copyfile(Path(__file__), directory / "harness_snapshot.py")
    old.write_new(directory / "planned_jobs.json", jobs())
    check_sources()
    manifest = {"schema_version": "v1_followup/1", "batch": name, "created_utc": old.now(),
        "source_batch": SOURCE_BATCH, "source_batch_manifest_sha256": sha(source / "manifest.json"),
        "source_content_sha256": snapshot["source_content_sha256"],
        "source_manifest_sha256": snapshot["source_manifest_sha256"],
        "design": original["design"], "decisions": original["decisions"], "instances": instances,
        "references": references, "budget": BUDGET, "planned_jobs_sha256": old.fingerprint(jobs()),
        "additional_sources": extra, "micro_evidence": micro,
        "followup_harness_sha256": sha(__file__), "environment": old.environment(),
        "input_policy": "byte copies of five existing N40 inputs; no SQLite access/resampling",
        "comparison_policy": "same 5 seeds, no new independent samples from reused T3 references",
        "prepare_wall_sec": time.monotonic()-started}
    old.write_new(directory / "manifest.json", manifest)
    return {"batch": str(directory), "new_solver_runs": 0, "planned_attempts": len(jobs()),
            "prepare_wall_sec": manifest["prepare_wall_sec"]}


def load(name):
    snapshot = check_sources()
    directory = old.batch_path(name)
    manifest = old.read_json(directory / "manifest.json")
    for key in ("source_content_sha256", "source_manifest_sha256"):
        if snapshot[key] != manifest[key]:
            raise ValueError("frozen source identity changed: " + key)
    if manifest["followup_harness_sha256"] != sha(__file__) or sha(directory / "harness_snapshot.py") != sha(__file__):
        raise ValueError("follow-up harness changed after preparation")
    if manifest["additional_sources"] != additional_identities():
        raise ValueError("follow-up script/test/protocol changed after preparation")
    micro = manifest["micro_evidence"]
    if sha(micro["path"]) != micro["sha256"]:
        raise ValueError("prior micro gate changed")
    old.require_micro(micro["path"], manifest)
    if manifest["budget"] != BUDGET or old.read_json(directory / "planned_jobs.json") != jobs() or manifest["planned_jobs_sha256"] != old.fingerprint(jobs()):
        raise ValueError("approved budget/plan changed")
    if manifest["design"] != snapshot["design"] or manifest["decisions"] != snapshot["decisions"]:
        raise ValueError("batch design/decisions differ from frozen sources")
    source = old.batch_path(SOURCE_BATCH)
    if sha(source / "manifest.json") != manifest["source_batch_manifest_sha256"]:
        raise ValueError("original batch manifest changed")
    for identifier in manifest["instances"]:
        old.load_instance(directory, manifest, identifier)
    for item in manifest["references"].values():
        path = ROOT / item["record_path"]
        if not path.resolve().is_relative_to(source.resolve()) or sha(path) != item["record_sha256"] or sha(path.parent / "raw.json") != item["raw_sha256"]:
            raise ValueError("reused reference changed")
    return directory, manifest


def replay_one(raw, config, orders):
    from intercity_delivery.v1.flexible_validation import validate_flexible_schedule
    if raw.get("schedule") is None:
        return {"status": "NO_SCHEDULE"}
    complete = bool(raw.get("completed"))
    evidence = validate_flexible_schedule(config, [V1Order(**o) for o in orders], flexible_profile(),
        raw["schedule"], completed=complete,
        completed_horizon=config.T if complete else raw.get("current_time", 0),
        reported_objective=raw.get("total_cost") if complete else None)
    return {"status": "PASSED", "evidence": evidence}


def checked_record(path, manifest, directory):
    record = old.read_json(path)
    job = record["job"]
    if record["source_content_sha256"] != manifest["source_content_sha256"]:
        raise ValueError("record source mismatch")
    instance = old.load_instance(directory, manifest, job["instance"])
    config = config_for(manifest, job)
    if record["orders_sha256"] != instance["orders_sha256"] or record["input_sha256"] != instance["input_sha256"] or record["config_sha256"] != old.fingerprint(asdict(config)):
        raise ValueError("record input/config mismatch")
    if old.fingerprint(record["config"]) != record["config_sha256"]:
        raise ValueError("saved record config changed")
    if "raw_sha256" in record:
        raw_path = path.parent / "raw.json"
        if sha(raw_path) != record["raw_sha256"]:
            raise ValueError("record raw changed")
        raw = old.read_json(raw_path)
        record["metrics"] = summarize_result(raw, record["config"], instance["identity"]["orders"])
        record["solver_certificate"] = {k: raw.get(k) for k in
            ("status_name", "optimal", "best_bound", "mip_gap", "bound_scope", "completed", "budget_exceeded")}
        record["direct_quantity"] = (raw.get("validation") or {}).get("direct_quantity")
        record["solver_optimize_attempts"] = raw.get("solver_optimize_attempts")
    return record


def cost_difference_interval(left, right):
    """Full-horizon feasible incumbents U, solver bounds L: [Lr-Ul, Ur-Ll]."""
    lm, rm = left.get("metrics", {}), right.get("metrics", {})
    if not (lm.get("completed_feasible") and rm.get("completed_feasible")):
        return None
    if any(record.get("status") not in ("OPTIMAL", "TIME_LIMIT") for record in (left, right)):
        return None
    lb, rb = lm.get("full_horizon_best_bound"), rm.get("full_horizon_best_bound")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in (lb, rb)):
        return None
    if any(bound > metrics["total_cost"] + max(1e-6, abs(metrics["total_cost"])*1e-4)
           for bound, metrics in ((lb, lm), (rb, rm))):
        return None
    return [rb-lm["total_cost"], rm["total_cost"]-lb]


def records_for(name):
    directory, manifest = load(name)
    records = {}
    for identifier, entry in manifest["references"].items():
        records[identifier] = checked_record(ROOT / entry["record_path"], manifest, directory)
        records[identifier]["reused_reference"] = True
    for job in jobs():
        target = directory / "jobs" / job["id"]
        if (target / "record.json").exists():
            records[job["id"]] = checked_record(target / "record.json", manifest, directory)
        elif target.exists():
            records[job["id"]] = {"job": job, "status": "INTERRUPTED_NO_FINAL_RECORD"}
        else:
            records[job["id"]] = {"job": job, "status": "NOT_RUN"}
    return directory, manifest, records


def analyze(name):
    started = time.monotonic()
    directory, manifest, records = records_for(name)
    pairs = []
    for label, left, right in COMPARISONS:
        details, differences = [], []
        for seed in SEEDS:
            def identifier(choice):
                return f"n40_s{seed}_{choice[0]}_m{choice[1]:g}_full"
            l, r = records[identifier(left)], records[identifier(right)]
            eligible = all(x.get("metrics", {}).get("completed_feasible") for x in (l, r))
            row = {"seed": seed, "left": identifier(left), "right": identifier(right), "eligible": eligible,
                   "left_status": l["status"], "right_status": r["status"]}
            if eligible:
                difference = r["metrics"]["total_cost"]-l["metrics"]["total_cost"]
                differences.append(difference)
                row.update(incumbent_cost_difference_right_minus_left=difference,
                    optimal_cost_difference_certificate_interval=cost_difference_interval(l, r),
                    left_optimal=l["metrics"]["global_optimal_oracle"], right_optimal=r["metrics"]["global_optimal_oracle"])
            details.append(row)
        pairs.append({"comparison": label, "incumbent_difference_summary": mean_ci95(differences), "seeds": details})
    sessions = [old.read_json(p) for p in (directory / "sessions").glob("*.json")]
    return {"schema_version": "v1_followup_analysis/1", "generated_utc": old.now(),
        "elapsed_sec": time.monotonic()-started, "source_content_sha256": manifest["source_content_sha256"],
        "followup_harness_sha256": manifest["followup_harness_sha256"], "records": records, "pairs": pairs,
        "planned_new_attempts": 30, "reused_reference_runs": 5,
        "counted_wall_sec": manifest["prepare_wall_sec"]+sum(s["wall_sec"] for s in sessions), "sessions": sessions,
        "scope": "five-seed descriptive incumbent differences; bound intervals enclose optimal policy cost differences; reused references add no samples; no general theorem"}


def replay(name):
    started = time.monotonic()
    directory, manifest, records = records_for(name)
    output = []
    for identifier, record in records.items():
        row = {"job": identifier, "replay": "NO_SCHEDULE"}
        if "raw_sha256" in record:
            path = (ROOT / manifest["references"][identifier]["record_path"]).parent if identifier in manifest["references"] else directory / "jobs" / identifier
            instance = old.load_instance(directory, manifest, record["job"]["instance"])
            try:
                row.update(replay_one(old.read_json(path / "raw.json"), config_for(manifest, record["job"]), instance["identity"]["orders"]))
                row["replay"] = row.pop("status")
            except ValueError as exc:
                row.update(replay="FAILED", error=str(exc))
        output.append(row)
    check_sources()
    return {"generated_utc": old.now(), "elapsed_sec": time.monotonic()-started, "runs": output,
            "failed": sum(r["replay"] == "FAILED" for r in output), "new_solver_runs": 0}


def export(name):
    """Exclusive derived outputs from readonly analysis; never changes raw."""
    analysis = analyze(name)
    directory = old.batch_path(name) / "tables"
    directory.mkdir(exist_ok=False)
    old.write_new(directory / "analysis.json", analysis)
    rows = []
    for identifier, record in analysis["records"].items():
        metrics, certificate = record.get("metrics", {}), record.get("solver_certificate", {})
        rows.append({"job_id": identifier, "reused_reference": record.get("reused_reference", False),
            "status": record["status"], "source_content_sha256": record.get("source_content_sha256"),
            "input_sha256": record.get("input_sha256"), "orders_sha256": record.get("orders_sha256"),
            "config_sha256": record.get("config_sha256"), "raw_sha256": record.get("raw_sha256"),
            "followup_harness_sha256": analysis["followup_harness_sha256"],
            "total_cost": metrics.get("total_cost"), "full_horizon_best_bound": metrics.get("full_horizon_best_bound"),
            "mip_gap": certificate.get("mip_gap"), "global_optimal_oracle": metrics.get("global_optimal_oracle"),
            "served_quantity": metrics.get("served_quantity"), "served_rate": metrics.get("served_rate"),
            "direct_share": metrics.get("direct_share"),
            "direct_quantity": record.get("direct_quantity"),
            "solver_optimize_attempts": record.get("solver_optimize_attempts"),
            "explicit_empty_vehicle_tasks": metrics.get("explicit_empty_vehicle_tasks"),
            "zero_cargo_direct_vehicle_tasks": metrics.get("zero_cargo_direct_vehicle_tasks"),
            "parent_wall_sec": record.get("parent_wall_sec"),
            "completed_feasible": metrics.get("completed_feasible")})
    pairs = [{"comparison": pair["comparison"], **row,
              "optimal_cost_difference_certificate_interval": json.dumps(row.get("optimal_cost_difference_certificate_interval"))}
             for pair in analysis["pairs"] for row in pair["seeds"]]
    for filename, items in (("run_summary.csv", rows), ("paired_results.csv", pairs)):
        fields = list(dict.fromkeys(key for row in items for key in row))
        with (directory / filename).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(items)
    old.write_new(directory / "paired_results.json", analysis["pairs"])
    return {"tables": str(directory), "run_rows": len(rows), "paired_rows": len(pairs), "new_solver_runs": 0}


def worker(name, identifier):
    directory, manifest = load(name)
    job = next(j for j in jobs() if j["id"] == identifier)
    target = directory / "jobs" / identifier
    request = old.read_json(target / "request.json")
    config = config_for(manifest, job)
    instance = old.load_instance(directory, manifest, job["instance"])
    if request["job"] != job or request["config_sha256"] != old.fingerprint(asdict(config)) or request["input_sha256"] != instance["input_sha256"] or request["parent_pid"] != old.read_json(directory / "run.lock")["pid"]:
        raise ValueError("unregistered or mutated worker request")
    if request["budget"] != BUDGET or request["followup_harness_sha256"] != sha(__file__):
        raise ValueError("worker budget/harness mismatch")
    if (target / "raw.json").exists() or (target / "record.json").exists():
        raise ValueError("worker refuses a prior result")
    old.write_new(target / "worker_claim.json", {"pid": os.getpid(), "claimed_utc": old.now()})
    from intercity_delivery.v1.flexible_runner import run_flexible
    result = run_flexible(config, [V1Order(**o) for o in instance["identity"]["orders"]], flexible_profile(), 5., output_flag=1)
    old.write_new(target / "raw.json", result)
    return {"status": result["status_name"]}


def run(name):
    started = time.monotonic()
    directory, manifest = load(name)
    execution_environment = old.environment()
    previous = [old.read_json(p) for p in (directory / "sessions").glob("*.json")]
    if any(s.get("stop_reason") for s in previous):
        raise ValueError("previous failure stops this batch; coordinator review required")
    spent = manifest["prepare_wall_sec"]+sum(s["wall_sec"] for s in previous)
    available = BUDGET["total_wall_sec"]-spent
    (directory / "jobs").mkdir(exist_ok=True)
    old.write_new(directory / "run.lock", {"pid": os.getpid(), "created_utc": old.now()})
    attempted, retained, not_run, reason = [], [], [], None
    try:
        for job in jobs():
            target = directory / "jobs" / job["id"]
            if target.exists():
                retained.append(job["id"])
                continue
            if available-(time.monotonic()-started) < 16.:
                not_run.append(job["id"])
                continue
            if sum(p.is_dir() for p in (directory / "jobs").iterdir()) >= BUDGET["maximum_attempts"]:
                raise ValueError("30-attempt budget exhausted")
            config = asdict(config_for(manifest, job))
            instance = old.load_instance(directory, manifest, job["instance"])
            target.mkdir(exist_ok=False)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", job["id"], "--batch", name]
            request = {"job": job, "started_utc": old.now(), "parent_pid": os.getpid(), "command": command,
                "input_sha256": instance["input_sha256"], "orders_sha256": instance["orders_sha256"],
                "config": config, "config_sha256": old.fingerprint(config), "budget": BUDGET,
                "source_content_sha256": manifest["source_content_sha256"], "followup_harness_sha256": sha(__file__)}
            old.write_new(target / "request.json", request)
            timeout = max(0., min(15., available-(time.monotonic()-started)-.1))
            outcome = old.execute_attempt(command, timeout) if timeout > 0 else {
                "worker_exit_status": None, "process_status": "NOT_RUN_BUDGET", "stdout": "", "stderr": "budget exhausted", "parent_wall_sec": 0.}
            for key in ("stdout", "stderr"):
                with (target / (key+".log")).open("x", encoding="utf-8") as stream:
                    stream.write(outcome.pop(key))
            record = {**request, **outcome, "hard_process_timeout_sec": timeout, "finished_utc": old.now(), "status": outcome["process_status"]}
            if outcome["process_status"] == "EXITED" and outcome["worker_exit_status"] == 0 and (target / "raw.json").exists():
                raw = old.read_json(target / "raw.json")
                record.update(status=raw["status_name"], raw_sha256=sha(target / "raw.json"),
                    metrics=summarize_result(raw, config, instance["identity"]["orders"]))
                try:
                    record["independent_replay"] = replay_one(raw, config_for(manifest, job), instance["identity"]["orders"])
                except ValueError as exc:
                    record.update(status="REPLAY_ERROR", replay_error=str(exc))
                if record["status"] == "OPTIMAL" and not record["metrics"].get("completed_feasible"):
                    reason = "missing complete independently validated incumbent"
            else:
                reason = "worker/time-limit/license/dependency failure: " + outcome["process_status"]
            if record["status"] not in ("OPTIMAL", "TIME_LIMIT"):
                reason = "unexpected solver/correctness status: " + record["status"]
            record["stop_reason"] = reason
            old.write_new(target / "record.json", record)
            attempted.append(job["id"])
            print(json.dumps({"job": job["id"], "status": record["status"], "parent_wall_sec": outcome["parent_wall_sec"]}), flush=True)
            if reason:
                break
        check_sources()
    except Exception as exc:
        reason = "controller failure: " + repr(exc)
        raise
    finally:
        session = {"created_utc": old.now(), "wall_sec": time.monotonic()-started, "spent_before_sec": spent,
            "attempted": attempted, "existing_retained": retained,
            "not_run": [j["id"] for j in jobs() if not (directory / "jobs" / j["id"]).exists()],
            "stop_reason": reason, "budget": BUDGET, "environment": execution_environment,
            "scope": "prepare plus orchestration sessions; analysis/replay and idle human wait excluded; no retries"}
        (directory / "sessions").mkdir(exist_ok=True)
        old.write_new(directory / "sessions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json"), session)
        (directory / "run.lock").unlink()
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "analyze", "replay", "export"):
        mode.add_argument("--"+name, action="store_true")
    mode.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--batch", default=DEFAULT_BATCH)
    args = parser.parse_args()
    result = worker(args.batch, args.worker) if args.worker else next(
        function(args.batch) for flag, function in ((args.prepare, prepare), (args.run, run), (args.analyze, analyze), (args.replay, replay), (args.export, export)) if flag)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
