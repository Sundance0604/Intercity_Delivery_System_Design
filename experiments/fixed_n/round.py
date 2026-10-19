"""Fixed N, four directions, immutable inputs and isolated no-retry attempts.

--plan (default), --prepare, --run, --analyze, --replay share one entry.
Run needs an explicit cumulative budget covering prepare plus every session.
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
import re
import subprocess
import sys
import time

from experiments.specs.run_v1_p1 import (canonical, environment, execute_attempt,
    fingerprint, freeze_snapshot, load_instance, now, read_json, require_preflight, write_new)
from experiments.specs.v1_p0_preflight import config_from_design
from intercity_delivery.v1.configuration import V1Order
from intercity_delivery.v1.experiment_metrics import mean_ci95, summarize_result
from intercity_delivery.v1.flexible_configuration import V1FConfig, flexible_profile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESIGN = Path(__file__).with_name("default_design.json")
STAGES = ("mechanism", "algorithms", "sensitivity", "pooling")


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return value


def validate_design(design):
    if set(design) != set(read_json(DEFAULT_DESIGN)) or design["schema_version"] != "fixed_n_design/1":
        raise ValueError("unsupported fixed design fields/schema")
    for key in ("direct_hourly_rates", "t2_minimum_loads", "prediction_horizons"):
        values = design[key]
        if not isinstance(values, list) or len(values) != 2 or len(set(values)) != 2:
            raise ValueError(key + " needs two distinct sensitivity levels")
        for value in values:
            positive(value, key)
    for key in ("baseline_horizon", "rolling_step", "extension_horizon", "empty_hourly_rate"):
        positive(design[key], key)
    for value in design["prediction_horizons"] + [design[k] for k in ("baseline_horizon", "rolling_step", "extension_horizon")]:
        if type(value) is not int or value < design["rolling_step"]:
            raise ValueError("integer RH horizons must be >= rolling_step")
    if .1 in design["t2_minimum_loads"] or 25 in design["direct_hourly_rates"] or design["baseline_horizon"] in design["prediction_horizons"]:
        raise ValueError("sensitivity levels must differ from baseline")
    return design


def validate_round(n, seeds, per_run_sec, worker_timeout_sec):
    if type(n) is not int or n < 2:
        raise ValueError("orders must be an integer >=2 for the CFS sampler")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    positive(per_run_sec, "per-run-sec")
    positive(worker_timeout_sec, "worker-timeout-sec")
    if worker_timeout_sec <= per_run_sec:
        raise ValueError("worker timeout must exceed logical solve budget")


def batch_path(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name):
        raise ValueError("batch must be a simple safe name <=80 characters")
    return ROOT / "simulation/fixed_n_runs" / name


def source_suite():
    files = sorted((ROOT / "experiments/fixed_n").glob("*.py"))
    files += sorted((ROOT / "experiments/fixed_n").glob("*.json"))
    files += [ROOT / "experiments/run_fixed_n.py", ROOT / "tests/test_fixed_n_round.py",
              ROOT / "tests/test_fixed_n_pooling.py", ROOT / "research/theory/fixed_n_round_contract.md"]
    entries = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {"files": entries, "sha256": fingerprint(entries)}


def config_object(values):
    values = dict(values)
    for key in ("a", "b", "rho", "N_manual", "N_auto", "terminal_target"):
        if values.get(key) is not None:
            values[key] = {int(k): v for k, v in values[key].items()}
    return V1FConfig(**values)


def make_plan(n, seeds, design, base_config, per_run_sec, worker_timeout_sec):
    validate_round(n, seeds, per_run_sec, worker_timeout_sec)
    validate_design(design)
    cases, refs = [], {s: [] for s in STAGES}
    for seed in seeds:
        prefix = f"n{n}_s{seed}"
        def add(label, stage, mechanism="T3", minimum=0., algorithm="full", rate=25., horizon=None, pooling="pooled"):
            cfg = replace(base_config, service_mechanism=mechanism,
                min_direct_load_per_vehicle=minimum, c_direct_hour=rate,
                c_empty_hour=design["empty_hourly_rate"])
            case = {"id": prefix+"_"+label, "label": label, "n": n, "seed": seed,
                "instance": prefix, "stage": stage, "config": json.loads(canonical(asdict(cfg))),
                "algorithm": algorithm, "pooling_model": pooling,
                "information": "offline" if algorithm == "full" else "release_revealed",
                "H": horizon if horizon is not None else design["baseline_horizon"],
                "h": design["rolling_step"], "E": design["extension_horizon"],
                "logical_budget_sec": per_run_sec, "worker_timeout_sec": worker_timeout_sec,
                "solver": {"Threads": 1, "Seed": 0}}
            cases.append(case)
            refs[stage].append(case["id"])
            return case["id"]
        add("T0", "mechanism", mechanism="T0")
        add("T1", "mechanism", mechanism="T1")
        add("T2_m0.1", "mechanism", mechanism="T2", minimum=.1)
        baseline = add("T3", "mechanism")
        add("T3_m0.1", "mechanism", minimum=.1)
        refs["algorithms"].append(baseline)
        add("RH_dense", "algorithms", algorithm="dense")
        add("RH_pruned", "algorithms", algorithm="pruned")
        refs["sensitivity"].extend([prefix+"_T2_m0.1", baseline, prefix+"_RH_pruned"])
        for m in design["t2_minimum_loads"]:
            add(f"T2_m{m:g}", "sensitivity", mechanism="T2", minimum=m)
        for rate in design["direct_hourly_rates"]:
            add(f"direct_rate{rate:g}", "sensitivity", rate=rate)
        for horizon in design["prediction_horizons"]:
            add(f"RH_H{horizon}", "sensitivity", algorithm="pruned", horizon=horizon)
        refs["pooling"].append(baseline)
        add("separate", "pooling", pooling="separate")
    cases.sort(key=lambda c: STAGES.index(c["stage"]))
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("sensitivity labels collide after numeric formatting")
    return {"schema_version": "fixed_n_plan/1", "n": n, "seeds": seeds,
        "cases": cases, "stage_references": refs, "unique_jobs": len(cases),
        "scope": "four directions at fixed N; offline/RH information differs"}


def prepare(args):
    started = time.monotonic()
    gate, core = require_preflight(), freeze_snapshot()
    design = validate_design(read_json(args.design or DEFAULT_DESIGN))
    config = config_from_design(core["design"], core["decisions"])
    plan = make_plan(args.orders, args.seeds, design, config, args.per_run_sec, args.worker_timeout_sec)
    directory = batch_path(args.batch)
    if directory.exists():
        raise ValueError("prepare refuses existing batch; use --run to resume")
    source_directory, source_manifest = None, None
    if args.input_batch:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.input_batch):
            raise ValueError("unsafe input-batch")
        source_directory = ROOT / "simulation/v1_runs" / args.input_batch
        source_manifest = read_json(source_directory / "manifest.json")
        if source_manifest["source_content_sha256"] != core["source_content_sha256"]:
            raise ValueError("imported inputs have a different frozen core")
    elif not Path(args.source).is_file():
        raise ValueError("existing SQLite source required")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "inputs").mkdir()
    instances = {}
    for seed in args.seeds:
        identifier = f"n{args.orders}_s{seed}"
        if source_manifest:
            old = load_instance(source_directory, source_manifest, identifier)
            identity = old["identity"]
            copied = {"batch": args.input_batch, "file_sha256": source_manifest["instances"][identifier]["file_sha256"],
                      "scope": "inputs only; no old solve reused"}
        else:
            from intercity_delivery.v1.experiment_inputs import load_cfs_job_orders
            orders, metadata = load_cfs_job_orders(Path(args.source).resolve(), num_orders=args.orders, seed=seed,
                city_pair=("06-348", "06-488"), T=config.T, t0_minutes=config.t0_minutes,
                tau_periods=config.tau_periods, penalty=core["design"]["baseline"]["penalty_per_unit"])
            identity = {"orders": [asdict(o) for o in orders], "source_metadata": metadata, "config": asdict(config)}
            copied = None
        if len(identity["orders"]) != args.orders:
            raise ValueError("input order count differs")
        payload = {"id": identifier, "n": args.orders, "seed": seed, "identity": identity,
            "input_sha256": fingerprint(identity), "orders_sha256": fingerprint(identity["orders"]), "imported_from": copied}
        path = directory / "inputs" / (identifier+".json")
        write_new(path, payload)
        instances[identifier] = {"path": path.relative_to(directory).as_posix(),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "input_sha256": payload["input_sha256"], "orders_sha256": payload["orders_sha256"]}
    write_new(directory / "planned_jobs.json", plan)
    manifest = {"schema_version": "fixed_n_round/1", "batch": args.batch, "created_utc": now(),
        "n": args.orders, "seeds": args.seeds, "design": design, "core": core, "suite": source_suite(),
        "preflight": gate, "environment": environment(), "instances": instances,
        "plan_sha256": fingerprint(plan), "per_run_sec": args.per_run_sec,
        "worker_timeout_sec": args.worker_timeout_sec, "prepare_wall_sec": time.monotonic()-started,
        "budget_scope": "prepare + all controller sessions; excludes separately reported replay/analysis and human idle"}
    write_new(directory / "manifest.json", manifest)
    return {"batch": str(directory), "unique_jobs": plan["unique_jobs"], "new_solver_runs": 0}


def load_round(name):
    require_preflight()
    directory = batch_path(name)
    manifest, current = read_json(directory / "manifest.json"), freeze_snapshot()
    if any(current[k] != manifest["core"][k] for k in ("source_content_sha256", "source_manifest_sha256", "harness_sha256")):
        raise ValueError("frozen core bytes changed")
    if source_suite() != manifest["suite"]:
        raise ValueError("fixed-N suite changed; prepare new batch")
    plan = read_json(directory / "planned_jobs.json")
    rebuilt = make_plan(manifest["n"], manifest["seeds"], manifest["design"],
        config_from_design(manifest["core"]["design"], manifest["core"]["decisions"]),
        manifest["per_run_sec"], manifest["worker_timeout_sec"])
    if plan != rebuilt or fingerprint(plan) != manifest["plan_sha256"]:
        raise ValueError("immutable predeclared plan changed")
    return directory, manifest, plan


def instance_for(directory, manifest, case):
    return load_instance(directory, manifest, case["instance"])


def semantic_identity(case, instance, manifest, budget):
    return {"case": case, "input_sha256": instance["input_sha256"], "orders_sha256": instance["orders_sha256"],
        "core_sha256": manifest["core"]["source_content_sha256"], "suite_sha256": manifest["suite"]["sha256"],
        "execution_budget": budget}


def independent_replay(case, instance, raw):
    config, orders = config_object(case["config"]), [V1Order(**o) for o in instance["identity"]["orders"]]
    if raw.get("schedule") is None:
        return None
    if case["pooling_model"] == "separate":
        from experiments.fixed_n.pooling import validate_separate
        return validate_separate(config, orders, raw)
    from intercity_delivery.v1.flexible_validation import validate_flexible_schedule
    completed = bool(raw.get("completed"))
    return validate_flexible_schedule(config, orders, flexible_profile(case["information"]), raw["schedule"],
        completed=completed, completed_horizon=config.T if completed else raw.get("current_time", 0),
        reported_objective=raw.get("total_cost") if completed else None)


def validate_raw_identity(case, instance, manifest, raw):
    """A feasible schedule alone cannot identify the model that produced it."""
    meta = raw["metadata"]
    provenance = meta["provenance"]
    expected = {"config": case["config"], "orders": instance["identity"]["orders"],
                "profile": asdict(flexible_profile(case["information"]))}
    if fingerprint(provenance["input"]) != fingerprint(expected) or provenance["input_sha256"] != fingerprint(expected):
        raise ValueError("raw model inputs differ from requested case")
    binding = provenance["source_binding"]
    if (binding["current_sources_match"] is not True or
        binding["manifest_sha256"] != manifest["core"]["source_manifest_sha256"]):
        raise ValueError("raw frozen source binding differs")
    expected_model = "v1_comparison_local_separate_v1" if case["pooling_model"] == "separate" else "v1_flexible_arc_wave_empty_v1"
    if raw["model_id"] != expected_model:
        raise ValueError("raw capacity model differs")
    if meta["threads"] != 1 or meta["solver_seed"] != 0 or raw["requested_time_limit_sec"] != case["logical_budget_sec"]:
        raise ValueError("raw solver controls differ")
    expected_mode = "full" if case["algorithm"] == "full" else "rolling"
    if meta["run_mode"] != expected_mode:
        raise ValueError("raw algorithm mode differs")
    if case["algorithm"] != "full":
        expected_options = {"prediction_horizon": case["H"], "rolling_step": case["h"],
                            "extension_horizon": case["E"], "prune": case["algorithm"] == "pruned"}
        if meta["rolling_options"] != expected_options:
            raise ValueError("raw rolling options differ")


def worker(name, case_id):
    directory, manifest, plan = load_round(name)
    case = next(c for c in plan["cases"] if c["id"] == case_id)
    target = directory / "jobs" / case_id
    request, budget = read_json(target / "request.json"), read_json(directory / "execution_budget.json")
    instance = instance_for(directory, manifest, case)
    identity = semantic_identity(case, instance, manifest, budget)
    if request["identity"] != identity or request["identity_sha256"] != fingerprint(identity):
        raise ValueError("worker identity differs")
    lock = read_json(directory / "run.lock")
    if request["parent_pid"] != lock["pid"] or request["session"] != lock["session"]:
        raise ValueError("worker lacks active parent registration")
    if (target / "raw.json").exists() or (target / "record.json").exists():
        raise ValueError("attempt already completed")
    write_new(target / "worker_claim.json", {"pid": os.getpid(), "claimed_utc": now(), "identity_sha256": fingerprint(identity)})
    config, orders = config_object(case["config"]), [V1Order(**o) for o in instance["identity"]["orders"]]
    if case["pooling_model"] == "separate":
        from experiments.fixed_n.pooling import run_separate
        raw = run_separate(config, orders, case["logical_budget_sec"], output_flag=1)
    else:
        from intercity_delivery.v1.flexible_runner import run_flexible, run_flexible_rolling
        if case["algorithm"] == "full":
            raw = run_flexible(config, orders, flexible_profile(), case["logical_budget_sec"], output_flag=1)
        else:
            from intercity_delivery.v1.flexible_rolling import RollingOptions
            options = RollingOptions(case["H"], case["h"], case["E"], case["algorithm"] == "pruned")
            raw = run_flexible_rolling(config, orders, options, case["logical_budget_sec"], output_flag=1)
    write_new(target / "raw.json", raw)
    return {"status": raw.get("status_name"), "completed": raw.get("completed")}


def consumed_wall(directory, manifest):
    return manifest["prepare_wall_sec"] + sum(read_json(p)["elapsed_sec"] for p in (directory / "sessions").glob("*.json"))


def remaining_timeout(total_budget, already_spent, session_started, hard_timeout, clock=time.monotonic):
    return max(0., min(hard_timeout, total_budget-already_spent-(clock()-session_started)-.1))


def stop_reason(raw, process):
    if process["process_status"] != "EXITED" or process["worker_exit_status"] != 0:
        return "WORKER_ERROR"
    if raw is None:
        return "MISSING_RAW"
    errors = str(raw.get("error", "")) + str(raw.get("warnings", ""))
    if raw.get("status_name") in ("ERROR", "LICENSE_ERROR", "DEPENDENCY_ERROR", "DEPENDENCY_UNAVAILABLE", "SOLVER_ERROR",
                                 "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "NUMERIC") or "license" in errors.lower():
        return "SOLVER_OR_LICENSE_ERROR"
    if raw.get("schedule") is not None and not raw.get("schedule_valid"):
        return "INVALID_SCHEDULE"
    return None


def run_round(args, process_run=subprocess.run):
    positive(args.budget_sec, "budget-sec")
    session_started = time.monotonic()
    directory = batch_path(args.batch)
    if not directory.exists():
        prepare(args)
        session_started = time.monotonic()  # prepare charged separately
    directory, manifest, plan = load_round(args.batch)
    if args.orders != manifest["n"] or args.seeds != manifest["seeds"]:
        raise ValueError("requested N/seeds differ from frozen round")
    if args.per_run_sec != manifest["per_run_sec"] or args.worker_timeout_sec != manifest["worker_timeout_sec"]:
        raise ValueError("requested solve/worker budgets differ from frozen round")
    if args.design and validate_design(read_json(args.design)) != manifest["design"]:
        raise ValueError("requested design differs from frozen round")
    if (directory / "stop.json").exists():
        raise ValueError("round stopped after failure; inspect stop.json; no retry")
    budget = {"cumulative_wall_sec": args.budget_sec, "scope": manifest["budget_scope"]}
    budget_file = directory / "execution_budget.json"
    if budget_file.exists():
        if read_json(budget_file) != budget:
            raise ValueError("cumulative budget cannot be recharged/changed")
    else:
        write_new(budget_file, budget)
    (directory / "sessions").mkdir(exist_ok=True)
    (directory / "jobs").mkdir(exist_ok=True)
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    lock = directory / "run.lock"
    write_new(lock, {"pid": os.getpid(), "session": session, "started_utc": now()})
    spent, attempts, stopped = consumed_wall(directory, manifest), [], None
    try:
        for stage in STAGES:
            for case in (c for c in plan["cases"] if c["stage"] == stage):
                target = directory / "jobs" / case["id"]
                if target.exists():
                    continue  # even a failed/interrupted reservation is never retried
                timeout = remaining_timeout(args.budget_sec, spent, session_started, case["worker_timeout_sec"])
                if timeout <= 0:
                    stopped = "BUDGET_EXHAUSTED"
                    break
                instance = instance_for(directory, manifest, case)
                identity = semantic_identity(case, instance, manifest, budget)
                target.mkdir(exist_ok=False)
                write_new(target / "request.json", {"identity": identity, "identity_sha256": fingerprint(identity),
                    "parent_pid": os.getpid(), "session": session, "created_utc": now()})
                command = [sys.executable, "-B", str(ROOT / "experiments/run_fixed_n.py"), "--worker", case["id"],
                    "--batch", args.batch, "--orders", str(args.orders), "--seeds", ",".join(map(str, args.seeds))]
                timeout = remaining_timeout(args.budget_sec, spent, session_started, case["worker_timeout_sec"])
                if timeout <= 0:
                    process = {"worker_exit_status": None, "process_status": "NOT_RUN_BUDGET",
                        "parent_wall_sec": 0., "stdout": "", "stderr": ""}
                else:
                    process = execute_attempt(command, timeout, process_run=process_run)
                (target / "stdout.log").write_text(process.pop("stdout"), encoding="utf-8")
                (target / "stderr.log").write_text(process.pop("stderr"), encoding="utf-8")
                record = {"case": case, "identity_sha256": fingerprint(identity), "command": command,
                    "effective_process_timeout_sec": timeout, "finished_utc": now(), **process}
                raw_path = target / "raw.json"
                raw = read_json(raw_path) if raw_path.exists() else None
                reason = stop_reason(raw, process)
                if process["process_status"] == "NOT_RUN_BUDGET":
                    reason, stopped = None, "BUDGET_EXHAUSTED"
                if raw is not None:
                    record.update(raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                        solver_status=raw.get("status_name"), completed=raw.get("completed"))
                    try:
                        validate_raw_identity(case, instance, manifest, raw)
                        evidence = independent_replay(case, instance, raw)
                        record.update(independent_replay="PASSED" if evidence is not None else "NO_SCHEDULE", replay_evidence=evidence)
                    except Exception as exc:
                        record.update(independent_replay="FAILED", replay_error=str(exc))
                        reason = "REPLAY_FAILED"
                if reason:
                    record["stop_reason"] = reason
                write_new(target / "record.json", record)
                attempts.append(case["id"])
                if reason:
                    stopped = reason
                    write_new(directory / "stop.json", {"reason": reason, "case": case["id"], "timestamp_utc": now()})
                if stopped:
                    break
            stage_state = {"stage": stage, "timestamp_utc": now(), "planned_references": len(plan["stage_references"][stage]),
                "recorded_references": sum((directory / "jobs" / i / "record.json").exists() for i in plan["stage_references"][stage]),
                "session": session, "stop_reason": stopped}
            (directory / "stage_status").mkdir(exist_ok=True)
            write_new(directory / "stage_status" / (session+"_"+stage+".json"), stage_state)
            if stopped:
                break
    finally:
        write_new(directory / "sessions" / (session+".json"), {"session": session,
            "elapsed_sec": time.monotonic()-session_started, "attempts": attempts, "stopped": stopped})
        lock.unlink()
    return {"batch": str(directory), "new_attempts": len(attempts), "stop_reason": stopped,
        "cumulative_wall_sec": consumed_wall(directory, manifest), "budget_sec": args.budget_sec}


def collect(name):
    directory, manifest, plan = load_round(name)
    rows, replay = [], []
    budget_path = directory / "execution_budget.json"
    budget = read_json(budget_path) if budget_path.exists() else None
    for case in plan["cases"]:
        target = directory / "jobs" / case["id"]
        row = {"id": case["id"], "label": case["label"], "stage": case["stage"], "n": case["n"], "seed": case["seed"],
            "information": case["information"], "pooling_model": case["pooling_model"],
            "status": "INTERRUPTED" if target.exists() else "UNATTEMPTED", "completed_feasible": False,
            "cost": None, "served_rate": None, "parent_wall_sec": None, "solver_calls": 0,
            "global_bound": None, "global_gap": None,
            "orders_sha256": manifest["instances"][case["instance"]]["orders_sha256"]}
        replay_row = {"id": case["id"], "result": "NO_SCHEDULE"}
        if (target / "record.json").exists():
            record, request = read_json(target / "record.json"), read_json(target / "request.json")
            instance = instance_for(directory, manifest, case)
            identity = semantic_identity(case, instance, manifest, budget)
            if (request["identity"] != identity or request["identity_sha256"] != fingerprint(identity)
                or record["identity_sha256"] != fingerprint(identity) or record["case"] != case):
                raise ValueError("attempt identity differs: " + case["id"])
            row.update(status=record.get("solver_status", record["process_status"]),
                parent_wall_sec=record["parent_wall_sec"], process_status=record["process_status"])
            if "raw_sha256" in record:
                raw_path = target / "raw.json"
                if hashlib.sha256(raw_path.read_bytes()).hexdigest() != record["raw_sha256"]:
                    raise ValueError("raw changed: " + case["id"])
                raw = read_json(raw_path)
                validate_raw_identity(case, instance, manifest, raw)
                try:
                    evidence = independent_replay(case, instance, raw)
                    replay_row.update(result="PASSED" if evidence is not None else "NO_SCHEDULE", evidence=evidence)
                except Exception as exc:
                    replay_row.update(result="FAILED", error=str(exc))
                    evidence = None
                if evidence is not None:
                    checked = {**raw, "validation": evidence, "schedule_valid": True}
                    metrics = summarize_result(checked, case["config"], instance["identity"]["orders"])
                    row.update(completed_feasible=metrics["completed_feasible"], cost=metrics["total_cost"],
                        served_rate=metrics["served_rate"], direct_share=metrics["direct_share"],
                        explicit_empty_vehicle_tasks=metrics["explicit_empty_vehicle_tasks"],
                        zero_cargo_direct_vehicle_tasks=metrics["zero_cargo_direct_vehicle_tasks"],
                        lead_time_mean_minutes=metrics["lead_time_mean_minutes"], manual_busy_fraction=metrics["manual_busy_fraction"],
                        phase_timing_sec=metrics["phase_timing_sec"], windows=metrics["window_statistics"])
                row["solver_calls"] = raw.get("solver_optimize_attempts", 0)
                row.update(bound_scope=raw.get("bound_scope"), optimal=raw.get("optimal", False),
                    model_id=raw.get("model_id"), raw_budget_exceeded=raw.get("budget_exceeded"),
                    runner_elapsed_sec=raw.get("elapsed_sec"),
                    variable_count=raw.get("variable_count"), constraint_count=raw.get("constraint_count"))
                if case["algorithm"] == "full":
                    row.update(global_bound=raw.get("best_bound"), global_gap=raw.get("mip_gap"))
        rows.append(row)
        replay.append(replay_row)
    return directory, manifest, plan, rows, replay


def comparisons(plan):
    unique = {c["label"]: c for c in plan["cases"]}
    pairs = [("mechanism", "T0", x) for x in ("T1", "T2_m0.1", "T3")]
    pairs += [("mechanism", "T3", "T3_m0.1"), ("mechanism", "T2_m0.1", "T3_m0.1"), ("algorithms", "T3", "RH_dense"),
        ("algorithms", "T3", "RH_pruned"), ("algorithms", "RH_dense", "RH_pruned"), ("pooling", "T3", "separate")]
    for label, case in unique.items():
        if case["stage"] == "sensitivity":
            baseline = "T2_m0.1" if label.startswith("T2") else "RH_pruned" if label.startswith("RH") else "T3"
            pairs.append(("sensitivity", baseline, label))
    return pairs


def paired_rows(plan, rows):
    lookup = {(r["seed"], r["label"]): r for r in rows}
    output, aggregates = [], []
    for stage, baseline, treatment in comparisons(plan):
        values, service_values, speedups = [], [], []
        for seed in plan["seeds"]:
            first, second = lookup[(seed, baseline)], lookup[(seed, treatment)]
            matched = first["orders_sha256"] == second["orders_sha256"]
            complete = matched and first["completed_feasible"] and second["completed_feasible"]
            delta = second["cost"]-first["cost"] if complete else None
            interval = None
            if (complete and first["information"] == second["information"] == "offline"
                and first.get("global_bound") is not None and second.get("global_bound") is not None):
                interval = [second["global_bound"]-first["cost"], second["cost"]-first["global_bound"]]
            service = second["served_rate"]-first["served_rate"] if complete else None
            speedup = first["parent_wall_sec"]/second["parent_wall_sec"] if complete and second["parent_wall_sec"] else None
            output.append({"stage": stage, "baseline": baseline, "treatment": treatment, "seed": seed,
                "same_orders": matched, "baseline_status": first["status"], "treatment_status": second["status"],
                "eligible": complete, "cost_delta_treatment_minus_baseline": delta,
                "served_rate_delta": service, "wall_speedup_baseline_over_treatment": speedup,
                "optimal_cost_delta_interval_from_solver_bounds": interval,
                "interval_scope": "reported full-model bounds for each own objective" if interval is not None else "not_available; RH has no full-horizon bound",
                "changed_objective": treatment.startswith("direct_rate"),
                "baseline_global_gap": first.get("global_gap"), "treatment_global_gap": second.get("global_gap"),
                "information_comparison": "same" if first["information"] == second["information"] else "offline_oracle_vs_release_revealed",
                "policy_scope": "direct-channel price, including zero-cargo direct" if treatment.startswith("direct_rate") else "explicit case difference"})
            if delta is not None:
                values.append(delta)
                service_values.append(service)
            if speedup is not None:
                speedups.append(speedup)
        aggregates.append({"stage": stage, "baseline": baseline, "treatment": treatment,
            "cost_delta": mean_ci95(values, expected_n=len(plan["seeds"])),
            "served_rate_delta": mean_ci95(service_values, expected_n=len(plan["seeds"])),
            "wall_speedup": mean_ci95(speedups, expected_n=len(plan["seeds"])),
            "scope": "complete replay-valid incumbents only; differing gaps retained; descriptive seed variation"})
    return output, aggregates


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def report(name, replay_only=False):
    started = time.monotonic()
    directory, manifest, plan, rows, replay = collect(name)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = directory / "analyses" / stamp
    destination.mkdir(parents=True, exist_ok=False)
    write_new(destination / "replay.json", {"rows": replay, "elapsed_sec": time.monotonic()-started})
    failed = sum(r["result"] == "FAILED" for r in replay)
    if replay_only:
        return {"analysis": str(destination), "replayed": sum(r["result"] == "PASSED" for r in replay), "failed": failed}
    if failed:
        raise ValueError("replay failed; comparative interpretation not exported")
    pairs, aggregates = paired_rows(plan, rows)
    write_csv(destination / "runs.csv", rows)
    windows = [{"id": row["id"], "window_index": index,
        "bound_scope": "window_local_only_not_global", **window}
        for row in rows for index, window in enumerate(row.get("windows", []))]
    write_csv(destination / "windows.csv", windows)
    write_csv(destination / "paired.csv", pairs)
    stages = {}
    for stage in STAGES:
        ids = plan["stage_references"][stage]
        selected = [r for r in rows if r["id"] in ids]
        write_csv(destination / (stage+".csv"), selected)
        stages[stage] = {"planned_references": len(ids), "completed_feasible": sum(r["completed_feasible"] for r in selected),
            "status_counts": {s: sum(r["status"] == s for r in selected) for s in sorted(set(r["status"] for r in selected))}}
    from experiments.fixed_n.pooling import capacity_curves
    summary = {"schema_version": "fixed_n_analysis/1", "generated_utc": now(), "n": manifest["n"], "seeds": manifest["seeds"],
        "stages": stages, "planned_unique_jobs": len(rows), "completed_feasible": sum(r["completed_feasible"] for r in rows),
        "rows": rows, "paired": pairs, "paired_aggregates": aggregates, "capacity_curves": capacity_curves(),
        "core_sha256": manifest["core"]["source_content_sha256"], "suite_sha256": manifest["suite"]["sha256"],
        "cumulative_execution_wall_sec": consumed_wall(directory, manifest),
        "limits": ["RH window bounds/gaps are not global bounds", "partial/no-incumbent cases retain denominator; no full-cost ranking",
            "CI only for all five eligible seeds; timing order fixed without repeated timing trials",
            "separate ablates local pooling only; direct capacity unchanged"], "analysis_elapsed_sec": time.monotonic()-started}
    write_new(destination / "summary.json", summary)
    lines = [f"# Fixed-N round {name}", "", f"N={manifest['n']}; seeds={manifest['seeds']}", "",
        f"Complete independent replay: {summary['completed_feasible']}/{len(rows)} planned unique runs.", "",
        "| Direction | Planned references | Completed feasible |", "|---|---:|---:|"]
    lines.extend(f"| {s} | {v['planned_references']} | {v['completed_feasible']} |" for s, v in stages.items())
    lines.extend(["", "All directions share immutable orders; references reuse the same baseline run.",
        "Offline/RH information differs; RH window bounds are not full-horizon lower bounds.",
        "A finite minimum excludes positive loads below threshold; it does not ensure every vehicle is loaded.",
        "Direct-channel price sensitivity also changes the zero-cargo Direct channel; explicit empty stays fixed.",
        "See summary.json, paired.csv and direction CSVs for statuses and matched comparisons."])
    (destination / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return {"analysis": str(destination), "planned": len(rows), "completed": summary["completed_feasible"], "replay_failed": failed}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    for name in ("plan", "prepare", "run", "analyze", "replay"):
        mode.add_argument("--"+name, action="store_true")
    mode.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--orders", type=int, required=True)
    p.add_argument("--seeds", default="4101,4102,4103,4104,4105")
    p.add_argument("--batch", required=True)
    p.add_argument("--budget-sec", type=float)
    p.add_argument("--per-run-sec", type=float, default=5.)
    p.add_argument("--worker-timeout-sec", type=float, default=15.)
    p.add_argument("--source", default=str(ROOT / "data/cfs_2022_pums.sqlite"))
    p.add_argument("--input-batch")
    p.add_argument("--design", help="finite sensitivity levels JSON; see fixed_n/default_design.json")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.seeds = [int(s) for s in args.seeds.split(",")]
    validate_round(args.orders, args.seeds, args.per_run_sec, args.worker_timeout_sec)
    if args.run and args.budget_sec is None:
        raise ValueError("--run requires explicit --budget-sec; no solve started")
    if args.worker:
        output = worker(args.batch, args.worker)
    elif args.prepare:
        output = prepare(args)
    elif args.run:
        output = run_round(args)
        # Analysis is read-only postprocessing, outside the solver/controller cap.
        analysis_started = time.monotonic()
        try:
            output["analysis"] = report(args.batch)
        except Exception as exc:
            output["analysis_error"] = str(exc)
        output["postprocessing_wall_sec"] = time.monotonic()-analysis_started
    elif args.analyze or args.replay:
        output = report(args.batch, replay_only=args.replay)
    else:
        core = freeze_snapshot()
        plan = make_plan(args.orders, args.seeds, validate_design(read_json(args.design or DEFAULT_DESIGN)),
            config_from_design(core["design"], core["decisions"]), args.per_run_sec, args.worker_timeout_sec)
        output = {"mode": "PLAN_ONLY", "solver_runs": 0, "batch": str(batch_path(args.batch)), "plan": plan}
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
