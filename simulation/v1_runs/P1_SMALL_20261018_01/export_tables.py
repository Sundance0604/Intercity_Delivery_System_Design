"""Lossless small-pilot tables from the preserved raw/paired analysis.

This postprocessor never solves or changes the frozen execution sources.
Its own hash and the analysis hash are recorded beside the derived tables.
"""
import csv
import hashlib
import json
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent
ANALYSIS = BASE / "analyses/20261018T153421101497Z.json"


def read(path):
    return json.loads(path.read_bytes())


def save_csv(path, rows):
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    analysis = read(ANALYSIS)
    rows, windows, records, raw_by_id = [], [], {}, {}
    for path in sorted((BASE / "jobs").glob("*/record.json")):
        record = read(path)
        raw_path = path.parent / "raw.json"
        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != record["raw_sha256"]:
            raise ValueError("raw bytes changed")
        raw = read(raw_path)
        job, metrics = record["job"], record["metrics"]
        identifier = job["id"]
        records[identifier], raw_by_id[identifier] = record, raw
        row = {**job, "status": record["status"], "input_sha256": record["input_sha256"],
               "orders_sha256": record["orders_sha256"], "config_sha256": record["config_sha256"],
               "source_content_sha256": record["source_content_sha256"],
               "parent_wall_sec": record["parent_wall_sec"], "runner_wall_sec": raw["elapsed_sec"],
               "solver_optimize_attempts": raw["solver_optimize_attempts"],
               "best_bound": raw["best_bound"], "mip_gap": raw["mip_gap"],
               "budget_exceeded": raw["budget_exceeded"],
               "max_constraint_violation": raw["validation"]["max_constraint_violation"]}
        for key in ("completed_feasible", "global_optimal_oracle", "total_cost", "served_quantity", "unserved_quantity",
                    "served_rate", "direct_share", "explicit_empty_vehicle_tasks", "zero_cargo_direct_vehicle_tasks",
                    "manual_busy_fraction", "av_busy_fraction", "lead_time_mean_minutes", "lead_time_p50_minutes", "lead_time_p95_minutes"):
            row[key] = metrics.get(key)
        row.update({"phase_"+k: v for k, v in raw["phase_timing_sec"].items()})
        row.update({"cost_"+k: v for k, v in raw["cost_components"].items()})
        rows.append(row)
        for window in raw["windows"]:
            window_row = {"job_id": identifier, "n": job["n"], "seed": job["seed"], "algorithm": job["algorithm"]}
            for key in ("window_start", "start_window_end", "completion_window_end", "control_end", "status", "feasible",
                        "objective", "best_bound", "mip_gap", "bound_scope", "cargo_variable_count", "variable_count",
                        "binary_variable_count", "integer_variable_count", "continuous_variable_count", "constraint_count",
                        "vehicle_selector_domain_sha256", "elapsed_sec"):
                window_row[key] = window.get(key)
            window_row["known_order_count"] = len(window["known_order_ids"])
            window_row.update({"phase_"+k: v for k, v in window["phase_timing_sec"].items()})
            pruning = window.get("pruning")
            window_row["dense_cargo_candidates"] = sum(pruning["before"].values()) if pruning else window["cargo_variable_count"]
            window_row["retained_cargo_candidates"] = sum(pruning["after"].values()) if pruning else window["cargo_variable_count"]
            windows.append(window_row)
    paired = []
    for pair in analysis["pilot_pairs"]:
        for item in pair["seeds"]:
            paired.append({"kind": "pilot", "comparison": pair["right"]+"_minus_"+pair["left"], "n": pair["n"], **item})
    for pair in analysis["mechanism_sensitivity_pairs"]:
        for item in pair["seeds"]:
            paired.append({"kind": "mechanism_sensitivity", "comparison": pair["comparison"], "n": 10, **item})
    output = BASE / "tables"
    output.mkdir(exist_ok=False)
    save_csv(output / "v1_paired_results.csv", paired)
    save_csv(output / "v1_run_summary.csv", rows)
    save_csv(output / "v1_window_summary.csv", windows)
    pilot = []
    for n in (10, 20, 40):
        for algorithm in ("full", "dense", "pruned"):
            group = [r for r in rows if r["n"] == n and r["algorithm"] == algorithm and r["mechanism"] == "T3" and r["minimum"] == 0]
            pilot.append({"n": n, "algorithm": algorithm, "count": len(group),
                "mean_cost": statistics.mean(r["total_cost"] for r in group),
                "mean_served_rate": statistics.mean(r["served_rate"] for r in group),
                "mean_parent_wall_sec": statistics.mean(r["parent_wall_sec"] for r in group),
                "mean_manual_busy_fraction": statistics.mean(r["manual_busy_fraction"] for r in group),
                "mean_av_busy_fraction": statistics.mean(r["av_busy_fraction"] for r in group)})
    fair_domains = []
    for n in (10, 20, 40):
        for seed in range(4101, 4106):
            dense = raw_by_id[f"n{n}_s{seed}_T3_m0_dense"]
            pruned = raw_by_id[f"n{n}_s{seed}_T3_m0_pruned"]
            left = {w["window_start"]: w for w in dense["windows"]}
            before = after = 0
            same = True
            for window in pruned["windows"]:
                same = same and left[window["window_start"]]["vehicle_selector_domain_sha256"] == window["vehicle_selector_domain_sha256"]
                before += sum(window["pruning"]["before"].values())
                after += sum(window["pruning"]["after"].values())
            fair_domains.append({"n": n, "seed": seed, "all_vehicle_selector_signatures_equal": same,
                "pruned_path_dense_cargo_candidates": before, "retained_cargo_candidates": after,
                "reduction": 1-after/before if before else None,
                "scope": "candidate reduction follows pruned trajectory; domain equality is not equality of historical constraints"})
    session_wall = sum(read(p)["wall_sec"] for p in (BASE / "sessions").glob("*.json"))
    manifest = read(BASE / "manifest.json")
    micro = read(BASE.parents[1] / "v1_micro/P1_SMALL_20261018_01/gate_evidence.json")
    micro_calls = sum(r.get("solver_optimize_attempts", r.get("raw_result", {}).get("solver_optimize_attempts", 0)) for r in micro["runs"])
    summary = {"evidence_class": "bounded_small_scale_full_model_experiment_not_large_scale_publication_package",
        "planned_batch_runs": 75, "completed_feasible_batch_runs": sum(r["completed_feasible"] for r in rows),
        "total_top_level_runs_including_micro": len(rows)+len(micro["runs"]),
        "total_optimize_attempts_including_micro": sum(r["solver_optimize_attempts"] for r in rows)+micro_calls,
        "budget_execution_wall_sec": manifest["sampling_seconds"]+micro["elapsed_sec"]+session_wall,
        "prepare_wall_sec": manifest["sampling_seconds"], "micro_wall_sec": micro["elapsed_sec"], "sessions_wall_sec": session_wall,
        "budget_limit_sec": 600, "max_residual_mixed_units": max(r["max_constraint_violation"] for r in rows),
        "run_status_counts": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
        "time_limited_rows": [{k: r[k] for k in ("id", "total_cost", "best_bound", "mip_gap", "budget_exceeded")} for r in rows if r["status"] == "TIME_LIMIT"],
        "pilot": pilot, "domain_and_pruning_checks": fair_domains,
        "pilot_pairs": analysis["pilot_pairs"], "mechanism_sensitivity_pairs": analysis["mechanism_sensitivity_pairs"],
        "analysis_sha256": hashlib.sha256(ANALYSIS.read_bytes()).hexdigest(),
        "postprocessor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_content_sha256": manifest["source_content_sha256"],
        "new_solver_runs_by_postprocessor": 0}
    with (output / "v1_paired_results.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: summary[k] for k in ("completed_feasible_batch_runs", "total_top_level_runs_including_micro",
        "total_optimize_attempts_including_micro", "budget_execution_wall_sec", "run_status_counts", "time_limited_rows", "pilot")}, indent=2))


if __name__ == "__main__":
    main()
