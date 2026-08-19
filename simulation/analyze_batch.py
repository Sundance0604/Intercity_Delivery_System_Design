"""Summarize one simulation batch into JSON and Markdown reports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def window_summary(detail: dict[str, Any]) -> dict[str, Any]:
    windows = detail.get("windows", [])
    times = [float(item["solve_time_sec"]) for item in windows]
    reductions = [
        float(item["diagnostics"]["non_direct_arc_reduction_rate"])
        for item in windows
        if item.get("diagnostics", {}).get("non_direct_arc_reduction_rate")
        is not None
    ]
    gaps = [
        float(item["mip_gap"])
        for item in windows
        if item.get("mip_gap") is not None
    ]
    variables = [
        int(item["diagnostics"]["variables"])
        for item in windows
        if item.get("diagnostics", {}).get("variables") is not None
    ]
    constraints = [
        int(item["diagnostics"]["constraints"])
        for item in windows
        if item.get("diagnostics", {}).get("constraints") is not None
    ]
    ratio_violations = [
        bool(item.get("diagnostics", {}).get("direct_ratio_violation"))
        for item in windows
        if "direct_ratio_violation" in item.get("diagnostics", {})
    ]
    return {
        "window_count": len(windows),
        "window_time_mean_sec": statistics.fmean(times) if times else None,
        "window_time_p95_sec": percentile(times, 0.95),
        "window_time_max_sec": max(times) if times else None,
        "arc_reduction_mean": statistics.fmean(reductions) if reductions else None,
        "arc_reduction_min": min(reductions) if reductions else None,
        "arc_reduction_max": max(reductions) if reductions else None,
        "mip_gap_mean": statistics.fmean(gaps) if gaps else None,
        "variable_count_max": max(variables) if variables else None,
        "constraint_count_max": max(constraints) if constraints else None,
        "direct_ratio_violation_windows": sum(ratio_violations),
    }


def analyze(batch_dir: Path) -> dict[str, Any]:
    result_files = sorted(
        (batch_dir / "results").glob("full_experiment_results_*.json")
    )
    if len(result_files) != 1:
        raise ValueError(
            f"预期恰好一个完整结果 JSON，实际找到 {len(result_files)} 个。"
        )
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    rows = []
    for experiment in payload["experiments"]:
        for result in experiment["solver_results"]:
            total_demand = sum(
                float(order["quantity"])
                for order in experiment["orders"].values()
            )
            row = {
                "experiment_id": experiment["experiment_id"],
                "seed": experiment["generation_parameters"]["seed"],
                "num_orders": len(experiment["orders"]),
                "total_demand": total_demand,
                "solver": result["solver"],
                "status": result["status"],
                "solve_time_sec": result["solve_time_sec"],
                "total_cost": result["total_cost"],
                "unserved_rate": result["unserved_rate"],
                "auto_usage": result["auto_usage"],
                "manual_usage": result["manual_usage"],
                **window_summary(result.get("detail") or {}),
            }
            rows.append(row)
    return {
        "batch_id": batch_dir.name,
        "source_result": str(result_files[0]),
        "city_pair": payload.get("city_pair"),
        "city_names": payload.get("city_names"),
        "rows": rows,
    }


def write_report(batch_dir: Path, analysis: dict[str, Any]) -> None:
    analysis_path = batch_dir / "analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# Batch analysis: {analysis['batch_id']}",
        "",
        "| Solver | Cost | Unserved rate | Total time (s) | Max window (s) | Mean arc reduction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["rows"]:
        reduction = row["arc_reduction_mean"]
        reduction_text = "" if reduction is None else f"{reduction:.2%}"
        lines.append(
            "| {solver} | {cost:.3f} | {unserved:.2%} | {time:.3f} | "
            "{window:.4f} | {reduction} |".format(
                solver=row["solver"],
                cost=float(row["total_cost"]),
                unserved=float(row["unserved_rate"]),
                time=float(row["solve_time_sec"]),
                window=float(row["window_time_max_sec"] or 0.0),
                reduction=reduction_text,
            )
        )
    lines.extend(
        [
            "",
            "说明：该报告由 `simulation/analyze_batch.py` 从完整批次 JSON 自动生成。",
            "",
        ]
    )
    (batch_dir / "stage_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dir", type=Path)
    args = parser.parse_args()
    batch_dir = args.batch_dir.resolve()
    analysis = analyze(batch_dir)
    write_report(batch_dir, analysis)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
