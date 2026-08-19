"""Verify paired inputs and create a cross-batch simulation comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_full_result(batch: Path) -> dict[str, Any]:
    files = sorted((batch / "results").glob("full_experiment_results_*.json"))
    if len(files) != 1:
        raise ValueError(f"{batch} 中完整结果 JSON 数量不是 1：{len(files)}")
    return json.loads(files[0].read_text(encoding="utf-8"))


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batches", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    batches = [path.resolve() for path in args.batches]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    order_hashes: dict[str, dict[str, str]] = defaultdict(dict)
    rows: list[dict[str, Any]] = []
    for batch in batches:
        payload = load_full_result(batch)
        analysis = json.loads((batch / "analysis.json").read_text(encoding="utf-8"))
        analysis_rows = {
            (row["experiment_id"], row["solver"]): row for row in analysis["rows"]
        }
        for experiment in payload["experiments"]:
            experiment_id = experiment["experiment_id"]
            order_hashes[experiment_id][batch.name] = canonical_hash(
                experiment["orders"]
            )
            for result in experiment["solver_results"]:
                rows.append(
                    {
                        "batch": batch.name,
                        **analysis_rows[(experiment_id, result["solver"])],
                    }
                )

    pairing = {
        experiment_id: {
            "hashes": hashes,
            "identical": len(set(hashes.values())) == 1,
        }
        for experiment_id, hashes in sorted(order_hashes.items())
    }
    if not all(item["identical"] for item in pairing.values()):
        raise ValueError("批次之间存在相同 Exp_ID 但订单不一致的情况。")

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    by_experiment: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(int(row["num_orders"]), row["solver"])].append(row)
        by_experiment[row["experiment_id"]][row["solver"]] = row

    aggregates = []
    for (num_orders, solver), items in sorted(grouped.items()):
        aggregates.append(
            {
                "num_orders": num_orders,
                "solver": solver,
                "replications": len(items),
                "mean_total_cost": mean([float(item["total_cost"]) for item in items]),
                "mean_unserved_rate": mean(
                    [float(item["unserved_rate"]) for item in items]
                ),
                "mean_solve_time_sec": mean(
                    [float(item["solve_time_sec"]) for item in items]
                ),
                "max_window_time_sec": max(
                    float(item["window_time_max_sec"] or 0.0) for item in items
                ),
                "mean_arc_reduction": mean(
                    [
                        float(item["arc_reduction_mean"])
                        for item in items
                        if item["arc_reduction_mean"] is not None
                    ]
                ),
            }
        )

    paired_differences = []
    for experiment_id, solver_rows in sorted(by_experiment.items()):
        benchmark = solver_rows.get("flexible_direct_mip")
        candidate = solver_rows.get("paper_candidate_mip")
        heuristic = solver_rows.get("paper_priority_heuristic")
        rolling = solver_rows.get("flexible_direct_rolling")
        record: dict[str, Any] = {"experiment_id": experiment_id}
        if benchmark and candidate:
            base_cost = max(abs(float(benchmark["total_cost"])), 1e-9)
            record["candidate_vs_exact_cost_gap"] = (
                float(candidate["total_cost"]) - float(benchmark["total_cost"])
            ) / base_cost
            record["candidate_vs_exact_unserved_difference"] = (
                float(candidate["unserved_rate"])
                - float(benchmark["unserved_rate"])
            )
        if candidate and heuristic:
            candidate_cost = max(abs(float(candidate["total_cost"])), 1e-9)
            record["heuristic_vs_candidate_cost_gap"] = (
                float(heuristic["total_cost"]) - float(candidate["total_cost"])
            ) / candidate_cost
            candidate_time = max(
                float(candidate["solve_time_sec"]),
                float(candidate["window_time_mean_sec"] or 0.0)
                * int(candidate["window_count"]),
                1e-6,
            )
            heuristic_time = max(
                float(heuristic["solve_time_sec"]),
                float(heuristic["window_time_mean_sec"] or 0.0)
                * int(heuristic["window_count"]),
                1e-6,
            )
            record["heuristic_speedup_vs_candidate"] = (
                candidate_time / heuristic_time
            )
        if benchmark and rolling:
            record["rolling_vs_exact_unserved_difference"] = (
                float(rolling["unserved_rate"])
                - float(benchmark["unserved_rate"])
            )
        paired_differences.append(record)

    report = {
        "batches": [str(batch) for batch in batches],
        "paired_input_check": pairing,
        "all_paired_inputs_identical": True,
        "rows": rows,
        "aggregates": aggregates,
        "paired_differences": paired_differences,
    }
    (output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Stage B paired comparison",
        "",
        f"- 批次：{', '.join(batch.name for batch in batches)}",
        f"- 配对订单校验：通过（{len(pairing)} 个算例）",
        "",
        "| Orders | Solver | N | Mean cost | Mean unserved | Mean time (s) | Max window (s) | Mean arc reduction |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        reduction = item["mean_arc_reduction"]
        lines.append(
            "| {num_orders} | {solver} | {replications} | {cost:.2f} | "
            "{unserved:.2%} | {time:.3f} | {window:.4f} | {reduction} |".format(
                num_orders=item["num_orders"],
                solver=item["solver"],
                replications=item["replications"],
                cost=item["mean_total_cost"],
                unserved=item["mean_unserved_rate"],
                time=item["mean_solve_time_sec"],
                window=item["max_window_time_sec"],
                reduction="" if reduction is None else f"{reduction:.2%}",
            )
        )
    lines.extend(
        [
            "",
            "逐算例目标差距与订单 SHA-256 见 `comparison.json`。",
            "",
        ]
    )
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["aggregates"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
