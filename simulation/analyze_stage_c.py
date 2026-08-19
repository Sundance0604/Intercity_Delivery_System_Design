"""Aggregate reproducible Stage C batch summaries without loading detail JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


SCENARIO_PATTERN = re.compile(r"^simulation_stage_c_(.+)_fleet_([0-9.]+)$")
METRICS = (
    "Total_Cost",
    "Unserved_Rate",
    "Solve_Time_Sec",
    "MIP_Gap",
    "Direct_Ratio",
    "Manual_Usage",
    "Auto_Usage",
)


def load_rows(batch_dirs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for batch_dir in batch_dirs:
        summary_path = batch_dir / "run_summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for source in payload["result_rows"]:
            row = dict(source)
            key = (str(row["Exp_ID"]), str(row["Solver"]))
            if key in seen:
                continue
            seen.add(key)
            match = SCENARIO_PATTERN.match(str(row.get("Scenario", "")))
            row["Mechanism"] = match.group(1) if match else "unspecified"
            row["Fleet_Scale"] = float(match.group(2)) if match else None
            row["Source_Batch"] = batch_dir.name
            rows.append(row)
    return rows


def numeric(values: list[object]) -> list[float]:
    return [float(value) for value in values if value is not None and value != ""]


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("Num_Orders"),
            row.get("Model_penalty_lost"),
            row.get("Mechanism"),
            row.get("Fleet_Scale"),
            row.get("Solver"),
        )
        grouped[key].append(row)

    output: list[dict] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        result = {
            "Num_Orders": key[0],
            "Penalty_Lost": key[1],
            "Mechanism": key[2],
            "Fleet_Scale": key[3],
            "Solver": key[4],
            "Replications": len(group),
            "Optimal_Rate": sum(row.get("Status") == 2 for row in group) / len(group),
        }
        for metric in METRICS:
            values = numeric([row.get(metric) for row in group])
            result[f"Mean_{metric}"] = statistics.fmean(values) if values else None
            result[f"SD_{metric}"] = statistics.stdev(values) if len(values) > 1 else None
            result[f"Max_{metric}"] = max(values) if values else None
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总阶段 C 的 run_summary.json。")
    parser.add_argument("batch_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    batch_dirs = [path.expanduser().resolve() for path in args.batch_dirs]
    rows = load_rows(batch_dirs)
    summary = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage_c_rows.csv", rows)
    write_csv(args.output_dir / "stage_c_summary.csv", summary)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_batches": [str(path) for path in batch_dirs],
        "row_count": len(rows),
        "summary": summary,
    }
    (args.output_dir / "stage_c_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Loaded {len(rows)} rows from {len(batch_dirs)} batches.")
    print(f"Wrote {len(summary)} grouped rows to {args.output_dir}.")


if __name__ == "__main__":
    main()
