"""Plot saved small-run evidence, keeping unsuccessful attempts visible."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.specs.run_v1_p1 import load_batch, load_instance
from intercity_delivery.v1.experiment_metrics import summarize_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    args = parser.parse_args()
    directory, manifest = load_batch(args.batch)
    destination = directory / "figures"
    destination.mkdir(exist_ok=False)
    rows = []
    for path in sorted((directory / "jobs").glob("*/record.json")):
        record = json.loads(path.read_bytes())
        raw_path = path.parent / "raw.json"
        if raw_path.is_file() and record.get("raw_sha256"):
            if hashlib.sha256(raw_path.read_bytes()).hexdigest() != record["raw_sha256"]:
                raise ValueError("raw source changed")
            instance = load_instance(directory, manifest, record["job"]["instance"])
            metrics = summarize_result(json.loads(raw_path.read_bytes()), record["config"], instance["identity"]["orders"])
        else:
            metrics = {"completed_feasible": False}
        rows.append({**record, "metrics": metrics})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.dpi": 180})
    colors = {"full": "#273b63", "dense": "#ce7135", "pruned": "#24857e"}
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for algorithm in ("full", "dense", "pruned"):
        groups = [[r for r in rows if r["job"]["n"] == n and r["job"]["algorithm"] == algorithm
                   and r["job"]["mechanism"] == "T3" and r["job"]["minimum"] == 0] for n in (10, 20, 40)]
        times = [statistics.median(r["parent_wall_sec"] for r in group) if group else float("nan") for group in groups]
        costs = [statistics.mean(r["metrics"]["total_cost"] for r in group if r["metrics"].get("completed_feasible"))
                 if any(r["metrics"].get("completed_feasible") for r in group) else float("nan") for group in groups]
        completions = [sum(bool(r["metrics"].get("completed_feasible")) for r in group)/5 for group in groups]
        for axis, values in zip(axes, (times, costs, completions)):
            axis.plot((10, 20, 40), values, marker="o", color=colors[algorithm], label=algorithm)
    for axis, title, ylabel in zip(axes,
        ("All attempted runs", "Completed, independently feasible only", "Five planned seeds per point"),
        ("Median process end-to-end time (s)", "Mean replayed full-horizon cost", "Completed feasible / planned")):
        axis.set(title=title, xlabel="Synthetic jobs", ylabel=ylabel, xticks=[10, 20, 40])
        axis.grid(alpha=.18)
    axes[0].legend()
    axes[2].set_ylim(-.03, 1.03)
    figure.suptitle("LA–SF bounded pilot; full has advance information, RH reveals orders at release", fontsize=11)
    figure.savefig(destination / "pilot_runtime_cost_completion.png")
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for seed in range(4101, 4106):
        selected = {r["job"]["mechanism"]: r for r in rows if r["job"]["n"] == 10
                    and r["job"]["seed"] == seed and r["job"]["algorithm"] == "full"
                    and r["job"]["minimum"] == (.1 if r["job"]["mechanism"] == "T2" else 0)}
        base = selected.get("T0", {}).get("metrics", {})
        if base.get("completed_feasible"):
            differences = [selected[m]["metrics"]["total_cost"]-base["total_cost"]
                           if m in selected and selected[m]["metrics"].get("completed_feasible") else float("nan")
                           for m in ("T0", "T1", "T2", "T3")]
            axes[0].plot(("T0", "T1", "T2", "T3"), differences, marker="o", alpha=.7, label=str(seed))
        sensitivity = {r["job"]["minimum"]: r for r in rows if r["job"]["n"] == 10
                       and r["job"]["seed"] == seed and r["job"]["mechanism"] == "T2"}
        costs = [sensitivity[m]["metrics"]["total_cost"] if m in sensitivity
                 and sensitivity[m]["metrics"].get("completed_feasible") else float("nan") for m in (.05, .1, .2)]
        axes[1].plot((.05, .1, .2), costs, marker="o", alpha=.7, label=str(seed))
    axes[0].axhline(0, color="gray", lw=.8)
    axes[0].set(title="Per-seed mechanism differences", xlabel="Policy (T2 additionally has m=0.1)", ylabel="Replayed cost minus T0")
    axes[1].set(title="T2 minimum average load sensitivity", xlabel="Minimum job-equivalents per vehicle", ylabel="Replayed cost", xticks=[.05, .1, .2])
    for axis in axes:
        axis.grid(alpha=.18)
    axes[1].legend(title="Seed", fontsize=8)
    figure.suptitle("N=10; completed feasible results only; missing points remain missing", fontsize=11)
    figure.savefig(destination / "mechanism_minimum_load.png")
    plt.close(figure)
    print(json.dumps({"figures": [str(p) for p in destination.glob("*.png")], "records": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
