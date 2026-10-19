"""Plot saved N40 follow-up results only; never imports an optimizer."""

import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "simulation/v1_runs/P1_FOLLOWUP_20261019_01"


def main():
    started = time.monotonic()
    path = BATCH / "tables/analysis.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    pair = next(p for p in data["pairs"] if p["comparison"] == "T2_m01_minus_T0")
    seeds = [r["seed"] for r in pair["seeds"]]
    differences = [r["incumbent_cost_difference_right_minus_left"] for r in pair["seeds"]]
    intervals = [r["optimal_cost_difference_certificate_interval"] for r in pair["seeds"]]
    errors = [[v - bounds[0] for v, bounds in zip(differences, intervals)],
              [bounds[1] - v for v, bounds in zip(differences, intervals)]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    axes[0].errorbar(range(5), differences, yerr=errors, fmt="o", capsize=5,
                     color="#6f42a4", label="Incumbent difference + solver-bound interval")
    axes[0].axhline(0, color="gray", linewidth=1, linestyle="--")
    axes[0].set(xticks=range(5), xticklabels=seeds, xlabel="Fixed instance seed",
                ylabel="Cost difference (T2 m=0.1 minus T0)", title="Cost: direct policy versus transfer-only")
    axes[0].legend(loc="upper left", fontsize=7)
    for prefix, label, marker, color in (("T0_m0", "T0: transfer-only", "s", "#54718b"),
                                         ("T2_m0.1", "T2: direct, m=0.1", "o", "#6f42a4")):
        service = [100 * data["records"][f"n40_s{s}_{prefix}_full"]["metrics"]["served_rate"] for s in seeds]
        axes[1].plot(range(5), service, marker=marker, color=color, label=label)
    axes[1].set(xticks=range(5), xticklabels=seeds, xlabel="Fixed instance seed",
                ylabel="Served continuous quantity (%)", ylim=(65, 102), title="Service in saved feasible incumbents")
    axes[1].legend(loc="lower left", fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("N40 follow-up: five existing synthetic instances\n"
                 "Bound intervals are optimization certificates; service uses feasible incumbents", fontsize=10)
    directory = BATCH / "figures_final"
    directory.mkdir(exist_ok=False)
    output = directory / "direct_policy_cost_service.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    metadata = {"analysis_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "figure_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "elapsed_sec": time.monotonic() - started, "new_solver_runs": 0,
                "scope": "derived plot of fixed saved results; no new experiment"}
    with (directory / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(json.dumps({"figure": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
