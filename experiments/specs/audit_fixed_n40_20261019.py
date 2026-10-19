"""Coordinator-only statistics audit of saved data; never imports a solver."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "simulation/fixed_n_runs/FIXED_N40_20261019_01"
ANALYSIS = BATCH / "analyses/20261019T012741615782Z/summary.json"


def main():
    started = time.monotonic()
    a = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    rows = {(r["seed"], r["label"]): r for r in a["rows"]}
    checks = 0

    def equal(left, right):
        nonlocal checks
        if isinstance(left, list):
            assert isinstance(right, list) and len(left) == len(right)
            for x, y in zip(left, right):
                equal(x, y)
        elif left is None or right is None:
            assert left is right
        else:
            assert math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-9), (left, right)
        checks += 1

    for pair in a["paired"]:
        l, r = rows[pair["seed"], pair["baseline"]], rows[pair["seed"], pair["treatment"]]
        assert pair["eligible"] and l["orders_sha256"] == r["orders_sha256"]
        equal(pair["cost_delta_treatment_minus_baseline"], r["cost"]-l["cost"])
        equal(pair["served_rate_delta"], r["served_rate"]-l["served_rate"])
        equal(pair["wall_speedup_baseline_over_treatment"], l["parent_wall_sec"]/r["parent_wall_sec"])
        expected = ([r["global_bound"]-l["cost"], r["cost"]-l["global_bound"]]
                    if l["information"] == r["information"] == "offline" else None)
        equal(pair["optimal_cost_delta_interval_from_solver_bounds"], expected)

    for group in a["paired_aggregates"]:
        pairs = [p for p in a["paired"] if p["stage"] == group["stage"]
                 and p["baseline"] == group["baseline"] and p["treatment"] == group["treatment"]]
        assert len(pairs) == 5
        for field, key in (("cost_delta_treatment_minus_baseline", "cost_delta"),
                           ("served_rate_delta", "served_rate_delta"),
                           ("wall_speedup_baseline_over_treatment", "wall_speedup")):
            values = [p[field] for p in pairs]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            half = float(t.ppf(.975, 4))*sd/math.sqrt(5)
            saved = group[key]
            for stat, val in (("n", 5), ("planned_n", 5), ("mean", mean), ("sd", sd),
                              ("median", statistics.median(values)), ("min", min(values)),
                              ("max", max(values)), ("ci95", [mean-half, mean+half])):
                equal(saved[stat], val)

    assert len(a["rows"]) == 70 and len(a["paired"]) == 75 and len(a["paired_aggregates"]) == 15
    windows = sum(len(r["windows"]) for r in a["rows"])
    assert windows == 240 and sum(r["solver_calls"] for r in a["rows"]) == 290
    result = {"scope": "coordinator recomputation; not an independent Referee verdict",
              "analysis_sha256": hashlib.sha256(ANALYSIS.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "rows": 70, "pairs": 75, "paired_groups": 15, "rh_windows": windows,
              "numeric_comparisons_passed": checks, "elapsed_sec": time.monotonic()-started,
              "new_solver_runs": 0, "passed": True}
    with (BATCH / "coordinator_statistics_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
