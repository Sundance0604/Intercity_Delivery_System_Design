"""Bounded, reproducible fluid-wave / reset-Bellman toy check; no Gurobi.

Run: python -B research/toy_models/qvi_wave_consistency_check.py
Output is JSON on stdout; this script writes no files.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class Parameters:
    arrival_rate: float = 10.0
    a: float = 0.02
    b: float = 0.3
    rho: float = 0.2
    c: float = 10.0
    c_auto: float = 4.0
    tau: float = 2.0
    omega: float = 1.0
    manual_capacity: float = 8.0
    auto_capacity: float = 20.0
    theta: float = 0.05
    handling: float = 0.1


def original_steady(q, p):
    """Numerically minimize original n-dependent expression, not clipped roots."""
    s = p.a + p.b / math.sqrt(q)
    upper = min(q, p.manual_capacity)
    loads, parts = [], []
    for weight in (p.omega, p.omega / 2):
        def local(n):
            return 2 * p.c * p.rho / n + weight * s * n
        fit = minimize_scalar(local, bounds=(upper * 1e-10, upper),
                              method="bounded", options={"xatol": 1e-11})
        # Bounded scipy minimization does not itself evaluate interval endpoints.
        candidates = [(fit.fun, float(fit.x)), (local(upper), upper)]
        value, n = min(candidates)
        loads.append(n)
        parts.append(value)
    cost = (2 * p.c * s + sum(parts)
            + p.c_auto * p.tau / min(q, p.auto_capacity)
            + p.omega * (q / (2 * p.arrival_rate) + 3 * p.rho + p.tau + 2 * p.theta)
            + 2 * p.handling)
    return float(cost), loads


def steady_search(p, upper):
    """Partition outer domain, explicitly retain linehaul kink and endpoints."""
    cuts = sorted(set([1e-5, p.auto_capacity, upper]
                      + list(np.geomspace(1e-4, upper, 28))))
    cuts = [x for x in cuts if 1e-5 <= x <= upper]
    choices = [(original_steady(x, p)[0], x) for x in cuts]
    for left, right in zip(cuts, cuts[1:]):
        fit = minimize_scalar(lambda q: original_steady(q, p)[0],
                              bounds=(left, right), method="bounded",
                              options={"xatol": 1e-9})
        choices.append((float(fit.fun), float(fit.x)))
    value, q = min(choices)
    return {"q": q, "per_shipment": value, "rate": p.arrival_rate * value,
            "loads": original_steady(q, p)[1]}


def intervention_cost(q, p):
    """Independent closed-form load path: service cost only, no accumulation wait."""
    q = np.asarray(q, dtype=float)
    s = p.a + p.b / np.sqrt(q)
    n1 = np.minimum(np.minimum(q, p.manual_capacity),
                    np.sqrt(2 * p.c * p.rho / (p.omega * s)))
    n2 = np.minimum(np.minimum(q, p.manual_capacity),
                    np.sqrt(4 * p.c * p.rho / (p.omega * s)))
    per_shipment = (p.c * (2 * s + 2 * p.rho / n1 + 2 * p.rho / n2)
                    + p.c_auto * p.tau / np.minimum(q, p.auto_capacity)
                    + p.omega * (s * (n1 + n2 / 2) + 3 * p.rho + p.tau + 2 * p.theta)
                    + 2 * p.handling)
    return q * per_shipment


def bellman(p, step, upper, *, holding=True, poisson=False):
    """Solve normalized average-cost reset Bellman by bisection, not wave argmin.

    q=0 cannot dispatch; upper is a forced dispatch boundary. At other states,
    compare stopping/reset with waiting one grid step. Fluid wait cost uses
    the exact area under q(t); Poisson uses expected exponential holding.
    """
    if poisson and step != 1:
        raise ValueError("Unit Poisson arrivals require unit backlog grid")
    size = int(round(upper / step))
    q = np.arange(size + 1, dtype=float) * step
    dispatch = np.full(size + 1, np.inf)
    dispatch[1:] = intervention_cost(q[1:], p)
    dt = step / p.arrival_rate
    holding_cost = (p.omega * (q[:-1] if poisson else q[:-1] + step / 2) * dt
                    if holding else np.zeros(size))

    def sweep(rate):
        value = np.empty(size + 1)
        value[-1] = dispatch[-1]
        for index in range(size - 1, -1, -1):
            wait = holding_cost[index] - rate * dt + value[index + 1]
            value[index] = min(dispatch[index], wait)
        return value

    lower, higher = 0.0, 1.0
    while sweep(higher)[0] > 0:
        higher *= 2
    for _ in range(70):
        mid = (lower + higher) / 2
        if sweep(mid)[0] > 0:
            lower = mid
        else:
            higher = mid
    rate = (lower + higher) / 2
    value = sweep(rate)
    wait_gap = holding_cost - rate * dt + value[1:] - value[:-1]
    obstacle = dispatch - value
    residual = np.minimum(wait_gap, obstacle[:-1])
    # The first dispatch reached along the trajectory from q=0 determines policy.
    action = np.flatnonzero(obstacle[1:] <= 1e-9)
    threshold = float(q[int(action[0]) + 1])
    return {"step": step, "upper": upper, "threshold": threshold, "rate": rate,
            "per_shipment": rate / p.arrival_rate,
            "normalization_abs": abs(float(value[0])),
            "discrete_qvi_max_abs": float(np.max(np.abs(residual))),
            "minimum_obstacle": float(np.min(obstacle[1:])),
            "minimum_wait_gap": float(np.min(wait_gap)),
            "forced_boundary": True}


def main():
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__).resolve(),
                    root / "research/paper/steady_state_model.tex",
                    root / "research/paper/steady_state_model_v1_linear.tex",
                    root / "research/theory/Common Wave 的内生化与非均匀需求扩展.md"]
    sources = {str(path.relative_to(root)).replace("\\", "/"):
               hashlib.sha256(path.read_bytes()).hexdigest().upper()
               for path in source_paths}
    base = Parameters()
    cases = {
        "small_wave_load_bounded_by_wave": replace(base, arrival_rate=0.1),
        "linehaul_capacity_kink": replace(base, arrival_rate=2, auto_capacity=8),
        "large_wave_manual_capacity": replace(base, arrival_rate=50),
    }
    output = {"scope": "deterministic fluid common-wave, no sorting, delta=0, no deadline; per direction",
              "cost_units": "time=hour, q=shipment; rate=currency/hour, G=currency/shipment",
              "runtime": {"python": sys.version, "numpy": np.__version__,
                          "scipy": scipy.__version__}, "source_sha256": sources,
              "cases": {}, "passed": False}
    for name, p in cases.items():
        upper = 160.0
        steady = steady_search(p, upper)
        grids = [bellman(p, step, upper) for step in (0.2, 0.1, 0.05)]
        enlarged = bellman(p, 0.05, 2 * upper)
        probes = np.array([0.1, 0.5, 1, 2, 5, p.auto_capacity, 25, 80])
        errors = [abs(float(intervention_cost(q, p)) / q + p.omega * q / (2 * p.arrival_rate)
                      - original_steady(float(q), p)[0]) for q in probes]
        assert max(errors) < 1e-7
        assert abs(grids[-1]["threshold"] - steady["q"]) <= 0.051
        assert abs(grids[-1]["rate"] - steady["rate"]) <= 0.001
        assert abs(enlarged["threshold"] - grids[-1]["threshold"]) < 1e-9
        for result in grids + [enlarged]:
            assert result["discrete_qvi_max_abs"] < 1e-8
            assert result["minimum_obstacle"] > -1e-8
            assert result["minimum_wait_gap"] > -1e-8
        output["cases"][name] = {"parameters": asdict(p), "steady": steady,
                                   "fluid_bellman": grids,
                                   "enlarged_boundary": enlarged,
                                   "independent_cost_path_max_error": max(errors)}

    p = base
    fluid_integer = bellman(p, 1, 160)
    poisson = bellman(p, 1, 160, poisson=True)
    assert fluid_integer["threshold"] == poisson["threshold"]
    assert abs(fluid_integer["rate"] - poisson["rate"] - p.omega / 2) < 1e-8
    output["unit_poisson_integer_comparison"] = {
        "parameters": asdict(p), "fluid_integer": fluid_integer, "poisson": poisson,
        "rate_difference_expected": p.omega / 2,
        "note": "Same integer action domain and same C; mean rates differ by constant omega/2."}

    matched = bellman(p, 0.1, 80)
    missing_waiting = bellman(p, 0.1, 80, holding=False)
    missing_waiting_larger = bellman(p, 0.1, 160, holding=False)
    assert abs(matched["threshold"] - missing_waiting["threshold"]) > 1
    assert missing_waiting_larger["threshold"] == 160
    assert missing_waiting_larger["rate"] < missing_waiting["rate"]
    output["objective_mismatch_counterexample"] = {
        "parameters": asdict(p), "matched": matched, "holding_omitted": missing_waiting,
        "holding_omitted_larger_boundary": missing_waiting_larger,
        "note": "Different objective, not QVI failure: removing accumulation holding pushes the bounded optimum to 80, then 160; these are artificial truncation boundaries, not unbounded finite optima."}
    output["elapsed_sec"] = time.perf_counter() - started
    output["passed"] = True
    print(json.dumps(output, ensure_ascii=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
