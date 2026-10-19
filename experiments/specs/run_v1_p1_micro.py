"""Five bounded real solver checks before the P1 paired pilot.

Default is plan only. --run requires a passing P0 gate and a new output file.
All five must pass. Failures are saved; later checks stop after a failure.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.specs.v1_p0_preflight import check_p0
from intercity_delivery.v1.configuration import V1Order
from intercity_delivery.v1.flexible_configuration import V1FConfig, flexible_profile
from intercity_delivery.v1.flexible_optimizer import V1FlexibleOptimizer
from intercity_delivery.v1.flexible_validation import validate_flexible_schedule


def base_config(mechanism="T2"):
    return V1FConfig(T=4, t0_minutes=60, tau_periods=1,
        N_manual={1: 1, 2: 0}, N_auto={1: 0, 2: 0},
        a={1: 10, 2: 10}, b={1: 0, 2: 0}, rho={1: 0, 2: 0},
        M=2, M_auto=2, c_manual_hour=1, c_auto_hour=1, c_direct_hour=2,
        service_mechanism=mechanism, min_direct_load_per_vehicle=.1 if mechanism == "T2" else 0)


def full_check(name, force_local_start=None, disabled=False, forced_load=1.0):
    config = base_config("T0" if disabled else "T2")
    orders = [V1Order("direct_job", "+", 1, 0, 4, 100)]
    started = time.monotonic()
    optimizer = V1FlexibleOptimizer(config, orders, flexible_profile())
    result = {"name": name, "config": asdict(config), "orders": [asdict(o) for o in orders],
              "per_run_limit_sec": 3, "expected_status": "INFEASIBLE" if disabled or force_local_start == 1 or forced_load < .1 else "OPTIMAL"}
    result["forced_tasks"] = {"loaded_direct": [0, 2, "+", forced_load],
                              "vehicle_only_local_start": force_local_start}
    encoded = json.dumps({k: result[k] for k in ("config", "orders", "forced_tasks")},
                         sort_keys=True, separators=(",", ":")).encode()
    result["input_sha256"] = hashlib.sha256(encoded).hexdigest()
    try:
        optimizer.build_model()
        model = optimizer.model
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Seed = 0
        result["solver_parameters"] = {"Threads": 1, "Seed": 0}
        # One loaded direct wave, completing at city 2's abstract node at t=2.
        model.addConstr(optimizer.w_direct[0, 2, "+"] == 1)
        model.addConstr(optimizer.h_direct[0, 2, "+", "direct_job"] == forced_load)
        if force_local_start is not None:
            model.addConstr(optimizer.x_manual[force_local_start, force_local_start+1, 2, "+"] == 1)
        remaining = 3 - (time.monotonic()-started)
        if remaining <= 0:
            raise TimeoutError("micro budget exhausted before optimization")
        model.Params.TimeLimit = remaining
        result["solver_optimize_attempts"] = 1
        model.optimize()
        result.update(status=int(model.Status), solution_count=int(model.SolCount))
        if result["expected_status"] == "INFEASIBLE":
            result["passed"] = model.Status == 3
        else:
            expected_cost = 4 + (1 if force_local_start is not None else 0)
            result["expected_cost"] = expected_cost
            if model.SolCount:
                schedule = optimizer.extract_schedule()
                result["validation"] = validate_flexible_schedule(config, orders, flexible_profile(),
                    schedule, reported_objective=float(model.ObjVal))
                result["schedule"] = {group: {str(tuple(k)): float(v) for k, v in values.items() if v != 0}
                                      for group, values in schedule.items()}
                result["cost"] = float(model.ObjVal)
                result["passed"] = model.Status == 2 and abs(model.ObjVal-expected_cost) < 1e-6
            else:
                result["passed"] = False
        return result
    finally:
        if optimizer.model is not None:
            optimizer.model.dispose()
        result["wall_time_sec"] = time.monotonic()-started


def rolling_check():
    from intercity_delivery.v1.flexible_rolling import RollingOptions
    from intercity_delivery.v1.flexible_runner import run_flexible_rolling
    config = base_config()
    orders = [V1Order("direct_job", "+", 1, 0, 4, 100)]
    result = run_flexible_rolling(config, orders,
        RollingOptions(prediction_horizon=2, rolling_step=1, extension_horizon=2), 3)
    return {"name": "direct_committed_across_windows", "config": asdict(config),
            "orders": [asdict(o) for o in orders], "raw_result": result,
            "expected_cost": 4, "per_run_limit_sec": 3,
            "passed": bool(result["completed"] and result["schedule_valid"]
                           and abs(result["total_cost"]-4) < 1e-6)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    plan = {"max_new_runs": 5, "per_run_limit_sec": 3, "threads": 1, "solver_seed": 0,
            "checks": ["T2_forced_load005_below_minimum_infeasible", "direct_then_same_node_local_cost5",
                       "direct_then_early_local_infeasible", "T0_forced_direct_infeasible",
                       "direct_committed_across_windows_cost4"]}
    if not args.run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.output is None or args.output.exists():
        parser.error("--run requires a new --output file")
    preflight = check_p0()
    if not preflight["p0_checks_passed"]:
        print(json.dumps(preflight, indent=2))
        return 1
    checks = [lambda: full_check("T2_forced_load005_below_minimum_infeasible", forced_load=.05),
              lambda: full_check("direct_then_same_node_local_cost5", 2),
              lambda: full_check("direct_then_early_local_infeasible", 1),
              lambda: full_check("T0_forced_direct_infeasible", disabled=True), rolling_check]
    from intercity_delivery.v1.flexible_runner import _source_binding
    binding = _source_binding(ROOT)
    if not binding["current_sources_match"]:
        raise ValueError("micro source manifest does not match current sources")
    report = {"schema_version": "v1_p1_micro/1", "plan": plan, "preflight": preflight,
              "source_binding": binding, "runs": [],
              "source_content_sha256": binding["manifest"]["content_sha256"],
              "source_manifest_sha256": binding["manifest_sha256"],
              "local_continuity_scope": "forced local task carries no cargo; checks vehicle reuse and inventory timing"}
    # Pre-open the target before consuming any solver budget.
    with args.output.open("x", encoding="utf-8") as target:
        started = time.monotonic()
        for check in checks:
            try:
                result = check()
            except Exception as exc:
                result = {"passed": False, "error_type": type(exc).__name__, "error": str(exc)}
            report["runs"].append(result)
            if not result["passed"]:
                break
        report["all_passed"] = len(report["runs"]) == 5 and all(r["passed"] for r in report["runs"])
        report["successful_runs"] = sum(bool(r["passed"]) for r in report["runs"])
        report["elapsed_sec"] = time.monotonic()-started
        json.dump(report, target, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"all_passed": report["all_passed"], "executed": len(report["runs"]),
                      "output": str(args.output)}, indent=2))
    return not report["all_passed"]


if __name__ == "__main__":
    raise SystemExit(main())
