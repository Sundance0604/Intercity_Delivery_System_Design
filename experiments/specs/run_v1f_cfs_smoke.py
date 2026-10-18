"""Bounded, opt-in CFS execution check for V1-F/full/RH/pruned RH.

Default invocation only samples and prints the plan. The PI executes --run
once under the shared 45-second CFS solver budget. This is not a benchmark
replication or a calibration of CFS shipment weights into customer stops.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intercity_delivery.v1.adapter import load_cfs_stop_orders
from intercity_delivery.v1.flexible_configuration import V1FConfig, flexible_profile


def compact_result(result):
    """Losslessly export nonzero decisions; omitted mother-domain keys are 0.

    No numerical tolerance is applied: every nonzero float is retained at its
    JSON precision. Keep all ten group names, inputs, validation and policies.
    """
    compact = dict(result)
    schedule = result.get("schedule")
    if isinstance(schedule, dict):
        compact["schedule"] = {
            name: {key: value for key, value in values.items() if value != 0}
            for name, values in schedule.items()
        }
        compact["schedule_export"] = {
            "format": "nonzero values; all omitted admissible mother-domain keys equal zero",
            "rounding_applied": False,
            "original_entry_count": sum(len(values) for values in schedule.values()),
            "exported_entry_count": sum(len(values) for values in compact["schedule"].values()),
            "reconstruction": "rebuild domains from config/orders/profile; assign missing decisions zero",
        }
    if isinstance(result.get("windows"), list):
        compact["windows"] = []
        for window in result["windows"]:
            summary = {key: value for key, value in window.items()
                       if key not in ("schedule", "solution", "indices")}
            if any(key in window for key in ("schedule", "solution", "indices")):
                summary["window_plan_export"] = "summary only; final committed nonzero schedule exported once"
            compact["windows"].append(summary)
    return compact


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "data/cfs_processed_sqlite/cfs_model_orders.json")
    parser.add_argument("--city-a", default="06-348")
    parser.add_argument("--city-b", default="06-488")
    parser.add_argument("--orders", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--time-limit", type=float, default=15.0)
    parser.add_argument("--run", action="store_true", help="Explicitly execute the three bounded solves")
    parser.add_argument("--output", type=Path, help="New JSON file; existing output is never overwritten")
    args = parser.parse_args(argv)
    if not 8 <= args.orders <= 12:
        parser.error("--orders must be 8..12 under this smoke protocol")
    if not 0 < args.time_limit <= 15:
        parser.error("--time-limit must be positive and <=15 seconds per run")
    if args.output is not None and args.output.exists():
        parser.error("--output already exists; choose an isolated new path")
    return args


def main(argv=None):
    args = parse_args(argv)
    output_stream = None
    if args.run and args.output is not None:
        # Check output permission before any solver or its budget is consumed.
        output_stream = args.output.open("x", encoding="utf-8")
    config = V1FConfig(
        T=24, t0_minutes=60, tau_periods=9,
        N_manual={1: 2, 2: 2}, N_auto={1: 1, 2: 1},
        a={1: .05, 2: .05}, b={1: .1, 2: .1}, rho={1: 5, 2: 5},
        M=4, M_auto=8, c_manual_hour=20, c_auto_hour=15, c_direct_hour=25,
    )
    orders, source = load_cfs_stop_orders(
        args.source, num_orders=args.orders, seed=args.seed,
        city_pair=(args.city_a, args.city_b), T=config.T,
        t0_minutes=config.t0_minutes, tau_periods=config.tau_periods, penalty=1000,
    )
    command = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=False)
    payload = {
        "schema_version": "v1_f_cfs_execution_smoke/1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "git_revision": command.stdout.strip(),
        "python": sys.version, "platform": platform.platform(),
        "source": source, "config": asdict(config), "orders": [asdict(o) for o in orders],
        "options": {"prediction_horizon": 8, "rolling_step": 2, "extension_horizon": 6},
        "time_limit_per_run_sec": args.time_limit, "max_solver_budget_sec": 3*args.time_limit,
        "threads": 1, "solver_seed": 0,
        "interpretation": "CFS execution smoke only; parameters and shipment-stop convention are not calibrated",
        "penalty_reason": "1000 per shipment-stop intentionally encourages positive service in smoke",
        "run_requested": args.run, "runs": [],
    }
    if not args.run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    from intercity_delivery.v1.flexible_rolling import RollingOptions
    from intercity_delivery.v1.flexible_runner import run_flexible, run_flexible_rolling

    started = time.monotonic()
    for name, prune in (("v1_flexible_mip", None), ("v1_flexible_rolling", False),
                        ("v1_flexible_pruned_rolling", True)):
        try:
            if prune is None:
                result = run_flexible(config, orders, flexible_profile(), args.time_limit)
            else:
                options = RollingOptions(prediction_horizon=8, rolling_step=2,
                                         extension_horizon=6, prune=prune)
                result = run_flexible_rolling(config, orders, options, args.time_limit)
        except Exception as error:
            result = {"completed": False, "status_name": "EXCEPTION",
                      "error_type": type(error).__name__, "error": str(error)}
        payload["runs"].append({"solver": name, "result": compact_result(result)})
    payload["elapsed_sec"] = time.monotonic()-started
    payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["all_completed_with_positive_service"] = all(
        row["result"].get("completed", False)
        and row["result"].get("validation") is not None
        and row["result"].get("served_quantity", 0) > 0
        for row in payload["runs"])
    # Preserve the full raw result even when any run ends partial or fails its
    # execution criterion. The caller receives a nonzero exit in that case.
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    if output_stream is not None:
        with output_stream:
            output_stream.write(encoded)
    print("BHH_CFS_EVIDENCE_BEGIN")
    print(encoded)
    print("BHH_CFS_EVIDENCE_END")
    return 0 if payload["all_completed_with_positive_service"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
