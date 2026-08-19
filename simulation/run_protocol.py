"""Run reproducible real-data simulation batches under ``simulation/runs``.

The regular CLI quick scenario intentionally uses project defaults.  This runner
creates explicit ExperimentSpec objects so a real city pair's calibrated travel
time is applied before the orders and time-space network are built.  All compared
solvers are executed in one suite and therefore consume the same in-memory order
instance for every experiment id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intercity_delivery.configuration import (  # noqa: E402
    DeliveryConfig,
    OrderGenerationConfig,
    RollingHorizonConfig,
)
from intercity_delivery.data.cfs_catalog import (  # noqa: E402
    cfs_area_name,
    find_city_pair,
    inspect_cfs_sqlite,
)
from intercity_delivery.data.cfs_processor import (  # noqa: E402
    ProcessorConfig,
    calibrate_city_pair_travel_time as calibrate_pair_travel_time,
)
from intercity_delivery.experiments.core import (  # noqa: E402
    ExperimentSpec,
    load_real_orders_with_metadata,
    run_experiment_suite,
)


DEFAULT_SOLVERS = ("paper_candidate_mip", "paper_priority_heuristic")


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("列表必须包含正整数。")
    return values


def parse_str_list(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("列表不能为空。")
    return values


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("列表必须包含正数。")
    return values

def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibrate_city_pair_travel_time(args: argparse.Namespace) -> dict:
    """Use the production full-pair calibration shared with the GUI."""

    config = ProcessorConfig(
        num_orders=2,
        planning_periods=args.planning_periods,
        period_hours=args.period_minutes / 60.0,
        buffer_min_periods=args.buffer_min,
        buffer_max_periods=args.buffer_max,
    )
    return calibrate_pair_travel_time(
        Path(args.database), args.city_a, args.city_b, config
    )

MECHANISM_RATIOS = {
    "transshipment_only": (0.0, 0.0),
    "flexible": (0.0, 1.0),
    "direct_only": (1.0, 1.0),
}

MECHANISM_SLUGS = {
    "transshipment_only": "T",
    "flexible": "F",
    "direct_only": "D",
}


def build_configs(
    args: argparse.Namespace,
    num_orders: int,
    penalty_lost: float | None = None,
    mechanism: str = "flexible",
    fleet_scale: float = 1.0,
):
    ratio_min, ratio_max = MECHANISM_RATIOS[mechanism]
    defaults = DeliveryConfig()
    model = DeliveryConfig(
        T=args.planning_periods,
        t_0=args.period_minutes,
        travel_time_periods=args.travel_time,
        penalty_lost=(
            args.penalty_lost if penalty_lost is None else penalty_lost
        ),
        direct_travel_time_periods=args.travel_time,
        N_manual={
            city: max(1, round(value * fleet_scale))
            for city, value in defaults.N_manual.items()
        },
        N_auto={
            city: max(1, round(value * fleet_scale))
            for city, value in defaults.N_auto.items()
        },
        direct_ratio_min=ratio_min,
        direct_ratio_max=ratio_max,
    )
    algorithm = RollingHorizonConfig(
        prediction_horizon=args.prediction_horizon,
        rolling_step=args.rolling_step,
        extension_horizon=args.extension_horizon,
    )
    orders = OrderGenerationConfig(
        num_orders=num_orders,
        buffer_range=(args.buffer_min, args.buffer_max),
    )
    algorithm.validate()
    orders.validate()
    return model, algorithm, orders


def build_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    penalties = args.penalty_values or [args.penalty_lost]
    for num_orders in args.order_counts:
        for seed in args.seeds:
            for penalty_lost in penalties:
                for mechanism in args.mechanisms:
                    for fleet_scale in args.fleet_scales:
                        model, algorithm, orders = build_configs(
                            args,
                            num_orders,
                            penalty_lost=penalty_lost,
                            mechanism=mechanism,
                            fleet_scale=fleet_scale,
                        )
                        penalty_slug = str(penalty_lost).replace(".", "p")
                        fleet_slug = round(fleet_scale * 100)
                        if args.stage == "stage_c":
                            experiment_id = (
                                f"C_N{num_orders}_S{seed}_P{penalty_slug}_"
                                f"M{MECHANISM_SLUGS[mechanism]}_F{fleet_slug}"
                            )
                            scenario = (
                                f"simulation_{args.stage}_{mechanism}_"
                                f"fleet_{fleet_scale:g}"
                            )
                        else:
                            experiment_id = (
                                f"{args.stage.upper()}_REAL_N{num_orders}_S{seed}"
                            )
                            scenario = f"simulation_{args.stage}"
                        specs.append(
                            ExperimentSpec(
                                experiment_id=experiment_id,
                                scenario=scenario,
                                config=model,
                                algorithm_config=algorithm,
                                order_config=orders,
                                seed=seed,
                                time_limit=args.time_limit,
                                save_detail=True,
                            )
                        )
    return specs

def preflight(
    args: argparse.Namespace, batch_dir: Path, calibration: dict
) -> dict:
    catalog = inspect_cfs_sqlite(args.database)
    pair = find_city_pair(catalog, args.city_a, args.city_b)
    if pair is None:
        raise ValueError(
            f"SQLite 中不存在双向城市对 {args.city_a} <-> {args.city_b}。"
        )

    preview_count = max(args.order_counts)
    model, _algorithm, orders = build_configs(
        args,
        preview_count,
        penalty_lost=(args.penalty_values or [args.penalty_lost])[0],
        mechanism=args.mechanisms[0],
        fleet_scale=args.fleet_scales[0],
    )
    orders_tuple, metadata = load_real_orders_with_metadata(
        str(args.database),
        model,
        orders,
        seed=args.seeds[0],
        city_pair=(args.city_a, args.city_b),
    )
    all_orders = orders_tuple[2]
    direction_counts = {
        "+": sum(order.flow == "+" for order in all_orders.values()),
        "-": sum(order.flow == "-" for order in all_orders.values()),
    }
    windows = [
        order.latest_completion - order.earliest_start
        for order in all_orders.values()
    ]
    recommendations = metadata.get("model_recommendations", {})
    recommended_travel = recommendations.get("travel_time_periods")

    payload = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "database": str(args.database),
        "columns": [
            {"name": name, "type": sql_type} for name, sql_type in catalog.columns
        ],
        "bidirectional_city_pair_count": len(catalog.city_pairs),
        "selected_pair": {
            "city_a": pair.city_a,
            "city_a_name": cfs_area_name(pair.city_a),
            "city_b": pair.city_b,
            "city_b_name": cfs_area_name(pair.city_b),
            "records_a_to_b": pair.records_a_to_b,
            "records_b_to_a": pair.records_b_to_a,
        },
        "preview": {
            "seed": args.seeds[0],
            "num_orders": len(all_orders),
            "direction_counts": direction_counts,
            "total_quantity": sum(order.quantity for order in all_orders.values()),
            "earliest_start_min": min(
                order.earliest_start for order in all_orders.values()
            ),
            "earliest_start_max": max(
                order.earliest_start for order in all_orders.values()
            ),
            "latest_completion_min": min(
                order.latest_completion for order in all_orders.values()
            ),
            "latest_completion_max": max(
                order.latest_completion for order in all_orders.values()
            ),
            "window_width_min": min(windows),
            "window_width_max": max(windows),
        },
        "sampling_metadata": metadata,
        "city_pair_travel_time_calibration": calibration,
        "checks": {
            "has_both_directions": all(value > 0 for value in direction_counts.values()),
            "all_time_windows_valid": all(
                0 <= order.earliest_start < order.latest_completion <= model.T
                for order in all_orders.values()
            ),
            "travel_time_matches_full_pair_calibration": (
                args.travel_time == int(calibration["travel_time_periods"])
            ),
        },
        "diagnostics": {
            "configured_travel_time_periods": args.travel_time,
            "sample_based_recommended_travel_time_periods": recommended_travel,
            "sample_recommendation_matches_fixed_calibration": (
                recommended_travel is None
                or args.travel_time == int(recommended_travel)
            ),
        },        "preview_orders": {
            str(order_id): asdict(order)
            for order_id, order in all_orders.items()
        },
    }
    write_json(batch_dir / "preflight.json", payload)
    if not all(payload["checks"].values()):
        raise ValueError("预检失败，详情见 preflight.json。")
    return payload


def write_batch_readme(
    path: Path,
    args: argparse.Namespace,
    status: str,
    result_rows: int = 0,
) -> None:
    content = f"""# Simulation batch: {args.batch_id}

- 状态：{status}
- 阶段：{args.stage}
- 数据：`{args.database}`
- 城市对：{cfs_area_name(args.city_a)} (`{args.city_a}`) — {cfs_area_name(args.city_b)} (`{args.city_b}`)
- 订单规模：{args.order_counts}
- 随机种子：{args.seeds}
- 求解器：{args.solvers}
- 单算例总时间限制：{args.time_limit} 秒
- 已生成汇总结果行数：{result_rows}

重要文件：

- `manifest.json`：环境、代码、数据和参数快照；
- `preflight.json`：SQLite、城市对、模型建议和实际订单预检；
- `logs/`：完整运行日志；
- `results/`：CSV 汇总、完整批次 JSON 和逐算例 JSON。
"""
    path.write_text(content, encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行可复现的 CFS 仿真批次。")
    parser.add_argument(
        "--stage",
        default="stage_a",
        choices=["stage_a", "stage_b", "stage_c"],
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--city-a", default="06-348")
    parser.add_argument("--city-b", default="06-488")
    parser.add_argument("--order-counts", type=parse_int_list, default=[20])
    parser.add_argument("--seeds", type=parse_int_list, default=[42])
    parser.add_argument(
        "--solvers", type=parse_str_list, default=list(DEFAULT_SOLVERS)
    )
    parser.add_argument("--time-limit", type=int, default=180)
    parser.add_argument("--planning-periods", type=int, default=24)
    parser.add_argument("--period-minutes", type=float, default=60.0)
    parser.add_argument("--penalty-lost", type=float, default=10.0)
    parser.add_argument(
        "--penalty-values",
        type=parse_float_list,
        default=None,
        help="可选罚金列表，用于修复后罚金校准。",
    )
    parser.add_argument(
        "--mechanisms",
        type=parse_str_list,
        default=["flexible"],
        help="transshipment_only,flexible,direct_only",
    )
    parser.add_argument(
        "--fleet-scales",
        type=parse_float_list,
        default=[1.0],
    )
    parser.add_argument(
        "--travel-time",
        type=int,
        default=None,
        help="固定城际时段数；省略时按城市对全部合格记录自动校准。",
    )
    parser.add_argument("--prediction-horizon", type=int, default=8)
    parser.add_argument("--rolling-step", type=int, default=2)
    parser.add_argument("--extension-horizon", type=int, default=6)
    parser.add_argument("--buffer-min", type=int, default=0)
    parser.add_argument("--buffer-max", type=int, default=5)
    parser.add_argument("--hash-database", action="store_true")
    args = parser.parse_args(argv)
    args.database = args.database.expanduser().resolve()
    if not args.database.is_file():
        parser.error(f"数据库不存在：{args.database}")
    if args.time_limit <= 0:
        parser.error("--time-limit 必须大于 0。")
    unknown_mechanisms = set(args.mechanisms) - set(MECHANISM_RATIOS)
    if unknown_mechanisms:
        parser.error("未知机制：" + ", ".join(sorted(unknown_mechanisms)))
    return args


def safe_batch_directory_name(batch_id: str, max_length: int = 24) -> str:
    """Keep nested simulation paths below the legacy Windows path limit."""

    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in batch_id
    ).strip("_-") or "batch"
    if len(safe) <= max_length:
        return safe
    digest = hashlib.sha256(batch_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:max_length - 11].rstrip('_-')}_{digest}"

def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    calibration = calibrate_city_pair_travel_time(args)
    if args.travel_time is None:
        args.travel_time = int(calibration["travel_time_periods"])
    batch_directory_name = safe_batch_directory_name(args.batch_id)
    batch_dir = PROJECT_ROOT / "simulation" / "runs" / batch_directory_name
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(f"批次目录已经存在且非空：{batch_dir}")
    (batch_dir / "logs").mkdir(parents=True, exist_ok=True)
    (batch_dir / "results").mkdir(parents=True, exist_ok=True)
    write_batch_readme(batch_dir / "README.md", args, "初始化")

    data_stat = args.database.stat()
    manifest = {
        "batch_id": args.batch_id,
        "batch_directory_name": batch_directory_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "git": {
            "commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "status_short": git_output("status", "--short").splitlines(),
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "database": {
            "path": str(args.database),
            "size_bytes": data_stat.st_size,
            "modified_at": datetime.fromtimestamp(data_stat.st_mtime).isoformat(),
            "sha256": sha256_file(args.database) if args.hash_database else None,
        },
        "city_pair_travel_time_calibration": calibration,
        "arguments": vars(args),
    }
    try:
        import gurobipy as gp

        manifest["environment"]["gurobi"] = list(gp.gurobi.version())
    except Exception as error:  # pragma: no cover - diagnostic path
        manifest["environment"]["gurobi_error"] = repr(error)
    write_json(batch_dir / "manifest.json", manifest)

    preflight(args, batch_dir, calibration)
    write_batch_readme(batch_dir / "README.md", args, "预检通过")

    specs = build_specs(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    previous_cwd = Path.cwd()
    try:
        os.chdir(batch_dir)
        frame = run_experiment_suite(
            specs,
            args.solvers,
            timestamp=timestamp,
            data_source="real",
            real_data_path=str(args.database),
            real_city_pair=(args.city_a, args.city_b),
        )
    except Exception:
        write_batch_readme(batch_dir / "README.md", args, "运行失败")
        raise
    finally:
        os.chdir(previous_cwd)

    write_json(
        batch_dir / "run_summary.json",
        {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "result_rows": json.loads(frame.to_json(orient="records")),
        },
    )
    write_batch_readme(
        batch_dir / "README.md", args, "运行完成", result_rows=len(frame)
    )
    print(f"\n批次完成：{batch_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
