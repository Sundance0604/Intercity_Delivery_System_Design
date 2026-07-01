"""程序入口：默认启动 GUI，也提供与 GUI 共用实验核心的完整 CLI。"""

import argparse
from dataclasses import asdict
from datetime import datetime

from experiment_core import (
    ExperimentPlan,
    build_specs,
    get_sensitivity_parameters,
    levels_to_text,
    parse_parameter_levels,
    planned_run_count,
    run_experiment_suite,
)


def parse_args():
    parser = argparse.ArgumentParser(description="城际配送系统仿真实验平台")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="使用命令行模式；不加参数时启动可视化界面。",
    )
    parser.add_argument(
        "--scenario",
        choices=["quick", "sensitivity", "all"],
        default="quick",
        help="选择快速测试、灵敏度分析或两者。",
    )
    parser.add_argument(
        "--solver",
        choices=[
            "exact_mip",
            "rolling_horizon",
            "flexible_direct_mip",
            "flexible_direct_rolling",
            "all",
        ],
        default="exact_mip",
        help="选择求解器。",
    )
    parser.add_argument("--seeds", type=int, default=3, help="每个参数水平的种子数。")
    parser.add_argument(
        "--time-limit",
        type=int,
        default=500,
        help="每次求解的总时间限制（秒）。",
    )
    parser.add_argument(
        "--level",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help=(
            "覆盖灵敏度水平，可重复使用，例如 "
            "--level algorithm.prediction_horizon=[6,8,10]"
        ),
    )
    parser.add_argument(
        "--list-parameters",
        action="store_true",
        help="列出三类动态参数及默认灵敏度水平后退出。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印实验计划，不运行求解器。",
    )
    return parser.parse_args()


def _expand_choice(name, all_values):
    return list(all_values) if name == "all" else [name]


def _build_plan_from_args(args) -> ExperimentPlan:
    parameters = {parameter.key: parameter for parameter in get_sensitivity_parameters()}
    plan = ExperimentPlan(seed_count=args.seeds, time_limit=args.time_limit)
    levels = dict(plan.sensitivity_levels)

    for assignment in args.level:
        if "=" not in assignment:
            raise ValueError(f"--level 必须使用 KEY=JSON 格式：{assignment}")
        key, text = assignment.split("=", 1)
        key = key.strip()
        if key not in parameters:
            raise ValueError(
                f"未知动态参数 {key}；可使用 --list-parameters 查看参数名。"
            )
        levels[key] = parse_parameter_levels(
            text.strip(), parameters[key].base_value
        )

    return ExperimentPlan(
        seed_count=args.seeds,
        time_limit=args.time_limit,
        sensitivity_levels=levels,
    )


def _print_parameters():
    current_source = None
    source_labels = {
        "model": "模型参数",
        "algorithm": "算法参数",
        "order": "订单参数",
    }
    for parameter in get_sensitivity_parameters():
        if parameter.source != current_source:
            current_source = parameter.source
            print(f"\n[{source_labels[current_source]}]")
        print(
            f"{parameter.key:<36} "
            f"默认水平={levels_to_text(parameter.default_levels)}"
        )


def run_cli(args):
    if args.list_parameters:
        _print_parameters()
        return

    plan = _build_plan_from_args(args)
    scenarios = _expand_choice(args.scenario, ("quick", "sensitivity"))
    solver_names = _expand_choice(
        args.solver,
        (
            "exact_mip",
            "rolling_horizon",
            "flexible_direct_mip",
            "flexible_direct_rolling",
        ),
    )
    specs = build_specs(scenarios, plan)

    print(f"算例规格数：{len(specs)}")
    print(f"实际求解次数：{planned_run_count(specs, solver_names)}")
    print(f"求解器：{', '.join(solver_names)}")

    if args.dry_run:
        for spec in specs:
            print(asdict(spec))
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_experiment_suite(specs, solver_names, timestamp)


def main():
    args = parse_args()
    if args.cli or args.list_parameters:
        run_cli(args)
        return

    from experiment_gui import launch_gui

    launch_gui()


if __name__ == "__main__":
    main()
