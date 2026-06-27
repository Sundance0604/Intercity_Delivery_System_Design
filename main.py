"""程序入口。

这个文件故意保持很薄：它只负责判断用户想打开图形界面，还是用命令行批处理。
真正的实验生成、数据构造、求解器调用都在 experiment_core.py 和 solvers.py 中；
可视化界面在 experiment_gui.py 中。这样后续新增 rolling horizon 等算法时，
不需要把 main.py 改成一个越来越复杂的大文件。
"""

import argparse
from dataclasses import asdict
from datetime import datetime

from experiment_core import ExperimentPlan, build_specs, run_experiment_suite


def parse_args():
    """解析命令行参数。

    默认直接运行 `python main.py` 会打开图形界面。
    如果要在终端批量运行，则加上 `--cli`。
    """

    parser = argparse.ArgumentParser(description="城际配送系统仿真实验入口")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="使用命令行模式运行；不加该参数时会打开可视化窗口。",
    )
    parser.add_argument(
        "--scenario",
        choices=["quick", "sensitivity", "all"],
        default="quick",
        help="命令行模式下选择实验场景。",
    )
    parser.add_argument(
        "--solver",
        choices=["exact_mip", "rolling_horizon", "all"],
        default="exact_mip",
        help="命令行模式下选择求解方式。rolling_horizon 目前是预留接口。",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="每个参数水平使用的随机种子数量。",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=500,
        help="每个算例的 Gurobi 求解时间限制，单位为秒。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印实验计划，不实际求解。",
    )
    return parser.parse_args()


def expand_scenario_name(name: str):
    """把命令行中的场景名称转换为内部场景列表。"""

    if name == "all":
        return ["quick", "sensitivity"]
    return [name]


def expand_solver_name(name: str):
    """把命令行中的求解器名称转换为内部求解器列表。"""

    if name == "all":
        return ["exact_mip", "rolling_horizon"]
    return [name]


def run_cli(args):
    """命令行批处理入口。

    命令行模式适合长时间跑实验，或者在没有图形界面的服务器上运行。
    GUI 和 CLI 最终都会调用 run_experiment_suite，因此结果字段保持一致。
    """

    plan = ExperimentPlan(seed_count=args.seeds, time_limit=args.time_limit)
    specs = build_specs(expand_scenario_name(args.scenario), plan)
    solver_names = expand_solver_name(args.solver)

    print(f"计划算例数：{len(specs)}")
    print(f"求解器：{', '.join(solver_names)}")

    if args.dry_run:
        for spec in specs:
            print(asdict(spec))
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_experiment_suite(specs, solver_names, timestamp)


def main():
    """根据参数启动 GUI 或 CLI。"""

    args = parse_args()
    if args.cli:
        run_cli(args)
    else:
        from experiment_gui import launch_gui

        launch_gui()


if __name__ == "__main__":
    main()
