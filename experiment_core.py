"""仿真实验核心逻辑。

本文件只负责“生成实验、生成订单、组装数据、调用求解器、保存结果”。
它不包含任何图形界面代码，因此后续无论使用 customtkinter、网页界面，
还是命令行批处理，都可以复用这里的逻辑。
"""

import json
import os
import random
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from itertools import product
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import DeliveryConfig
from data_loader import DataLoader, DeliveryData, OrderBatch
from solvers import SOLVER_REGISTRY


@dataclass
class ExperimentPlan:
    """用户可调整的批量实验参数。

    这些字段对应 GUI 里的输入框。后续如果要新增一个灵敏度参数，推荐先在这里
    加字段，再在 build_sensitivity_specs 中使用，最后在 GUI 中加输入框。
    """

    seed_count: int = 3
    time_limit: int = 500
    quick_orders: int = 20
    baseline_order_sizes: List[int] = field(default_factory=lambda: [20, 50])
    scale_order_sizes: List[int] = field(default_factory=lambda: [100, 200, 500, 1000])
    scale_auto_fleet: int = 50
    scale_manual_fleet: int = 100
    sensitivity_orders: int = 100
    sensitivity_base_auto: int = 30
    sensitivity_base_manual: int = 60
    auto_fleet_levels: List[int] = field(default_factory=lambda: [0, 5, 10, 20, 30, 50])
    auto_cost_levels: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0, 20.0, 25.0])
    manual_fleet_levels: List[int] = field(default_factory=lambda: [10, 20, 30, 50, 80])
    time_window_buffers: List[Tuple[int, int]] = field(
        default_factory=lambda: [(0, 1), (0, 3), (0, 5), (0, 8)]
    )
    large_order_probs: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7])


@dataclass
class ExperimentSpec:
    """一个具体算例的完整描述。

    注意：这个结构只描述数据和参数，不描述求解器。求解器在 run_experiment_suite
    中单独传入，因此同一个 ExperimentSpec 可以被多个求解器复用。
    """

    experiment_id: str
    scenario: str
    config: DeliveryConfig
    num_orders: int
    seed: int
    buffer_range: Tuple[int, int] = (0, 5)
    large_order_prob: float = 0.3
    time_limit: int = 500
    save_detail: bool = False


def parse_int_list(text: str) -> List[int]:
    """把 '20, 50, 100' 解析为整数列表。"""

    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("请输入至少一个整数。")
    return [int(item) for item in values]


def parse_float_list(text: str) -> List[float]:
    """把 '0.1, 0.3, 0.5' 解析为小数列表。"""

    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("请输入至少一个数字。")
    return [float(item) for item in values]


def parse_buffer_ranges(text: str) -> List[Tuple[int, int]]:
    """把 '0-1, 0-3, 0-5' 解析为时间窗缓冲区间列表。"""

    ranges = []
    for item in [part.strip() for part in text.split(",") if part.strip()]:
        if "-" not in item:
            raise ValueError("时间窗缓冲区间格式应为 0-1, 0-3 这样的形式。")
        left, right = item.split("-", 1)
        ranges.append((int(left.strip()), int(right.strip())))
    if not ranges:
        raise ValueError("请输入至少一个时间窗缓冲区间。")
    return ranges


def int_list_to_text(values: List[int]) -> str:
    return ", ".join(str(value) for value in values)


def float_list_to_text(values: List[float]) -> str:
    return ", ".join(f"{value:g}" for value in values)


def buffer_ranges_to_text(values: List[Tuple[int, int]]) -> str:
    return ", ".join(f"{left}-{right}" for left, right in values)


def generate_random_orders(
    config: DeliveryConfig,
    num_orders: int = 50,
    seed: int = 42,
    buffer_range: Tuple[int, int] = (0, 5),
    large_order_prob: float = 0.3,
    small_quantity_range: Tuple[int, int] = (10, 50),
    large_quantity_range: Tuple[int, int] = (100, 300),
):
    """生成一批随机订单。

    这里是保证“同一输入数据可比较”的关键之一：只要 config、num_orders、
    seed、buffer_range、large_order_prob 相同，生成的订单就完全相同。
    在 run_experiment_suite 中，订单只生成一次，然后传给所有求解器。
    """

    random.seed(seed)
    np.random.seed(seed)

    pos_orders = {}
    neg_orders = {}
    all_orders = {}
    min_buffer, max_buffer = buffer_range

    for l in range(1, num_orders + 1):
        flow = "+" if random.random() > 0.5 else "-"
        min_duration = config.travel_time_periods + 1
        max_start = config.T - min_duration - max_buffer - 1

        if max_start <= 0:
            earliest_start = 0
            latest_completion = config.T
        else:
            earliest_start = random.randint(0, max_start)
            buffer = random.randint(min_buffer, max_buffer)
            latest_completion = min(config.T, earliest_start + min_duration + buffer)

        if random.random() < large_order_prob:
            quantity = random.randint(*large_quantity_range)
        else:
            quantity = random.randint(*small_quantity_range)

        order = OrderBatch(
            batch_id=l,
            flow=flow,
            quantity=quantity,
            earliest_start=earliest_start,
            latest_completion=latest_completion,
            penalty_lost=config.penalty_lost,
        )

        all_orders[l] = order
        if flow == "+":
            pos_orders[l] = order
        else:
            neg_orders[l] = order

    return pos_orders, neg_orders, all_orders


def build_delivery_data(config: DeliveryConfig, orders_tuple) -> DeliveryData:
    """把订单和参数转换为优化模型需要的数据结构。"""

    pos, neg, _ = orders_tuple
    loader = DataLoader(config)
    m1, m2 = loader.generate_arcs_manual()
    auto = loader.generate_arcs_auto()
    sets_m1, sets_m2, sets_auto = loader.generate_sets(m1, m2, auto)
    epsilon = loader.generate_epsilon_sets(pos, neg, m1, m2)
    coeff1, coeff2 = loader.pre_inverse_count(m1, m2)

    return DeliveryData(
        arcs_manual_1=m1,
        arcs_manual_2=m2,
        arcs_auto=auto,
        sets_manual_1=sets_m1,
        sets_manual_2=sets_m2,
        sets_auto=sets_auto,
        cap_coeff_1=coeff1,
        cap_coeff_2=coeff2,
        pos_orders=pos,
        neg_orders=neg,
        all_orders=orders_tuple[2],
        epsilon_sets=epsilon,
    )


def make_seed_list(start: int, count: int) -> List[int]:
    """生成连续随机种子，便于重复实验。"""

    return [start + i for i in range(count)]


def build_quick_specs(plan: ExperimentPlan) -> List[ExperimentSpec]:
    return [
        ExperimentSpec(
            experiment_id=f"QUICK_N{plan.quick_orders}_S42",
            scenario="quick",
            config=DeliveryConfig(),
            num_orders=plan.quick_orders,
            seed=42,
            time_limit=plan.time_limit,
            save_detail=True,
        )
    ]


def build_baseline_specs(plan: ExperimentPlan) -> List[ExperimentSpec]:
    """构建小规模基准算例，用于和精确解或未来算法做对比。"""

    specs = []
    base_cfg = DeliveryConfig()
    for num_orders in plan.baseline_order_sizes:
        for seed in make_seed_list(1001, plan.seed_count):
            specs.append(
                ExperimentSpec(
                    experiment_id=f"BASE_N{num_orders}_S{seed}",
                    scenario="baseline",
                    config=base_cfg,
                    num_orders=num_orders,
                    seed=seed,
                    time_limit=plan.time_limit,
                    save_detail=num_orders == min(plan.baseline_order_sizes),
                )
            )
    return specs


def build_scale_specs(plan: ExperimentPlan) -> List[ExperimentSpec]:
    """构建规模扩展算例，用于观察订单数增加后的求解表现。"""

    specs = []
    scale_cfg = DeliveryConfig(
        N_auto={1: plan.scale_auto_fleet, 2: plan.scale_auto_fleet},
        N_manual={1: plan.scale_manual_fleet, 2: plan.scale_manual_fleet},
    )
    for num_orders in plan.scale_order_sizes:
        for seed in make_seed_list(2001, plan.seed_count):
            specs.append(
                ExperimentSpec(
                    experiment_id=f"SCALE_N{num_orders}_S{seed}",
                    scenario="scale",
                    config=scale_cfg,
                    num_orders=num_orders,
                    seed=seed,
                    time_limit=plan.time_limit,
                    save_detail=num_orders == min(plan.scale_order_sizes),
                )
            )
    return specs


def build_sensitivity_specs(plan: ExperimentPlan) -> List[ExperimentSpec]:
    """构建单因素灵敏度算例。"""

    specs = []
    seeds = make_seed_list(3001, plan.seed_count)
    base_cfg = DeliveryConfig(
        N_auto={1: plan.sensitivity_base_auto, 2: plan.sensitivity_base_auto},
        N_manual={1: plan.sensitivity_base_manual, 2: plan.sensitivity_base_manual},
    )

    for n_auto, seed in product(plan.auto_fleet_levels, seeds):
        specs.append(
            ExperimentSpec(
                experiment_id=f"SENS_AUTO_{n_auto}_S{seed}",
                scenario="sens_auto_fleet",
                config=replace(base_cfg, N_auto={1: n_auto, 2: n_auto}),
                num_orders=plan.sensitivity_orders,
                seed=seed,
                time_limit=plan.time_limit,
            )
        )

    for cost_auto, seed in product(plan.auto_cost_levels, seeds):
        specs.append(
            ExperimentSpec(
                experiment_id=f"SENS_AUTO_COST_{cost_auto:g}_S{seed}",
                scenario="sens_auto_cost",
                config=replace(base_cfg, cost_auto=cost_auto),
                num_orders=plan.sensitivity_orders,
                seed=seed,
                time_limit=plan.time_limit,
            )
        )

    for n_manual, seed in product(plan.manual_fleet_levels, seeds):
        specs.append(
            ExperimentSpec(
                experiment_id=f"SENS_MANUAL_{n_manual}_S{seed}",
                scenario="sens_manual_fleet",
                config=replace(base_cfg, N_manual={1: n_manual, 2: n_manual}),
                num_orders=plan.sensitivity_orders,
                seed=seed,
                time_limit=plan.time_limit,
            )
        )

    for buffer_range, seed in product(plan.time_window_buffers, seeds):
        specs.append(
            ExperimentSpec(
                experiment_id=f"SENS_WINDOW_{buffer_range[1]}_S{seed}",
                scenario="sens_time_window",
                config=base_cfg,
                num_orders=plan.sensitivity_orders,
                seed=seed,
                buffer_range=buffer_range,
                time_limit=plan.time_limit,
            )
        )

    for large_prob, seed in product(plan.large_order_probs, seeds):
        specs.append(
            ExperimentSpec(
                experiment_id=f"SENS_DEMAND_{large_prob:.1f}_S{seed}",
                scenario="sens_demand_mix",
                config=base_cfg,
                num_orders=plan.sensitivity_orders,
                seed=seed,
                large_order_prob=large_prob,
                time_limit=plan.time_limit,
            )
        )

    return specs


def build_specs(selected_scenarios: List[str], plan: ExperimentPlan) -> List[ExperimentSpec]:
    """根据用户勾选的场景生成算例列表。"""

    specs = []
    if "quick" in selected_scenarios:
        specs.extend(build_quick_specs(plan))
    if "baseline" in selected_scenarios:
        specs.extend(build_baseline_specs(plan))
    if "scale" in selected_scenarios:
        specs.extend(build_scale_specs(plan))
    if "sensitivity" in selected_scenarios:
        specs.extend(build_sensitivity_specs(plan))
    if not specs:
        raise ValueError("请至少选择一个实验场景。")
    return specs


def run_experiment_suite(
    specs: List[ExperimentSpec],
    solver_names: List[str],
    timestamp: str = None,
) -> pd.DataFrame:
    """批量运行实验并保存汇总 CSV。

    这里保证同一输入数据下的可比性：对每个 ExperimentSpec，先生成一次 orders_tuple
    和 DeliveryData，然后把同一份数据依次传给用户选择的每个求解器。
    """

    all_summaries = []
    os.makedirs("results", exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    for index, spec in enumerate(specs, start=1):
        print(f"\n[{index}/{len(specs)}] 构建算例 {spec.experiment_id}")
        orders_tuple = generate_random_orders(
            spec.config,
            num_orders=spec.num_orders,
            seed=spec.seed,
            buffer_range=spec.buffer_range,
            large_order_prob=spec.large_order_prob,
        )
        data = build_delivery_data(spec.config, orders_tuple)
        total_demand = sum(order.quantity for order in orders_tuple[2].values())

        for solver_name in solver_names:
            solver = SOLVER_REGISTRY[solver_name]
            print(f"  -> 使用求解器：{solver.display_name}")
            result = solver.solve(spec.config, data, orders_tuple, spec.time_limit)
            print(f"     {result.message}")

            all_summaries.append(
                {
                    "Scenario": spec.scenario,
                    "Exp_ID": spec.experiment_id,
                    "Solver": result.solver_name,
                    "Seed": spec.seed,
                    "Status": result.status,
                    "Solve_Time_Sec": result.solve_time_sec,
                    "Time_Limit_Sec": spec.time_limit,
                    "Num_Orders": spec.num_orders,
                    "Total_Demand": total_demand,
                    "Buffer_Min": spec.buffer_range[0],
                    "Buffer_Max": spec.buffer_range[1],
                    "Large_Order_Prob": spec.large_order_prob,
                    "Param_N_Auto": spec.config.N_auto[1],
                    "Param_N_Manual": spec.config.N_manual[1],
                    "Param_Cost_Auto": spec.config.cost_auto,
                    "Penalty_Lost": spec.config.penalty_lost,
                    "Total_Cost": result.total_cost,
                    "Best_Bound": result.best_bound,
                    "MIP_Gap": result.mip_gap,
                    "Unserved_Rate": result.unserved_rate,
                    "Auto_Usage": result.auto_usage,
                    "Manual_Usage": result.manual_usage,
                    "Message": result.message,
                }
            )

            if spec.save_detail and result.detail:
                detail_payload = {
                    "scenario": spec.scenario,
                    "experiment_id": spec.experiment_id,
                    "solver": result.solver_name,
                    "seed": spec.seed,
                    "buffer_range": spec.buffer_range,
                    "large_order_prob": spec.large_order_prob,
                    "config": asdict(spec.config),
                    "orders": {k: asdict(v) for k, v in orders_tuple[2].items()},
                    **result.detail,
                }
                detail_path = f"results/detail_{spec.experiment_id}_{result.solver_name}_{timestamp}.json"
                with open(detail_path, "w", encoding="utf-8") as f:
                    json.dump(detail_payload, f, indent=4, ensure_ascii=False)

    df = pd.DataFrame(all_summaries)
    csv_filename = f"results/full_experiment_summary_{timestamp}.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    print(f"\n所有测试完成！汇总结果已保存至: {csv_filename}")
    print(df)
    return df
