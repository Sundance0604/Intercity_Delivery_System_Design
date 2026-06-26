import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import DeliveryConfig
from data_loader import DataLoader, DeliveryData, OrderBatch


@dataclass
class ExperimentSpec:
    """A reproducible simulation case used by the paper-oriented benchmark."""

    experiment_id: str
    scenario: str
    config: DeliveryConfig
    num_orders: int
    seed: int
    buffer_range: Tuple[int, int] = (0, 5)
    large_order_prob: float = 0.3
    time_limit: int = 500
    save_detail: bool = False


def generate_random_orders(
    config: DeliveryConfig,
    num_orders: int = 50,
    seed: int = 42,
    buffer_range: Tuple[int, int] = (0, 5),
    large_order_prob: float = 0.3,
    small_quantity_range: Tuple[int, int] = (10, 50),
    large_quantity_range: Tuple[int, int] = (100, 300),
):
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
    pos, neg, all_ord = orders_tuple
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
        all_orders=all_ord,
        epsilon_sets=epsilon,
    )


def run_single_experiment(
    experiment_id: str,
    config: DeliveryConfig,
    orders_tuple,
    scenario: str = "custom",
    seed: Optional[int] = None,
    buffer_range: Tuple[int, int] = (0, 5),
    large_order_prob: float = 0.3,
    time_limit: int = 500,
):
    import gurobipy as gp
    from optimizer import Optimizer

    start_time = time.time()
    _, _, all_ord = orders_tuple

    data = build_delivery_data(config, orders_tuple)

    opt = Optimizer(config, data)
    opt.setup_variables()
    opt.set_objective()
    opt.set_constraints()

    opt.model.setParam("TimeLimit", time_limit)
    opt.model.setParam("OutputFlag", 0)
    opt.model.optimize()

    solve_time = time.time() - start_time
    total_demand = sum(o.quantity for o in all_ord.values())

    result_summary = {
        "Scenario": scenario,
        "Exp_ID": experiment_id,
        "Seed": seed,
        "Status": opt.model.Status,
        "Solve_Time_Sec": round(solve_time, 2),
        "Time_Limit_Sec": time_limit,
        "Num_Orders": len(all_ord),
        "Total_Demand": total_demand,
        "Buffer_Min": buffer_range[0],
        "Buffer_Max": buffer_range[1],
        "Large_Order_Prob": large_order_prob,
        "Param_N_Auto": config.N_auto[1],
        "Param_N_Manual": config.N_manual[1],
        "Param_Cost_Auto": config.cost_auto,
        "Total_Cost": None,
        "Best_Bound": None,
        "MIP_Gap": None,
        "Unserved_Rate": None,
        "Auto_Usage": 0,
        "Manual_Usage": 0,
    }

    detailed_log = None

    if opt.model.SolCount > 0:
        result_summary["Total_Cost"] = opt.model.ObjVal
        result_summary["Best_Bound"] = opt.model.ObjBound
        result_summary["MIP_Gap"] = opt.model.MIPGap
        unserved_amount = sum(v.X for v in opt.z_unserved.values())
        result_summary["Unserved_Rate"] = (
            round(unserved_amount / total_demand, 4) if total_demand > 0 else 0
        )
        result_summary["Auto_Usage"] = sum(v.X for v in opt.y_auto.values())
        result_summary["Manual_Usage"] = sum(v.X for v in opt.x_manual.values())

        if opt.model.Status == gp.GRB.OPTIMAL:
            print(f"  [成功] 找到全局最优解！Cost = {result_summary['Total_Cost']:.2f}")
        elif opt.model.Status == gp.GRB.TIME_LIMIT:
            print(f"  [警告] 达到时间限制 ({opt.model.Params.TimeLimit}s)！")
            print(f"  [提示] 当前解可能不是最优解 (MIP Gap: {opt.model.MIPGap * 100:.2f}%)")
            print(f"        当前找到的最好 Cost = {result_summary['Total_Cost']:.2f}")

        detailed_log = {
            "scenario": scenario,
            "experiment_id": experiment_id,
            "seed": seed,
            "buffer_range": buffer_range,
            "large_order_prob": large_order_prob,
            "config": asdict(config),
            "orders": {k: asdict(v) for k, v in all_ord.items()},
            "solution": {
                "y_auto": {str(k): v.X for k, v in opt.y_auto.items() if v.X > 0.1},
                "z_unserved": {k: v.X for k, v in opt.z_unserved.items() if v.X > 0.1},
            },
        }
        print(
            f"Exp {experiment_id} | Scenario={scenario} | Orders={len(all_ord)} "
            f"| Seed={seed} | Cost={result_summary['Total_Cost']}"
        )
    else:
        print(f"  [失败] 未找到任何可行解。Gurobi 状态码: {opt.model.Status}")

    return result_summary, detailed_log


def make_seed_list(start: int, count: int) -> List[int]:
    return [start + i for i in range(count)]


def build_baseline_specs(seed_count: int, time_limit: int) -> List[ExperimentSpec]:
    specs = []
    base_cfg = DeliveryConfig()
    for num_orders in [20, 50]:
        for seed in make_seed_list(1001, seed_count):
            specs.append(
                ExperimentSpec(
                    experiment_id=f"BASE_N{num_orders}_S{seed}",
                    scenario="baseline",
                    config=base_cfg,
                    num_orders=num_orders,
                    seed=seed,
                    time_limit=time_limit,
                    save_detail=num_orders == 20,
                )
            )
    return specs


def build_scale_specs(seed_count: int, time_limit: int) -> List[ExperimentSpec]:
    specs = []
    scale_cfg = DeliveryConfig(N_auto={1: 50, 2: 50}, N_manual={1: 100, 2: 100})
    for num_orders in [100, 200, 500, 1000]:
        for seed in make_seed_list(2001, seed_count):
            specs.append(
                ExperimentSpec(
                    experiment_id=f"SCALE_N{num_orders}_S{seed}",
                    scenario="scale",
                    config=scale_cfg,
                    num_orders=num_orders,
                    seed=seed,
                    time_limit=time_limit,
                    save_detail=num_orders <= 100,
                )
            )
    return specs


def build_sensitivity_specs(seed_count: int, time_limit: int) -> List[ExperimentSpec]:
    specs = []
    seeds = make_seed_list(3001, seed_count)
    base_cfg = DeliveryConfig(N_auto={1: 30, 2: 30}, N_manual={1: 60, 2: 60})

    for n_auto, seed in product([0, 5, 10, 20, 30, 50], seeds):
        specs.append(
            ExperimentSpec(
                experiment_id=f"SENS_AUTO_{n_auto}_S{seed}",
                scenario="sens_auto_fleet",
                config=replace(base_cfg, N_auto={1: n_auto, 2: n_auto}),
                num_orders=100,
                seed=seed,
                time_limit=time_limit,
            )
        )

    for cost_auto, seed in product([5.0, 10.0, 15.0, 20.0, 25.0], seeds):
        specs.append(
            ExperimentSpec(
                experiment_id=f"SENS_AUTO_COST_{cost_auto:g}_S{seed}",
                scenario="sens_auto_cost",
                config=replace(base_cfg, cost_auto=cost_auto),
                num_orders=100,
                seed=seed,
                time_limit=time_limit,
            )
        )

    for n_manual, seed in product([10, 20, 30, 50, 80], seeds):
        specs.append(
            ExperimentSpec(
                experiment_id=f"SENS_MANUAL_{n_manual}_S{seed}",
                scenario="sens_manual_fleet",
                config=replace(base_cfg, N_manual={1: n_manual, 2: n_manual}),
                num_orders=100,
                seed=seed,
                time_limit=time_limit,
            )
        )

    for buffer_range, seed in product([(0, 1), (0, 3), (0, 5), (0, 8)], seeds):
        specs.append(
            ExperimentSpec(
                experiment_id=f"SENS_WINDOW_{buffer_range[1]}_S{seed}",
                scenario="sens_time_window",
                config=base_cfg,
                num_orders=100,
                seed=seed,
                buffer_range=buffer_range,
                time_limit=time_limit,
            )
        )

    for large_prob, seed in product([0.1, 0.3, 0.5, 0.7], seeds):
        specs.append(
            ExperimentSpec(
                experiment_id=f"SENS_DEMAND_{large_prob:.1f}_S{seed}",
                scenario="sens_demand_mix",
                config=base_cfg,
                num_orders=100,
                seed=seed,
                large_order_prob=large_prob,
                time_limit=time_limit,
            )
        )

    return specs


def build_specs(scenario: str, seed_count: int, time_limit: int) -> List[ExperimentSpec]:
    if scenario == "quick":
        return [
            ExperimentSpec(
                experiment_id="QUICK_N20_S42",
                scenario="quick",
                config=DeliveryConfig(),
                num_orders=20,
                seed=42,
                time_limit=time_limit,
                save_detail=True,
            )
        ]
    if scenario == "baseline":
        return build_baseline_specs(seed_count, time_limit)
    if scenario == "scale":
        return build_scale_specs(seed_count, time_limit)
    if scenario == "sensitivity":
        return build_sensitivity_specs(seed_count, time_limit)
    if scenario == "all":
        return (
            build_baseline_specs(seed_count, time_limit)
            + build_scale_specs(seed_count, time_limit)
            + build_sensitivity_specs(seed_count, time_limit)
        )
    raise ValueError(f"Unsupported scenario: {scenario}")


def run_experiment_suite(specs: List[ExperimentSpec], timestamp: str) -> pd.DataFrame:
    all_summaries = []
    os.makedirs("results", exist_ok=True)

    for index, spec in enumerate(specs, start=1):
        print(f"\n[{index}/{len(specs)}] Running {spec.experiment_id}")
        orders = generate_random_orders(
            spec.config,
            num_orders=spec.num_orders,
            seed=spec.seed,
            buffer_range=spec.buffer_range,
            large_order_prob=spec.large_order_prob,
        )
        res, details = run_single_experiment(
            spec.experiment_id,
            spec.config,
            orders,
            scenario=spec.scenario,
            seed=spec.seed,
            buffer_range=spec.buffer_range,
            large_order_prob=spec.large_order_prob,
            time_limit=spec.time_limit,
        )
        all_summaries.append(res)

        if spec.save_detail and details:
            detail_path = f"results/detail_{spec.experiment_id}_{timestamp}.json"
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(details, f, indent=4, ensure_ascii=False)

    df = pd.DataFrame(all_summaries)
    csv_filename = f"results/full_experiment_summary_{timestamp}.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    print(f"\n所有测试完成！汇总结果已保存至: {csv_filename}")
    print(df)
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Run intercity delivery simulation experiments.")
    parser.add_argument(
        "--scenario",
        choices=["quick", "baseline", "scale", "sensitivity", "all"],
        default="quick",
        help="Experiment suite to run. Use quick for a smoke test.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of random seeds per experiment level.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=500,
        help="Gurobi time limit in seconds for each instance.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned experiment cases without solving them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    specs = build_specs(args.scenario, args.seeds, args.time_limit)

    print(f"Planned experiments: {len(specs)}")
    if args.dry_run:
        for spec in specs:
            print(asdict(spec))
    else:
        run_experiment_suite(specs, timestamp)
