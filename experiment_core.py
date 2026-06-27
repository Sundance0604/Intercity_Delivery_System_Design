"""仿真实验核心逻辑。

本文件负责实验参数发现、算例生成、订单生成、求解器调用和结果保存。
图形界面与命令行都只调用这里的公共函数，因此两种运行方式使用完全相同的
实验定义和输出格式。
"""

import json
import os
import random
import re
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from config import DeliveryConfig
from data_loader import DataLoader, DeliveryData, OrderBatch
from solvers import SOLVER_REGISTRY


# 订单生成参数不属于数学模型的 DeliveryConfig，但同样会影响实验输入数据。
# 它们与 config.py 中动态发现的字段一起参加单因素灵敏度分析。
INPUT_PARAMETER_DEFAULTS: Dict[str, Any] = {
    "num_orders": 100,
    "buffer_range": (0, 5),
    "large_order_prob": 0.3,
    "small_quantity_range": (10, 50),
    "large_quantity_range": (100, 300),
}

INPUT_PARAMETER_LEVELS: Dict[str, List[Any]] = {
    "num_orders": [20, 50, 100, 200],
    "buffer_range": [(0, 1), (0, 3), (0, 5), (0, 8)],
    "large_order_prob": [0.1, 0.3, 0.5, 0.7],
    "small_quantity_range": [(5, 25), (10, 50), (20, 100)],
    "large_quantity_range": [(50, 150), (100, 300), (200, 500)],
}


@dataclass(frozen=True)
class SensitivityParameter:
    """一个可进行灵敏度分析的参数定义。"""

    key: str
    label: str
    source: str
    field_name: str
    base_value: Any
    default_levels: List[Any]


@dataclass
class ExperimentPlan:
    """界面和命令行共同使用的批量实验计划。

    sensitivity_levels 的键由 get_sensitivity_parameters 动态生成。用户在
    DeliveryConfig 中新增 dataclass 字段后，该字段会自动进入这里，无需修改 GUI。
    """

    seed_count: int = 3
    time_limit: int = 500
    quick_orders: int = 20
    sensitivity_levels: Dict[str, List[Any]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.sensitivity_levels:
            self.sensitivity_levels = {
                parameter.key: parameter.default_levels
                for parameter in get_sensitivity_parameters()
            }


@dataclass
class ExperimentSpec:
    """一个具体算例的完整描述。

    求解器不放在该结构中。同一个算例会先生成一次订单，再交给所有已选求解器，
    以保证不同算法比较时使用完全相同的输入数据。
    """

    experiment_id: str
    scenario: str
    config: DeliveryConfig
    num_orders: int
    seed: int
    buffer_range: Tuple[int, int] = (0, 5)
    large_order_prob: float = 0.3
    small_quantity_range: Tuple[int, int] = (10, 50)
    large_quantity_range: Tuple[int, int] = (100, 300)
    time_limit: int = 500
    sensitivity_parameter: str = ""
    sensitivity_value: Any = None
    sensitivity_level: int = 0
    save_detail: bool = False


def _scaled_default_levels(value: Any) -> List[Any]:
    """根据 config.py 中的默认值自动构造一组可编辑的初始水平。

    自动水平仅用于给 GUI 提供合理起点。正式论文实验仍应结合参数实际含义，
    在界面中修改水平范围。字典参数按相同比例同步缩放各城市的值。
    """

    if isinstance(value, bool):
        return [False, True]
    if isinstance(value, int):
        if value == 0:
            return [0, 1, 2]
        return sorted({max(0, round(value * 0.5)), value, round(value * 1.5)})
    if isinstance(value, float):
        if value == 0:
            return [0.0, 0.5, 1.0]
        return [round(value * factor, 6) for factor in (0.5, 1.0, 1.5)]
    if isinstance(value, dict) and value and all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        for item in value.values()
    ):
        return [
            {
                key: type(item)(round(item * factor))
                if isinstance(item, int)
                else round(item * factor, 6)
                for key, item in value.items()
            }
            for factor in (0.5, 1.0, 1.5)
        ]
    return [value]


def get_sensitivity_parameters() -> List[SensitivityParameter]:
    """动态返回全部模型参数和订单生成参数。

    DeliveryConfig 必须保持为 dataclass。只要用户在 config.py 中增加一个字段并
    给出默认值，这里就会发现它，GUI、算例生成、CSV 和 JSON 都会自动包含该字段。
    """

    default_config = DeliveryConfig()
    parameters = []
    for config_field in fields(DeliveryConfig):
        value = getattr(default_config, config_field.name)
        parameters.append(
            SensitivityParameter(
                key=f"config.{config_field.name}",
                label=f"模型参数 {config_field.name}",
                source="config",
                field_name=config_field.name,
                base_value=value,
                default_levels=_scaled_default_levels(value),
            )
        )

    for name, value in INPUT_PARAMETER_DEFAULTS.items():
        parameters.append(
            SensitivityParameter(
                key=f"input.{name}",
                label=f"订单参数 {name}",
                source="input",
                field_name=name,
                base_value=value,
                default_levels=INPUT_PARAMETER_LEVELS[name],
            )
        )
    return parameters


def _coerce_like(value: Any, template: Any) -> Any:
    """把 JSON 解析结果转换为与默认值相同的结构和基础类型。"""

    if isinstance(template, bool):
        if isinstance(value, bool):
            return value
        raise ValueError("布尔参数只能填写 true 或 false。")
    if isinstance(template, int):
        return int(value)
    if isinstance(template, float):
        return float(value)
    if isinstance(template, str):
        return str(value)
    if isinstance(template, tuple):
        if not isinstance(value, (list, tuple)) or len(value) != len(template):
            raise ValueError(f"参数水平必须包含 {len(template)} 个元素。")
        return tuple(_coerce_like(item, template[index]) for index, item in enumerate(value))
    if isinstance(template, list):
        if not isinstance(value, list):
            raise ValueError("该参数水平必须是列表。")
        if not template:
            return value
        return [_coerce_like(item, template[0]) for item in value]
    if isinstance(template, dict):
        if not isinstance(value, dict):
            raise ValueError("该参数水平必须是 JSON 对象。")
        converted = {}
        sample_key = next(iter(template), None)
        sample_value = next(iter(template.values()), None)
        for key, item in value.items():
            converted_key = _coerce_like(key, sample_key) if sample_key is not None else key
            converted_value = (
                _coerce_like(item, template.get(converted_key, sample_value))
                if sample_value is not None
                else item
            )
            converted[converted_key] = converted_value
        return converted
    return value


def levels_to_text(values: List[Any]) -> str:
    """把参数水平统一转换为 GUI 可编辑的 JSON 数组。"""

    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def parse_parameter_levels(text: str, template: Any) -> List[Any]:
    """解析 GUI 中的参数水平，并按该参数默认值恢复类型。

    统一使用 JSON 数组格式，可以无歧义地表示数值、布尔值、区间、列表和字典。
    """

    try:
        values = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"请输入合法的 JSON 数组：{exc.msg}") from exc
    if not isinstance(values, list) or not values:
        raise ValueError("参数水平必须是至少含一个元素的 JSON 数组。")
    return [_coerce_like(value, template) for value in values]


def generate_random_orders(
    config: DeliveryConfig,
    num_orders: int = 50,
    seed: int = 42,
    buffer_range: Tuple[int, int] = (0, 5),
    large_order_prob: float = 0.3,
    small_quantity_range: Tuple[int, int] = (10, 50),
    large_quantity_range: Tuple[int, int] = (100, 300),
):
    """按给定参数生成一批可复现的随机订单。"""

    random.seed(seed)
    np.random.seed(seed)

    pos_orders = {}
    neg_orders = {}
    all_orders = {}
    min_buffer, max_buffer = buffer_range

    for order_id in range(1, num_orders + 1):
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

        quantity_range = (
            large_quantity_range if random.random() < large_order_prob else small_quantity_range
        )
        quantity = random.randint(*quantity_range)
        order = OrderBatch(
            batch_id=order_id,
            flow=flow,
            quantity=quantity,
            earliest_start=earliest_start,
            latest_completion=latest_completion,
            penalty_lost=config.penalty_lost,
        )

        all_orders[order_id] = order
        if flow == "+":
            pos_orders[order_id] = order
        else:
            neg_orders[order_id] = order

    return pos_orders, neg_orders, all_orders


def build_delivery_data(config: DeliveryConfig, orders_tuple) -> DeliveryData:
    """把订单和参数转换为优化模型需要的数据结构。"""

    pos, neg, _ = orders_tuple
    loader = DataLoader(config)
    manual_1, manual_2 = loader.generate_arcs_manual()
    auto = loader.generate_arcs_auto()
    sets_manual_1, sets_manual_2, sets_auto = loader.generate_sets(
        manual_1, manual_2, auto
    )
    epsilon = loader.generate_epsilon_sets(pos, neg, manual_1, manual_2)
    coeff_1, coeff_2 = loader.pre_inverse_count(manual_1, manual_2)

    return DeliveryData(
        arcs_manual_1=manual_1,
        arcs_manual_2=manual_2,
        arcs_auto=auto,
        sets_manual_1=sets_manual_1,
        sets_manual_2=sets_manual_2,
        sets_auto=sets_auto,
        cap_coeff_1=coeff_1,
        cap_coeff_2=coeff_2,
        pos_orders=pos,
        neg_orders=neg,
        all_orders=orders_tuple[2],
        epsilon_sets=epsilon,
    )


def make_seed_list(start: int, count: int) -> List[int]:
    """生成连续随机种子，便于重复实验。"""

    if count <= 0:
        raise ValueError("随机种子数必须大于 0。")
    return [start + index for index in range(count)]


def build_quick_specs(plan: ExperimentPlan) -> List[ExperimentSpec]:
    """构建用于检查环境、模型和输出链路的快速测试。"""

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


def _safe_id_fragment(text: str) -> str:
    """把参数名转换为适合文件名和实验编号的片段。"""

    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").upper()


def _build_spec_for_level(
    plan: ExperimentPlan,
    parameter: SensitivityParameter,
    value: Any,
    level_index: int,
    seed: int,
) -> ExperimentSpec:
    """以默认输入为基准，只替换一个参数，构建单因素灵敏度算例。"""

    config = DeliveryConfig()
    input_values = dict(INPUT_PARAMETER_DEFAULTS)
    if parameter.source == "config":
        config = replace(config, **{parameter.field_name: value})
    else:
        input_values[parameter.field_name] = value

    parameter_id = _safe_id_fragment(parameter.key)
    return ExperimentSpec(
        experiment_id=f"SENS_{parameter_id}_L{level_index}_S{seed}",
        scenario="sensitivity",
        config=config,
        num_orders=input_values["num_orders"],
        seed=seed,
        buffer_range=input_values["buffer_range"],
        large_order_prob=input_values["large_order_prob"],
        small_quantity_range=input_values["small_quantity_range"],
        large_quantity_range=input_values["large_quantity_range"],
        time_limit=plan.time_limit,
        sensitivity_parameter=parameter.key,
        sensitivity_value=value,
        sensitivity_level=level_index,
    )


def build_sensitivity_specs(plan: ExperimentPlan) -> List[ExperimentSpec]:
    """动态构建覆盖全部模型参数和订单参数的单因素灵敏度算例。"""

    specs = []
    seeds = make_seed_list(3001, plan.seed_count)
    for parameter in get_sensitivity_parameters():
        levels = plan.sensitivity_levels.get(parameter.key, parameter.default_levels)
        if not levels:
            raise ValueError(f"{parameter.key} 至少需要一个参数水平。")
        for (level_index, value), seed in product(enumerate(levels, start=1), seeds):
            specs.append(_build_spec_for_level(plan, parameter, value, level_index, seed))
    return specs


def build_specs(selected_scenarios: List[str], plan: ExperimentPlan) -> List[ExperimentSpec]:
    """根据用户选择生成快速测试或灵敏度分析算例。"""

    unknown = set(selected_scenarios) - {"quick", "sensitivity"}
    if unknown:
        raise ValueError(f"不支持的实验场景：{', '.join(sorted(unknown))}")

    specs = []
    if "quick" in selected_scenarios:
        specs.extend(build_quick_specs(plan))
    if "sensitivity" in selected_scenarios:
        specs.extend(build_sensitivity_specs(plan))
    if not specs:
        raise ValueError("请至少选择一个实验场景。")
    return specs


def _json_value(value: Any) -> str:
    """把复杂参数稳定地写入 CSV 单元格。"""

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _config_csv_fields(config: DeliveryConfig) -> Dict[str, Any]:
    """动态展开全部 DeliveryConfig 字段，新增参数会自动进入 CSV。"""

    return {
        f"Config_{config_field.name}": _json_value(getattr(config, config_field.name))
        for config_field in fields(DeliveryConfig)
    }


def _generation_payload(spec: ExperimentSpec) -> Dict[str, Any]:
    """返回能够完整复现订单的全部生成参数。"""

    return {
        "num_orders": spec.num_orders,
        "seed": spec.seed,
        "buffer_range": spec.buffer_range,
        "large_order_prob": spec.large_order_prob,
        "small_quantity_range": spec.small_quantity_range,
        "large_quantity_range": spec.large_quantity_range,
    }


def run_experiment_suite(
    specs: List[ExperimentSpec],
    solver_names: List[str],
    timestamp: str = None,
) -> pd.DataFrame:
    """批量运行实验，并同时保存完整 CSV 和结构化 JSON。

    CSV 每行对应“一个算例 + 一个求解器”，适合统计分析；批次 JSON 按算例组织，
    订单只保存一次，旗下可包含多个求解器结果，适合复现实验和检查详细解。
    """

    unknown_solvers = set(solver_names) - set(SOLVER_REGISTRY)
    if unknown_solvers:
        raise ValueError(f"未知求解器：{', '.join(sorted(unknown_solvers))}")
    if not solver_names:
        raise ValueError("请至少选择一个求解器。")

    all_summaries = []
    json_experiments = []
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
            small_quantity_range=spec.small_quantity_range,
            large_quantity_range=spec.large_quantity_range,
        )
        data = build_delivery_data(spec.config, orders_tuple)
        total_demand = sum(order.quantity for order in orders_tuple[2].values())
        json_experiment = {
            "scenario": spec.scenario,
            "experiment_id": spec.experiment_id,
            "sensitivity_parameter": spec.sensitivity_parameter or None,
            "sensitivity_value": spec.sensitivity_value,
            "sensitivity_level": spec.sensitivity_level or None,
            "time_limit_sec": spec.time_limit,
            "config": asdict(spec.config),
            "generation_parameters": _generation_payload(spec),
            "orders": {str(key): asdict(value) for key, value in orders_tuple[2].items()},
            "solver_results": [],
        }

        for solver_name in solver_names:
            solver = SOLVER_REGISTRY[solver_name]
            print(f"  -> 使用求解器：{solver.display_name}")
            result = solver.solve(spec.config, data, orders_tuple, spec.time_limit)
            print(f"     {result.message}")

            summary = {
                "Scenario": spec.scenario,
                "Exp_ID": spec.experiment_id,
                "Solver": result.solver_name,
                "Seed": spec.seed,
                "Sensitivity_Parameter": spec.sensitivity_parameter,
                "Sensitivity_Value": _json_value(spec.sensitivity_value),
                "Sensitivity_Level": spec.sensitivity_level or None,
                "Status": result.status,
                "Solve_Time_Sec": result.solve_time_sec,
                "Time_Limit_Sec": spec.time_limit,
                "Num_Orders": spec.num_orders,
                "Total_Demand": total_demand,
                "Buffer_Min": spec.buffer_range[0],
                "Buffer_Max": spec.buffer_range[1],
                "Large_Order_Prob": spec.large_order_prob,
                "Small_Quantity_Min": spec.small_quantity_range[0],
                "Small_Quantity_Max": spec.small_quantity_range[1],
                "Large_Quantity_Min": spec.large_quantity_range[0],
                "Large_Quantity_Max": spec.large_quantity_range[1],
                **_config_csv_fields(spec.config),
                "Total_Cost": result.total_cost,
                "Best_Bound": result.best_bound,
                "MIP_Gap": result.mip_gap,
                "Unserved_Rate": result.unserved_rate,
                "Auto_Usage": result.auto_usage,
                "Manual_Usage": result.manual_usage,
                "Message": result.message,
            }
            all_summaries.append(summary)
            json_experiment["solver_results"].append(
                {
                    "solver": result.solver_name,
                    "status": result.status,
                    "solve_time_sec": result.solve_time_sec,
                    "total_cost": result.total_cost,
                    "best_bound": result.best_bound,
                    "mip_gap": result.mip_gap,
                    "unserved_rate": result.unserved_rate,
                    "auto_usage": result.auto_usage,
                    "manual_usage": result.manual_usage,
                    "message": result.message,
                    "detail": result.detail,
                }
            )

            if spec.save_detail:
                detail_path = (
                    f"results/detail_{spec.experiment_id}_{result.solver_name}_{timestamp}.json"
                )
                detail_payload = {**json_experiment, "solver_results": [json_experiment["solver_results"][-1]]}
                with open(detail_path, "w", encoding="utf-8") as file:
                    json.dump(detail_payload, file, indent=2, ensure_ascii=False)

        json_experiments.append(json_experiment)

    data_frame = pd.DataFrame(all_summaries)
    csv_filename = f"results/full_experiment_summary_{timestamp}.csv"
    json_filename = f"results/full_experiment_results_{timestamp}.json"
    data_frame.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    with open(json_filename, "w", encoding="utf-8") as file:
        json.dump(
            {
                "format_version": 2,
                "generated_at": timestamp,
                "experiment_count": len(specs),
                "solver_names": solver_names,
                "experiments": json_experiments,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n所有测试完成！CSV 汇总已保存至: {csv_filename}")
    print(f"完整批次 JSON 已保存至: {json_filename}")
    print(data_frame)
    return data_frame
