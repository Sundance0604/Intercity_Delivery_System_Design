"""仿真实验核心逻辑。

本文件负责三类参数发现、算例生成、订单生成、求解器调用和结果保存。
图形界面与命令行只调用这里的公共函数，因此两种入口使用完全相同的实验定义。
"""

import json
import os
import random
import re
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intercity_delivery.configuration import DeliveryConfig, OrderGenerationConfig, RollingHorizonConfig
from intercity_delivery.data.loader import DataLoader, DeliveryData, OrderBatch
from intercity_delivery.data.cfs_catalog import (
    cfs_area_filename_label,
    cfs_area_name,
)
from intercity_delivery.data.sqlite_store import is_sqlite_path
from intercity_delivery.experiments.solvers import (
    SOLVER_REGISTRY,
    get_solver_result_slug,
)

@dataclass(frozen=True)
class SensitivityParameter:
    """一个可进行灵敏度分析的参数定义。"""

    key: str
    label: str
    source: str
    field_name: str
    base_value: Any
    default_levels: List[Any]
    solver_names: Optional[Tuple[str, ...]] = None


@dataclass
class ExperimentPlan:
    """GUI 和 CLI 共用的批量实验计划。"""

    seed_count: int = 3
    time_limit: int = 500
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
    algorithm_config: RollingHorizonConfig
    order_config: OrderGenerationConfig
    seed: int
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
    if isinstance(value, tuple) and value and all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        for item in value
    ):
        levels = []
        for factor in (0.5, 1.0, 1.5):
            scaled = tuple(
                type(item)(round(item * factor))
                if isinstance(item, int)
                else round(item * factor, 6)
                for item in value
            )
            if scaled not in levels:
                levels.append(scaled)
        return levels
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


PARAMETER_CONFIGS = {
    "model": ("模型参数", DeliveryConfig),
    "algorithm": ("算法参数", RollingHorizonConfig),
    "order": ("订单参数", OrderGenerationConfig),
}


def get_sensitivity_parameters() -> List[SensitivityParameter]:
    """从三个 dataclass 动态发现全部单因素灵敏度参数。"""

    parameters = []
    for source, (_category_label, config_type) in PARAMETER_CONFIGS.items():
        default_config = config_type()
        for config_field in fields(config_type):
            value = getattr(default_config, config_field.name)
            parameters.append(
                SensitivityParameter(
                    key=f"{source}.{config_field.name}",
                    label=config_field.name,
                    source=source,
                    field_name=config_field.name,
                    base_value=value,
                    default_levels=list(
                        config_field.metadata.get(
                            "sensitivity_levels",
                            _scaled_default_levels(value),
                        )
                    ),
                    solver_names=(
                        tuple(config_field.metadata["solvers"])
                        if "solvers" in config_field.metadata
                        else None
                    ),
                )
            )
    return parameters


def get_parameter_groups() -> Dict[str, List[SensitivityParameter]]:
    """按模型、算法、订单三类返回动态参数，供 GUI 和 CLI 共用。"""

    groups = {source: [] for source in PARAMETER_CONFIGS}
    for parameter in get_sensitivity_parameters():
        groups[parameter.source].append(parameter)
    return groups


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
    order_config: OrderGenerationConfig,
    seed: int = 42,
):
    """按给定参数生成一批可复现的随机订单。"""

    order_config.validate()
    random.seed(seed)
    np.random.seed(seed)

    pos_orders = {}
    neg_orders = {}
    all_orders = {}
    min_buffer, max_buffer = order_config.buffer_range

    for order_id in range(1, order_config.num_orders + 1):
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
            order_config.large_quantity_range
            if random.random() < order_config.large_order_prob
            else order_config.small_quantity_range
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


def _sample_order_batches(
    source_orders: Dict[int, OrderBatch],
    config: DeliveryConfig,
    order_config: OrderGenerationConfig,
    seed: int,
):
    """Deterministically sample and validate an existing model-order pool."""

    requested = order_config.num_orders
    if requested > len(source_orders):
        raise ValueError(
            f"真实数据只有 {len(source_orders)} 条订单，当前算例要求 {requested} 条。"
        )
    selected_ids = sorted(random.Random(seed).sample(list(source_orders), requested))
    all_orders = {}
    for new_id, source_id in enumerate(selected_ids, start=1):
        source = source_orders[source_id]
        if source.flow not in {"+", "-"}:
            raise ValueError(f"订单 {source_id} 的 flow 必须为 '+' 或 '-'。")
        if not (0 <= source.earliest_start < source.latest_completion <= config.T):
            raise ValueError(
                f"订单 {source_id} 的时间窗 [{source.earliest_start}, "
                f"{source.latest_completion}] 超出当前 T={config.T}。"
            )
        all_orders[new_id] = replace(
            source,
            batch_id=new_id,
            penalty_lost=config.penalty_lost,
        )
    pos_orders = {key: order for key, order in all_orders.items() if order.flow == "+"}
    neg_orders = {key: order for key, order in all_orders.items() if order.flow == "-"}
    return pos_orders, neg_orders, all_orders


def resolve_real_city_pair(
    path: str,
    city_pair: Optional[Tuple[str, str]] = None,
) -> Tuple[str, str]:
    """Resolve and validate the city pair for either SQLite or legacy JSON."""

    from intercity_delivery.data.cfs_processor import _normalize_area

    input_path = Path(path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"真实数据文件不存在：{input_path}")

    if is_sqlite_path(input_path):
        if city_pair is None or len(city_pair) != 2:
            raise ValueError("SQLite 真实数据必须选择一个城市对。")
        city_a, city_b = (_normalize_area(item) for item in city_pair)
    else:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        payload_pair = payload.get("city_pair") or {}
        city_a = payload_pair.get("city_1")
        city_b = payload_pair.get("city_2")
        if city_pair is not None:
            city_a, city_b = city_pair
        if not city_a or not city_b:
            raise ValueError("真实数据 JSON 缺少 city_pair.city_1/city_2。")
        city_a, city_b = _normalize_area(city_a), _normalize_area(city_b)

    if city_a == city_b:
        raise ValueError("城市对中的两个 CFS Area 必须不同。")
    return city_a, city_b


def load_real_orders_with_metadata(
    path: str,
    config: DeliveryConfig,
    order_config: OrderGenerationConfig,
    seed: int = 42,
    city_pair: Optional[Tuple[str, str]] = None,
):
    """Load model orders directly from SQLite or from a processed JSON pool."""

    from intercity_delivery.data.cfs_processor import (
        ProcessorConfig,
        build_model_orders,
        load_processed_orders,
        sample_pair_records,
    )

    input_path = Path(path).expanduser()
    resolved_pair = resolve_real_city_pair(str(input_path), city_pair)
    city_a, city_b = resolved_pair

    if is_sqlite_path(input_path):
        processor_config = ProcessorConfig(
            num_orders=order_config.num_orders,
            seed=seed,
            planning_periods=config.T,
            period_hours=config.t_0 / 60.0,
            buffer_min_periods=order_config.buffer_range[0],
            buffer_max_periods=order_config.buffer_range[1],
            penalty_lost=config.penalty_lost,
        )
        processor_config.validate()
        records, sample_stats = sample_pair_records(
            input_path, city_a, city_b, processor_config
        )
        processed_orders, recommendations = build_model_orders(
            records, city_a, city_b, processor_config
        )
        source_orders = {
            item.batch_id: OrderBatch(**item.model_fields())
            for item in processed_orders
        }
        orders_tuple = _sample_order_batches(
            source_orders,
            config,
            replace(order_config, num_orders=len(source_orders)),
            seed,
        )
        metadata = {
            "source_kind": "sqlite",
            "city_pair": {"city_1": city_a, "city_2": city_b},
            "city_names": {
                "city_1": cfs_area_name(city_a),
                "city_2": cfs_area_name(city_b),
            },
            "sampling_statistics": sample_stats,
            "model_recommendations": recommendations,
        }
        return orders_tuple, metadata

    _, _, source_orders = load_processed_orders(input_path)
    orders_tuple = _sample_order_batches(source_orders, config, order_config, seed)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    metadata = {
        "source_kind": "processed_json",
        "city_pair": {"city_1": city_a, "city_2": city_b},
        "city_names": {
            "city_1": cfs_area_name(city_a),
            "city_2": cfs_area_name(city_b),
        },
        "model_recommendations": payload.get("model_recommendations", {}),
    }
    return orders_tuple, metadata


def load_real_orders(
    path: str,
    config: DeliveryConfig,
    order_config: OrderGenerationConfig,
    seed: int = 42,
    city_pair: Optional[Tuple[str, str]] = None,
):
    """Backward-compatible real-order loader returning only the order tuple."""

    input_path = Path(path).expanduser()
    if not is_sqlite_path(input_path) and city_pair is None:
        from intercity_delivery.data.cfs_processor import load_processed_orders

        _, _, source_orders = load_processed_orders(input_path)
        return _sample_order_batches(source_orders, config, order_config, seed)

    orders_tuple, _metadata = load_real_orders_with_metadata(
        path, config, order_config, seed=seed, city_pair=city_pair
    )
    return orders_tuple

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

    order_config = OrderGenerationConfig()
    return [
        ExperimentSpec(
            experiment_id=f"QUICK_N{order_config.num_orders}_S42",
            scenario="quick",
            config=DeliveryConfig(),
            algorithm_config=RollingHorizonConfig(),
            order_config=order_config,
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
    algorithm_config = RollingHorizonConfig()
    order_config = OrderGenerationConfig()
    if parameter.source == "model":
        config = replace(config, **{parameter.field_name: value})
    elif parameter.source == "algorithm":
        algorithm_config = replace(
            algorithm_config, **{parameter.field_name: value}
        )
    elif parameter.source == "order":
        order_config = replace(order_config, **{parameter.field_name: value})
    else:
        raise ValueError(f"未知参数类别：{parameter.source}")
    algorithm_config.validate()
    order_config.validate()

    parameter_id = _safe_id_fragment(parameter.key)
    return ExperimentSpec(
        experiment_id=f"SENS_{parameter_id}_L{level_index}_S{seed}",
        scenario="sensitivity",
        config=config,
        algorithm_config=algorithm_config,
        order_config=order_config,
        seed=seed,
        time_limit=plan.time_limit,
        sensitivity_parameter=parameter.key,
        sensitivity_value=value,
        sensitivity_level=level_index,
    )


def build_sensitivity_specs(plan: ExperimentPlan) -> List[ExperimentSpec]:
    """动态构建覆盖模型、算法和订单参数的单因素灵敏度算例。"""

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


def _dataclass_csv_fields(prefix: str, config_object) -> Dict[str, Any]:
    """动态展开任意一类 dataclass 参数并写入 CSV。"""

    return {
        f"{prefix}_{config_field.name}": _json_value(
            getattr(config_object, config_field.name)
        )
        for config_field in fields(type(config_object))
    }


def _generation_payload(spec: ExperimentSpec) -> Dict[str, Any]:
    """返回能够完整复现订单的全部生成参数。"""

    return {
        "seed": spec.seed,
        **asdict(spec.order_config),
    }


def applicable_solver_names(
    spec: ExperimentSpec, solver_names: List[str]
) -> List[str]:
    """过滤对当前灵敏度参数有实际响应的求解器。

    算法参数只交给声明支持该参数类别的算法，避免精确 MIP 产生完全重复的伪结果。
    """

    if not spec.sensitivity_parameter:
        return list(solver_names)
    source = spec.sensitivity_parameter.split(".", 1)[0]
    parameter = next(
        (
            item
            for item in get_sensitivity_parameters()
            if item.key == spec.sensitivity_parameter
        ),
        None,
    )
    return [
        name
        for name in solver_names
        if source in SOLVER_REGISTRY[name].sensitivity_sources
        and (
            parameter is None
            or parameter.solver_names is None
            or name in parameter.solver_names
        )
    ]


def planned_run_count(specs: List[ExperimentSpec], solver_names: List[str]) -> int:
    """返回实际会执行的“算例×适用求解器”数量，供 GUI 和 CLI 预览。"""

    return sum(len(applicable_solver_names(spec, solver_names)) for spec in specs)


def _safe_filename_fragment(value: str, max_length: int = 180) -> str:
    value = re.sub(r"[^A-Za-z0-9+_.-]+", "_", str(value)).strip("_.")
    return (value or "experiment")[:max_length].rstrip("_.-")


def build_result_context_tag(
    data_source: str,
    solver_names: List[str],
    city_pair: Optional[Tuple[str, str]] = None,
) -> str:
    """Build the human-readable city-pair and approach segment of filenames."""

    if data_source == "real":
        if not city_pair:
            raise ValueError("真实数据结果命名需要城市对。")
        city_a, city_b = city_pair
        data_label = (
            f"{cfs_area_filename_label(city_a)}_{city_a}_to_"
            f"{cfs_area_filename_label(city_b)}_{city_b}"
        )
    else:
        data_label = "generated"

    solver_label = "+".join(get_solver_result_slug(name) for name in solver_names)
    return _safe_filename_fragment(f"{data_label}__{solver_label}")

def run_experiment_suite(
    specs: List[ExperimentSpec],
    solver_names: List[str],
    timestamp: str = None,
    data_source: str = "generated",
    real_data_path: str = None,
    real_city_pair: Optional[Tuple[str, str]] = None,
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
    if data_source not in {"generated", "real"}:
        raise ValueError("data_source 必须为 generated 或 real。")
    if data_source == "real" and not real_data_path:
        raise ValueError("选择真实数据时必须提供 CFS SQLite 或处理后 JSON 文件。")

    selected_city_pair = (
        resolve_real_city_pair(real_data_path, real_city_pair)
        if data_source == "real"
        else None
    )
    result_context = build_result_context_tag(
        data_source, solver_names, selected_city_pair
    )

    all_summaries = []
    json_experiments = []
    os.makedirs("results", exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    for index, spec in enumerate(specs, start=1):
        print(f"\n[{index}/{len(specs)}] 构建算例 {spec.experiment_id}")
        if data_source == "real":
            orders_tuple, real_data_metadata = load_real_orders_with_metadata(
                real_data_path,
                spec.config,
                spec.order_config,
                seed=spec.seed,
                city_pair=selected_city_pair,
            )
        else:
            orders_tuple = generate_random_orders(
                spec.config,
                spec.order_config,
                seed=spec.seed,
            )
            real_data_metadata = None
        data = build_delivery_data(spec.config, orders_tuple)
        total_demand = sum(order.quantity for order in orders_tuple[2].values())
        json_experiment = {
            "data_source": data_source,
            "real_data_path": str(Path(real_data_path).resolve()) if real_data_path else None,
            "real_data_metadata": real_data_metadata,
            "scenario": spec.scenario,
            "experiment_id": spec.experiment_id,
            "sensitivity_parameter": spec.sensitivity_parameter or None,
            "sensitivity_value": spec.sensitivity_value,
            "sensitivity_level": spec.sensitivity_level or None,
            "time_limit_sec": spec.time_limit,
            "model_parameters": asdict(spec.config),
            "algorithm_parameters": asdict(spec.algorithm_config),
            "order_parameters": asdict(spec.order_config),
            "generation_parameters": _generation_payload(spec),
            "orders": {str(key): asdict(value) for key, value in orders_tuple[2].items()},
            "solver_results": [],
        }

        spec_solver_names = applicable_solver_names(spec, solver_names)
        if not spec_solver_names:
            print("  -> 当前所选求解器均不适用于该参数类别，跳过。")
        for solver_name in spec_solver_names:
            solver = SOLVER_REGISTRY[solver_name]
            print(f"  -> 使用求解器：{solver.display_name}")
            result = solver.solve(
                spec.config,
                data,
                orders_tuple,
                spec.time_limit,
                spec.algorithm_config,
            )
            print(f"     {result.message}")

            summary = {
                "Scenario": spec.scenario,
                "Exp_ID": spec.experiment_id,
                "Solver": result.solver_name,
                "Data_Source": data_source,
                "City_1_CFS_Area": selected_city_pair[0] if selected_city_pair else None,
                "City_1_Name": cfs_area_name(selected_city_pair[0]) if selected_city_pair else None,
                "City_2_CFS_Area": selected_city_pair[1] if selected_city_pair else None,
                "City_2_Name": cfs_area_name(selected_city_pair[1]) if selected_city_pair else None,
                "Seed": spec.seed,
                "Sensitivity_Parameter": spec.sensitivity_parameter,
                "Sensitivity_Value": _json_value(spec.sensitivity_value),
                "Sensitivity_Level": spec.sensitivity_level or None,
                "Status": result.status,
                "Solve_Time_Sec": result.solve_time_sec,
                "Time_Limit_Sec": spec.time_limit,
                "Parameter_Category": (
                    spec.sensitivity_parameter.split(".", 1)[0]
                    if spec.sensitivity_parameter
                    else None
                ),
                "Num_Orders": spec.order_config.num_orders,
                "Total_Demand": total_demand,
                **_dataclass_csv_fields("Model", spec.config),
                **_dataclass_csv_fields("Algorithm", spec.algorithm_config),
                **_dataclass_csv_fields("Order", spec.order_config),
                "Total_Cost": result.total_cost,
                "Best_Bound": result.best_bound,
                "MIP_Gap": result.mip_gap,
                "Unserved_Rate": result.unserved_rate,
                "Auto_Usage": result.auto_usage,
                "Manual_Usage": result.manual_usage,
                "Direct_Ratio": result.direct_ratio,
                "Direct_Volume": result.direct_volume,
                "Transshipment_Volume": result.transshipment_volume,
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
                    "direct_ratio": result.direct_ratio,
                    "direct_volume": result.direct_volume,
                    "transshipment_volume": result.transshipment_volume,
                    "message": result.message,
                    "detail": result.detail,
                }
            )

            if spec.save_detail:
                detail_context = build_result_context_tag(
                    data_source, [result.solver_name], selected_city_pair
                )
                detail_path = (
                    f"results/detail_{spec.experiment_id}_{detail_context}__"
                    f"{timestamp}.json"
                )
                detail_payload = {**json_experiment, "solver_results": [json_experiment["solver_results"][-1]]}
                with open(detail_path, "w", encoding="utf-8") as file:
                    json.dump(detail_payload, file, indent=2, ensure_ascii=False)

        json_experiments.append(json_experiment)

    data_frame = pd.DataFrame(all_summaries)
    csv_filename = (
        f"results/full_experiment_summary_{result_context}__{timestamp}.csv"
    )
    json_filename = (
        f"results/full_experiment_results_{result_context}__{timestamp}.json"
    )
    data_frame.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    with open(json_filename, "w", encoding="utf-8") as file:
        json.dump(
            {
                "format_version": 4,
                "generated_at": timestamp,
                "experiment_count": len(specs),
                "solver_run_count": len(all_summaries),
                "solver_names": solver_names,
                "data_source": data_source,
                "real_data_path": str(Path(real_data_path).resolve()) if real_data_path else None,
                "city_pair": (
                    {"city_1": selected_city_pair[0], "city_2": selected_city_pair[1]}
                    if selected_city_pair
                    else None
                ),
                "city_names": (
                    {
                        "city_1": cfs_area_name(selected_city_pair[0]),
                        "city_2": cfs_area_name(selected_city_pair[1]),
                    }
                    if selected_city_pair
                    else None
                ),
                "result_context": result_context,
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
