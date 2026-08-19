"""把 2022 CFS PUMS 微观货运记录转换为当前两城市模型的订单。

该脚本针对官方 PUMS CSV（约 680 MB）采用分块读取。它不会把 CFS 中不存在的
订单发布日期或截止时间当作观测数据；这些时序字段会按明确、可复现的规则构造，
并在输出元数据中记录参数。

典型用法：

    conda run -n pavane python cfs_data_processor.py \
        --input D:/download/cfs_2022_pums.csv.zip \
        --output-dir data/cfs_processed \
        --city-a 06-348 --city-b 06-488 --num-orders 100

省略 --city-a/--city-b 时，脚本自动选择双向加权货流最充足的都市区对。
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd


REQUIRED_COLUMNS = (
    "SHIPMT_ID",
    "ORIG_CFS_AREA",
    "DEST_CFS_AREA",
    "MODE",
    "SCTG",
    "SHIPMT_VALUE",
    "SHIPMT_WGHT",
    "SHIPMT_DIST_GC",
    "TEMP_CNTL_YN",
    "EXPORT_YN",
    "HAZMAT",
    "WGT_FACTOR",
)

CODE_COLUMNS = {
    "SHIPMT_ID": "string",
    "ORIG_CFS_AREA": "string",
    "DEST_CFS_AREA": "string",
    "MODE": "string",
    "SCTG": "string",
    "TEMP_CNTL_YN": "string",
    "EXPORT_YN": "string",
    "HAZMAT": "string",
}

NUMERIC_COLUMNS = (
    "SHIPMT_VALUE",
    "SHIPMT_WGHT",
    "SHIPMT_DIST_GC",
    "WGT_FACTOR",
)


@dataclass(frozen=True)
class ProcessorConfig:
    """从 CFS 到模型订单的全部可复现转换参数。"""

    modes: Tuple[str, ...] = ("111", "112")
    domestic_only: bool = True
    metro_only: bool = True
    min_distance_miles: float = 50.0
    num_orders: int = 100
    seed: int = 42
    chunksize: int = 250_000
    planning_periods: int = 24
    period_hours: float = 1.0
    truck_speed_mph: float = 50.0
    circuity_factor: float = 1.20
    local_service_periods_per_end: int = 1
    buffer_min_periods: int = 0
    buffer_max_periods: int = 5
    pounds_per_model_unit: float = 50.0
    min_quantity: int = 10
    max_quantity: int = 300
    penalty_lost: float = 10.0
    min_records_per_direction: int = 10

    def validate(self) -> None:
        if not self.modes:
            raise ValueError("modes 不能为空。")
        if self.num_orders < 2:
            raise ValueError("num_orders 至少为 2，才能生成双向订单。")
        if self.chunksize <= 0:
            raise ValueError("chunksize 必须大于 0。")
        if self.planning_periods <= 0 or self.period_hours <= 0:
            raise ValueError("规划期和单时段长度必须大于 0。")
        if self.truck_speed_mph <= 0 or self.circuity_factor < 1:
            raise ValueError("货车速度必须大于 0，绕行系数必须不小于 1。")
        if self.local_service_periods_per_end < 0:
            raise ValueError("两端服务时段不能为负数。")
        if not 0 <= self.buffer_min_periods <= self.buffer_max_periods:
            raise ValueError("时间窗缓冲必须满足 0 <= min <= max。")
        if self.pounds_per_model_unit <= 0:
            raise ValueError("pounds_per_model_unit 必须大于 0。")
        if not 1 <= self.min_quantity <= self.max_quantity:
            raise ValueError("模型货量必须满足 1 <= min <= max。")


@dataclass(frozen=True)
class ProcessedOrder:
    """兼容 OrderBatch 的模型字段以及可追溯的 CFS 来源字段。"""

    batch_id: int
    flow: str
    quantity: int
    earliest_start: int
    latest_completion: int
    penalty_lost: float
    source_shipmt_id: str
    origin_cfs_area: str
    destination_cfs_area: str
    source_mode: str
    source_sctg: str
    source_weight_pounds: float
    source_value_dollars: float
    source_distance_gc_miles: float
    source_weight_factor: float
    source_temperature_controlled: str
    source_hazmat: str
    estimated_route_miles: float
    estimated_linehaul_periods: int
    minimum_completion_periods: int

    def model_fields(self) -> dict:
        """返回可以直接构造 data_loader.OrderBatch 的字段。"""

        return {
            "batch_id": self.batch_id,
            "flow": self.flow,
            "quantity": self.quantity,
            "earliest_start": self.earliest_start,
            "latest_completion": self.latest_completion,
            "penalty_lost": self.penalty_lost,
        }


def _normalize_area(value: object) -> str:
    """规范 CFS Area，同时保留州 FIPS 的前导零。"""

    text = str(value).strip()
    if "-" not in text:
        raise ValueError(f"非法 CFS Area：{value!r}")
    state, metro = text.split("-", 1)
    return f"{state.zfill(2)}-{metro.zfill(3) if metro.isdigit() else metro}"


def _sqlite_query(
    config: Optional[ProcessorConfig],
    city_pair: Optional[Tuple[str, str]],
) -> Tuple[str, list]:
    columns = ", ".join(f'"{column}"' for column in REQUIRED_COLUMNS)
    if config is None:
        return f'SELECT {columns} FROM "shipments"', []

    mode_placeholders = ", ".join("?" for _ in config.modes)
    conditions = [
        f'"MODE" IN ({mode_placeholders})',
        '"ORIG_CFS_AREA" IS NOT NULL',
        '"DEST_CFS_AREA" IS NOT NULL',
        '"ORIG_CFS_AREA" <> "DEST_CFS_AREA"',
        '"SHIPMT_WGHT" > 0',
        '"SHIPMT_DIST_GC" >= ?',
        '"WGT_FACTOR" > 0',
    ]
    parameters: list = [*config.modes, config.min_distance_miles]
    if config.domestic_only:
        conditions.append('"EXPORT_YN" = ?')
        parameters.append("N")
    if city_pair is not None:
        city_a, city_b = city_pair
        conditions.append(
            '(("ORIG_CFS_AREA" = ? AND "DEST_CFS_AREA" = ?) OR '
            '("ORIG_CFS_AREA" = ? AND "DEST_CFS_AREA" = ?))'
        )
        parameters.extend((city_a, city_b, city_b, city_a))
    source = (
        '"shipments" INDEXED BY idx_shipments_od'
        if city_pair is not None
        else '"shipments"'
    )
    return f"SELECT {columns} FROM {source} WHERE " + " AND ".join(conditions), parameters


def _read_chunks(
    path: Path,
    chunksize: int,
    config: Optional[ProcessorConfig] = None,
    city_pair: Optional[Tuple[str, str]] = None,
) -> Iterator[pd.DataFrame]:
    """Read required columns from CSV/ZIP/GZ or the indexed SQLite store."""

    if path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
        query, parameters = _sqlite_query(config, city_pair)
        with closing(sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)) as connection:
            columns = {
                row[1] for row in connection.execute('PRAGMA table_info("shipments")')
            }
            missing = set(REQUIRED_COLUMNS) - columns
            if missing:
                raise ValueError(
                    "SQLite shipments 表缺少字段："
                    + ", ".join(sorted(missing))
                )
            for chunk in pd.read_sql_query(
                query,
                connection,
                params=parameters,
                chunksize=chunksize,
            ):
                for column in CODE_COLUMNS:
                    chunk[column] = chunk[column].astype("string").str.strip()
                for column in NUMERIC_COLUMNS:
                    chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
                yield chunk
        return

    try:
        reader = pd.read_csv(
            path,
            usecols=list(REQUIRED_COLUMNS),
            dtype=CODE_COLUMNS,
            chunksize=chunksize,
            low_memory=False,
            compression="infer",
        )
    except ValueError as exc:
        raise ValueError(
            "输入文件缺少 2022 CFS PUMS 必需字段；请确认传入的是官方 CSV，"
            f"需要字段：{', '.join(REQUIRED_COLUMNS)}"
        ) from exc

    for chunk in reader:
        chunk.columns = [str(column).strip().upper() for column in chunk.columns]
        for column in NUMERIC_COLUMNS:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk["ORIG_CFS_AREA"] = chunk["ORIG_CFS_AREA"].str.strip()
        chunk["DEST_CFS_AREA"] = chunk["DEST_CFS_AREA"].str.strip()
        yield chunk


def _base_filter(chunk: pd.DataFrame, config: ProcessorConfig) -> pd.DataFrame:
    valid = (
        chunk["MODE"].isin(config.modes)
        & chunk["ORIG_CFS_AREA"].notna()
        & chunk["DEST_CFS_AREA"].notna()
        & (chunk["ORIG_CFS_AREA"] != chunk["DEST_CFS_AREA"])
        & (chunk["SHIPMT_WGHT"] > 0)
        & (chunk["SHIPMT_DIST_GC"] >= config.min_distance_miles)
        & (chunk["WGT_FACTOR"] > 0)
    )
    if config.domestic_only:
        valid &= chunk["EXPORT_YN"].eq("N")
    if config.metro_only:
        origin_ma = chunk["ORIG_CFS_AREA"].str.split("-").str[-1]
        destination_ma = chunk["DEST_CFS_AREA"].str.split("-").str[-1]
        valid &= ~origin_ma.isin(("00000", "0000", "000", "99999"))
        valid &= ~destination_ma.isin(("00000", "0000", "000", "99999"))
    return chunk.loc[valid].copy()


def _unordered_pair(origin: str, destination: str) -> Tuple[str, str]:
    return (origin, destination) if origin < destination else (destination, origin)


def select_city_pair(input_path: Path, config: ProcessorConfig) -> Tuple[str, str, dict]:
    """选择两方向都有足够记录、且较弱方向加权货流最大的 CFS Area 对。"""

    weighted = defaultdict(lambda: [0.0, 0.0])
    counts = defaultdict(lambda: [0, 0])

    for chunk in _read_chunks(input_path, config.chunksize, config=config):
        filtered = _base_filter(chunk, config)
        if filtered.empty:
            continue
        for (origin, destination), group in filtered.groupby(
            ["ORIG_CFS_AREA", "DEST_CFS_AREA"], sort=False
        ):
            city_a, city_b = _unordered_pair(str(origin), str(destination))
            direction = 0 if str(origin) == city_a else 1
            weighted[(city_a, city_b)][direction] += float(group["WGT_FACTOR"].sum())
            counts[(city_a, city_b)][direction] += int(len(group))

    candidates = []
    for pair, direction_counts in counts.items():
        if min(direction_counts) < config.min_records_per_direction:
            continue
        direction_weights = weighted[pair]
        candidates.append(
            (
                min(direction_weights),
                sum(direction_weights),
                min(direction_counts),
                pair,
            )
        )
    if not candidates:
        raise ValueError(
            "没有找到满足双向最小记录数的 CFS Area 对。可降低 "
            "--min-records-per-direction、--min-distance-miles，或关闭 --metro-only。"
        )

    _, _, _, selected = max(candidates)
    stats = {
        "selection_rule": "maximize_minimum_directional_weight",
        "raw_records_a_to_b": counts[selected][0],
        "raw_records_b_to_a": counts[selected][1],
        "weighted_shipments_a_to_b": weighted[selected][0],
        "weighted_shipments_b_to_a": weighted[selected][1],
    }
    return selected[0], selected[1], stats


def _weighted_reservoir_add(
    heap: List[Tuple[float, int, dict]],
    row: dict,
    capacity: int,
    rng: random.Random,
    sequence: int,
) -> None:
    """Efraimidis-Spirakis 加权无放回蓄水池抽样。"""

    if capacity <= 0:
        return
    weight = max(float(row["WGT_FACTOR"]), 1e-12)
    priority = math.log(max(rng.random(), 1e-15)) / weight
    item = (priority, sequence, row)
    if len(heap) < capacity:
        heapq.heappush(heap, item)
    elif priority > heap[0][0]:
        heapq.heapreplace(heap, item)


def sample_pair_records(
    input_path: Path,
    city_a: str,
    city_b: str,
    config: ProcessorConfig,
) -> Tuple[List[dict], dict]:
    """从指定 OD 对按总体代表权重抽取近似平衡的双向微观记录。"""

    plus_target = (config.num_orders + 1) // 2
    minus_target = config.num_orders // 2
    plus_heap: List[Tuple[float, int, dict]] = []
    minus_heap: List[Tuple[float, int, dict]] = []
    rng = random.Random(config.seed)
    eligible_counts = {"+": 0, "-": 0}
    sequence = 0

    for chunk in _read_chunks(
        input_path,
        config.chunksize,
        config=config,
        city_pair=(city_a, city_b),
    ):
        filtered = _base_filter(chunk, config)
        pair_rows = filtered.loc[
            (
                filtered["ORIG_CFS_AREA"].eq(city_a)
                & filtered["DEST_CFS_AREA"].eq(city_b)
            )
            | (
                filtered["ORIG_CFS_AREA"].eq(city_b)
                & filtered["DEST_CFS_AREA"].eq(city_a)
            )
        ]
        for row in pair_rows.to_dict(orient="records"):
            sequence += 1
            flow = "+" if row["ORIG_CFS_AREA"] == city_a else "-"
            eligible_counts[flow] += 1
            _weighted_reservoir_add(
                plus_heap if flow == "+" else minus_heap,
                row,
                plus_target if flow == "+" else minus_target,
                rng,
                sequence,
            )

    if len(plus_heap) < plus_target or len(minus_heap) < minus_target:
        raise ValueError(
            f"所选 OD 对没有足够记录：+方向 {eligible_counts['+']} 条，"
            f"-方向 {eligible_counts['-']} 条；需要 {plus_target}/{minus_target} 条。"
        )

    sampled = [item[2] for item in plus_heap] + [item[2] for item in minus_heap]
    rng.shuffle(sampled)
    return sampled, {
        "eligible_records_a_to_b": eligible_counts["+"],
        "eligible_records_b_to_a": eligible_counts["-"],
        "sampled_records_a_to_b": plus_target,
        "sampled_records_b_to_a": minus_target,
        "sampling_method": "weighted_without_replacement_reservoir",
        "sampling_weight": "WGT_FACTOR",
    }


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    ordered = sorted(zip(values, weights), key=lambda pair: pair[0])
    threshold = sum(weight for _, weight in ordered) / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


def calibrate_city_pair_travel_time(
    input_path: Path,
    city_a: str,
    city_b: str,
    config: ProcessorConfig,
) -> dict:
    """Estimate one seed-independent linehaul time from all eligible OD rows."""

    city_a, city_b = _normalize_area(city_a), _normalize_area(city_b)
    distances: List[float] = []
    weights: List[float] = []
    direction_counts = {"a_to_b": 0, "b_to_a": 0}
    for chunk in _read_chunks(
        input_path,
        config.chunksize,
        config=config,
        city_pair=(city_a, city_b),
    ):
        filtered = _base_filter(chunk, config)
        pair_rows = filtered.loc[
            (
                filtered["ORIG_CFS_AREA"].eq(city_a)
                & filtered["DEST_CFS_AREA"].eq(city_b)
            )
            | (
                filtered["ORIG_CFS_AREA"].eq(city_b)
                & filtered["DEST_CFS_AREA"].eq(city_a)
            )
        ]
        for row in pair_rows.itertuples(index=False):
            distances.append(float(row.SHIPMT_DIST_GC))
            weights.append(float(row.WGT_FACTOR))
            key = "a_to_b" if row.ORIG_CFS_AREA == city_a else "b_to_a"
            direction_counts[key] += 1

    if not distances or min(direction_counts.values()) == 0:
        raise ValueError(
            f"城市对 {city_a} <-> {city_b} 没有满足筛选条件的双向记录。"
        )
    representative_gc_miles = _weighted_median(distances, weights)
    representative_route_miles = representative_gc_miles * config.circuity_factor
    linehaul_hours = representative_route_miles / config.truck_speed_mph
    travel_time_periods = max(
        1, math.ceil(linehaul_hours / config.period_hours)
    )
    return {
        "method": "full_eligible_pair_weighted_median",
        "eligible_record_count": len(distances),
        "direction_counts": direction_counts,
        "weight": "WGT_FACTOR",
        "representative_gc_miles": round(representative_gc_miles, 3),
        "circuity_factor": config.circuity_factor,
        "representative_route_miles": round(representative_route_miles, 3),
        "truck_speed_mph": config.truck_speed_mph,
        "representative_linehaul_hours": round(linehaul_hours, 3),
        "period_hours": config.period_hours,
        "travel_time_periods": travel_time_periods,
    }

def build_model_orders(
    records: Sequence[dict],
    city_a: str,
    city_b: str,
    config: ProcessorConfig,
    travel_calibration: Optional[dict] = None,
) -> Tuple[List[ProcessedOrder], dict]:
    """将抽样记录转换成模型订单；物理行程时间可由全 OD 对校准固定。"""

    if travel_calibration is None:
        route_miles = [
            float(record["SHIPMT_DIST_GC"]) * config.circuity_factor
            for record in records
        ]
        weights = [float(record["WGT_FACTOR"]) for record in records]
        representative_route_miles = _weighted_median(route_miles, weights)
        linehaul_hours = representative_route_miles / config.truck_speed_mph
        linehaul_periods = max(1, math.ceil(linehaul_hours / config.period_hours))
        calibration_method = "sample_weighted_median"
    else:
        representative_route_miles = float(
            travel_calibration["representative_route_miles"]
        )
        linehaul_hours = float(
            travel_calibration["representative_linehaul_hours"]
        )
        linehaul_periods = int(travel_calibration["travel_time_periods"])
        calibration_method = str(travel_calibration.get("method", "provided"))
    minimum_completion_periods = (
        linehaul_periods + 2 * config.local_service_periods_per_end
    )
    latest_possible_start = (
        config.planning_periods
        - minimum_completion_periods
        - config.buffer_max_periods
    )
    if latest_possible_start < 0:
        raise ValueError(
            "规划期过短：代表性最短完成时段为 "
            f"{minimum_completion_periods}，最大缓冲为 {config.buffer_max_periods}，"
            f"但 T={config.planning_periods}。请增加 --planning-periods、增大"
            " --period-hours，或调整速度/服务时段。"
        )

    rng = random.Random(config.seed + 1)
    orders: List[ProcessedOrder] = []
    for order_id, record in enumerate(records, start=1):
        flow = "+" if record["ORIG_CFS_AREA"] == city_a else "-"
        raw_quantity = round(
            float(record["SHIPMT_WGHT"]) / config.pounds_per_model_unit
        )
        quantity = min(
            config.max_quantity,
            max(config.min_quantity, int(raw_quantity)),
        )
        earliest_start = rng.randint(0, latest_possible_start)
        buffer = rng.randint(
            config.buffer_min_periods, config.buffer_max_periods
        )
        latest_completion = min(
            config.planning_periods,
            earliest_start + minimum_completion_periods + buffer,
        )
        estimated_route = float(record["SHIPMT_DIST_GC"]) * config.circuity_factor
        orders.append(
            ProcessedOrder(
                batch_id=order_id,
                flow=flow,
                quantity=quantity,
                earliest_start=earliest_start,
                latest_completion=latest_completion,
                penalty_lost=config.penalty_lost,
                source_shipmt_id=str(record["SHIPMT_ID"]),
                origin_cfs_area=str(record["ORIG_CFS_AREA"]),
                destination_cfs_area=str(record["DEST_CFS_AREA"]),
                source_mode=str(record["MODE"]),
                source_sctg=str(record["SCTG"]),
                source_weight_pounds=float(record["SHIPMT_WGHT"]),
                source_value_dollars=float(record["SHIPMT_VALUE"]),
                source_distance_gc_miles=float(record["SHIPMT_DIST_GC"]),
                source_weight_factor=float(record["WGT_FACTOR"]),
                source_temperature_controlled=str(record["TEMP_CNTL_YN"]),
                source_hazmat=str(record["HAZMAT"]),
                estimated_route_miles=round(estimated_route, 3),
                estimated_linehaul_periods=linehaul_periods,
                minimum_completion_periods=minimum_completion_periods,
            )
        )

    recommendations = {
        "city_1_cfs_area": city_a,
        "city_2_cfs_area": city_b,
        "representative_route_miles": round(representative_route_miles, 3),
        "representative_linehaul_hours": round(linehaul_hours, 3),
        "travel_calibration_method": calibration_method,
        "travel_calibration_record_count": (
            travel_calibration.get("eligible_record_count")
            if travel_calibration is not None
            else len(records)
        ),
        "travel_time_periods": linehaul_periods,
        "direct_travel_time_periods": linehaul_periods,
        "T": config.planning_periods,
        "t_0_minutes": config.period_hours * 60.0,
        "capacity_interpretation": (
            f"1 model quantity unit = {config.pounds_per_model_unit} pounds"
        ),
        "minimum_completion_periods": minimum_completion_periods,
    }
    return orders, recommendations


def write_outputs(
    output_dir: Path,
    input_path: Path,
    city_a: str,
    city_b: str,
    orders: Sequence[ProcessedOrder],
    config: ProcessorConfig,
    pair_stats: dict,
    sample_stats: dict,
    recommendations: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cfs_model_orders.csv"
    json_path = output_dir / "cfs_model_orders.json"
    metadata_path = output_dir / "cfs_processing_metadata.json"

    frame = pd.DataFrame([asdict(order) for order in orders])
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    order_payload = {
        "format_version": 1,
        "source": "2022 CFS Public Use Microdata Sample (PUMS)",
        "city_pair": {"city_1": city_a, "city_2": city_b},
        "model_recommendations": recommendations,
        "orders": [order.model_fields() for order in orders],
        "orders_with_source": [asdict(order) for order in orders],
    }
    json_path.write_text(
        json.dumps(order_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "source_file": str(input_path.resolve()),
        "source_dataset": "2022 CFS PUMS",
        "official_user_guide_issue": "January 2026",
        "processor_config": asdict(config),
        "city_pair": {"city_1": city_a, "city_2": city_b},
        "pair_statistics": pair_stats,
        "sampling_statistics": sample_stats,
        "model_recommendations": recommendations,
        "observed_fields": [
            "OD CFS areas",
            "mode",
            "SCTG commodity",
            "shipment value",
            "shipment weight",
            "great-circle distance",
            "temperature control",
            "HAZMAT",
            "WGT_FACTOR",
        ],
        "constructed_fields": [
            "flow",
            "quantity scaling and clipping",
            "estimated route miles",
            "linehaul periods",
            "earliest_start",
            "latest_completion",
            "penalty_lost",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "metadata": str(metadata_path),
    }


def load_processed_orders(path: str | Path):
    """读取脚本输出，返回项目求解器使用的 (pos, neg, all_orders)。"""

    from intercity_delivery.data.loader import OrderBatch

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    all_orders = {
        int(item["batch_id"]): OrderBatch(**item) for item in payload["orders"]
    }
    pos_orders = {
        order_id: order
        for order_id, order in all_orders.items()
        if order.flow == "+"
    }
    neg_orders = {
        order_id: order
        for order_id, order in all_orders.items()
        if order.flow == "-"
    }
    return pos_orders, neg_orders, all_orders


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 2022 CFS PUMS 转换为城际配送模型订单。"
    )
    parser.add_argument("--input", required=True, type=Path, help="官方 PUMS CSV/ZIP/GZ 或转换后的 SQLite。")
    parser.add_argument("--output-dir", type=Path, default=Path("data/cfs_processed"))
    parser.add_argument("--city-a", help="城市1的 CFS Area，如 06-348。")
    parser.add_argument("--city-b", help="城市2的 CFS Area，如 06-488。")
    parser.add_argument("--num-orders", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--modes", default="111,112", help="逗号分隔的 MODE 代码。")
    parser.add_argument("--min-distance-miles", type=float, default=50.0)
    parser.add_argument("--min-records-per-direction", type=int, default=10)
    parser.add_argument("--planning-periods", type=int, default=24)
    parser.add_argument("--period-hours", type=float, default=1.0)
    parser.add_argument("--truck-speed-mph", type=float, default=50.0)
    parser.add_argument("--circuity-factor", type=float, default=1.20)
    parser.add_argument("--local-service-periods-per-end", type=int, default=1)
    parser.add_argument("--buffer-min-periods", type=int, default=0)
    parser.add_argument("--buffer-max-periods", type=int, default=5)
    parser.add_argument("--pounds-per-model-unit", type=float, default=50.0)
    parser.add_argument("--min-quantity", type=int, default=10)
    parser.add_argument("--max-quantity", type=int, default=300)
    parser.add_argument("--penalty-lost", type=float, default=10.0)
    parser.add_argument(
        "--include-exports", action="store_true", help="不限制 EXPORT_YN=N。"
    )
    parser.add_argument(
        "--include-remainder-areas",
        action="store_true",
        help="允许州剩余区域和被抑制区域参与 OD 选择。",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = parse_args(argv)
    if bool(args.city_a) != bool(args.city_b):
        raise ValueError("--city-a 和 --city-b 必须同时提供或同时省略。")
    if not args.input.exists():
        raise FileNotFoundError(f"找不到输入文件：{args.input}")

    config = ProcessorConfig(
        modes=tuple(part.strip() for part in args.modes.split(",") if part.strip()),
        domestic_only=not args.include_exports,
        metro_only=not args.include_remainder_areas,
        min_distance_miles=args.min_distance_miles,
        num_orders=args.num_orders,
        seed=args.seed,
        chunksize=args.chunksize,
        planning_periods=args.planning_periods,
        period_hours=args.period_hours,
        truck_speed_mph=args.truck_speed_mph,
        circuity_factor=args.circuity_factor,
        local_service_periods_per_end=args.local_service_periods_per_end,
        buffer_min_periods=args.buffer_min_periods,
        buffer_max_periods=args.buffer_max_periods,
        pounds_per_model_unit=args.pounds_per_model_unit,
        min_quantity=args.min_quantity,
        max_quantity=args.max_quantity,
        penalty_lost=args.penalty_lost,
        min_records_per_direction=args.min_records_per_direction,
    )
    config.validate()

    if args.city_a:
        city_a = _normalize_area(args.city_a)
        city_b = _normalize_area(args.city_b)
        if city_a == city_b:
            raise ValueError("city-a 与 city-b 必须不同。")
        pair_stats = {"selection_rule": "explicit_user_selection"}
    else:
        city_a, city_b, pair_stats = select_city_pair(args.input, config)

    travel_calibration = calibrate_city_pair_travel_time(
        args.input, city_a, city_b, config
    )
    records, sample_stats = sample_pair_records(
        args.input, city_a, city_b, config
    )
    orders, recommendations = build_model_orders(
        records,
        city_a,
        city_b,
        config,
        travel_calibration=travel_calibration,
    )
    paths = write_outputs(
        args.output_dir,
        args.input,
        city_a,
        city_b,
        orders,
        config,
        pair_stats,
        sample_stats,
        recommendations,
    )
    print(f"CFS Area 对：{city_a} <-> {city_b}")
    print(f"生成订单：{len(orders)}")
    print(
        "推荐 DeliveryConfig："
        f"T={recommendations['T']}, t_0={recommendations['t_0_minutes']}, "
        f"travel_time_periods={recommendations['travel_time_periods']}"
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    return paths


if __name__ == "__main__":
    main()
