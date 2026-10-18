"""Opt-in V1 adapters and traceable, bounded CFS shipment-stop sampling.

CFS weights are retained as provenance. They are never silently interpreted as
BHH stops by the dedicated CFS loader: one source shipment is one stop unit.
This is an execution smoke assumption, not a calibrated demand conversion.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from .configuration import V1Order


V1_F_SOLVERS = ("v1_flexible_mip", "v1_flexible_rolling", "v1_flexible_pruned_rolling")


def adapt_delivery_config(config):
    """Explicit V0 container to V1 conversion; do not merge incompatible units."""
    from .flexible_configuration import V1FConfig

    if config.direct_travel_time_periods != config.travel_time_periods:
        raise ValueError("V1-F uses one physical tau: direct_travel_time_periods must equal travel_time_periods")
    if config.capacity_direct != config.capacity_manual:
        raise ValueError("V1-F uses one manual stop capacity M: capacity_direct must equal capacity_manual")
    return V1FConfig(
        T=config.T, t0_minutes=config.t_0, tau_periods=config.travel_time_periods,
        N_manual=dict(config.N_manual), N_auto=dict(config.N_auto),
        a={1: config.service_a_1, 2: config.service_a_2},
        b={1: config.service_b_1, 2: config.service_b_2},
        rho={1: config.v1_rho_1, 2: config.v1_rho_2},
        M=config.capacity_manual, M_auto=config.capacity_auto,
        c_manual_hour=config.cost_manual, c_auto_hour=config.cost_auto,
        c_direct_hour=config.cost_direct,
        transfer_time_periods=config.transfer_time_periods,
        transfer_cost_per_unit=config.transfer_cost_per_unit,
        direct_ratio_min=config.direct_ratio_min, direct_ratio_max=config.direct_ratio_max,
    )


def adapt_order_batches(orders_tuple):
    """Preserve caller-selected quantities; conversion belongs before data build."""
    return [V1Order(str(key), order.flow, order.quantity, order.earliest_start,
                    order.latest_completion, order.penalty_lost)
            for key, order in orders_tuple[2].items()]


def adapt_solver_payload(payload, input_metadata=None):
    """Project V1 result into the existing experiment result fields."""
    evidence = payload.get("validation") or {}
    volumes = evidence.get("volumes", evidence)
    direct = payload.get("direct_quantity", payload.get("direct_volume", volumes.get("direct_volume")))
    transfer = payload.get("transshipment_quantity", payload.get("transshipment_volume", volumes.get("transshipment_volume")))
    served = payload.get("served_quantity", payload.get("served_amount", volumes.get("served_amount")))
    rejected = payload.get("unserved_quantity", payload.get("unserved_amount", volumes.get("unserved_amount")))
    if served is None and direct is not None and transfer is not None:
        served = direct+transfer
    total = input_metadata.get("total_demand") if input_metadata else None
    detail = dict(payload)
    detail["adapter_input"] = input_metadata or {
        "demand_semantics": "caller_supplied_continuous_stops", "source_traceability": "unavailable"}
    return {
        "status": payload.get("status"), "solve_time_sec": payload.get("elapsed_sec", 0.0),
        "total_cost": payload.get("total_cost"), "best_bound": payload.get("best_bound"),
        "mip_gap": payload.get("mip_gap"),
        "unserved_rate": payload.get("unserved_rate", rejected/total if rejected is not None and total else None),
        "auto_usage": payload.get("auto_usage", 0.0), "manual_usage": payload.get("manual_usage", 0.0),
        "detail": detail, "direct_ratio": payload.get("direct_ratio", direct/served if direct is not None and served else 0.0),
        "direct_volume": direct, "transshipment_volume": transfer,
        "message": payload.get("message", payload.get("status_name", "V1-F run finished")),
    }


def load_cfs_stop_orders(path, *, num_orders=4, seed=42, city_pair=None,
                         T=24, t0_minutes=60.0, tau_periods=None,
                         buffer_range=(0, 5), penalty=1000.0):
    """Read existing processed JSON or one indexed SQLite OD pair, read-only.

    JSON retains the previously synthesized windows. SQLite reuses the old
    weighted reservoir sampler and seed+1 window generator. No raw CSV scan,
    database reconstruction, full-database hash or output file is performed.
    """
    if isinstance(num_orders, bool) or not isinstance(num_orders, int) or num_orders < 2:
        raise ValueError("num_orders must be an integer >= 2 for bidirectional sampling")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rng = random.Random(seed)
    sqlite = source.suffix.lower() in {".sqlite", ".sqlite3", ".db"}
    if sqlite:
        if city_pair is None or len(city_pair) != 2 or city_pair[0] == city_pair[1]:
            raise ValueError("SQLite requires an explicit distinct city_pair")
        from intercity_delivery.data.cfs_processor import (
            ProcessorConfig, build_model_orders, sample_pair_records,
        )
        processor = ProcessorConfig(
            num_orders=num_orders, seed=seed, planning_periods=T,
            period_hours=t0_minutes / 60.0, penalty_lost=penalty,
            buffer_min_periods=buffer_range[0], buffer_max_periods=buffer_range[1],
        )
        processor.validate()
        records, sample = sample_pair_records(source, *city_pair, processor)
        calibration = None
        if tau_periods is not None:
            calibration = {
                "travel_time_periods": tau_periods,
                "representative_route_miles": 0.0,
                "representative_linehaul_hours": tau_periods*t0_minutes/60.0,
                "method": "explicit_fixed_periods_for_smoke_not_recalibrated",
            }
        processed, recommendations = build_model_orders(
            records, *city_pair, processor, travel_calibration=calibration)
        chosen = [asdict(order) for order in processed]
        file_hash = None
        window_method = "old build_model_orders seed+1 synthetic windows"
    else:
        raw = source.read_bytes()
        payload = json.loads(raw)
        pool = payload.get("orders_with_source")
        if not isinstance(pool, list) or not pool:
            raise ValueError("CFS JSON requires nonempty orders_with_source; source IDs may not be fabricated")
        pair = payload.get("city_pair", {})
        detected_pair = (pair.get("city_1"), pair.get("city_2"))
        if not all(detected_pair) or detected_pair[0] == detected_pair[1]:
            raise ValueError("CFS JSON requires two distinct source city areas")
        if city_pair is not None and tuple(city_pair) != detected_pair:
            raise ValueError("requested city_pair disagrees with processed JSON")
        city_pair = detected_pair
        targets = {"+": (num_orders+1)//2, "-": num_orders//2}
        chosen = []
        for direction, count in targets.items():
            candidates = [row for row in pool if row.get("flow") == direction]
            if len(candidates) < count:
                raise ValueError(f"insufficient processed records for flow {direction}")
            chosen.extend(rng.sample(candidates, count))
        rng.shuffle(chosen)
        recommendations = payload.get("model_recommendations", {})
        if tau_periods is not None and recommendations.get("travel_time_periods") != tau_periods:
            raise ValueError("fixed travel duration disagrees with processed CFS calibration")
        sample = {"sampling_method": "seeded balanced sample from existing processed pool",
                  "pool_records": len(pool), "seed": seed}
        file_hash = hashlib.sha256(raw).hexdigest()
        window_method = "retained old synthesized integer windows from processed JSON"
    result = []
    seen = set()
    for row in chosen:
        shipment = str(row.get("source_shipmt_id", "")).strip()
        if not shipment or shipment in seen:
            raise ValueError("source SHIPMT_ID must be present and unique in selected sample")
        seen.add(shipment)
        start, end = row["earliest_start"], row["latest_completion"]
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, int) or not isinstance(end, int)
                or not 0 <= start <= end <= T):
            raise ValueError("source time window must satisfy integer 0 <= s <= e <= T")
        result.append(V1Order(shipment, row["flow"], 1.0, start, end, penalty))
    stat = source.stat()
    metadata = {
        "source_path": str(source), "source_kind": "sqlite" if sqlite else "processed_json",
        "source_dataset": "2022 CFS PUMS", "source_sha256": file_hash,
        "source_size_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
        "source_hash_scope": "entire processed JSON" if file_hash else "not computed for multi-GB SQLite",
        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "city_pair": {"city_1": city_pair[0], "city_2": city_pair[1]},
        "demand_semantics": "shipment_stop", "stop_units_per_source_shipment": 1.0,
        "calibration_status": "execution smoke assumption; no calibrated weight-to-stops conversion",
        "source_weights_used_as_demand": False, "time_window_method": window_method,
        "sampling": sample, "model_recommendations": recommendations,
        "source_records": chosen,
    }
    return result, metadata
