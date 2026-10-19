"""Author-confirmed job units for prospective numerical experiments.

The legacy smoke loader remains unchanged. This wrapper explicitly relabels
the same one-shipment demand mapping and preserves observed/synthetic metadata.
It does not download data, modify the cache, or invoke optimization.
"""

import hashlib
import json
from pathlib import Path

from .adapter import load_cfs_stop_orders


CONTRACT_PATH = Path(__file__).resolve().parents[2] / "experiments/specs/experiment_data_contract.json"


def load_cfs_job_orders(source, *, contract_path=CONTRACT_PATH, **options):
    raw_contract = Path(contract_path).read_bytes()
    contract = json.loads(raw_contract)
    if (contract.get("schema_version") != "v1_experiment_data_contract/1"
            or contract.get("unit_status") != "AUTHOR_CONFIRMED"
            or contract.get("model_unit") != "synthetic_job_equivalent"
            or isinstance(contract.get("quantity_per_shipment"), bool)
            or contract.get("quantity_per_shipment") != 1
            or contract.get("continuous_divisible") is not True
            or contract.get("source_weight_used_as_demand") is not False):
        raise ValueError("unsupported or unconfirmed experiment demand contract")
    options.setdefault("penalty", contract["synthetic_parameter_policy"]["base_penalty_per_job"])
    orders, legacy = load_cfs_stop_orders(source, **options)
    if any(order.quantity != 1 for order in orders):
        raise ValueError("each source shipment must map to exactly one job-equivalent")
    records = []
    for record in legacy["source_records"]:
        row = dict(record)
        row.update({
            "source_weight_lb": record["source_weight_pounds"],
            "model_quantity": 1.0,
            "model_unit": contract["model_unit"],
            "conversion_rule": contract["conversion_rule"],
            "seed": legacy["seed"],
            "model_quantity_is_synthetic": True,
            "source_weight_is_observed": True,
            "time_windows_are_synthetic": True,
        })
        records.append(row)
    metadata = dict(legacy)
    metadata.update({
        "legacy_loader_semantics": legacy["demand_semantics"],
        "demand_semantics": contract["model_unit"],
        "model_unit": contract["model_unit"],
        "unit_decision_id": contract["decision_id"],
        "data_contract_sha256": hashlib.sha256(raw_contract).hexdigest(),
        "calibration_status": "author-confirmed synthetic job mapping; no measured-stop calibration",
        "source_records": records,
        "parameter_status": "synthetic design; operational effects not yet tested",
    })
    recommendations = dict(legacy.get("model_recommendations", {}))
    metadata["legacy_model_recommendations"] = recommendations.copy()
    recommendations["capacity_interpretation"] = "M and M_auto are synthetic job-equivalents per vehicle"
    metadata["model_recommendations"] = recommendations
    return orders, metadata
