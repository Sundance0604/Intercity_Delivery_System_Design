"""Read-only gate for the P0 model/data freeze; no Gurobi import or solve.

Source bytes, author decisions, units, policy configurations, and the capacity
mechanism are checked. A pass is permission to prepare a budgeted pilot, not
permission to run one, a license check, or an operational feasibility result.
"""

from dataclasses import fields
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.specs.freeze_v1_p0_sources import MANIFEST, check_manifest
from intercity_delivery.v1.capacity import local_capacity
from intercity_delivery.v1.flexible_configuration import V1FConfig, flexible_profile, validate_f_instance


def config_from_design(design, decisions, mechanism="T3"):
    if mechanism not in ("T0", "T1", "T2", "T3"):
        raise ValueError("unknown service mechanism")
    if decisions.get("schema_version") != "v1_p0_decisions/1":
        raise ValueError("unsupported author decision schema")
    if any(decisions[name]["status"] != "AUTHOR_CONFIRMED"
           for name in ("unit", "direct_endpoint", "t2_policy")):
        raise ValueError("author decisions are not all confirmed")
    supported = {
        "unit": "one_shipment_one_synthetic_job_equivalent",
        "direct_endpoint": "aggregate_destination_service_node_original_formula",
        "t2_policy": "minimum_average_direct_load",
    }
    for name, choice in supported.items():
        if decisions[name].get("choice") != choice:
            raise ValueError(f"author choice requires a different implementation: {name}")
    minimum = decisions["t2_policy"].get("minimum_per_vehicle")
    from intercity_delivery.v1.capacity import _real
    _real("confirmed T2 minimum_per_vehicle", minimum, positive=True)
    baseline = design["baseline"]
    values = {field.name: baseline[field.name] for field in fields(V1FConfig)
              if field.name in baseline}
    values["rho"] = {int(k): v for k, v in baseline["rho_minutes"].items()}
    for key in ("N_manual", "N_auto", "a", "b"):
        values[key] = {int(k): v for k, v in values[key].items()}
    values["service_mechanism"] = mechanism
    values["min_direct_load_per_vehicle"] = (
        minimum if mechanism == "T2" else 0)
    config = V1FConfig(**values)
    validate_f_instance(config, [], flexible_profile())
    return config


def check_p0(root=ROOT):
    errors = []
    manifest = json.loads((root / MANIFEST.relative_to(ROOT)).read_text(encoding="utf-8"))
    errors.extend(check_manifest(manifest, root))
    directory = root / "experiments/specs"
    decisions = json.loads((directory / "v1_p0_decisions.json").read_text(encoding="utf-8"))
    contract = json.loads((directory / "experiment_data_contract.json").read_text(encoding="utf-8"))
    design = json.loads((directory / "v1_p0_capacity_design.json").read_text(encoding="utf-8"))
    for name in ("unit", "direct_endpoint", "t2_policy"):
        if decisions[name]["status"] != "AUTHOR_CONFIRMED":
            errors.append(f"pending author decision: {name}")
    if (contract["model_unit"] != design["unit_contract"]["quantity"]
            or contract["decision_id"] != decisions["unit"]["decision_id"]
            or contract["source_weight_used_as_demand"] is not False):
        errors.append("data unit contract differs from parameter design or author decision")
    mechanism_configs = {}
    for mechanism in ("T0", "T1", "T2", "T3"):
        try:
            config = config_from_design(design, decisions, mechanism)
            mechanism_configs[mechanism] = {"min_direct_load_per_vehicle": config.min_direct_load_per_vehicle,
                                            "terminal_target": config.terminal_target}
        except ValueError as exc:
            errors.append(f"{mechanism}: {exc}")
    probes = {}
    for scenario in design["scenarios"]:
        pooled = local_capacity(scenario["duration_minutes"], 2, scenario["a"],
                                scenario["b"], scenario["rho_minutes"], scenario["M"])
        separate = 2 * local_capacity(scenario["duration_minutes"], 1, scenario["a"],
                                      scenario["b"], scenario["rho_minutes"], scenario["M"])
        probes[scenario["id"]] = {"two_vehicle_pooled": pooled, "two_vehicle_separate": separate}
        if scenario["id"] in ("routing_binding", "near_transition") and pooled <= separate + 1e-9:
            errors.append(f"pooling not activated: {scenario['id']}")
        if scenario["id"] == "load_binding" and abs(pooled - separate) > 1e-9:
            errors.append("negative control is not load-saturated")
    return {"p0_checks_passed": not errors, "errors": errors,
            "frozen_source_files": len(manifest["files"]),
            "mechanism_configs": mechanism_configs, "capacity_probes": probes,
            "evidence_scope": "source/units/configuration/pure capacity checks only",
            "license_status": "NOT_CHECKED", "new_solver_runs": 0,
            "solver_budget": decisions.get("solver_budget", "NOT_AUTHORIZED")}


def main():
    try:
        result = check_p0()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"p0_checks_passed": False, "errors": [str(exc)], "new_solver_runs": 0}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return not result["p0_checks_passed"]


if __name__ == "__main__":
    raise SystemExit(main())
