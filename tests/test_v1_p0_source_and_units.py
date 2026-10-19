"""Reject source drift and silent reintroduction of weight-based model units."""

import hashlib
import json
from pathlib import Path
from copy import deepcopy
import unittest
from unittest.mock import patch

from experiments.specs.freeze_v1_p0_sources import check_manifest, content_fingerprint
from experiments.specs.v1_p0_preflight import config_from_design
from intercity_delivery.v1.configuration import V1Order
from intercity_delivery.v1.experiment_inputs import CONTRACT_PATH, load_cfs_job_orders


class SourceFreezeTests(unittest.TestCase):
    def manifest(self, content=b"exact source"):
        entries = {"paper.tex": {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}}
        return {"schema_version": "v1_p0_source_manifest/1", "files": entries,
                "content_sha256": content_fingerprint(entries)}

    def test_content_change_is_detected(self):
        with patch("experiments.specs.freeze_v1_p0_sources.source_paths", return_value=["paper.tex"]), \
                patch.object(Path, "read_bytes", return_value=b"exact source"):
            self.assertEqual(check_manifest(self.manifest()), [])
        with patch("experiments.specs.freeze_v1_p0_sources.source_paths", return_value=["paper.tex"]), \
                patch.object(Path, "read_bytes", return_value=b"changed source"):
            self.assertTrue(any("bytes differ" in error for error in check_manifest(self.manifest())))

    def test_new_source_is_not_silently_ignored(self):
        with patch("experiments.specs.freeze_v1_p0_sources.source_paths", return_value=["paper.tex", "new.tex"]), \
                patch.object(Path, "read_bytes", return_value=b"exact source"):
            self.assertIn("source inventory differs (added or missing files)", check_manifest(self.manifest()))

    def test_manifest_fingerprint_is_checked(self):
        manifest = self.manifest()
        manifest["content_sha256"] = "incorrect"
        with patch("experiments.specs.freeze_v1_p0_sources.source_paths", return_value=["paper.tex"]), \
                patch.object(Path, "read_bytes", return_value=b"exact source"):
            self.assertIn("manifest content fingerprint differs", check_manifest(manifest))


class JobUnitsTests(unittest.TestCase):
    def test_job_mapping_retains_weight_without_using_it_as_demand(self):
        legacy = {"seed": 12, "demand_semantics": "shipment_stop",
                  "source_records": [{"source_shipmt_id": "CFS1", "source_weight_pounds": 20000,
                                      "quantity": 300}],
                  "model_recommendations": {"capacity_interpretation": "50 pounds per unit", "travel_time_periods": 9}}
        order = V1Order("CFS1", "+", 1, 0, 24, 100)
        with patch("intercity_delivery.v1.experiment_inputs.load_cfs_stop_orders", return_value=([order], legacy)) as loader:
            orders, evidence = load_cfs_job_orders("unused.sqlite", num_orders=2)
        self.assertEqual(loader.call_args.kwargs["penalty"], 100)
        self.assertEqual(orders[0].quantity, 1)
        self.assertEqual(evidence["model_unit"], "synthetic_job_equivalent")
        self.assertEqual(evidence["source_records"][0]["source_weight_lb"], 20000)
        self.assertTrue(evidence["source_records"][0]["time_windows_are_synthetic"])
        self.assertEqual(legacy["source_records"][0]["quantity"], 300)
        self.assertNotIn("pounds", evidence["model_recommendations"]["capacity_interpretation"])

    def test_unconfirmed_or_weight_mapping_rejected_before_loading_data(self):
        base = json.loads(CONTRACT_PATH.read_bytes())
        for field, value in (("unit_status", "PENDING"), ("model_unit", "pounds"),
                             ("quantity_per_shipment", True), ("source_weight_used_as_demand", True)):
            contract = dict(base, **{field: value})
            with self.subTest(field=field), patch.object(Path, "read_bytes", return_value=json.dumps(contract).encode()), \
                    patch("intercity_delivery.v1.experiment_inputs.load_cfs_stop_orders") as loader:
                with self.assertRaisesRegex(ValueError, "contract"):
                    load_cfs_job_orders("unused.sqlite")
                loader.assert_not_called()


class PreflightPolicyTests(unittest.TestCase):
    def decisions(self):
        return {
            "schema_version": "v1_p0_decisions/1",
            "unit": {"status": "AUTHOR_CONFIRMED", "choice": "one_shipment_one_synthetic_job_equivalent"},
            "direct_endpoint": {"status": "AUTHOR_CONFIRMED", "choice": "aggregate_destination_service_node_original_formula"},
            "t2_policy": {"status": "AUTHOR_CONFIRMED", "choice": "minimum_average_direct_load", "minimum_per_vehicle": .1},
        }

    def design(self):
        return json.loads((CONTRACT_PATH.parent / "v1_p0_capacity_design.json").read_bytes())

    def test_confirmation_of_different_choice_cannot_run_old_formula(self):
        for name in ("unit", "direct_endpoint", "t2_policy"):
            decisions = self.decisions()
            decisions[name]["choice"] = "a_different_policy"
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "different implementation"):
                config_from_design(self.design(), decisions)

    def test_pending_and_malformed_minimum_cannot_bypass_gate(self):
        for value in (None, 0, True, float("nan"), float("inf"), -.1):
            decisions = self.decisions()
            decisions["t2_policy"]["minimum_per_vehicle"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                config_from_design(self.design(), decisions)
        decisions = self.decisions()
        decisions["direct_endpoint"]["status"] = "PENDING_AUTHOR_REPLY"
        with self.assertRaisesRegex(ValueError, "not all confirmed"):
            config_from_design(self.design(), decisions)

    def test_all_mechanisms_from_supported_decisions_preserve_baseline(self):
        design = self.design()
        original = deepcopy(design)
        for mechanism in ("T0", "T1", "T2", "T3"):
            cfg = config_from_design(design, self.decisions(), mechanism)
            self.assertEqual(cfg.service_mechanism, mechanism)
            self.assertEqual(cfg.min_direct_load_per_vehicle, .1 if mechanism == "T2" else 0)
            self.assertEqual(cfg.rho, {1: 5, 2: 5})
        self.assertEqual(design, original)


if __name__ == "__main__":
    unittest.main()
