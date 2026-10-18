"""V1 opt-in interfaces and CFS stop semantics; no optimization calls."""

from pathlib import Path
from contextlib import contextmanager
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from intercity_delivery.v1.adapter import load_cfs_stop_orders, adapt_delivery_config, V1_F_SOLVERS


class CFSStopAdapterTests(unittest.TestCase):
    @contextmanager
    def fixture(self, **overrides):
        payload = {
            "city_pair": {"city_1": "06-348", "city_2": "06-488"},
            "model_recommendations": {"travel_time_periods": 2},
            "orders_with_source": [
                {"batch_id": i, "flow": "+" if i % 2 else "-", "quantity": 300,
                 "source_shipmt_id": f"source-{i}", "source_weight_pounds": 15000,
                 "source_weight_factor": 21711.9, "earliest_start": 0,
                 "latest_completion": 6} for i in range(1, 9)
            ],
        }
        payload.update(overrides)
        raw = json.dumps(payload).encode("utf-8")
        path = Path("cfs_fixture.json")
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "read_bytes", return_value=raw), \
                patch.object(Path, "stat", return_value=SimpleNamespace(st_size=len(raw), st_mtime_ns=1)):
            yield path

    def test_fixed_seed_balanced_stop_sample_preserves_source(self):
        with self.fixture() as path:
            before = path.read_bytes()
            orders, evidence = load_cfs_stop_orders(path, T=6, num_orders=4, seed=42)
            again, _ = load_cfs_stop_orders(path, T=6, num_orders=4, seed=42)
            self.assertEqual(orders, again)
            self.assertEqual(sum(o.flow == "+" for o in orders), 2)
            self.assertEqual({o.quantity for o in orders}, {1.0})
            self.assertTrue(all(o.order_id.startswith("source-") for o in orders))
            self.assertEqual(evidence["source_records"][0]["quantity"], 300)
            self.assertEqual(evidence["source_records"][0]["source_weight_pounds"], 15000)
            self.assertEqual(evidence["demand_semantics"], "shipment_stop")
            self.assertFalse(evidence["source_weights_used_as_demand"])
            self.assertEqual(path.read_bytes(), before)

    def test_missing_source_records_rejected(self):
        with self.fixture(orders_with_source=[]) as path:
            with self.assertRaises(ValueError):
                load_cfs_stop_orders(path)

    def test_missing_source_id_rejected(self):
        rows = [{"flow": flow, "earliest_start": 0, "latest_completion": 2}
                for flow in ("+", "-")]
        with self.fixture(orders_with_source=rows) as path:
            with self.assertRaises(ValueError):
                load_cfs_stop_orders(path, num_orders=2)

    def test_metadata_conflict_or_short_horizon_rejected(self):
        with self.fixture() as path:
            for kwargs in ({"tau_periods": 3}, {"T": 5},
                           {"city_pair": ("wrong", "pair")}, {"num_orders": True}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    load_cfs_stop_orders(path, **kwargs)


class V1EntryTests(unittest.TestCase):
    def test_smoke_evidence_keeps_nonzero_precision_and_group_names(self):
        from experiments.specs.run_v1f_cfs_smoke import compact_result
        original = {"schedule": {"x_manual": {"zero": 0.0, "tiny": 1e-12, "one": 1.0},
                                  "g_auto": {}},
                    "windows": [{"status": 2, "pruning": {"removed": 100},
                                 "schedule": {"duplicate": 1}}],
                    "metadata": {"policy": "preserve"}}
        compact = compact_result(original)
        self.assertEqual(set(compact["schedule"]), {"x_manual", "g_auto"})
        self.assertEqual(compact["schedule"]["x_manual"], {"tiny": 1e-12, "one": 1.0})
        self.assertEqual(original["schedule"]["x_manual"]["zero"], 0)
        self.assertEqual(compact["metadata"], original["metadata"])
        self.assertNotIn("schedule", compact["windows"][0])
        self.assertEqual(compact["windows"][0]["pruning"], {"removed": 100})

    def test_registry_and_cli_opt_in_preserve_default(self):
        import main
        from intercity_delivery.experiments.solvers import SOLVER_REGISTRY
        for name in V1_F_SOLVERS:
            self.assertIn(name, SOLVER_REGISTRY)
            with patch("sys.argv", ["main.py", "--solver", name]):
                self.assertEqual(main.parse_args().solver, name)
        with patch("sys.argv", ["main.py"]):
            self.assertEqual(main.parse_args().solver, "exact_mip")

    def test_conversion_exposes_stem_and_rejects_inconsistent_physics(self):
        from dataclasses import replace
        from intercity_delivery.configuration import DeliveryConfig
        source = DeliveryConfig(v1_rho_1=5, v1_rho_2=8)
        mapped = adapt_delivery_config(source)
        self.assertEqual(mapped.rho, {1: 5, 2: 8})
        self.assertEqual(mapped.M, source.capacity_manual)
        for bad in (replace(source, capacity_direct=3),
                    replace(source, direct_travel_time_periods=3)):
            with self.assertRaises(ValueError):
                adapt_delivery_config(bad)

    def test_core_maps_each_solver_in_isolation(self):
        from intercity_delivery.experiments.core import _solver_orders_and_metadata
        from intercity_delivery.data.loader import OrderBatch
        row = OrderBatch(1, "+", 300, 0, 6, 1000)
        original = ({1: row}, {}, {1: row})
        mapped, evidence = _solver_orders_and_metadata("v1_flexible_mip", original, {"source_kind": "processed_json"})
        self.assertEqual(mapped[2][1].quantity, 1)
        self.assertEqual(evidence["total_demand"], 1)
        old, evidence_old = _solver_orders_and_metadata("flexible_direct_mip", original, {})
        self.assertIs(old, original)
        self.assertIsNone(evidence_old)
        self.assertEqual(original[2][1].quantity, 300)

    def test_generated_v1_is_explicit_continuous_stop_data(self):
        from intercity_delivery.experiments.core import _solver_orders_and_metadata
        from intercity_delivery.data.loader import OrderBatch
        row = OrderBatch(1, "+", 3.5, 0, 6, 1000)
        original = ({1: row}, {}, {1: row})
        mapped, evidence = _solver_orders_and_metadata("v1_flexible_rolling", original)
        self.assertIs(mapped, original)
        self.assertEqual(evidence["total_demand"], 3.5)
        self.assertEqual(evidence["demand_semantics"], "synthetic_continuous_stops")

    def test_cfs_json_tau_adopted_only_for_v1_with_explicit_override_preserved(self):
        from intercity_delivery.experiments.core import _solver_model_config
        from intercity_delivery.configuration import DeliveryConfig
        config = DeliveryConfig()
        source = {"source_kind": "processed_json", "model_recommendations": {
            "travel_time_periods": 9, "t_0_minutes": 60,
            "representative_linehaul_hours": 8.496}}
        mapped, evidence = _solver_model_config("v1_flexible_mip", config, source)
        self.assertEqual(mapped.travel_time_periods, 9)
        self.assertEqual(mapped.direct_travel_time_periods, 9)
        self.assertEqual(config.travel_time_periods, 4)
        old, _ = _solver_model_config("flexible_direct_mip", config, source)
        self.assertIs(old, config)
        explicit, evidence = _solver_model_config("v1_flexible_mip", config, source,
                                                  "model.travel_time_periods")
        self.assertEqual(explicit.travel_time_periods, 4)
        self.assertIn("explicit", evidence["policy"])

    def test_cfs_json_bad_travel_metadata_is_not_silently_accepted(self):
        from intercity_delivery.experiments.core import _solver_model_config
        from intercity_delivery.configuration import DeliveryConfig
        for tau in (True, -1, 0, 9.5, float("nan")):
            source = {"source_kind": "processed_json", "model_recommendations": {"travel_time_periods": tau}}
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                _solver_model_config("v1_flexible_mip", DeliveryConfig(), source)


if __name__ == "__main__":
    unittest.main()
