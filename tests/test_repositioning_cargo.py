"""Independent cargo adversarial cases and one <=2 s registry rolling run."""

import ast
import copy
import json
from types import SimpleNamespace
import unittest

from intercity_delivery.repositioning.cargo_validation import validate_cargo


def cargo_fixture(quantity=0.05, extra_orders=0):
    cfg = SimpleNamespace(T=5, t_0=60.0, travel_time_periods=1,
        direct_travel_time_periods=1, transfer_time_periods=0,
        service_a_1=60.0, service_a_2=60.0, service_b_1=0.0,
        service_b_2=0.0, capacity_auto=1.0, capacity_direct=1.0,
        direct_ratio_min=0.0, direct_ratio_max=1.0)
    orders = {oid: SimpleNamespace(flow="+", quantity=quantity,
        earliest_start=0, latest_completion=3)
        for oid in range(1, 2 + extra_orders)}
    manual = [(i, j) for i in range(cfg.T) for j in range(i + 1, cfg.T + 1)]
    data = SimpleNamespace(all_orders=orders, arcs_manual_1=manual,
        arcs_manual_2=manual, arcs_auto=[(i, i + 1) for i in range(cfg.T)])
    decisions = {"x_manual": {(0, 1, 1, "+"): 1, (2, 3, 2, "+"): 1},
        "y_auto": {(1, 2, "+"): 1},
        "g_manual": {(0, 1, 1, "+", 1): quantity, (2, 3, 2, "+", 1): quantity},
        "g_auto": {(1, 2, "+", 1): quantity}}
    return cfg, data, decisions


class CargoValidationTests(unittest.TestCase):
    def test_complete_chain_and_summary_are_json_safe(self):
        cfg, data, decisions = cargo_fixture()
        result = validate_cargo(cfg, data, decisions, completed=True,
            order_summaries={"r_transshipment": {"1": 0.05},
                "q_direct": {"1": 0}, "z_unserved": {"1": 0}})
        self.assertAlmostEqual(result["served_quantity"], 0.05)
        self.assertEqual(result["orders"]["1"]["state"]["origin_hub"], 0)
        self.assertTrue(result["order_summaries_checked"])
        json.dumps(result)

    def test_partial_pickup_is_reserved_and_not_delivered_or_balanced(self):
        cfg, data, _ = cargo_fixture()
        result = validate_cargo(cfg, data, {"x_manual": {(0, 1, 1, "+"): 1},
            "g_manual": {(0, 1, 1, "+", 1): 0.05}}, 1)
        self.assertEqual(result["served_quantity"], 0)
        self.assertAlmostEqual(result["orders"]["1"]["state"]["origin_hub"], 0.05)
        self.assertIn("final_stage_balance", result["deferred"])

    def test_partial_inflight_pickup_preserves_future_completion(self):
        cfg, data, _ = cargo_fixture()
        result = validate_cargo(cfg, data, {"x_manual": {(0, 2, 1, "+"): 1},
            "g_manual": {(0, 2, 1, "+", 1): 0.05}}, 1)
        self.assertAlmostEqual(result["orders"]["1"]["state"]["pickup_in_progress"], 0.05)
        self.assertEqual(result["orders"]["1"]["state"]["origin_hub"], 0)
        self.assertEqual(result["orders"]["1"]["committed_events"][0]["end"], 2)

    def test_partial_av_in_transit_is_not_destination_stock(self):
        cfg, data, decisions = cargo_fixture()
        cfg.travel_time_periods = 2
        data.arcs_auto = [(i, i + 2) for i in range(4)]
        decisions["g_auto"] = {(1, 3, "+", 1): 0.05}
        decisions["y_auto"] = {(1, 3, "+"): 1}
        decisions["g_manual"].pop((2, 3, 2, "+", 1))
        decisions["x_manual"].pop((2, 3, 2, "+"))
        result = validate_cargo(cfg, data, decisions, 2)
        self.assertEqual(result["served_quantity"], 0)
        self.assertAlmostEqual(result["orders"]["1"]["state"]["av_in_transit"], 0.05)
        self.assertEqual(result["orders"]["1"]["state"]["destination_hub"], 0)

    def test_smaller_positive_cargo_preserved(self):
        cfg, data, decisions = cargo_fixture(0.00001)
        self.assertAlmostEqual(validate_cargo(cfg, data, decisions, completed=True)["served_quantity"], 0.00001)

    def test_direct_only_complete_schedule(self):
        cfg, data, _ = cargo_fixture()
        result = validate_cargo(cfg, data, {"w_direct": {(0, 2, "+"): 1},
            "h_direct": {(0, 2, "+", 1): 0.05}}, completed=True)
        self.assertAlmostEqual(result["direct_quantity"], 0.05)

    def test_unstarted_future_commitment_rejected(self):
        cfg, data, decisions = cargo_fixture()
        with self.assertRaisesRegex(ValueError, "uninitiated"):
            validate_cargo(cfg, data, decisions, 1)

    def test_linehaul_cannot_depart_before_pickup_completion(self):
        cfg, data, decisions = cargo_fixture()
        decisions["g_auto"] = {(0, 1, "+", 1): 0.05}
        decisions["y_auto"] = {(0, 1, "+"): 1}
        with self.assertRaisesRegex(ValueError, "origin cargo causality"):
            validate_cargo(cfg, data, decisions, completed=True)

    def test_destination_cannot_depart_before_av_arrival(self):
        cfg, data, decisions = cargo_fixture()
        decisions["g_manual"][(1, 2, 2, "+", 1)] = decisions["g_manual"].pop((2, 3, 2, "+", 1))
        decisions["x_manual"] = {(0, 1, 1, "+"): 1, (1, 2, 2, "+"): 1}
        with self.assertRaisesRegex(ValueError, "destination cargo causality"):
            validate_cargo(cfg, data, decisions, completed=True)

    def test_order_identity_prevents_borrowing_other_order_stock(self):
        cfg, data, decisions = cargo_fixture(extra_orders=1)
        decisions["g_auto"] = {(1, 2, "+", 2): 0.05}
        decisions["g_manual"][(2, 3, 2, "+", 2)] = decisions["g_manual"].pop((2, 3, 2, "+", 1))
        with self.assertRaisesRegex(ValueError, "causality"):
            validate_cargo(cfg, data, decisions, completed_horizon=3)

    def test_transfer_processing_delay_enforced(self):
        cfg, data, decisions = cargo_fixture()
        cfg.transfer_time_periods = 1
        with self.assertRaisesRegex(ValueError, "origin cargo causality"):
            validate_cargo(cfg, data, decisions, completed=True)

    def test_delayed_transfer_chain_valid(self):
        cfg, data, decisions = cargo_fixture()
        cfg.transfer_time_periods = 1
        data.all_orders[1].latest_completion = 5
        decisions["g_auto"] = {(2, 3, "+", 1): 0.05}
        decisions["y_auto"] = {(2, 3, "+"): 1}
        decisions["g_manual"][(4, 5, 2, "+", 1)] = decisions["g_manual"].pop((2, 3, 2, "+", 1))
        decisions["x_manual"] = {(0, 1, 1, "+"): 1, (4, 5, 2, "+"): 1}
        self.assertTrue(validate_cargo(cfg, data, decisions, completed=True)["valid"])

    def test_destination_processing_delay_enforced(self):
        cfg, data, decisions = cargo_fixture()
        cfg.transfer_time_periods = 1
        data.all_orders[1].latest_completion = 4
        decisions["g_auto"] = {(2, 3, "+", 1): 0.05}
        decisions["y_auto"] = {(2, 3, "+"): 1}
        decisions["g_manual"][(3, 4, 2, "+", 1)] = decisions["g_manual"].pop((2, 3, 2, "+", 1))
        decisions["x_manual"] = {(0, 1, 1, "+"): 1, (3, 4, 2, "+"): 1}
        with self.assertRaisesRegex(ValueError, "destination cargo causality"):
            validate_cargo(cfg, data, decisions, completed=True)

    def test_av_capacity_is_shared_across_orders(self):
        cfg, data, decisions = cargo_fixture(0.6, 1)
        for key, value in list(decisions["g_manual"].items()):
            decisions["g_manual"][(*key[:-1], 2)] = value
        decisions["g_auto"][(1, 2, "+", 2)] = 0.6
        decisions["x_manual"] = {key: 2 for key in decisions["x_manual"]}
        with self.assertRaisesRegex(ValueError, "AV shared capacity"):
            validate_cargo(cfg, data, decisions, completed=True)

    def test_manual_capacity_is_shared_across_orders(self):
        cfg, data, decisions = cargo_fixture(0.6, 1)
        for key, value in list(decisions["g_manual"].items()):
            decisions["g_manual"][(*key[:-1], 2)] = value
        decisions["g_auto"][(1, 2, "+", 2)] = 0.6
        decisions["y_auto"][(1, 2, "+")] = 2
        with self.assertRaisesRegex(ValueError, "manual shared capacity"):
            validate_cargo(cfg, data, decisions, completed=True)

    def test_direct_shared_capacity(self):
        cfg, data, _ = cargo_fixture(0.6, 1)
        with self.assertRaisesRegex(ValueError, "direct shared capacity"):
            validate_cargo(cfg, data, {"w_direct": {(0, 3, "+"): 1},
                "h_direct": {(0, 3, "+", 1): 0.6, (0, 3, "+", 2): 0.6}}, completed=True)

    def test_late_delivery_rejected(self):
        cfg, data, decisions = cargo_fixture()
        data.all_orders[1].latest_completion = 2
        with self.assertRaisesRegex(ValueError, "deadline"):
            validate_cargo(cfg, data, decisions, completed=True)

    def test_pickup_before_release_rejected(self):
        cfg, data, decisions = cargo_fixture()
        data.all_orders[1].earliest_start = 1
        with self.assertRaisesRegex(ValueError, "release"):
            validate_cargo(cfg, data, decisions, completed=True)

    def test_duplicate_demand_direct_plus_pickup_rejected(self):
        cfg, data, decisions = cargo_fixture()
        decisions.update({"w_direct": {(0, 2, "+"): 1}, "h_direct": {(0, 2, "+", 1): 0.05}})
        with self.assertRaisesRegex(ValueError, "reserved demand"):
            validate_cargo(cfg, data, decisions, completed=True)

    def test_full_leftover_hub_inventory_rejected(self):
        cfg, data, _ = cargo_fixture()
        with self.assertRaisesRegex(ValueError, "transshipment imbalance"):
            validate_cargo(cfg, data, {"x_manual": {(0, 1, 1, "+"): 1},
                "g_manual": {(0, 1, 1, "+", 1): 0.05}}, completed=True)

    def test_tampered_full_summary_rejected(self):
        cfg, data, decisions = cargo_fixture()
        with self.assertRaisesRegex(ValueError, "r_transshipment"):
            validate_cargo(cfg, data, decisions, completed=True,
                order_summaries={"r_transshipment": {1: 0}, "q_direct": {1: 0}, "z_unserved": {1: 0}})

    def test_direct_ratio_checked_only_at_full_completion(self):
        cfg, data, decisions = cargo_fixture()
        cfg.direct_ratio_min = 0.5
        with self.assertRaisesRegex(ValueError, "direct ratio"):
            validate_cargo(cfg, data, decisions, completed=True)
        result = validate_cargo(cfg, data, {}, 0)
        self.assertIn("aggregate_direct_ratio", result["deferred"])

    def test_bad_scalar_direction_order_and_vehicle_inputs_rejected(self):
        cfg, data, decisions = cargo_fixture()
        for key, value in [((1, 2, "+", 1), float("nan")),
                           ((1, 2, "-", 1), 0.05),
                           ((1, 2, "+", 99), 0.05),
                           ((1, 2, "+", 1), -1)]:
            bad = copy.deepcopy(decisions)
            bad["g_auto"] = {key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_cargo(cfg, data, bad, completed=True)
        decisions["y_auto"][(1, 2, "+")] = 0.5
        with self.assertRaisesRegex(ValueError, "nonintegral"):
            validate_cargo(cfg, data, decisions, completed=True)

    def test_local_capacity_does_not_add_v1_manual_M_truncation(self):
        cfg, data, decisions = cargo_fixture(1.5)
        cfg.capacity_manual = 1  # V0-F does not truncate its inverse at this M.
        cfg.service_a_1 = cfg.service_a_2 = 30.0
        cfg.capacity_auto = 2
        self.assertTrue(validate_cargo(cfg, data, decisions, completed=True)["valid"])

    def test_serialized_registry_keys_supported(self):
        cfg, data, decisions = cargo_fixture()
        serialized = {name: {str(key): value for key, value in values.items()}
                      for name, values in decisions.items()}
        self.assertTrue(validate_cargo(cfg, data, serialized, completed=True)["valid"])

    def test_nonfinite_invalid_config_never_certifies_capacity(self):
        for name, value in [("capacity_auto", float("nan")),
                            ("capacity_direct", float("inf")),
                            ("service_a_1", float("nan")),
                            ("service_a_2", 0), ("service_b_1", -1),
                            ("service_b_2", float("inf")), ("t_0", float("nan")),
                            ("capacity_auto", 0), ("direct_travel_time_periods", 1.5),
                            ("travel_time_periods", True), ("T", False),
                            ("direct_ratio_min", -0.1), ("direct_ratio_max", 1.1),
                            ("direct_ratio_min", float("nan"))]:
            cfg, data, decisions = cargo_fixture(2)
            cfg.service_a_1 = cfg.service_a_2 = 20
            setattr(cfg, name, value)
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_cargo(cfg, data, decisions, completed=True)

    def test_validation_control_scalars_are_strict(self):
        cfg, data, decisions = cargo_fixture()
        for kwargs in ({"tolerance": True}, {"tolerance": "tiny"},
                       {"tolerance": float("nan")}, {"completed": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_cargo(cfg, data, decisions, **kwargs)

    def test_nonfinite_or_unordered_order_windows_rejected(self):
        for name, value in [("earliest_start", float("nan")),
                            ("latest_completion", float("nan")),
                            ("latest_completion", float("inf")),
                            ("earliest_start", True), ("earliest_start", 4),
                            ("quantity", float("nan"))]:
            cfg, data, decisions = cargo_fixture()
            setattr(data.all_orders[1], name, value)
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_cargo(cfg, data, decisions, completed=True)

    def test_finite_fractional_order_window_keeps_v0_threshold_semantics(self):
        cfg, data, decisions = cargo_fixture()
        data.all_orders[1].earliest_start = -0.5
        data.all_orders[1].latest_completion = 3.5
        self.assertTrue(validate_cargo(cfg, data, decisions, completed=True)["valid"])

    def test_inverse_overflow_rejected_instead_of_nonfinite_capacity(self):
        cfg, data, decisions = cargo_fixture()
        cfg.service_a_1 = 1e308
        with self.assertRaisesRegex(ValueError, "overflow"):
            validate_cargo(cfg, data, decisions, completed=True)
        cfg.service_a_1 = 1e-308
        cfg.t_0 = 1e154
        with self.assertRaisesRegex(ValueError, "overflow"):
            validate_cargo(cfg, data, decisions, completed=True)


class CargoRollingMicroTests(unittest.TestCase):
    def test_registry_pickup_av_delivery_across_three_control_windows(self):
        from intercity_delivery.configuration import RollingHorizonConfig
        from intercity_delivery.data.loader import OrderBatch
        from intercity_delivery.experiments.solvers import SOLVER_REGISTRY
        from tests.test_repositioning_optimizer import tiny_instance

        cfg, data = tiny_instance([OrderBatch(1, "+", 0.05, 0, 3, 1000.0)],
            N_manual={1: 1, 2: 1}, N_auto={1: 1, 2: 0}, cost_manual=2.0,
            cost_auto=3.0, transfer_cost_per_unit=4.0, direct_ratio_max=0.0,
            service_a_1=60.0, service_a_2=60.0, capacity_auto=1.0)
        result = SOLVER_REGISTRY["flexible_direct_empty_rolling"].solve(
            cfg, data, (data.pos_orders, data.neg_orders, data.all_orders),
            time_limit=2, algorithm_config=RollingHorizonConfig(
                prediction_horizon=1, rolling_step=1, extension_horizon=2))
        detail = result.detail
        self.assertTrue(detail["algorithm"]["completed"])
        self.assertAlmostEqual(result.total_cost, 7.2)
        self.assertAlmostEqual(result.direct_volume, 0.0)
        self.assertAlmostEqual(result.transshipment_volume, 0.05)
        self.assertIsNone(result.best_bound)
        self.assertTrue(detail["cargo_validation"]["valid"])
        self.assertAlmostEqual(detail["cargo_validation"]["served_quantity"], 0.05)
        committed = detail["solution"]["committed_decisions"]
        report = validate_cargo(cfg, data, committed, cfg.T, True)
        self.assertTrue(report["valid"])
        self.assertEqual({ast.literal_eval(k)[:2] for k in committed["g_manual"]}, {(0, 1), (2, 3)})
        self.assertEqual({ast.literal_eval(k)[:2] for k in committed["g_auto"]}, {(1, 2)})
        states = report["orders"]["1"]["events"]
        self.assertAlmostEqual(states[1]["av_in_transit"], 0.05)
        self.assertAlmostEqual(states[2]["delivery_in_progress"], 0.05)
        self.assertAlmostEqual(states[3]["delivered"], 0.05)


if __name__ == "__main__":
    unittest.main()
