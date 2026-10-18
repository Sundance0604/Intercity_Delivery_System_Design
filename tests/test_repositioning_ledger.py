"""Small deterministic ledger/accounting tests; no solver import or solve."""

from dataclasses import replace
import unittest

from intercity_delivery.configuration import DeliveryConfig
from intercity_delivery.data.loader import OrderBatch
from intercity_delivery.repositioning.config import RepositioningPolicy
from intercity_delivery.repositioning.ledger import (
    Task, state, validate_ledger, check_terminal_target, evaluate_committed,
)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = DeliveryConfig(T=8, t_0=30, direct_travel_time_periods=2,
                                  travel_time_periods=2, N_manual={1: 1, 2: 0},
                                  N_auto={1: 1, 2: 0}, cost_direct=10,
                                  cost_manual=6, transfer_cost_per_unit=3)
        self.policy = RepositioningPolicy().resolve(self.cfg)
        self.move = Task("a", "empty", 1, 2, 0, 2)
        self.order = OrderBatch(batch_id=1, flow="+", quantity=1.0, penalty_lost=100,
                                earliest_start=0, latest_completion=8)

    def evaluate(self, decisions, orders=None, cut=8, completed=True):
        return evaluate_committed(self.cfg, orders or {}, decisions,
                                  self.policy, cut, completed)

    def test_explicit_compatibility_defaults_disclosed(self):
        self.assertEqual(self.policy.empty_travel_time_periods, 2)
        self.assertEqual(self.policy.empty_cost_per_hour, 10)
        self.assertEqual(self.policy.metadata()["empty_time_source"], "cfg.direct_travel_time_periods")
        self.assertEqual(self.policy.resolve(self.cfg).metadata(), self.policy.metadata())
        self.assertIsNone(self.policy.terminal_target)

    def test_invalid_numeric_policy_rejected(self):
        for bad in (float("nan"), float("inf"), -1, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RepositioningPolicy(empty_cost_per_hour=bad).resolve(self.cfg)
        for bad in (0, -1, 2.0, True):
            with self.subTest(tau=bad), self.assertRaises(ValueError):
                RepositioningPolicy(empty_travel_time_periods=bad).resolve(self.cfg)

    def test_invalid_fleet_float_and_bool_rejected(self):
        for bad in (1.0, True, -1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_ledger({1: bad, 2: 0}, [], 2)
        with self.assertRaises(ValueError):
            validate_ledger({True: 1, 2: 0}, [], 2)

    def test_strict_task_schema(self):
        for update in ({"start": True}, {"vehicles": 1.0}, {"vehicles": False},
                       {"cargo": float("nan")}, {"cargo": float("inf")},
                       {"cargo": -1}, {"end": 2.0}, {"task_id": ""}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_ledger(self.cfg.N_manual, [replace(self.move, **update)], 2)

    def test_duplicate_task_id_rejected(self):
        with self.assertRaises(ValueError):
            validate_ledger(self.cfg.N_manual, [self.move, self.move], 2)

    def test_empty_no_orders_cost_once_and_no_service(self):
        result = self.evaluate({"empty_vehicles": {(0, 2, "+"): 1}})
        self.assertEqual(result["total_cost"], 10)
        self.assertEqual(result["cost_components"]["empty"], 10)
        self.assertEqual(result["empty_vehicle_trips"], 1)
        self.assertEqual(result["served_quantity"], 0)
        self.assertEqual(result["direct_share"], 0)
        self.assertEqual(result["cost_components"]["transfer"], 0)

    def test_small_continuous_direct_delivery_is_not_rounded(self):
        result = self.evaluate({"w_direct": {(0, 3, "+"): 1},
                                "h_direct": {(0, 3, "+", 1): 0.05}}, {1: self.order})
        self.assertAlmostEqual(result["direct_quantity"], 0.05)
        self.assertEqual(result["cost_components"]["direct"], 15)
        self.assertAlmostEqual(result["cost_components"]["unserved"], 95)
        self.assertEqual(result["cost_components"]["empty"], 0)

    def test_empty_cannot_carry_cargo(self):
        with self.assertRaises(ValueError):
            validate_ledger(self.cfg.N_manual, [replace(self.move, cargo=0.05)], 2)

    def test_empty_duration_and_local_city(self):
        for task in (replace(self.move, end=3), replace(self.move, end=0),
                     replace(self.move, kind="local")):
            with self.assertRaises(ValueError):
                validate_ledger(self.cfg.N_manual, [task], 2)

    def test_loaded_zero_cargo_is_honestly_preserved(self):
        task = replace(self.move, kind="loaded", cargo=0)
        self.assertTrue(validate_ledger(self.cfg.N_manual, [task], 2))
        result = self.evaluate({"w_direct": {(0, 3, "+"): 1}})
        self.assertEqual(result["direct_service_vehicle_trips"], 1)
        self.assertEqual(result["positive_cargo_direct_arc_count"], 0)
        self.assertEqual(result["served_quantity"], 0)

    def test_inflight_not_available_until_arrival(self):
        early = Task("local", "local", 2, 2, 1, 3)
        with self.assertRaises(ValueError):
            validate_ledger(self.cfg.N_manual, [self.move, early], 2)
        self.assertTrue(validate_ledger(self.cfg.N_manual,
                                        [self.move, replace(early, start=2)], 2))
        self.assertEqual(state(self.cfg.N_manual, [self.move], 1)["in_transit"], 1)

    def test_completed_history_persists_and_no_window_reset(self):
        later = Task("local", "local", 2, 2, 5, 6)
        self.assertTrue(validate_ledger(self.cfg.N_manual, [self.move, later], 2))
        self.assertEqual(state(self.cfg.N_manual, [self.move], 7)["idle"], {1: 0, 2: 1})
        with self.assertRaises(ValueError):
            validate_ledger(self.cfg.N_manual, [later], 2)
        with self.assertRaises(ValueError):
            validate_ledger({1: 0, 2: 1}, [self.move, later], 2)

    def test_real_terminal_counts_empty_and_rejects_future_task(self):
        back = Task("back", "empty", 2, 1, 3, 5)
        self.assertTrue(check_terminal_target(self.cfg.N_manual, [self.move, back], 5,
                                              self.cfg.N_manual, 2))
        self.assertFalse(check_terminal_target(self.cfg.N_manual, [self.move], 5,
                                               self.cfg.N_manual, 2))
        future = Task("future", "empty", 1, 2, 6, 8)
        self.assertFalse(check_terminal_target(self.cfg.N_manual, [future], 5,
                                               self.cfg.N_manual, 2))

    def test_partial_delivery_counts_completion_only_and_no_future_penalty(self):
        result = self.evaluate({"w_direct": {(0, 3, "+"): 1},
                                "h_direct": {(0, 3, "+", 1): 0.05}},
                               {1: self.order}, cut=1, completed=False)
        self.assertIsNone(result["total_cost"])
        self.assertEqual(result["committed_cost"], 15)
        self.assertEqual(result["served_quantity"], 0)
        self.assertEqual(result["cost_components"]["unserved"], 0)
        self.assertEqual(result["terminal_state"]["in_transit"], 1)

    def test_expired_partial_demand_is_penalized(self):
        self.order.latest_completion = 2
        result = self.evaluate({}, {1: self.order}, cut=2, completed=False)
        self.assertEqual(result["cost_components"]["unserved"], 100)
        self.assertEqual(result["penalty_scope"], "expired demand only")

    def test_destination_delivery_only_handling_accounting(self):
        # Evaluator explicitly does not establish cargo-network feasibility.
        result = self.evaluate({"g_manual": {(0, 1, 1, "+", 1): 0.4,
                                              (3, 4, 2, "+", 1): 0.05}}, {1: self.order})
        self.assertAlmostEqual(result["served_quantity"], 0.05)
        self.assertAlmostEqual(result["cost_components"]["transfer"], 0.15)
        self.assertIn("NOT validated", result["validation_scope"])

    def test_decision_negative_nan_noninteger_and_future_are_rejected(self):
        for bad in (-1, float("nan"), 0.5, True):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.evaluate({"empty_vehicles": {(0, 2, "+"): bad}})
        with self.assertRaises(ValueError):
            self.evaluate({"empty_vehicles": {(2, 4, "+"): 1}}, cut=1, completed=False)

    def test_terminal_policy_conserves_fleet_and_enforces_true_end(self):
        with self.assertRaises(ValueError):
            RepositioningPolicy(terminal_target={1: 2, 2: 0}).resolve(self.cfg)
        target = RepositioningPolicy(terminal_target={1: 1, 2: 0}).resolve(self.cfg)
        with self.assertRaises(ValueError):
            evaluate_committed(self.cfg, {}, {"empty_vehicles": {(0, 2, "+"): 1}}, target, 8, True)


if __name__ == "__main__":
    unittest.main()
