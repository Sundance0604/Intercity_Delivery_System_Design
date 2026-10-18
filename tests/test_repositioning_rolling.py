"""Controller boundaries, with one real rolling run capped at 2 s total."""

import ast
import unittest
from unittest.mock import patch, Mock

from intercity_delivery.configuration import RollingHorizonConfig
from intercity_delivery.data.loader import OrderBatch
from intercity_delivery.repositioning.config import RepositioningPolicy
from intercity_delivery.repositioning.rolling import (
    commit_control_window, solve_rolling, visible_data,
)
from intercity_delivery.repositioning.runner import run_rolling
from tests.test_repositioning_optimizer import tiny_instance


class RepositioningRollingTests(unittest.TestCase):
    def test_registry_rolling_empty_migration_then_cross_window_delivery(self):
        from intercity_delivery.experiments.solvers import SOLVER_REGISTRY

        cfg, data = tiny_instance([OrderBatch(1, "-", 0.05, 1, 5, 1000.0)])
        algorithm = RollingHorizonConfig(
            prediction_horizon=2, rolling_step=1, extension_horizon=2
        )
        result = SOLVER_REGISTRY["flexible_direct_empty_rolling"].solve(
            cfg, data, (data.pos_orders, data.neg_orders, data.all_orders),
            time_limit=2, algorithm_config=algorithm,
        )
        detail = result.detail
        self.assertTrue(detail["algorithm"]["completed"])
        self.assertEqual(detail["algorithm"]["completed_horizon"], 5)
        self.assertEqual(detail["windows"][0]["known_order_ids"], [])
        self.assertEqual(detail["windows"][1]["known_order_ids"], [1])
        self.assertAlmostEqual(result.direct_volume, 0.05)
        self.assertAlmostEqual(result.total_cost, 3.0)
        self.assertAlmostEqual(result.unserved_rate, 0.0)
        self.assertIsNone(result.best_bound)
        self.assertIsNone(result.mip_gap)
        self.assertAlmostEqual(detail["execution_accounting"]["empty_vehicle_trips"], 1.0)
        committed = detail["solution"]["committed_decisions"]
        empty = [(ast.literal_eval(key), value)
                 for key, value in committed["empty_vehicles"].items()]
        loaded = [(ast.literal_eval(key), value)
                  for key, value in committed["w_direct"].items()]
        self.assertEqual(len(empty), 1)
        self.assertEqual(len(loaded), 1)
        empty_key, empty_value = empty[0]
        loaded_key, loaded_value = loaded[0]
        self.assertEqual(empty_key[2], "+")
        self.assertEqual(loaded_key[2], "-")
        self.assertGreaterEqual(empty_key[0], 1)  # Never anticipate the hidden order.
        self.assertGreaterEqual(loaded_key[0], empty_key[1])
        self.assertLessEqual(loaded_key[1], cfg.T)
        self.assertEqual(empty_value, 1.0)
        self.assertEqual(loaded_value, 1.0)
        # Departure times can tie in objective; no unapproved early-dispatch
        # tie-break is assumed. Both [1,2)+[2,4) and later feasible plans cost 3.

    def test_budget_stop_preserves_first_window_commitments(self):
        cfg, data = tiny_instance()
        fake = Mock()
        fake.model.Status = 2
        fake.model.SolCount = 1
        fake.model.ObjVal = 999.0
        fake.model.ObjBound = 0.0
        fake.model.MIPGap = 0.0
        fake.build_model.return_value = fake
        fake.extract_decisions.return_value = {"empty_vehicles": {(0, 1, "+"): 1.0}}
        with patch("intercity_delivery.repositioning.rolling.RepositioningOptimizer") as cls:
            cls.return_value = fake
            cls.vehicle_groups = frozenset(("x_manual", "y_auto", "w_direct", "empty_vehicles"))
            with patch("intercity_delivery.repositioning.rolling.time.monotonic",
                       side_effect=[0.0, 0.0, 0.0, 3.0, 3.0]):
                raw = solve_rolling(cfg, data, RollingHorizonConfig(
                    prediction_horizon=2, rolling_step=1, extension_horizon=1), 2)
        self.assertFalse(raw["completed"])
        self.assertEqual(raw["current_time"], 1)
        self.assertEqual(raw["decisions"]["empty_vehicles"], {(0, 1, "+"): 1.0})
        self.assertEqual(fake.model.optimize.call_count, 1)

    def test_commit_empty_is_integer_and_small_cargo_survives(self):
        result = commit_control_window({
            "empty_vehicles": {(0, 2, "+"): 0.99999999, (2, 4, "-"): 1},
            "h_direct": {(0, 3, "+", 1): 0.05},
        }, 0, 1)
        self.assertEqual(result["empty_vehicles"], {(0, 2, "+"): 1.0})
        self.assertEqual(result["h_direct"], {(0, 3, "+", 1): 0.05})
        with self.assertRaises(ValueError):
            commit_control_window({"empty_vehicles": {(0, 2, "+"): 0.5}}, 0, 1)

    def test_revealed_expired_orders_remain_but_future_orders_are_hidden(self):
        cfg, data = tiny_instance([
            OrderBatch(1, "+", 1, 0, 1, 10),
            OrderBatch(2, "-", 1, 4, 5, 10),
        ])
        known = visible_data(cfg, data, 2)
        self.assertEqual(set(known.all_orders), {1})
        self.assertEqual(set(known.pos_orders), {1})
        self.assertFalse(known.neg_orders)

    def test_rolling_real_target_fails_explicitly_before_build(self):
        cfg, data = tiny_instance()
        with self.assertRaisesRegex(ValueError, "reachability"):
            solve_rolling(cfg, data, RollingHorizonConfig(), 2,
                          RepositioningPolicy(terminal_target={1: 1, 2: 0}))

    def test_partial_report_has_no_complete_total_or_global_bound(self):
        cfg, data = tiny_instance()
        raw = {
            "decisions": {"empty_vehicles": {(0, 1, "+"): 1.0}},
            "windows": [{"objective": 999.0}], "current_time": 1,
            "completed": False, "status": 2, "solve_time_sec": 0.1,
            "policy": RepositioningPolicy().resolve(cfg), "message": "partial",
        }
        with patch("intercity_delivery.repositioning.runner.solve_rolling", return_value=raw):
            result = run_rolling(cfg, data, RollingHorizonConfig(), 2)
        self.assertIsNone(result["total_cost"])
        self.assertIsNone(result["best_bound"])
        self.assertEqual(result["status"], 9)
        self.assertEqual(result["detail"]["algorithm"]["last_window_status"], 2)
        self.assertFalse(result["completed"])
        self.assertEqual(result["empty_vehicle_trips"], 1)
        self.assertNotEqual(result["detail"]["execution_accounting"]["total_cost"], 999)


if __name__ == "__main__":
    unittest.main()
