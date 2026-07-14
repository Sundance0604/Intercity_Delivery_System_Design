import json
import tempfile
import unittest
from pathlib import Path

from config import DeliveryConfig, OrderGenerationConfig, RollingHorizonConfig
from experiment_core import build_delivery_data, generate_random_orders, load_real_orders
from solvers import SOLVER_REGISTRY


class PaperSolutionApproachTests(unittest.TestCase):
    def setUp(self):
        self.config = DeliveryConfig(
            T=12,
            travel_time_periods=2,
            direct_travel_time_periods=2,
            N_manual={1: 4, 2: 4},
            N_auto={1: 2, 2: 2},
        )
        self.order_config = OrderGenerationConfig(num_orders=4, buffer_range=(2, 3))
        self.algorithm_config = RollingHorizonConfig(
            prediction_horizon=4, rolling_step=2, extension_horizon=4
        )
        self.orders = generate_random_orders(self.config, self.order_config, seed=7)
        self.data = build_delivery_data(self.config, self.orders)

    def test_both_paper_approaches_run_through_registry(self):
        for name in ("paper_candidate_mip", "paper_priority_heuristic"):
            with self.subTest(name=name):
                result = SOLVER_REGISTRY[name].solve(
                    self.config, self.data, self.orders, 20, self.algorithm_config
                )
                self.assertIsNotNone(result.total_cost)
                self.assertTrue(result.detail["algorithm"]["completed"])
                self.assertEqual(result.detail["algorithm"]["approach"], name)
                self.assertGreater(len(result.detail["windows"]), 1)

    def test_real_order_loader_samples_and_relabels(self):
        payload = {"orders": [
            {"batch_id": i, "flow": "+" if i % 2 else "-", "quantity": 10 + i,
             "earliest_start": 0, "latest_completion": 8, "penalty_lost": 1.0}
            for i in range(1, 7)
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cfs_model_orders.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            first = load_real_orders(path, self.config, self.order_config, seed=3)
            second = load_real_orders(path, self.config, self.order_config, seed=3)

        self.assertEqual(first, second)
        self.assertEqual(list(first[2]), [1, 2, 3, 4])
        self.assertTrue(all(
            order.penalty_lost == self.config.penalty_lost
            for order in first[2].values()
        ))


if __name__ == "__main__":
    unittest.main()
