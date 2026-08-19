import unittest

from intercity_delivery.configuration import (
    DeliveryConfig,
    OrderGenerationConfig,
    RollingHorizonConfig,
)
from intercity_delivery.experiments.core import (
    build_delivery_data,
    generate_random_orders,
)
from intercity_delivery.experiments.solvers import SOLVER_REGISTRY


class RegressionFixTests(unittest.TestCase):
    def test_vehicle_rates_are_converted_from_minutes_to_hours(self):
        self.assertEqual(DeliveryConfig(t_0=30).period_hours, 0.5)
        self.assertEqual(DeliveryConfig(t_0=90).period_hours, 1.5)

    def test_heuristic_solution_is_used_as_reduced_mip_start(self):
        config = DeliveryConfig(T=16, penalty_lost=100)
        order_config = OrderGenerationConfig(
            num_orders=4, buffer_range=(4, 6)
        )
        orders = generate_random_orders(config, order_config, seed=7)
        data = build_delivery_data(config, orders)
        result = SOLVER_REGISTRY["paper_priority_heuristic"].solve(
            config,
            data,
            orders,
            30,
            RollingHorizonConfig(
                prediction_horizon=10,
                rolling_step=2,
                extension_horizon=6,
            ),
        )

        self.assertIsNotNone(result.total_cost)
        for window in result.detail["windows"]:
            diagnostics = window["diagnostics"]
            self.assertIn("heuristic_start_objective", diagnostics)
            if diagnostics["heuristic_start_objective"] is not None:
                self.assertLessEqual(
                    window["objective"],
                    diagnostics["heuristic_start_objective"] + 1e-6,
                )

    def test_full_window_baseline_handles_travel_longer_than_start_window(self):
        config = DeliveryConfig(
            T=24,
            travel_time_periods=9,
            direct_travel_time_periods=9,
            penalty_lost=1000,
        )
        order_config = OrderGenerationConfig(
            num_orders=4, buffer_range=(4, 5)
        )
        orders = generate_random_orders(config, order_config, seed=11)
        data = build_delivery_data(config, orders)
        result = SOLVER_REGISTRY["flexible_direct_rolling"].solve(
            config,
            data,
            orders,
            30,
            RollingHorizonConfig(
                prediction_horizon=8,
                rolling_step=2,
                extension_horizon=6,
            ),
        )

        self.assertIsNotNone(result.total_cost)
        self.assertLess(result.unserved_rate, 1.0)
        self.assertIsNotNone(result.direct_ratio)
        total_demand = sum(order.quantity for order in orders[2].values())
        committed = result.detail["solution"]["committed_decisions"]
        flow_values = [
            value
            for group in ("g_manual", "g_auto", "h_direct")
            for value in committed[group].values()
        ]
        self.assertTrue(flow_values)
        self.assertLessEqual(max(flow_values), total_demand)


if __name__ == "__main__":
    unittest.main()
