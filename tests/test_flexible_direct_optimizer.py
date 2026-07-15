"""直送/换装共存模型的求解级冒烟测试。

运行：
    python -m unittest tests.test_flexible_direct_optimizer

需要可用的 gurobipy 和 Gurobi license。
"""

import unittest

from intercity_delivery.configuration import DeliveryConfig, OrderGenerationConfig, RollingHorizonConfig
from intercity_delivery.experiments.core import build_delivery_data, generate_random_orders
from intercity_delivery.experiments.solvers import SOLVER_REGISTRY


class FlexibleDirectOptimizerTests(unittest.TestCase):
    def _instance(self, **overrides):
        config = DeliveryConfig(
            T=16,
            penalty_lost=1000,
            **overrides,
        )
        order_config = OrderGenerationConfig(
            num_orders=4,
            buffer_range=(4, 6),
        )
        orders = generate_random_orders(config, order_config, seed=7)
        data = build_delivery_data(config, orders)
        return config, data, orders

    def test_low_direct_cost_produces_direct_delivery(self):
        config, data, orders = self._instance(cost_direct=1.0)
        result = SOLVER_REGISTRY["flexible_direct_mip"].solve(
            config,
            data,
            orders,
            30,
            RollingHorizonConfig(prediction_horizon=10, rolling_step=2),
        )

        self.assertEqual(result.unserved_rate, 0)
        self.assertGreater(result.direct_volume, 0)

    def test_fixed_direct_ratio_supported_by_exact_and_rolling(self):
        config, data, orders = self._instance(
            cost_direct=1.0,
            direct_ratio_min=0.5,
            direct_ratio_max=0.5,
        )
        algorithm_config = RollingHorizonConfig(
            prediction_horizon=10,
            rolling_step=2,
        )

        exact = SOLVER_REGISTRY["flexible_direct_mip"].solve(
            config, data, orders, 30, algorithm_config
        )
        rolling = SOLVER_REGISTRY["flexible_direct_rolling"].solve(
            config, data, orders, 30, algorithm_config
        )

        self.assertAlmostEqual(
            exact.direct_ratio, 0.5, places=6
        )
        self.assertAlmostEqual(
            rolling.direct_ratio, 0.5, places=6
        )


if __name__ == "__main__":
    unittest.main()
