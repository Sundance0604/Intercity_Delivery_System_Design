"""Independent original-work checks for the explicit local-separate model."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from experiments.fixed_n import pooling as p
from intercity_delivery.v1.configuration import V1Order
from intercity_delivery.v1.flexible_configuration import V1FConfig, flexible_profile
from intercity_delivery.v1.flexible_optimizer import V1FlexibleOptimizer
from intercity_delivery.v1.flexible_validation import GROUPS, validate_flexible_schedule


def config():
    return V1FConfig(T=4, t0_minutes=60., tau_periods=1,
        N_manual={1: 2, 2: 2}, N_auto={1: 1, 2: 0},
        a={1: 5., 2: 5.}, b={1: 20., 2: 20.}, rho={1: 5., 2: 5.},
        M=4., M_auto=8., c_manual_hour=20., c_auto_hour=15., c_direct_hour=25.)


def transfer(q):
    schedule = {name: {} for name in GROUPS}
    for i, j, city in ((0, 1, 1), (2, 3, 2)):
        schedule["x_manual"][i, j, city, "+"] = 2
        schedule["u_wave"][i, j, city, "+", 2] = 1
        schedule["g_manual"][i, j, city, "+", "o"] = q
    schedule["y_auto"][1, 2, "+"] = 1
    schedule["g_auto"][1, 2, "+", "o"] = q
    return schedule


class PoolingTests(unittest.TestCase):
    def test_capacity_inclusion_and_single_vehicle(self):
        for b in (0, 10, 20, 40):
            for duration in (0, 10, 60, 120):
                self.assertAlmostEqual(p.separate_capacity(duration, 1, 5, b, 5, 4),
                                       p.local_capacity(duration, 1, 5, b, 5, 4))
                for v in (0, 2, 3, 5):
                    self.assertLessEqual(p.separate_capacity(duration, v, 5, b, 5, 4),
                                         p.local_capacity(duration, v, 5, b, 5, 4)+1e-12)

    def test_strict_baseline_and_saturated_control(self):
        curves = p.capacity_curves()
        for row in curves:
            if row["b"] == 10 or row["vehicles"] == 1:
                self.assertAlmostEqual(row["pooled"], row["separate"])
            else:
                self.assertGreater(row["pooled"], row["separate"])

    def test_bad_domains_and_zero_budget(self):
        for v in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                p.separate_capacity(60, v, 5, 20, 5, 4)
        self.assertEqual(p.separate_capacity(10, 2, 5, 20, 5, 4), 0)

    def test_only_instance_local_table_changes(self):
        optimizer = V1FlexibleOptimizer(config(), (), flexible_profile())
        original = dict(optimizer.capacities)
        direct = dict(optimizer.direct_capacities)
        manual_arcs = optimizer.arcs_manual
        p.configure_separate(optimizer)
        self.assertEqual(optimizer.direct_capacities, direct)
        self.assertEqual(optimizer.arcs_manual, manual_arcs)
        self.assertEqual(set(optimizer.capacities), set(original))
        self.assertTrue(any(optimizer.capacities[k] < v for k, v in original.items()))

    def test_pooled_feasible_is_not_sufficient(self):
        cfg, schedule = config(), transfer(7)
        orders = (V1Order("o", "+", 7., 0, 4, 100.),)
        validate_flexible_schedule(cfg, orders, flexible_profile(), schedule)
        with self.assertRaisesRegex(ValueError, "separate_original_f_budget"):
            p.validate_separate(cfg, orders, {"schedule": schedule, "completed": True})

    def test_independent_validator_never_uses_kernel_table(self):
        cfg, schedule = config(), transfer(6)
        orders = (V1Order("o", "+", 6., 0, 4, 100.),)
        with patch.object(p, "separate_capacity", side_effect=AssertionError("K-table used")):
            evidence = p.validate_separate(cfg, orders, {"schedule": schedule, "completed": True})
        self.assertEqual(evidence["validation_model"], p.MODEL_ID)
        self.assertEqual(evidence["served_quantity"], 6.)

    def test_partial_and_json_keys(self):
        cfg, schedule = config(), transfer(6)
        for name in GROUPS:
            schedule[name] = {str(key): val for key, val in schedule[name].items() if key[0] < 1}
        orders = (V1Order("o", "+", 6., 0, 4, 100.),)
        evidence = p.validate_separate(cfg, orders,
            {"schedule": schedule, "completed": False, "current_time": 1})
        self.assertIsNone(evidence["objective"])
        self.assertEqual(evidence["local_separate"]["checked_local_waves"], 1)

    def test_configuration_after_build_rejected(self):
        optimizer = V1FlexibleOptimizer(config(), (), flexible_profile())
        optimizer.model = object()
        with self.assertRaisesRegex(ValueError, "before build"):
            p.configure_separate(optimizer)


if __name__ == "__main__":
    unittest.main()
