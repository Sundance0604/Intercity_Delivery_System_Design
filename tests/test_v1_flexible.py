"""Pure F inputs/capacities/history plus four PI-run <=2 s micro solves."""

from dataclasses import replace
import math
import time
from types import SimpleNamespace
import unittest

from intercity_delivery.v1.configuration import V1Order
from intercity_delivery.v1.flexible_configuration import (
    V1FConfig, empty_duration, empty_hourly_cost, flexible_profile, validate_f_instance)
from intercity_delivery.v1.flexible_capacity import direct_arcs, direct_capacity, empty_arcs
from intercity_delivery.v1.flexible_optimizer import V1FlexibleOptimizer


def config(**changes):
    cfg = V1FConfig(3, 1, 1, {1: 1, 2: 0}, {1: 0, 2: 0},
                   {1: 1, 2: 1}, {1: 0, 2: 0}, {1: 0, 2: 0},
                   1, 1, 60, 60, 60)
    return replace(cfg, **changes)


class FlexiblePureTests(unittest.TestCase):
    def test_profiles_and_empty_fallback(self):
        cfg = config()
        for information in ("offline", "release_revealed"):
            validate_f_instance(cfg, [], flexible_profile(information))
        self.assertEqual(empty_duration(cfg), 1)
        self.assertEqual(empty_hourly_cost(cfg), 60)
        self.assertEqual(empty_duration(replace(cfg, empty_tau_periods=2)), 2)

    def test_extension_scalar_domains(self):
        for field in ("c_direct_hour", "c_empty_hour", "transfer_cost_per_unit",
                      "direct_ratio_min", "direct_ratio_max"):
            for value in (True, -1, math.inf, math.nan):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_f_instance(config(**{field: value}), [], flexible_profile())
        for field in ("empty_tau_periods", "transfer_time_periods"):
            with self.assertRaises(ValueError):
                validate_f_instance(config(**{field: 0.5}), [], flexible_profile())
        with self.assertRaises(ValueError):
            validate_f_instance(config(direct_ratio_min=.8, direct_ratio_max=.2), [], flexible_profile())

    def test_terminal_target_conservation_and_city_keys(self):
        for target in ({1: 1, 2: 1}, {1: 0.5, 2: 0.5}, {True: 1, 2: 0}):
            with self.subTest(target=target), self.assertRaises(ValueError):
                validate_f_instance(config(terminal_target=target), [], flexible_profile())
        validate_f_instance(config(terminal_target={1: 0, 2: 1}), [], flexible_profile())

    def test_direction_and_origin_stem(self):
        cfg = config(T=8, rho={1: 1, 2: 0}, M=100,
                     a={1: 1, 2: 1}, b={1: 2, 2: 5})
        self.assertGreater(direct_capacity(cfg, 0, 8, "-", 2),
                           direct_capacity(cfg, 0, 8, "+", 2))
        q = direct_capacity(cfg, 0, 8, "+", 2)
        duration = 2+q/2+2*math.sqrt(q)/2+1+q/2+5*math.sqrt(q/2)
        self.assertAlmostEqual(duration, 8)
        self.assertEqual(direct_capacity(cfg, 0, 1, "+", 2), 0)
        self.assertEqual(direct_capacity(cfg, 0, 8, "+", 0), 0)

    def test_mother_arcs_do_not_need_orders(self):
        cfg = config(T=5, empty_tau_periods=2)
        self.assertIn((0, 1), direct_arcs(cfg))
        self.assertEqual(empty_arcs(cfg), ((0, 2), (1, 3), (2, 4), (3, 5)))
        model = V1FlexibleOptimizer(cfg, [], flexible_profile())
        self.assertIn((0, 1, 2, 1), model.capacities)
        self.assertEqual(model.selector_bound, "global_manual_fleet")

    def test_history_implicit_zeros_and_new_bounds_without_solver(self):
        cfg = config(T=5)
        opt = V1FlexibleOptimizer(cfg, [], flexible_profile())
        v = lambda: SimpleNamespace(LB=0, UB=math.inf)
        old, old_zero, future, downstream = v(), v(), v(), v()
        opt.model = SimpleNamespace(update=lambda: None)
        opt.groups = {"empty_vehicles": {(0, 1, "+"): old, (0, 1, "-"): old_zero,
                                         (2, 3, "+"): future},
                      "x_manual": {(2, 3, 2, "+"): downstream}, "z_unserved": {}}
        opt.configure_window(1, 2, 4, {"empty_vehicles": {(0, 1, "+"): 1}})
        self.assertEqual((old.LB, old.UB), (1, 1))
        self.assertEqual((old_zero.LB, old_zero.UB), (0, 0))
        self.assertEqual(future.UB, 0)
        self.assertEqual(downstream.UB, math.inf)

    def test_nonzero_unknown_history_and_rolling_target_rejected(self):
        opt = V1FlexibleOptimizer(config(), [], flexible_profile())
        opt.model = SimpleNamespace(update=lambda: None)
        opt.groups = {"empty_vehicles": {}, "z_unserved": {}}
        with self.assertRaises(ValueError):
            opt.configure_window(1, 2, 3, {"empty_vehicles": {(0, 1, "+"): 1}})
        opt.config = config(terminal_target={1: 1, 2: 0})
        with self.assertRaisesRegex(ValueError, "tail reachability"):
            opt.configure_window(0, 1, 3, {})


class FlexibleMicroSolveTests(unittest.TestCase):
    """Four real runs, each total construction+solve capped at two seconds."""

    def solve(self, cfg, orders, history=None):
        started = time.monotonic()
        opt = V1FlexibleOptimizer(cfg, orders, flexible_profile()).build_model()
        try:
            if history is not None:
                opt.configure_window(1, 2, cfg.T, history)
            remaining = 2-(time.monotonic()-started)
            self.assertGreater(remaining, 0)
            opt.model.Params.TimeLimit = remaining
            opt.model.optimize()
            self.assertGreater(opt.model.SolCount, 0)
            self.assertEqual(opt.model.Status, 2)
            return opt.extract_schedule(), opt.model.ObjVal
        finally:
            opt.model.dispose()

    def test_01_continuous_direct_split(self):
        schedule, objective = self.solve(config(), [V1Order("a", "+", 1.5, 0, 3, 100)])
        self.assertAlmostEqual(sum(schedule["h_direct"].values()), 1)
        self.assertAlmostEqual(schedule["z_unserved"]["a",], .5)
        self.assertAlmostEqual(objective, 53)

    def test_02_independent_empty_then_reverse_direct(self):
        cfg = config(T=4, c_empty_hour=30)
        schedule, objective = self.solve(cfg, [V1Order("a", "-", 1, 1, 4, 100)])
        self.assertAlmostEqual(schedule["empty_vehicles"][0, 1, "+"], 1)
        self.assertAlmostEqual(sum(schedule["h_direct"].values()), 1)
        self.assertAlmostEqual(objective, 3.5)

    def test_03_migrated_vehicle_can_do_local_city_initially_zero(self):
        cfg = config(T=4, N_auto={1: 0, 2: 1}, direct_ratio_max=0, c_direct_hour=10000,
                     c_empty_hour=60)
        schedule, objective = self.solve(cfg, [V1Order("a", "-", 1, 1, 4, 100)])
        self.assertAlmostEqual(schedule["g_manual"][1, 2, 2, "-", "a"], 1)
        self.assertAlmostEqual(schedule["z_unserved"]["a",], 0)
        self.assertAlmostEqual(objective, 5)

    def test_04_committed_empty_is_retained_with_no_orders(self):
        cfg = config(T=4)
        schedule, objective = self.solve(cfg, [], {"empty_vehicles": {(0, 1, "+"): 1}})
        self.assertAlmostEqual(schedule["empty_vehicles"][0, 1, "+"], 1)
        self.assertAlmostEqual(objective, 1)
        self.assertEqual(sum(v for key, v in schedule["w_direct"].items() if key[0] < 1), 0)


if __name__ == "__main__":
    unittest.main()
