"""Pure independent-checker and mocked runner tests; no solver calls."""

from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from intercity_delivery.v1.configuration import V1Order
from intercity_delivery.v1.flexible_configuration import V1FConfig, flexible_profile
from intercity_delivery.v1.flexible_validation import GROUPS, validate_flexible_schedule
from intercity_delivery.v1.flexible_runner import run_flexible, run_flexible_rolling
from intercity_delivery.v1.flexible_rolling import RollingOptions


def config(**kw):
    values = dict(T=6, t0_minutes=60., tau_periods=1, N_manual={1: 1, 2: 1},
                  N_auto={1: 1, 2: 0}, a={1: 60., 2: 60.}, b={1: 0., 2: 0.},
                  rho={1: 0., 2: 0.}, M=10., M_auto=10., c_manual_hour=1.,
                  c_auto_hour=1., c_direct_hour=2.)
    values.update(kw)
    return V1FConfig(**values)


ORDER = V1Order("one", "+", 1., 0, 6, 100.)


def blank():
    return {n: {} for n in GROUPS}


def direct_schedule(flow="+", start=0, end=3, quantity=1.):
    s = blank()
    s["w_direct"][(start, end, flow)] = 1
    s["u_direct"][(start, end, flow, 1)] = 1
    s["h_direct"][(start, end, flow, "one")] = quantity
    return s


def transfer_schedule():
    s = blank()
    for i, j, city in ((0, 1, 1), (2, 3, 2)):
        s["x_manual"][(i, j, city, "+")] = 1
        s["u_wave"][(i, j, city, "+", 1)] = 1
        s["g_manual"][(i, j, city, "+", "one")] = 1.
    s["y_auto"][(1, 2, "+")] = 1
    s["g_auto"][(1, 2, "+", "one")] = 1.
    return s


class FlexibleCheckerTests(unittest.TestCase):
    def check(self, s, cfg=None, orders=(ORDER,), **kw):
        return validate_flexible_schedule(cfg or config(), orders, flexible_profile(), s, **kw)

    def test_direct_original_formula_and_hour_cost(self):
        v = self.check(direct_schedule(), reported_objective=6.)
        self.assertEqual(v["served_quantity"], 1)
        self.assertEqual(v["costs"]["direct"], 6)
        self.assertEqual(v["empty_trip_count"], 0)

    def test_complete_transfer_cost(self):
        v = self.check(transfer_schedule(), config(transfer_cost_per_unit=2))
        self.assertEqual(v["objective"], 5)
        self.assertEqual(v["transfer_quantity"], 1)

    def test_json_keys_round_trip(self):
        s = {n: {str(k): v for k, v in values.items()} for n, values in direct_schedule().items()}
        self.assertEqual(self.check(json.loads(json.dumps(s)))["objective"], 6)

    def test_empty_without_orders(self):
        s = blank()
        s["empty_vehicles"][(0, 1, "+")] = 1
        v = self.check(s, orders=())
        self.assertEqual(v["objective"], 2)
        self.assertEqual(v["served_quantity"], 0)

    def test_migrated_vehicle_selector_uses_total_fleet(self):
        s = blank()
        s["empty_vehicles"][(0, 1, "+")] = 2
        s["x_manual"][(1, 2, 2, "+")] = 2
        s["u_wave"][(1, 2, 2, "+", 2)] = 1
        self.check(s, config(N_manual={1: 2, 2: 0}), orders=())

    def test_partial_inflight_whole_task_cost(self):
        v = self.check(direct_schedule(), completed=False, completed_horizon=1)
        self.assertIsNone(v["objective"])
        self.assertEqual(v["costs"]["direct"], 6)
        self.assertEqual(v["served_quantity"], 0)
        self.assertEqual(v["costs"]["penalty"], 0)
        self.assertIn("future_completion_reachability", v["deferred_checks"])

    def test_partial_hub_stock_and_expired_penalty(self):
        s = blank()
        s["x_manual"][(0, 1, 1, "+")] = 1
        s["u_wave"][(0, 1, 1, "+", 1)] = 1
        s["g_manual"][(0, 1, 1, "+", "one")] = 1
        o = replace(ORDER, latest_completion=2)
        self.assertEqual(self.check(s, orders=(o,), completed=False,
                                    completed_horizon=2)["costs"]["penalty"], 100)
        with self.assertRaisesRegex(ValueError, "stage balance"):
            self.check(s, orders=(o,))

    def test_local_joint_load_not_per_order(self):
        s = transfer_schedule()
        s["g_manual"][(0, 1, 1, "+", "two")] = 1
        o = replace(ORDER, order_id="two")
        with self.assertRaisesRegex(ValueError, "local original f"):
            self.check(s, orders=(ORDER, o))

    def test_directional_origin_stem(self):
        cfg = config(rho={1: 60., 2: 0.})
        with self.assertRaisesRegex(ValueError, "direct original directional"):
            self.check(direct_schedule(), cfg)
        self.check(direct_schedule(flow="-"), cfg, orders=(replace(ORDER, flow="-"),))

    def test_zero_capacity_tau_arc(self):
        with self.assertRaisesRegex(ValueError, "zero-capacity direct"):
            self.check(direct_schedule(end=1))

    def test_shared_av_capacity(self):
        with self.assertRaisesRegex(ValueError, "shared AV"):
            self.check(transfer_schedule(), config(M_auto=.5))

    def test_manual_city_inventory(self):
        with self.assertRaisesRegex(ValueError, "manual city inventory"):
            self.check(direct_schedule(), config(N_manual={1: 0, 2: 2}))

    def test_av_city_inventory(self):
        with self.assertRaisesRegex(ValueError, "AV city inventory"):
            self.check(transfer_schedule(), config(N_auto={1: 0, 2: 1}))

    def test_empty_physical_duration(self):
        s = blank()
        s["empty_vehicles"][(0, 2, "+")] = 1
        with self.assertRaisesRegex(ValueError, "empty physical duration"):
            self.check(s)

    def test_direct_physical_capacity(self):
        s = direct_schedule(quantity=2)
        with self.assertRaisesRegex(ValueError, "physical cap"):
            self.check(s, config(M=1), orders=(replace(ORDER, quantity=2),))

    def test_cargo_causality(self):
        s = transfer_schedule()
        s["g_manual"].pop((0, 1, 1, "+", "one"))
        with self.assertRaisesRegex(ValueError, "pickup before AV"):
            self.check(s)

    def test_handling_delay_on_both_hubs(self):
        with self.assertRaisesRegex(ValueError, "pickup before AV"):
            self.check(transfer_schedule(), config(transfer_time_periods=1))
        s = transfer_schedule()
        s["x_manual"] = {(0, 1, 1, "+"): 1, (3, 4, 2, "+"): 1}
        s["u_wave"] = {(0, 1, 1, "+", 1): 1, (3, 4, 2, "+", 1): 1}
        s["g_manual"] = {(0, 1, 1, "+", "one"): 1, (3, 4, 2, "+", "one"): 1}
        s["y_auto"] = {(2, 3, "+"): 1}
        s["g_auto"] = {(2, 3, "+", "one"): 1}
        with self.assertRaisesRegex(ValueError, "AV before delivery"):
            self.check(s, config(transfer_time_periods=1))

    def test_time_window_release_and_deadline(self):
        for order in (replace(ORDER, earliest_start=1), replace(ORDER, latest_completion=2)):
            with self.subTest(order=order), self.assertRaises(ValueError):
                self.check(direct_schedule(), orders=(order,))

    def test_complete_ratio_and_partial_deferred(self):
        cfg = config(direct_ratio_max=0)
        with self.assertRaisesRegex(ValueError, "maximum direct ratio"):
            self.check(direct_schedule(), cfg)
        self.check(direct_schedule(), cfg, completed=False, completed_horizon=1)

    def test_explicit_terminal_target_and_no_default_reset(self):
        self.check(direct_schedule())
        with self.assertRaisesRegex(ValueError, "terminal fleet target"):
            self.check(direct_schedule(), config(terminal_target={1: 1, 2: 1}))

    def test_duplicate_normalized_key(self):
        s = direct_schedule()
        s["w_direct"]["(0, 3, '+')"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate normalized"):
            self.check(s)

    def test_numeric_domains_nan_inf_bool_noninteger(self):
        for bad in (float("nan"), float("inf"), True, .5, -2):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                s = direct_schedule()
                s["w_direct"][(0, 3, "+")] = bad
                self.check(s)

    def test_tolerance_validated_and_absolute(self):
        for tol in (0, -1, .1, float("nan"), True):
            with self.subTest(tol=tol), self.assertRaises(ValueError):
                self.check(direct_schedule(), tolerance=tol)
        with self.assertRaisesRegex(ValueError, "objective mismatch"):
            self.check(direct_schedule(), reported_objective=6.001)

    def test_unserved_balance(self):
        s = blank()
        s["z_unserved"][("one",)] = 1
        self.assertEqual(self.check(s)["objective"], 100)
        s["z_unserved"][("one",)] = 2
        with self.assertRaises(ValueError):
            self.check(s)

    def test_partial_uninitiated_or_unserved_forbidden(self):
        for s in (direct_schedule(start=1, end=4), dict(blank(), z_unserved={("one",): 1})):
            with self.assertRaises(ValueError):
                self.check(s, completed=False, completed_horizon=1)

    def test_partial_has_no_reported_full_objective(self):
        with self.assertRaises(ValueError):
            self.check(blank(), completed=False, completed_horizon=0, reported_objective=0)

    def test_overflow_in_original_f_is_not_feasibility(self):
        with self.assertRaises(ValueError):
            self.check(blank(), config(a={1: 1e308, 2: 60.}, M=1e308))


class FlexibleRunnerPureTests(unittest.TestCase):
    def test_full_mocked_optimal_replay_and_json(self):
        model = SimpleNamespace(Params=SimpleNamespace(), Status=2, SolCount=1,
                                ObjVal=6., ObjBound=6., MIPGap=0., Runtime=.01,
                                NumVars=10, NumConstrs=20, optimize=Mock(), dispose=Mock())
        optimizer = SimpleNamespace(model=model, build_model=Mock(),
                                    extract_schedule=lambda: direct_schedule())
        with patch("intercity_delivery.v1.flexible_optimizer.V1FlexibleOptimizer", return_value=optimizer):
            r = run_flexible(config(), (ORDER,), flexible_profile(), 2)
        self.assertTrue(r["optimal"])
        self.assertTrue(r["completed"])
        self.assertEqual(r["total_cost"], 6)
        self.assertEqual(r["metadata"]["selector_bound"], "global_manual_fleet")
        self.assertEqual(r["metadata"]["empty_cost_source"], "c_direct_hour fallback")
        self.assertEqual(len(r["metadata"]["provenance"]["input_sha256"]), 64)
        json.dumps(r, allow_nan=False)
        model.dispose.assert_called_once()

    def test_full_no_incumbent_never_fabricates_cost(self):
        model = SimpleNamespace(Params=SimpleNamespace(), Status=9, SolCount=0,
                                ObjBound=0., Runtime=.01, NumVars=10, NumConstrs=20,
                                optimize=Mock(), dispose=Mock())
        optimizer = SimpleNamespace(model=model, build_model=Mock())
        with patch("intercity_delivery.v1.flexible_optimizer.V1FlexibleOptimizer", return_value=optimizer):
            r = run_flexible(config(), (ORDER,), flexible_profile(), 2)
        self.assertFalse(r["completed"])
        self.assertFalse(r["schedule_valid"])
        self.assertIsNone(r["total_cost"])
        self.assertIsNone(r["schedule"])

    def test_rh_partial_inflight_no_global_bound(self):
        raw = dict(schedule=direct_schedule(), windows=[{"feasible": True, "status": 2,
                    "best_bound": 6}], current_time=1, completed=False,
                   status="PARTIAL_TIME_LIMIT", message="bounded")
        with patch("intercity_delivery.v1.flexible_rolling.solve_flexible_rolling", return_value=raw):
            r = run_flexible_rolling(config(), (ORDER,), RollingOptions(), 2)
        self.assertIsNone(r["total_cost"])
        self.assertIsNone(r["best_bound"])
        self.assertIsNone(r["mip_gap"])
        self.assertFalse(r["optimal"])
        self.assertEqual(r["validation"]["costs"]["direct"], 6)
        json.dumps(r, allow_nan=False)

    def test_rh_complete_reconstructs_z_without_global_optimality(self):
        raw = dict(schedule=direct_schedule(), windows=[{"feasible": True, "status": 2}],
                   current_time=6, completed=True, status="COMPLETED", message="done")
        with patch("intercity_delivery.v1.flexible_rolling.solve_flexible_rolling", return_value=raw):
            r = run_flexible_rolling(config(), (ORDER,), RollingOptions(), 2)
        self.assertEqual(r["total_cost"], 6)
        self.assertEqual(r["status"], "COMPLETED")
        self.assertFalse(r["optimal"])
        self.assertIsNone(r["best_bound"])
        self.assertEqual(r["schedule"]["z_unserved"]["('one',)"], 0)

    def test_mode_profiles_and_rolling_target_rejected(self):
        with self.assertRaises(ValueError):
            run_flexible(config(), (ORDER,), flexible_profile("release_revealed"), 2)
        with self.assertRaises(ValueError):
            run_flexible_rolling(config(), (ORDER,), RollingOptions(), 2,
                                 profile=flexible_profile())
        with self.assertRaises(ValueError):
            run_flexible_rolling(config(terminal_target={1: 1, 2: 1}),
                                 (ORDER,), RollingOptions(), 2)

    def test_invalid_rolling_options_fail_before_provenance_or_solver(self):
        for options in ({}, RollingOptions(prediction_horizon=0)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_flexible_rolling(config(), (ORDER,), options, 2)


if __name__ == "__main__":
    unittest.main()
