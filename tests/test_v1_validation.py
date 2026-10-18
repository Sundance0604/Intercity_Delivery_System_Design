"""Hand-written schedules, independent of Gurobi and capacity lookups."""

import copy
import json
import math
import unittest
from dataclasses import replace

from intercity_delivery.v1.configuration import V1MConfig, V1MProfile, V1Order
from intercity_delivery.v1.validation import validate_schedule


def fixture():
    profile = V1MProfile("offline", "arc", "continuous_stops_zero_allowed",
                        "minutes_hourly", "tour_end_same_time",
                        "integer_windows_within_horizon")
    cfg = V1MConfig(3, 30, 1, {1: 1, 2: 1}, {1: 1, 2: 0},
                    {1: 30, 2: 30}, {1: 0, 2: 0}, {1: 0, 2: 0},
                    1, 1, 10, 4)
    orders = [V1Order("a", "+", 1, 0, 3, 100)]
    schedule = {"x_manual": {(0, 1, 1, "+"): 1, (2, 3, 2, "+"): 1},
                "y_auto": {(1, 2, "+"): 1},
                "u_wave": {(0, 1, 1, "+", 1): 1, (2, 3, 2, "+", 1): 1},
                "g_manual": {(0, 1, 1, "+", "a"): 1,
                             (2, 3, 2, "+", "a"): 1},
                "g_auto": {(1, 2, "+", "a"): 1},
                "z_unserved": {("a",): 0}}
    return cfg, orders, profile, schedule


def empty_schedule():
    return {name: {} for name in ("x_manual", "y_auto", "u_wave", "g_manual",
                                  "g_auto", "z_unserved")}


class V1IndependentValidationTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.orders, self.profile, self.schedule = fixture()

    def check(self, *, cfg=None, orders=None, schedule=None, **kw):
        return validate_schedule(cfg or self.cfg,
                                 self.orders if orders is None else orders,
                                 self.profile,
                                 self.schedule if schedule is None else schedule,
                                 **kw)

    def test_manual_same_time_handoff_and_hourly_cost_oracle(self):
        result = self.check(reported_objective=12)
        self.assertEqual(result["costs"], {"manual": 10, "auto": 2,
                                          "penalty": 0, "total": 12})
        self.assertEqual(result["orders"]["a"]["linehaul"], 1)
        json.dumps(result, allow_nan=False)

    def test_string_tuple_keys_json_roundtrip(self):
        schedule = {group: {str(key): value for key, value in data.items()}
                    for group, data in self.schedule.items()}
        result = self.check(schedule=json.loads(json.dumps(schedule)))
        self.assertTrue(result["schedule_valid"])

    def test_reverse_direction_uses_city_two_origin(self):
        cfg = replace(self.cfg, N_auto={1: 0, 2: 1})
        orders = [replace(self.orders[0], flow="-")]
        schedule = empty_schedule()
        schedule["x_manual"] = {(0, 1, 2, "-"): 1, (2, 3, 1, "-"): 1}
        schedule["u_wave"] = {(0, 1, 2, "-", 1): 1, (2, 3, 1, "-", 1): 1}
        schedule["g_manual"] = {(0, 1, 2, "-", "a"): 1,
                                 (2, 3, 1, "-", "a"): 1}
        schedule["y_auto"] = {(1, 2, "-"): 1}
        schedule["g_auto"] = {(1, 2, "-", "a"): 1}
        self.assertEqual(self.check(cfg=cfg, orders=orders, schedule=schedule)
                         ["objective"], 12)

    def test_av_can_turn_around_at_exact_arrival_time(self):
        schedule = empty_schedule()
        schedule["y_auto"] = {(0, 1, "+"): 1, (1, 2, "-"): 1,
                               (2, 3, "+"): 1}
        self.assertEqual(self.check(orders=[], schedule=schedule)["objective"], 6)

    def test_manual_touching_arcs_use_half_open_occupancy(self):
        schedule = empty_schedule()
        schedule["x_manual"] = {(0, 1, 1, "+"): 1, (1, 2, 1, "-"): 1}
        schedule["u_wave"] = {(0, 1, 1, "+", 1): 1, (1, 2, 1, "-", 1): 1}
        self.assertEqual(self.check(orders=[], schedule=schedule)["objective"], 10)

    def test_looser_feasibility_tolerance_does_not_relax_integer_domain(self):
        self.schedule["x_manual"][(0, 1, 1, "+")] = 1.0009
        with self.assertRaisesRegex(ValueError, "noninteger"):
            self.check(tolerance=1e-3)

    def test_fractional_service_and_continuous_z(self):
        orders = [replace(self.orders[0], quantity=1.5, penalty=1)]
        self.schedule["z_unserved"][("a",)] = .5
        self.assertEqual(self.check(orders=orders)["costs"]["penalty"], .5)

    def test_small_quantity_preserved_with_zero_fleet(self):
        cfg = replace(self.cfg, N_manual={1: 0, 2: 0}, N_auto={1: 0, 2: 0})
        schedule = empty_schedule()
        schedule["z_unserved"][("a",)] = .05
        orders = [replace(self.orders[0], quantity=.05, penalty=2)]
        result = self.check(cfg=cfg, orders=orders, schedule=schedule)
        self.assertEqual(result["costs"]["total"], .1)
        self.assertEqual(result["orders"]["a"]["unserved"], .05)

    def test_no_orders_no_fleet_and_tau_beyond_horizon(self):
        cfg = replace(self.cfg, N_manual={1: 0, 2: 0}, N_auto={1: 0, 2: 0},
                      tau_periods=5)
        self.assertEqual(self.check(cfg=cfg, orders=[], schedule=empty_schedule())
                         ["costs"]["total"], 0)

    def test_zero_demand(self):
        orders = [replace(self.orders[0], quantity=0)]
        self.assertEqual(self.check(orders=orders, schedule=empty_schedule())
                         ["orders"]["a"]["served"], 0)

    def test_joint_bhh_wave_two_vehicles_cost_five(self):
        # f(1)=2; one minute needs v=2, so each local leg costs 2.
        cfg = replace(self.cfg, t0_minutes=1, a={1: 1, 2: 1}, b={1: 1, 2: 1},
                      N_manual={1: 2, 2: 2}, c_manual_hour=60, c_auto_hour=60)
        self.schedule["x_manual"] = {key: 2 for key in self.schedule["x_manual"]}
        self.schedule["u_wave"] = {key[:4] + (2,): 1
                                    for key in self.schedule["u_wave"]}
        self.assertEqual(self.check(cfg=cfg)["objective"], 5)

    def test_zero_capacity_source_arc_may_run_empty_even_under_stem(self):
        cfg = replace(self.cfg, rho={1: 30, 2: 30})
        schedule = empty_schedule()
        schedule["x_manual"][(0, 1, 1, "+")] = 1
        schedule["u_wave"][(0, 1, 1, "+", 1)] = 1
        schedule["z_unserved"][("a",)] = 1
        self.assertEqual(self.check(cfg=cfg, schedule=schedule)["objective"], 105)

    def test_stem_blocks_positive_cargo(self):
        with self.assertRaisesRegex(ValueError, "BHH"):
            self.check(cfg=replace(self.cfg, rho={1: 15, 2: 15}))

    def test_original_f_blocks_insufficient_tour_time(self):
        with self.assertRaisesRegex(ValueError, "BHH"):
            self.check(cfg=replace(self.cfg, b={1: 1, 2: 1}))

    def test_shared_manual_physical_capacity(self):
        with self.assertRaisesRegex(ValueError, "physical capacity"):
            self.check(cfg=replace(self.cfg, M=.5))

    def test_shared_av_capacity_across_orders(self):
        orders = [replace(self.orders[0], quantity=.75),
                  V1Order("b", "+", .75, 0, 3, 100)]
        cfg = replace(self.cfg, M=2, a={1: 10, 2: 10})
        self.schedule["g_manual"] = {
            (0, 1, 1, "+", oid): .75 for oid in ("a", "b")}
        self.schedule["g_manual"].update({
            (2, 3, 2, "+", oid): .75 for oid in ("a", "b")})
        self.schedule["g_auto"] = {(1, 2, "+", oid): .75 for oid in ("a", "b")}
        with self.assertRaisesRegex(ValueError, "shared AV"):
            self.check(cfg=cfg, orders=orders)

    def test_wave_vehicle_link_missing_selector(self):
        self.schedule["u_wave"].pop((0, 1, 1, "+", 1))
        with self.assertRaisesRegex(ValueError, "wave vehicle link"):
            self.check()

    def test_multiple_wave_selectors(self):
        cfg = replace(self.cfg, N_manual={1: 2, 2: 1})
        self.schedule["u_wave"][(0, 1, 1, "+", 2)] = 1
        with self.assertRaisesRegex(ValueError, "one-hot"):
            self.check(cfg=cfg)

    def test_manual_overlapping_directions_share_city_fleet(self):
        self.schedule["x_manual"][(0, 1, 1, "-")] = 1
        self.schedule["u_wave"][(0, 1, 1, "-", 1)] = 1
        with self.assertRaisesRegex(ValueError, "manual city occupancy"):
            self.check()

    def test_av_occupancy(self):
        self.schedule["y_auto"][(1, 2, "-")] = 1
        with self.assertRaisesRegex(ValueError, "AV occupancy"):
            self.check()

    def test_av_wrong_hub_despite_sufficient_total_fleet(self):
        with self.assertRaisesRegex(ValueError, "hub inventory"):
            self.check(cfg=replace(self.cfg, N_auto={1: 0, 2: 1}))

    def test_empty_av_reposition_without_order_and_no_terminal_return(self):
        schedule = empty_schedule()
        schedule["y_auto"][(0, 1, "+")] = 1
        result = self.check(orders=[], schedule=schedule)
        self.assertEqual(result["costs"]["total"], 2)

    def test_av_departure_cannot_precede_pickup_return(self):
        self.schedule["y_auto"] = {(0, 1, "+"): 1}
        self.schedule["g_auto"] = {(0, 1, "+", "a"): 1}
        with self.assertRaisesRegex(ValueError, "pickup before AV"):
            self.check()

    def test_distribution_cannot_precede_av_arrival(self):
        self.schedule["x_manual"][(1, 2, 2, "+")] = self.schedule["x_manual"].pop((2, 3, 2, "+"))
        self.schedule["u_wave"][(1, 2, 2, "+", 1)] = self.schedule["u_wave"].pop((2, 3, 2, "+", 1))
        self.schedule["g_manual"][(1, 2, 2, "+", "a")] = self.schedule["g_manual"].pop((2, 3, 2, "+", "a"))
        with self.assertRaisesRegex(ValueError, "AV arrival before"):
            self.check()

    def test_release_window(self):
        with self.assertRaisesRegex(ValueError, "release window"):
            self.check(orders=[replace(self.orders[0], earliest_start=1)])

    def test_deadline(self):
        with self.assertRaisesRegex(ValueError, "deadline"):
            self.check(orders=[replace(self.orders[0], latest_completion=2)])

    def test_missing_delivery_and_unserved_balance(self):
        self.schedule["g_manual"].pop((2, 3, 2, "+", "a"))
        with self.assertRaisesRegex(ValueError, "delivery demand balance"):
            self.check()

    def test_excess_z(self):
        self.schedule["z_unserved"][("a",)] = 2
        with self.assertRaisesRegex(ValueError, "unserved exceeds"):
            self.check()

    def test_objective_tampering(self):
        with self.assertRaisesRegex(ValueError, "objective mismatch"):
            self.check(reported_objective=12.01)

    def test_absolute_objective_tolerance_does_not_scale_with_cost(self):
        cfg = replace(self.cfg, c_manual_hour=1e12)
        good = self.check(cfg=cfg)["objective"]
        with self.assertRaisesRegex(ValueError, "objective mismatch"):
            self.check(cfg=cfg, reported_objective=good + 1)

    def test_illegal_values_including_nan_bool_and_noninteger(self):
        for group, value in (("g_auto", math.nan), ("g_auto", math.inf),
                             ("g_auto", -1), ("x_manual", True),
                             ("x_manual", 1.1), ("u_wave", 2)):
            with self.subTest(group=group, value=value):
                schedule = copy.deepcopy(self.schedule)
                schedule[group][next(iter(schedule[group]))] = value
                with self.assertRaises(ValueError):
                    self.check(schedule=schedule)

    def test_tolerance_and_objective_finiteness(self):
        for value in (0, -1, .01, math.inf, math.nan, True, "1e-6"):
            with self.subTest(tolerance=value), self.assertRaises(ValueError):
                self.check(tolerance=value)
        for value in (math.nan, math.inf, True):
            with self.subTest(objective=value), self.assertRaises(ValueError):
                self.check(reported_objective=value)

    def test_overflow_cannot_create_false_pass(self):
        orders = [replace(self.orders[0], quantity=1e308, penalty=1e308)]
        schedule = empty_schedule()
        schedule["z_unserved"][("a",)] = 1e308
        with self.assertRaisesRegex(ValueError, "overflow"):
            self.check(orders=orders, schedule=schedule)

    def test_domains_reject_wrong_tau_city_flow_order_and_horizon(self):
        mutations = [("y_auto", (0, 2, "+")),
                     ("x_manual", (0, 1, 3, "+")),
                     ("x_manual", (0, 1, True, "+")),
                     ("x_manual", (0, 1, 1, "bad")),
                     ("u_wave", (0, 1, 1, "+", 2)),
                     ("g_manual", (0, 1, 1, "+", "missing")),
                     ("g_manual", (0, 1, 1, "-", "a")),
                     ("x_manual", (-1, 0, 1, "+")),
                     ("x_manual", (3, 4, 1, "+")),
                     ("x_manual", (0, 3, 1, "+"))]
        for group, key in mutations:
            with self.subTest(group=group, key=key):
                schedule = copy.deepcopy(self.schedule)
                schedule[group][key] = 0
                with self.assertRaises(ValueError):
                    self.check(schedule=schedule)

    def test_malformed_and_duplicate_normalized_keys(self):
        for key in ("__import__('os').system('bad')", "([0],1,1,'+')", "(0,1)"):
            schedule = copy.deepcopy(self.schedule)
            schedule["x_manual"][key] = 0
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(schedule=schedule)
        self.schedule["x_manual"]["(0,1,1,'+')"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate normalized"):
            self.check()

    def test_missing_or_unknown_group_not_silently_ignored(self):
        self.schedule["made_up"] = {}
        with self.assertRaises(ValueError):
            self.check()
        self.schedule.pop("made_up")
        self.schedule.pop("g_auto")
        with self.assertRaises(ValueError):
            self.check()


if __name__ == "__main__":
    unittest.main()
