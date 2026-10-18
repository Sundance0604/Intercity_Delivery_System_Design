"""Conditional V1-M: pure input checks and exactly eight bounded MIP runs.

The PI executes the solver class once under the shared 20-second first-pass
authorization. Each run deducts construction from its total 2-second budget,
uses one thread, T<=3, <=2 orders, <=2 HV/city, and <=1 AV in total.
No data files, benchmark outputs, V0 optimizer, or registry are changed.
"""

import ast
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import unittest

from intercity_delivery.v1.configuration import (
    V1MConfig, V1MProfile, V1Order, validate_instance,
)
from intercity_delivery.v1.runner import run_mandatory


def explicit_profile():
    return V1MProfile("offline", "arc", "continuous_stops_zero_allowed",
                      "minutes_hourly", "tour_end_same_time",
                      "integer_windows_within_horizon")


def tiny_config(**changes):
    config = V1MConfig(T=3, t0_minutes=1, tau_periods=1,
                      N_manual={1: 1, 2: 1}, N_auto={1: 1, 2: 0},
                      a={1: 1, 2: 1}, b={1: 0, 2: 0}, rho={1: 0, 2: 0},
                      M=1, M_auto=1, c_manual_hour=0, c_auto_hour=0)
    return replace(config, **changes)


def order(quantity=1, penalty=100, **changes):
    return replace(V1Order("order-A", "+", quantity, 0, 3, penalty), **changes)


class ConfigurationTests(unittest.TestCase):
    def test_profile_has_no_defaults(self):
        with self.assertRaises(TypeError):
            V1MProfile()
        with self.assertRaises(TypeError):
            V1MProfile("offline", "arc", "continuous_stops_zero_allowed",
                       "minutes_hourly", "tour_end_same_time")

    def test_explicit_profile_and_zero_inputs(self):
        cfg = tiny_config(N_manual={1: 0, 2: 0}, N_auto={1: 0, 2: 0})
        validate_instance(cfg, [order(0, 0)], explicit_profile())
        validate_instance(cfg, [], explicit_profile())

    def test_unsupported_profile_and_missing_profile_rejected(self):
        for profile in (None, replace(explicit_profile(), wave="start"),
                        replace(explicit_profile(), information="online"),
                        replace(explicit_profile(), units="hours_hourly")):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                validate_instance(tiny_config(), [order()], profile)

    def test_config_scalar_domains(self):
        for name in ("T", "tau_periods", "t0_minutes", "M", "M_auto",
                     "c_manual_hour", "c_auto_hour"):
            for value in (True, math.nan, math.inf, -1):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    validate_instance(tiny_config(**{name: value}), [], explicit_profile())
        for name in ("T", "tau_periods", "t0_minutes", "M", "M_auto"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_instance(tiny_config(**{name: 0}), [], explicit_profile())
        for name in ("T", "tau_periods"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_instance(tiny_config(**{name: 1.5}), [], explicit_profile())

    def test_city_mapping_domains(self):
        for name in ("N_manual", "N_auto", "a", "b", "rho"):
            for values in ({1: 1}, {1: 1, 3: 1}, {True: 1, 2: 1},
                           {1.0: 1, 2: 1}, {1: math.nan, 2: 1},
                           {1: True, 2: 1}):
                with self.subTest(name=name, values=values), self.assertRaises(ValueError):
                    validate_instance(tiny_config(**{name: values}), [], explicit_profile())
        for name in ("N_manual", "N_auto"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_instance(tiny_config(**{name: {1: 0.5, 2: 1}}), [], explicit_profile())
        with self.assertRaises(ValueError):
            validate_instance(tiny_config(a={1: 0, 2: 1}), [], explicit_profile())

    def test_order_domains_duplicates_and_one_shot_iterators(self):
        bad_orders = [replace(order(), order_id=""), replace(order(), order_id=" "),
                      replace(order(), flow="x"), replace(order(), quantity=math.nan),
                      replace(order(), penalty=math.inf), replace(order(), quantity=True),
                      replace(order(), earliest_start=0.5),
                      replace(order(), latest_completion=4),
                      replace(order(), earliest_start=3, latest_completion=2)]
        for invalid in bad_orders:
            with self.subTest(order=invalid), self.assertRaises(ValueError):
                validate_instance(tiny_config(), [invalid], explicit_profile())
        with self.assertRaises(ValueError):
            validate_instance(tiny_config(), [order(), order()], explicit_profile())
        with self.assertRaises(ValueError):
            validate_instance(tiny_config(), iter([order()]), explicit_profile())

    def test_invalid_optimizer_input_never_creates_solver_model(self):
        from intercity_delivery.v1.mandatory_optimizer import V1MandatoryOptimizer
        with self.assertRaises(ValueError):
            V1MandatoryOptimizer(tiny_config(), [order()], None)

    def test_pure_metadata_normalizes_non_native_reals(self):
        from fractions import Fraction
        from intercity_delivery.v1.runner import _plain
        value = _plain(tiny_config(t0_minutes=Fraction(1, 2)))
        self.assertEqual(value["t0_minutes"], 0.5)
        self.assertEqual(value["N_manual"], {"1": 1, "2": 1})
        json.dumps(value, allow_nan=False)


class MandatoryMicroSolveTests(unittest.TestCase):
    """Exactly eight run_mandatory calls; no license-failure skips."""

    def run_case(self, config, orders, expected):
        result = run_mandatory(config, orders, explicit_profile(), time_limit=2,
                               output_flag=0)
        self.assertTrue(result["has_incumbent"], result)
        self.assertTrue(result["schedule_valid"], result)
        self.assertTrue(result["optimal"], result)
        self.assertEqual(result["status_name"], "OPTIMAL")
        self.assertAlmostEqual(result["total_cost"], expected, places=6)
        self.assertAlmostEqual(result["validation"]["objective"], expected, places=6)
        self.assertEqual(result["semantic_status"], "author_confirmed_profile")
        self.assertEqual(result["profile_decision_id"], "V1M-D1-D6-A-2026-10-18")
        self.assertEqual(result["profile"]["wave"], "arc")
        self.assertEqual(result["threads"], 1)
        self.assertEqual(result["bound_scope"], "full_horizon_adopted_V1_M_profile_only")
        self.assertLessEqual(result["solver_time_limit_sec"], 2)
        self.assertEqual(set(result["schedule"]),
                         {"x_manual", "y_auto", "u_wave", "g_manual", "g_auto", "z_unserved"})
        self.assertEqual(len(result["provenance"]["input_sha256"]), 64)
        inputs = result["provenance"]["input"]
        digest = hashlib.sha256(json.dumps(
            inputs, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest()
        self.assertEqual(digest, result["provenance"]["input_sha256"])
        source = Path(__file__).resolve().parents[1] / "research/paper/main_v2_BHH.tex"
        self.assertEqual(result["provenance"]["source_sha256"]["research/paper/main_v2_BHH.tex"],
                         hashlib.sha256(source.read_bytes()).hexdigest())
        json.dumps(result, allow_nan=False)
        return result

    @staticmethod
    def group(result, name):
        return {ast.literal_eval(key): value for key, value in result["schedule"][name].items()}

    def test_01_F2_shared_AV_two_orders(self):
        cfg = tiny_config(a={1: 0.5, 2: 0.5}, M=2)
        result = self.run_case(cfg, [order(0.75, 1),
                                    order(0.75, 1, order_id="order-B")], 0.5)
        self.assertAlmostEqual(sum(self.group(result, "z_unserved").values()), 0.5)
        flow = self.group(result, "g_auto")
        self.assertAlmostEqual(sum(v for (i, j, d, oid), v in flow.items()
                                   if (i, j, d) == (1, 2, "+")), 1)

    def test_02_F3_continuous_unserved(self):
        cfg = tiny_config(a={1: 0.5, 2: 0.5}, M=2)
        result = self.run_case(cfg, [order(1.5, 1)], 0.5)
        self.assertAlmostEqual(self.group(result, "z_unserved")["order-A",], 0.5)

    def test_03_F4_units_same_time_handoffs(self):
        cfg = tiny_config(t0_minutes=30, a={1: 30, 2: 30},
                          c_manual_hour=10, c_auto_hour=4)
        result = self.run_case(cfg, [order()], 12)
        self.assertAlmostEqual(result["cost_components"]["manual"], 10)
        self.assertAlmostEqual(result["cost_components"]["auto"], 2)
        self.assertAlmostEqual(self.group(result, "g_manual")[0, 1, 1, "+", "order-A"], 1)
        self.assertAlmostEqual(self.group(result, "g_auto")[1, 2, "+", "order-A"], 1)
        self.assertAlmostEqual(self.group(result, "g_manual")[2, 3, 2, "+", "order-A"], 1)

    def test_04_F4_deadline_rejects_incomplete_chain(self):
        cfg = tiny_config(t0_minutes=30, a={1: 30, 2: 30},
                          c_manual_hour=10, c_auto_hour=4)
        result = self.run_case(cfg, [order(latest_completion=2)], 100)
        self.assertAlmostEqual(self.group(result, "z_unserved")["order-A",], 1)

    def test_05_F5_tiny_rejection_zero_fleet_zero_demand_tau_outside(self):
        cfg = tiny_config(N_manual={1: 0, 2: 0}, N_auto={1: 0, 2: 0}, tau_periods=4)
        result = self.run_case(cfg, [order(0.05, 2), order(0, 0, order_id="zero")], 0.1)
        self.assertAlmostEqual(self.group(result, "z_unserved")["order-A",], 0.05)
        self.assertEqual(self.group(result, "z_unserved")["zero",], 0)
        self.assertEqual(result["schedule"]["u_wave"], {})
        self.assertEqual(result["schedule"]["y_auto"], {})
        self.assertEqual(result["schedule"]["g_auto"], {})

    def test_06_joint_wave_strictly_changes_capacity(self):
        cfg = tiny_config(N_manual={1: 2, 2: 2}, b={1: 1, 2: 1},
                          c_manual_hour=60, c_auto_hour=60)
        result = self.run_case(cfg, [order()], 5)
        self.assertAlmostEqual(self.group(result, "u_wave")[0, 1, 1, "+", 2], 1)
        self.assertAlmostEqual(self.group(result, "u_wave")[2, 3, 2, "+", 2], 1)
        self.assertAlmostEqual(self.group(result, "z_unserved")["order-A",], 0)

    def test_07_F1_stem_zero_capacity_mother_arc_is_retained(self):
        cfg = tiny_config(rho={1: 0.5, 2: 0.5}, c_manual_hour=60, c_auto_hour=60)
        result = self.run_case(cfg, [order()], 100)
        self.assertIn((0, 1, 1, "+"), self.group(result, "x_manual"))
        self.assertAlmostEqual(self.group(result, "g_manual")[0, 1, 1, "+", "order-A"], 0)
        self.assertAlmostEqual(self.group(result, "z_unserved")["order-A",], 1)

    def test_08_AV_empty_direction_needed_for_reverse_order(self):
        cfg = tiny_config(c_manual_hour=60, c_auto_hour=60)
        result = self.run_case(cfg, [order(flow="-")], 4)
        auto = self.group(result, "y_auto")
        self.assertAlmostEqual(auto[0, 1, "+"], 1)
        self.assertAlmostEqual(auto[1, 2, "-"], 1)
        self.assertFalse(any(key[2] == "+" for key in self.group(result, "g_auto")))
        self.assertAlmostEqual(self.group(result, "z_unserved")["order-A",], 0)


if __name__ == "__main__":
    unittest.main()
