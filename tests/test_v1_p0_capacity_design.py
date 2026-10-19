"""Pure P0 capacity-design checks; no solver or CFS access."""

import json
import math
from pathlib import Path
import unittest

from intercity_delivery.v1.capacity import local_capacity


DESIGN = Path(__file__).resolve().parents[1] / "experiments/specs/v1_p0_capacity_design.json"


def bisect_capacity(budget, a, b, upper):
    """Independent inverse by the original time inequality, not the saved inverse."""
    low, high = 0.0, upper
    for _ in range(100):
        mid = (low + high) / 2
        if a * mid + b * math.sqrt(mid) <= budget:
            low = mid
        else:
            high = mid
    return low


class CapacityDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = json.loads(DESIGN.read_text(encoding="utf-8"))

    def test_all_curve_points_match_independent_original_inequality(self):
        for case in self.design["scenarios"]:
            for point in case["curve"]:
                with self.subTest(case=case["id"], v=point["vehicles"]):
                    v, a, b = point["vehicles"], case["a"], case["b"]
                    budget = case["duration_minutes"] - 2 * case["rho_minutes"]
                    cap = v * case["M"]
                    reference = bisect_capacity(v * budget, a, b, cap)
                    separate = v * bisect_capacity(budget, a, b, case["M"])
                    self.assertAlmostEqual(point["pooled_capacity"], reference, places=10)
                    self.assertAlmostEqual(point["separate_capacity"], separate, places=10)
                    actual = local_capacity(case["duration_minutes"], v, a, b,
                                            case["rho_minutes"], case["M"])
                    self.assertAlmostEqual(actual, reference, places=10)
                    self.assertLessEqual(a * actual + b * math.sqrt(actual), v * budget + 1e-9)
                    self.assertLessEqual(actual, cap + 1e-9)

    def test_binding_regimes_and_hand_calculated_anchor(self):
        cases = {case["id"]: case for case in self.design["scenarios"]}
        routing = cases["routing_binding"]["curve"]
        self.assertAlmostEqual(routing[1]["pooled_capacity"], 4.0)
        # f(4)=100 = 2*B: q=4 at v=2 is an independent hand anchor.
        for point in routing:
            self.assertLess(point["pooled_capacity"], 4 * point["vehicles"])
        for point in cases["load_binding"]["curve"]:
            self.assertEqual(point["pooled_capacity"], 4 * point["vehicles"])
            self.assertEqual(point["separate_capacity"], 4 * point["vehicles"])
        transition = cases["near_transition"]["curve"]
        self.assertLess(transition[0]["pooled_capacity"], 4)
        for point in transition[1:]:
            self.assertEqual(point["pooled_capacity"], 4 * point["vehicles"])
            self.assertLess(point["separate_capacity"], point["pooled_capacity"])

    def test_chain_support_and_latest_origin_admission(self):
        baseline = self.design["baseline"]
        for case in self.design["scenarios"]:
            durations = [math.ceil((2 * case["rho_minutes"] + case["a"] * q
                                   + case["b"] * math.sqrt(q)) / 60) for q in (1, 4)]
            full_periods = 1 if case["id"] == "load_binding" else 2
            self.assertEqual(durations, [1, full_periods])
            self.assertEqual([2 * length + baseline["tau_periods"] for length in durations],
                             [11, 2 * full_periods + 9])
        rh = self.design["candidate_rh"]
        latest_i = rh["H"] - 1
        self.assertLessEqual(latest_i + 13, rh["H"] + rh["E"])
        self.assertGreater((8 - 1) + 13, 8 + 6)

    def test_asymmetric_city_tables_are_distinct(self):
        cities = self.design["asymmetric_extension"]["cities"]
        values = [local_capacity(60, 2, c["a"], c["b"], c["rho_minutes"], 4) for c in cities]
        self.assertGreater(values[0], values[1])
        for c in cities:
            self.assertEqual(math.ceil((2*c["rho_minutes"] + c["a"] + c["b"])/60), 1)

    def test_preregistered_scope_preserves_confirmed_abstract_endpoint(self):
        self.assertEqual(self.design["status"], "frozen_synthetic_parameter_design_v1")
        self.assertEqual(self.design["baseline"]["scenario_id"], "near_transition")
        self.assertEqual(self.design["baseline"]["penalty_per_unit"], 100)
        self.assertEqual(self.design["baseline"]["physical_direct_endpoint_status"],
                         "author_confirmed_aggregate_city_node_not_physical_hub_route")
        self.assertEqual(self.design["candidate_rh"]["status"],
                         "frozen_P1_pilot_design_not_runtime_default")


if __name__ == "__main__":
    unittest.main()
