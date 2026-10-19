"""Direct arrival continuity using independent replay, without a solver."""

from copy import deepcopy
import unittest

from intercity_delivery.v1.configuration import V1Order
from intercity_delivery.v1.flexible_configuration import V1FConfig, flexible_profile
from intercity_delivery.v1.flexible_validation import GROUPS, validate_flexible_schedule


def config():
    return V1FConfig(T=8, t0_minutes=60, tau_periods=1,
                    N_manual={1: 1, 2: 0}, N_auto={1: 1, 2: 0},
                    a={1: 60, 2: 60}, b={1: 0, 2: 0}, rho={1: 0, 2: 0},
                    M=4, M_auto=4, c_manual_hour=1,
                    c_auto_hour=1, c_direct_hour=2)


def blank():
    return {group: {} for group in GROUPS}


def direct(schedule, i, j, flow, oid):
    schedule["w_direct"][(i, j, flow)] = 1
    schedule["u_direct"][(i, j, flow, 1)] = 1
    schedule["h_direct"][(i, j, flow, oid)] = 1


def local(schedule, i, city, oid=None):
    schedule["x_manual"][(i, i+1, city, "+")] = 1
    schedule["u_wave"][(i, i+1, city, "+", 1)] = 1
    if oid is not None:
        schedule["g_manual"][(i, i+1, city, "+", oid)] = 1


class DirectContinuityTests(unittest.TestCase):
    def check(self, schedule, orders, **kwargs):
        return validate_flexible_schedule(config(), orders, flexible_profile(), schedule, **kwargs)

    def test_loaded_direct_then_loaded_destination_local_at_same_node(self):
        orders = [V1Order("direct", "+", 1, 0, 8, 100),
                  V1Order("transfer", "+", 1, 0, 8, 100)]
        for local_start, feasible in ((4, True), (3, False)):
            with self.subTest(local_start=local_start):
                schedule = blank()
                local(schedule, 0, 1, "transfer")
                schedule["y_auto"][(1, 2, "+")] = 1
                schedule["g_auto"][(1, 2, "+", "transfer")] = 1
                direct(schedule, 1, 4, "+", "direct")
                local(schedule, local_start, 2, "transfer")
                if feasible:
                    result = self.check(schedule, orders, reported_objective=9)
                    self.assertEqual(result["served_quantity"], 2)
                    self.assertEqual(result["objective"], 9)
                else:
                    with self.assertRaisesRegex(ValueError, "manual city inventory"):
                        self.check(schedule, orders)

    def test_loaded_direct_then_reverse_loaded_direct_at_same_node(self):
        orders = [V1Order("out", "+", 1, 0, 8, 100),
                  V1Order("back", "-", 1, 0, 8, 100)]
        for start in (3, 2):
            with self.subTest(start=start):
                schedule = blank()
                direct(schedule, 0, 3, "+", "out")
                direct(schedule, start, start+3, "-", "back")
                if start == 3:
                    self.assertEqual(self.check(schedule, orders, reported_objective=12)["objective"], 12)
                else:
                    with self.assertRaisesRegex(ValueError, "manual city inventory"):
                        self.check(schedule, orders)

    def test_loaded_direct_then_empty_at_same_node(self):
        orders = [V1Order("out", "+", 1, 0, 8, 100)]
        for start in (3, 2):
            with self.subTest(start=start):
                schedule = blank()
                direct(schedule, 0, 3, "+", "out")
                schedule["empty_vehicles"][(start, start+1, "-")] = 1
                if start == 3:
                    result = self.check(schedule, orders, reported_objective=8)
                    self.assertEqual(result["costs"]["direct"], 6)
                    self.assertEqual(result["costs"]["empty"], 2)
                else:
                    with self.assertRaisesRegex(ValueError, "manual city inventory"):
                        self.check(schedule, orders)

    def test_prior_window_whole_task_retained_and_cost_not_counted_twice(self):
        orders = [V1Order("out", "+", 1, 0, 8, 100)]
        prior = blank()
        direct(prior, 0, 3, "+", "out")
        initial = self.check(prior, orders, completed=False, completed_horizon=1)
        self.assertEqual(initial["costs"]["direct"], 6)
        self.assertEqual(initial["served_quantity"], 0)
        self.assertIsNone(initial["objective"])
        continuation = deepcopy(prior)
        local(continuation, 3, 2)
        partial = self.check(continuation, orders, completed=False, completed_horizon=4)
        self.assertEqual(partial["costs"]["direct"], 6)
        self.assertEqual(partial["served_quantity"], 1)
        self.assertEqual(self.check(continuation, orders, reported_objective=7)["objective"], 7)
        # Counterexample: truncating inherited 0->3 to the previous cut 1
        # leaves a positive load on a tau-only direct arc, hence is rejected.
        truncated = blank()
        direct(truncated, 0, 1, "+", "out")
        with self.assertRaisesRegex(ValueError, "zero-capacity direct"):
            self.check(truncated, orders, completed=False, completed_horizon=1)


if __name__ == "__main__":
    unittest.main()
