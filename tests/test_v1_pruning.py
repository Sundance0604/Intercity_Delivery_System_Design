"""Pure V1-F candidate filtering checks; no optimizer or solver run."""

from dataclasses import replace
import unittest

from intercity_delivery.v1.configuration import V1Order
from intercity_delivery.v1.flexible_configuration import V1FConfig
from intercity_delivery.v1.pruning import prune_candidates


def tiny_config(**changes):
    cfg = V1FConfig(T=6, t0_minutes=60, tau_periods=1,
                    N_manual={1: 1, 2: 1}, N_auto={1: 1, 2: 0},
                    a={1: 60, 2: 60}, b={1: 0, 2: 0}, rho={1: 0, 2: 0},
                    M=1, M_auto=1, c_manual_hour=1, c_auto_hour=1,
                    c_direct_hour=1)
    return replace(cfg, **changes)


def one_order(**changes):
    return replace(V1Order("A", "+", 0.05, 0, 6, 1000), **changes)


class PruningTests(unittest.TestCase):
    def test_real_removals_release_deadline_and_positive_capacity(self):
        result = prune_candidates(tiny_config(), [one_order(earliest_start=1,
                                  latest_completion=4)], 1, 3, 5, {})
        self.assertGreater(result["statistics"]["removed"], 0)
        self.assertEqual(result["statistics"]["scope"], "cargo_variables_only")
        for i, j, flow, oid in result["candidates"]["h_direct"]:
            self.assertGreaterEqual(i, 1)
            self.assertLess(i, 3)
            self.assertLessEqual(j, 4)
            self.assertGreater(j-i, 1)
        self.assertTrue(result["statistics"]["vehicle_arcs_retained"])

    def test_three_leg_chain_respects_both_transfer_delays(self):
        cfg = tiny_config(T=5, transfer_time_periods=1)
        keys = prune_candidates(cfg, [one_order(latest_completion=5)], 0, 1, 5, {})["candidates"]
        self.assertIn((0, 1, 1, "+", "A"), keys["g_manual"])
        self.assertIn((2, 3, "+", "A"), keys["g_auto"])
        self.assertIn((4, 5, 2, "+", "A"), keys["g_manual"])
        self.assertNotIn((1, 2, "+", "A"), keys["g_auto"])

    def test_committed_pickup_seeds_av_without_new_pickup(self):
        past = {"g_manual": {(0, 1, 1, "+", "A"): 0.05}}
        keys = prune_candidates(tiny_config(T=3), [one_order(latest_completion=3)],
                                1, 2, 3, past)["candidates"]
        self.assertIn((0, 1, 1, "+", "A"), keys["g_manual"])
        self.assertIn((1, 2, "+", "A"), keys["g_auto"])
        self.assertIn((2, 3, 2, "+", "A"), keys["g_manual"])

    def test_committed_av_seeds_delivery_without_new_pickup_or_av(self):
        past = {"g_auto": {(0, 1, "+", "A"): 0.05}}
        keys = prune_candidates(tiny_config(T=2), [one_order(latest_completion=2)],
                                1, 2, 2, past)["candidates"]
        self.assertIn((0, 1, "+", "A"), keys["g_auto"])
        self.assertIn((1, 2, 2, "+", "A"), keys["g_manual"])

    def test_city_with_no_initial_hv_can_receive_future_vehicle(self):
        cfg = tiny_config(N_manual={1: 0, 2: 1})
        keys = prune_candidates(cfg, [one_order(earliest_start=1)],
                                0, 4, 6, {})["candidates"]
        self.assertIn((1, 2, 1, "+", "A"), keys["g_manual"])
        self.assertIn((1, 3, "+", "A"), keys["h_direct"])

    def test_no_manual_or_auto_fleet_does_not_create_cargo_support(self):
        cfg = tiny_config(N_manual={1: 0, 2: 0}, N_auto={1: 0, 2: 0})
        keys = prune_candidates(cfg, [one_order()], 0, 3, 6, {})["candidates"]
        self.assertTrue(all(not values for values in keys.values()))

    def test_history_retained_after_deadline_and_tiny_quantity(self):
        history = {"h_direct": {(0, 2, "+", "A"): 1e-12},
                   "g_auto": {(0, 1, "+", "A"): 1e-12},
                   "g_manual": {(0, 1, 1, "+", "A"): 1e-12}}
        keys = prune_candidates(tiny_config(), [one_order(latest_completion=2)],
                                4, 5, 6, history)["candidates"]
        for name, values in history.items():
            self.assertTrue(set(values) <= keys[name])

    def test_invalid_window_rejected(self):
        for bounds in ((2, 1, 3), (0, 4, 3), (0, 1, 7), (True, 1, 3)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                prune_candidates(tiny_config(), [one_order()], *bounds, {})

    def test_impossible_three_leg_deadline_prunes_transshipment(self):
        keys = prune_candidates(tiny_config(T=2), [one_order(latest_completion=2)],
                                0, 2, 2, {})["candidates"]
        self.assertFalse(keys["g_manual"])
        self.assertFalse(keys["g_auto"])
        self.assertTrue(keys["h_direct"])


if __name__ == "__main__":
    unittest.main()
