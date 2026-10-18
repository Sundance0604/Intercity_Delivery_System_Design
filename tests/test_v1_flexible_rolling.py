"""Pure controller checks plus three PI-only bounded end-to-end RH runs.

The micro class contains exactly three total-run <=2 second solves. The PI
executes it under the shared budget; running ControllerTests uses no solver.
"""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from intercity_delivery.v1.flexible_rolling import (
    RollingOptions, commit_schedule, revealed_orders, solve_flexible_rolling,
)
from tests.test_v1_pruning import tiny_config, one_order


class ControllerTests(unittest.TestCase):
    def test_options_validate(self):
        RollingOptions().validate()
        for opts in (RollingOptions(prediction_horizon=0),
                     RollingOptions(rolling_step=True),
                     RollingOptions(prediction_horizon=1, rolling_step=2),
                     RollingOptions(extension_horizon=-1),
                     RollingOptions(prune=1)):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                opts.validate()

    def test_revealed_orders_retains_expired_and_hides_future(self):
        past = one_order(latest_completion=1)
        future = one_order(order_id="B", earliest_start=4)
        self.assertEqual(revealed_orders([past, future], 3), (past,))

    def test_commit_keeps_all_history_types_and_tiny_cargo(self):
        schedule = {"u_wave": {(1, 2, 1, "+", 1): 0.999999999},
                    "w_direct": {(0, 2, "+"): 1, (1, 3, "+"): 1.000000001},
                    "h_direct": {(1, 3, "+", "A"): 1e-12},
                    "z_unserved": {("A",): 0.9}}
        result = commit_schedule(schedule, 1, 2)
        self.assertEqual(result["u_wave"][1, 2, 1, "+", 1], 1)
        self.assertEqual(result["w_direct"], {(1, 3, "+"): 1})
        self.assertEqual(result["h_direct"][1, 3, "+", "A"], 1e-12)
        self.assertNotIn("z_unserved", result)

    def test_commit_rejects_nonfinite_negative_and_fractional_vehicle(self):
        for name, value in (("x_manual", 0.5), ("g_auto", float("nan")),
                            ("g_auto", float("inf")), ("g_auto", -1)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                commit_schedule({name: {(0, 1, "+"): value}}, 0, 1)

    def test_total_budget_deducts_build_and_prior_windows(self):
        clock, instances = [0.0], []

        class FakeOptimizer:
            def __init__(self, config, orders, profile, candidates=None):
                self.known = orders
                self.groups = {name: {} for name in ("g_manual", "g_auto", "h_direct")}
                self.model = SimpleNamespace(SolCount=1, Status=2, ObjVal=0,
                                             ObjBound=0, MIPGap=0)
                self.params, self.disposed = {}, False
                self.model.setParam = self.params.__setitem__
                self.model.optimize = lambda: clock.__setitem__(0, clock[0]+0.75)
                self.model.dispose = lambda: setattr(self, "disposed", True)
                instances.append(self)

            def build_model(self):
                clock[0] += 0.5
                return self

            def configure_window(self, current, start_end, completion_end, committed):
                self.history = {key: dict(value) for key, value in committed.items()}
                self.current = current

            def extract_schedule(self):
                return {"empty_vehicles": {(self.current, self.current+1, "+"): 1}}

        cfg = tiny_config(T=3)
        order = one_order(earliest_start=1, latest_completion=3)
        with patch("intercity_delivery.v1.flexible_rolling.time.monotonic", lambda: clock[0]), \
             patch("intercity_delivery.v1.flexible_rolling.V1FlexibleOptimizer", FakeOptimizer):
            raw = solve_flexible_rolling(cfg, [order], RollingOptions(1, 1, 1), 2.5)
        self.assertFalse(raw["completed"])
        self.assertEqual(raw["status"], "PARTIAL_TIME_LIMIT")
        self.assertEqual(raw["current_time"], 2)
        self.assertEqual([item.params["TimeLimit"] for item in instances], [2, 0.75])
        self.assertEqual(instances[0].known, ())
        self.assertEqual(instances[1].known, (order,))
        self.assertEqual(instances[1].history["empty_vehicles"], {(0, 1, "+"): 1})
        self.assertTrue(all(item.disposed for item in instances))
        self.assertTrue(all(item.params["Threads"] == 1 and item.params["Seed"] == 0
                            for item in instances))

    def test_budget_exhausted_during_build_never_optimizes(self):
        clock = [0.0]
        class BuildOnly:
            def __init__(self, *args, **kwargs):
                self.model = SimpleNamespace(setParam=lambda *a: None,
                                             dispose=lambda: None)
            def build_model(self):
                clock[0] = 2
                return self
            def configure_window(self, *args):
                pass
        with patch("intercity_delivery.v1.flexible_rolling.time.monotonic", lambda: clock[0]), \
             patch("intercity_delivery.v1.flexible_rolling.V1FlexibleOptimizer", BuildOnly):
            raw = solve_flexible_rolling(tiny_config(), [], RollingOptions(), 1)
        self.assertEqual(raw["windows"], [])
        self.assertEqual(raw["current_time"], 0)

    def test_terminal_target_rejected_before_build(self):
        cfg = tiny_config(terminal_target={1: 1, 2: 1})
        with self.assertRaisesRegex(ValueError, "reachability"):
            solve_flexible_rolling(cfg, [], RollingOptions(), 1)

    def test_no_incumbent_keeps_control_boundary_and_disposes(self):
        disposed = []
        class Infeasible:
            def __init__(self, *args, **kwargs):
                self.groups = {name: {} for name in ("g_manual", "g_auto", "h_direct")}
                self.model = SimpleNamespace(SolCount=0, Status=3,
                    setParam=lambda *args: None, optimize=lambda: None,
                    dispose=lambda: disposed.append(True))
            def build_model(self):
                return self
            def configure_window(self, *args):
                pass
        with patch("intercity_delivery.v1.flexible_rolling.V1FlexibleOptimizer", Infeasible):
            raw = solve_flexible_rolling(tiny_config(), [], RollingOptions(), 1)
        self.assertFalse(raw["completed"])
        self.assertEqual(raw["status"], "PARTIAL_NO_INCUMBENT")
        self.assertEqual(raw["current_time"], 0)
        self.assertEqual(raw["schedule"], {})
        self.assertEqual(disposed, [True])
        self.assertIsNone(raw["windows"][0]["objective"])

    def test_bad_runtime_rejected(self):
        for value in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                solve_flexible_rolling(tiny_config(), [], RollingOptions(), value)


class FlexibleRollingMicroTests(unittest.TestCase):
    """Exactly three <=2-second RH runs; the PI alone executes this class."""

    @staticmethod
    def _run(prune, *, empty=False):
        from intercity_delivery.v1.flexible_runner import run_flexible_rolling
        if empty:
            cfg = tiny_config(T=5, N_manual={1: 1, 2: 0},
                              N_auto={1: 0, 2: 0}, c_empty_hour=0.5)
            order = one_order(flow="-", earliest_start=1, latest_completion=5)
            options = RollingOptions(2, 1, 2, prune)
        else:
            cfg = tiny_config(T=3, c_direct_hour=100)
            order = one_order(latest_completion=3)
            options = RollingOptions(1, 1, 2, prune)
        return run_flexible_rolling(cfg, [order], options, 2)

    def test_unpruned_three_leg_cross_window(self):
        payload = self._run(False)
        self.assertTrue(payload["completed"])
        self.assertAlmostEqual(payload["total_cost"], 3)
        self.assertIsNone(payload["best_bound"])
        self.assertEqual(payload["windows"][0]["known_order_ids"], ["A"])

    def test_pruned_three_leg_has_same_hand_cost_and_real_removal(self):
        payload = self._run(True)
        self.assertTrue(payload["completed"])
        self.assertAlmostEqual(payload["total_cost"], 3)
        self.assertGreater(sum(window["pruning"]["removed"] for window in payload["windows"]), 0)

    def test_pruned_empty_then_direct_future_order_hidden(self):
        payload = self._run(True, empty=True)
        self.assertTrue(payload["completed"])
        # A strictly cheaper independent empty movement removes the tie with
        # the manuscript's zero-cargo tau-length direct-vehicle special case.
        self.assertAlmostEqual(payload["total_cost"], 2.5)
        self.assertEqual(payload["windows"][0]["known_order_ids"], [])
        self.assertEqual(payload["windows"][1]["known_order_ids"], ["A"])
        self.assertTrue(any(value > 0 for value in payload["schedule"]["empty_vehicles"].values()))


if __name__ == "__main__":
    unittest.main()
