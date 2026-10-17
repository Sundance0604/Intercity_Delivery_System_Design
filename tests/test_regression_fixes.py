import unittest

from intercity_delivery.algorithms.bhh_priority_heuristic import (
    DynamicBHHPriorityApproach,
)
from intercity_delivery.algorithms.paper_rolling_horizon import (
    PaperCommittedState,
    PaperRollingHorizonController,
    PaperWindowContext,
    PaperWindowSolution,
)
from intercity_delivery.algorithms.state_dependent_mip import (
    ReducedFlexibleDirectOptimizer,
    StateDependentCandidateGenerator,
    StateDependentMIPApproach,
)
from intercity_delivery.algorithms.local_branching_mip import (
    LocalBranchingStateDependentMIPApproach,
)
from intercity_delivery.algorithms.warm_started_mip import (
    WarmStartedStateDependentMIPApproach,
)
from intercity_delivery.configuration import (
    DeliveryConfig,
    OrderGenerationConfig,
    RollingHorizonConfig,
)
from intercity_delivery.experiments.core import (
    build_delivery_data,
    generate_random_orders,
)
from intercity_delivery.data.loader import OrderBatch
from intercity_delivery.experiments.solvers import SOLVER_REGISTRY


class RegressionFixTests(unittest.TestCase):
    def _long_haul_context(
        self,
        penalty: float = 1000.0,
        remaining_time: float = 30.0,
        heuristic_trigger_time_sec: float = 1.0,
    ):
        config = DeliveryConfig(
            T=24,
            travel_time_periods=9,
            direct_travel_time_periods=9,
            penalty_lost=penalty,
            direct_ratio_max=0.0,
            N_manual={1: 4, 2: 4},
            N_auto={1: 2, 2: 2},
        )
        order = OrderBatch(
            batch_id=1,
            flow="+",
            quantity=10,
            earliest_start=0,
            latest_completion=14,
            penalty_lost=penalty,
        )
        pos_orders = {1: order}
        orders = (pos_orders, {}, dict(pos_orders))
        data = build_delivery_data(config, orders)
        algorithm_config = RollingHorizonConfig(
            prediction_horizon=8,
            rolling_step=2,
            extension_horizon=6,
            heuristic_trigger_time_sec=heuristic_trigger_time_sec,
        )
        context = PaperWindowContext(
            config=config,
            algorithm_config=algorithm_config,
            full_data=data,
            all_orders=orders[2],
            known_order_ids=frozenset({1}),
            current_time=0,
            start_end=8,
            completion_end=14,
            control_end=2,
            committed=PaperCommittedState(),
            remaining_time=remaining_time,
        )
        return config, orders, data, context

    def test_extended_downstream_arcs_restore_path_when_h_is_below_tau(self):
        config, _orders, _data, context = self._long_haul_context()
        network = StateDependentCandidateGenerator(context).generate()

        self.assertTrue(
            network.diagnostics[
                "transshipment_path_available_by_order"
            ][1]
        )
        self.assertTrue(
            any(i >= context.start_end for i, _j in network.data.arcs_manual_2)
        )
        optimizer = ReducedFlexibleDirectOptimizer(
            config,
            network.data,
            network.direct_arcs,
            first_stage_start_end=context.start_end,
        ).build_model()
        late_origin = [
            variable
            for key, variable in optimizer.g_manual.items()
            if key[2] == 1 and key[3] == "+" and key[0] >= context.start_end
        ]
        late_destination = [
            variable
            for key, variable in optimizer.g_manual.items()
            if key[2] == 2 and key[3] == "+" and key[0] >= context.start_end
        ]
        self.assertTrue(late_origin)
        self.assertTrue(late_destination)
        self.assertTrue(all(variable.ub == 0.0 for variable in late_origin))
        self.assertTrue(any(variable.ub > 0.0 for variable in late_destination))

        solution = StateDependentMIPApproach().solve_window(context)
        self.assertTrue(solution.feasible)
        self.assertGreater(solution.transshipment_volume, 0.0)
        self.assertEqual(solution.direct_volume, 0.0)
        self.assertLess(solution.unserved_by_order[1], 10.0)

    def test_economic_rule_rejects_service_costlier_than_penalty(self):
        _config, _orders, _data, context = self._long_haul_context(
            penalty=0.1
        )
        network = StateDependentCandidateGenerator(context).generate()
        solution = DynamicBHHPriorityApproach().solve_window(
            context, network=network
        )

        self.assertEqual(solution.unserved_by_order[1], 10.0)
        self.assertEqual(
            solution.diagnostics["economically_rejected_orders"], 1
        )
        self.assertEqual(solution.diagnostics["priority_iterations"], 0)

    def test_adaptive_heuristic_skips_easy_and_triggers_tight_window(self):
        _config, _orders, _data, easy_context = self._long_haul_context(
            remaining_time=30.0,
            heuristic_trigger_time_sec=1.0,
        )
        easy = WarmStartedStateDependentMIPApproach().solve_window(
            easy_context
        )
        self.assertFalse(easy.diagnostics["heuristic_triggered"])
        self.assertEqual(easy.diagnostics["partial_start_values"], 0)

        _config, _orders, _data, tight_context = self._long_haul_context(
            remaining_time=1.5,
            heuristic_trigger_time_sec=1.0,
        )
        first_tight = WarmStartedStateDependentMIPApproach().solve_window(
            tight_context
        )
        self.assertFalse(first_tight.diagnostics["heuristic_triggered"])
        tight_approach = WarmStartedStateDependentMIPApproach()
        tight_approach._last_mip_time_sec = 0.4
        tight = tight_approach.solve_window(tight_context)
        self.assertTrue(tight.diagnostics["heuristic_triggered"])
        self.assertEqual(
            tight.diagnostics["heuristic_trigger_reason"],
            "tight_average_window_budget",
        )
        self.assertGreater(tight.diagnostics["partial_start_values"], 0)

    def test_a3_local_branching_transfers_incumbent_then_unrestricts(self):
        _config, _orders, _data, context = self._long_haul_context(
            remaining_time=5.0,
            heuristic_trigger_time_sec=1.0,
        )
        approach = LocalBranchingStateDependentMIPApproach()
        approach._last_mip_time_sec = 2.0

        solution = approach.solve_window(context)

        self.assertTrue(solution.feasible)
        diagnostics = solution.diagnostics
        self.assertTrue(diagnostics["local_branching_triggered"])
        self.assertEqual(
            diagnostics["local_branching_trigger_reason"],
            "previous_window_mip_time",
        )
        self.assertGreater(
            diagnostics["local_branching_neighborhood_variables"],
            0,
        )
        self.assertGreater(
            diagnostics["local_branching_solution_count"],
            0,
        )
        self.assertTrue(diagnostics["local_incumbent_transferred"])
        self.assertLessEqual(
            solution.objective,
            diagnostics["local_branching_objective"] + 1e-6,
        )

    def test_heuristic_respects_a_hard_construction_budget(self):
        _config, _orders, _data, context = self._long_haul_context(
            remaining_time=30.0,
            heuristic_trigger_time_sec=1.0,
        )
        solution = DynamicBHHPriorityApproach().solve_window(
            context,
            max_time_sec=0.0,
        )
        self.assertTrue(
            solution.diagnostics["heuristic_time_limit_reached"]
        )
        self.assertEqual(solution.diagnostics["priority_iterations"], 0)
        self.assertEqual(solution.diagnostics["ranked_orders_before_limit"], 0)

    def test_control_window_drops_numerical_flow_dust(self):
        solution = PaperWindowSolution(
            feasible=True,
            status=2,
            objective=0.0,
            decisions={
                "x_manual": {
                    (3, 4, 2, "-"): 0.9999997,
                },
                "g_manual": {
                    (3, 4, 2, "-", 1): 7.2e-7,
                    (3, 4, 2, "-", 2): 5.00009,
                    (4, 5, 2, "-", 3): 6.0,
                },
            },
            unserved_by_order={},
        )

        committed = PaperRollingHorizonController._commit_control_window(
            solution, 2, 4
        )

        self.assertEqual(
            committed["x_manual"][(3, 4, 2, "-")], 1.0
        )
        self.assertNotIn(
            (3, 4, 2, "-", 1), committed["g_manual"]
        )
        self.assertEqual(
            committed["g_manual"][(3, 4, 2, "-", 2)], 5.00009
        )
        self.assertNotIn(
            (4, 5, 2, "-", 3), committed["g_manual"]
        )

    def test_history_retry_relaxes_only_committed_flows(self):
        config, _orders, _data, context = self._long_haul_context()
        network = StateDependentCandidateGenerator(context).generate()
        optimizer = ReducedFlexibleDirectOptimizer(
            config,
            network.data,
            network.direct_arcs,
            first_stage_start_end=context.start_end,
        ).build_model()
        eligible = [
            key
            for key, variable in optimizer.g_manual.items()
            if key[0] < 2 and variable.ub > 0.0
        ]
        committed_key, missing_key = eligible[:2]
        history = {"g_manual": {committed_key: 5.0}}

        optimizer.fix_committed_history(2, history)
        self.assertEqual(optimizer.g_manual[committed_key].lb, 5.0)
        self.assertEqual(optimizer.g_manual[committed_key].ub, 5.0)
        self.assertEqual(optimizer.g_manual[missing_key].lb, 0.0)
        self.assertEqual(optimizer.g_manual[missing_key].ub, 0.0)

        optimizer.relax_committed_flow_history(2, history)
        self.assertAlmostEqual(
            optimizer.g_manual[committed_key].lb, 4.99999
        )
        self.assertAlmostEqual(
            optimizer.g_manual[committed_key].ub, 5.00001
        )
        self.assertEqual(optimizer.g_manual[missing_key].lb, 0.0)
        self.assertEqual(optimizer.g_manual[missing_key].ub, 0.0)

    def test_vehicle_rates_are_converted_from_minutes_to_hours(self):
        self.assertEqual(DeliveryConfig(t_0=30).period_hours, 0.5)
        self.assertEqual(DeliveryConfig(t_0=90).period_hours, 1.5)

    def test_heuristic_solution_is_used_as_reduced_mip_start(self):
        config = DeliveryConfig(T=16, penalty_lost=100)
        order_config = OrderGenerationConfig(
            num_orders=4, buffer_range=(4, 6)
        )
        orders = generate_random_orders(config, order_config, seed=7)
        data = build_delivery_data(config, orders)
        result = SOLVER_REGISTRY["paper_priority_heuristic"].solve(
            config,
            data,
            orders,
            30,
            RollingHorizonConfig(
                prediction_horizon=10,
                rolling_step=2,
                extension_horizon=6,
            ),
        )

        self.assertIsNotNone(result.total_cost)
        for window in result.detail["windows"]:
            diagnostics = window["diagnostics"]
            self.assertIn("heuristic_start_objective", diagnostics)
            if diagnostics["heuristic_start_objective"] is not None:
                self.assertLessEqual(
                    window["objective"],
                    diagnostics["heuristic_start_objective"] + 1e-6,
                )

    def test_full_window_baseline_handles_travel_longer_than_start_window(self):
        config = DeliveryConfig(
            T=24,
            travel_time_periods=9,
            direct_travel_time_periods=9,
            penalty_lost=1000,
        )
        order_config = OrderGenerationConfig(
            num_orders=4, buffer_range=(4, 5)
        )
        orders = generate_random_orders(config, order_config, seed=11)
        data = build_delivery_data(config, orders)
        result = SOLVER_REGISTRY["flexible_direct_rolling"].solve(
            config,
            data,
            orders,
            30,
            RollingHorizonConfig(
                prediction_horizon=8,
                rolling_step=2,
                extension_horizon=6,
            ),
        )

        self.assertIsNotNone(result.total_cost)
        self.assertLess(result.unserved_rate, 1.0)
        self.assertIsNotNone(result.direct_ratio)
        total_demand = sum(order.quantity for order in orders[2].values())
        committed = result.detail["solution"]["committed_decisions"]
        flow_values = [
            value
            for group in ("g_manual", "g_auto", "h_direct")
            for value in committed[group].values()
        ]
        self.assertTrue(flow_values)
        self.assertLessEqual(max(flow_values), total_demand)


if __name__ == "__main__":
    unittest.main()
