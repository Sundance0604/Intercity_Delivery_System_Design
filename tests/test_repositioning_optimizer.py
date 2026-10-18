"""Tiny deterministic MILPs. Six solves, each at most 2 s and one thread."""

import json
import unittest

from intercity_delivery.configuration import DeliveryConfig
from intercity_delivery.data.loader import DataLoader, DeliveryData, OrderBatch
from intercity_delivery.repositioning.config import RepositioningPolicy
from intercity_delivery.repositioning.optimizer import RepositioningOptimizer
from intercity_delivery.repositioning.ledger import evaluate_committed
from intercity_delivery.repositioning.runner import run_full


def tiny_instance(orders=(), **kwargs):
    parameters = dict(
        T=5, t_0=60.0, travel_time_periods=1, direct_travel_time_periods=1,
        N_manual={1: 1, 2: 0}, N_auto={1: 0, 2: 0},
        cost_direct=1.0, service_a_1=1.0, service_a_2=1.0,
        service_b_1=0.0, service_b_2=0.0, capacity_manual=1.0,
    )
    parameters.update(kwargs)
    cfg = DeliveryConfig(**parameters)
    loader = DataLoader(cfg)
    m1, m2 = loader.generate_arcs_manual()
    auto = loader.generate_arcs_auto()
    s1, s2, sa = loader.generate_sets(m1, m2, auto)
    c1, c2 = loader.pre_inverse_count(m1, m2)
    all_orders = {order.batch_id: order for order in orders}
    pos = {key: order for key, order in all_orders.items() if order.flow == "+"}
    neg = {key: order for key, order in all_orders.items() if order.flow == "-"}
    return cfg, DeliveryData(
        m1, m2, auto, s1, s2, sa, c1, c2, pos, neg, all_orders,
        loader.generate_epsilon_sets(pos, neg, m1, m2),
    )


def solve_micro(optimizer):
    optimizer.model.setParam("OutputFlag", 0)
    optimizer.model.setParam("TimeLimit", 2)
    optimizer.model.setParam("Threads", 1)
    optimizer.model.optimize()


class RepositioningOptimizerTests(unittest.TestCase):
    def test_empty_move_without_any_order_or_window_return(self):
        cfg, data = tiny_instance()
        opt = RepositioningOptimizer(cfg, data).build_model()
        try:
            opt.configure_event_window(0, 1, 2, {})
            opt.empty_vehicles[0, 1, "+"].lb = 1
            solve_micro(opt)
            self.assertGreater(opt.model.SolCount, 0)
            self.assertEqual(opt.empty_vehicles[0, 1, "+"].VType, "I")
            self.assertAlmostEqual(opt.model.ObjVal, 1.0)
            self.assertAlmostEqual(opt._city_idle(2, 2).getValue(), 1.0)
        finally:
            opt.model.dispose()

    def test_real_target_restores_at_real_T_with_independent_empty_return(self):
        cfg, data = tiny_instance()
        opt = RepositioningOptimizer(
            cfg, data, RepositioningPolicy(terminal_target={1: 1, 2: 0})
        ).build_model()
        try:
            opt.empty_vehicles[0, 1, "+"].lb = 1
            solve_micro(opt)
            self.assertGreater(opt.model.SolCount, 0)
            self.assertAlmostEqual(opt._city_idle(1, cfg.T).getValue(), 1.0)
            self.assertAlmostEqual(sum(v.X for (i, j, f), v in opt.empty_vehicles.items()
                                       if f == "-"), 1.0)
        finally:
            opt.model.dispose()

    def test_completed_move_survives_next_window_inventory(self):
        cfg, data = tiny_instance()
        opt = RepositioningOptimizer(cfg, data).build_model()
        try:
            opt.configure_event_window(2, 3, 4, {"empty_vehicles": {(0, 1, "+"): 1}})
            opt.x_manual[2, 3, 2, "+"].lb = 1
            solve_micro(opt)
            self.assertGreater(opt.model.SolCount, 0)
            self.assertAlmostEqual(opt.empty_vehicles[0, 1, "+"].X, 1.0)
            self.assertAlmostEqual(opt._city_idle(2, 2).getValue(), 0.0)
        finally:
            opt.model.dispose()

    def test_in_transit_vehicle_cannot_be_used_before_arrival(self):
        cfg, data = tiny_instance()
        opt = RepositioningOptimizer(
            cfg, data, RepositioningPolicy(empty_travel_time_periods=2)
        ).build_model()
        try:
            opt.configure_event_window(1, 3, 4, {"empty_vehicles": {(0, 2, "+"): 1}})
            opt.x_manual[1, 2, 2, "+"].lb = 1
            solve_micro(opt)
            self.assertEqual(opt.model.SolCount, 0)
        finally:
            opt.model.dispose()

    def test_fractional_small_delivery_is_not_dropped(self):
        cfg, data = tiny_instance([OrderBatch(1, "+", 0.05, 0, 4, 1000.0)])
        opt = RepositioningOptimizer(cfg, data).build_model()
        try:
            # One two-period direct service plus an independent one-period return.
            opt.w_direct[0, 2, "+"].lb = 1
            opt.h_direct[0, 2, "+", 1].lb = 0.05
            opt.empty_vehicles[2, 3, "-"].lb = 1
            solve_micro(opt)
            self.assertGreater(opt.model.SolCount, 0)
            exported = opt.extract_decisions()
            self.assertAlmostEqual(sum(exported["h_direct"].values()), 0.05)
            self.assertAlmostEqual(opt.z_unserved[1].X, 0.0)
            self.assertAlmostEqual(opt.model.ObjVal, 3.0)
            ledger = evaluate_committed(cfg, data, exported, opt.policy, cfg.T, True)
            self.assertAlmostEqual(ledger["total_cost"], opt.model.ObjVal)
        finally:
            opt.model.dispose()

    def test_full_runner_independent_accounting_and_json_export(self):
        # Only city 2 has a car. Delivering city-1 cargo requires an independent
        # one-period empty move followed by a two-period direct service.
        cfg, data = tiny_instance(
            [OrderBatch(1, "+", 0.05, 1, 4, 1000.0)], N_manual={1: 0, 2: 1}
        )
        result = run_full(cfg, data, 2)
        self.assertTrue(result["completed"])
        self.assertAlmostEqual(result["total_cost"], 3.0)
        self.assertAlmostEqual(result["empty_vehicle_trips"], 1.0)
        self.assertAlmostEqual(result["direct_volume"], 0.05)
        self.assertTrue(result["detail"]["accounting_matches_objective"])
        self.assertAlmostEqual(result["detail"]["accounting_gap"], 0.0)
        self.assertEqual(result["detail"]["solution"]["model_order_summaries"]["z_unserved"], {"1": 0.0})
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
