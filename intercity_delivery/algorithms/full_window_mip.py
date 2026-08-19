"""Unpruned MILP baseline for the paper rolling-horizon controller."""

from __future__ import annotations

from intercity_delivery.algorithms.paper_rolling_horizon import PaperWindowContext
from intercity_delivery.algorithms.state_dependent_mip import (
    CandidateNetwork,
    StateDependentMIPApproach,
)
from intercity_delivery.data.loader import DataLoader, DeliveryData


class FullWindowCandidateGenerator:
    """Keep every time-feasible arc inside the paper start/completion windows."""

    def __init__(self, context: PaperWindowContext):
        self.ctx = context

    @staticmethod
    def _history_arcs(context, groups, city=None):
        arcs = set()
        for group in groups:
            for key, value in context.committed.decisions.get(group, {}).items():
                if value <= 1e-8:
                    continue
                if city is not None and int(key[2]) != city:
                    continue
                arcs.add((int(key[0]), int(key[1])))
        return arcs

    def generate(self) -> CandidateNetwork:
        ctx = self.ctx
        cfg = ctx.config
        loader = DataLoader(cfg)
        known_orders = {
            order_id: ctx.all_orders[order_id]
            for order_id in ctx.known_order_ids
        }

        manual = {}
        for city, source in (
            (1, ctx.full_data.arcs_manual_1),
            (2, ctx.full_data.arcs_manual_2),
        ):
            arcs = {
                (i, j)
                for i, j in source
                if ctx.current_time <= i < ctx.start_end
                and j <= ctx.completion_end
            }
            arcs.update(
                self._history_arcs(ctx, ("x_manual", "g_manual"), city)
            )
            manual[city] = sorted(arcs)

        auto_arcs = {
            (i, j)
            for i, j in ctx.full_data.arcs_auto
            if ctx.current_time <= i < ctx.start_end
            and j <= ctx.completion_end
        }
        auto_arcs.update(
            self._history_arcs(ctx, ("y_auto", "g_auto"))
        )
        auto_arcs = sorted(auto_arcs)

        direct_arcs = {
            (i, j)
            for i in range(ctx.current_time, ctx.start_end)
            for j in range(
                i + cfg.direct_travel_time_periods + 1,
                ctx.completion_end + 1,
            )
            if any(
                order.earliest_start <= i and j <= order.latest_completion
                for order in known_orders.values()
            )
        }
        direct_arcs.update(
            self._history_arcs(ctx, ("w_direct", "h_direct"))
        )

        known_pos = {
            key: order for key, order in known_orders.items() if order.flow == "+"
        }
        known_neg = {
            key: order for key, order in known_orders.items() if order.flow == "-"
        }
        sets_1, sets_2, sets_auto = loader.generate_sets(
            manual[1], manual[2], auto_arcs
        )
        coeff_1, coeff_2 = loader.pre_inverse_count(manual[1], manual[2])
        data = DeliveryData(
            arcs_manual_1=manual[1],
            arcs_manual_2=manual[2],
            arcs_auto=auto_arcs,
            sets_manual_1=sets_1,
            sets_manual_2=sets_2,
            sets_auto=sets_auto,
            cap_coeff_1=coeff_1,
            cap_coeff_2=coeff_2,
            pos_orders=known_pos,
            neg_orders=known_neg,
            all_orders=known_orders,
            epsilon_sets=loader.generate_epsilon_sets(
                known_pos, known_neg, manual[1], manual[2]
            ),
        )
        full_count = (
            len(ctx.full_data.arcs_manual_1)
            + len(ctx.full_data.arcs_manual_2)
            + len(ctx.full_data.arcs_auto)
        )
        window_count = len(manual[1]) + len(manual[2]) + len(auto_arcs)
        return CandidateNetwork(
            data=data,
            direct_arcs=tuple(sorted(direct_arcs)),
            manual_availability={},
            auto_availability={},
            diagnostics={
                "manual_arcs_city_1": len(manual[1]),
                "manual_arcs_city_2": len(manual[2]),
                "auto_arcs": len(auto_arcs),
                "direct_arcs": len(direct_arcs),
                "baseline_non_direct_arcs": full_count,
                "reduced_non_direct_arcs": window_count,
                "non_direct_arc_reduction_rate": 0.0,
                "candidate_policy": "all_time_feasible_window_arcs",
            },
        )


class FullWindowMIPApproach(StateDependentMIPApproach):
    """Same MILP as Algorithm 1, without state-dependent candidate pruning."""

    name = "flexible_direct_rolling"

    _candidate_generator = FullWindowCandidateGenerator
