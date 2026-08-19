"""Algorithm 2: use the BHH priority construction as a reduced-MIP start."""

from __future__ import annotations

import time

from typing import Dict

from intercity_delivery.algorithms.bhh_priority_heuristic import (
    DynamicBHHPriorityApproach,
)
from intercity_delivery.algorithms.paper_rolling_horizon import (
    PaperWindowContext,
    PaperWindowSolution,
)
from intercity_delivery.algorithms.state_dependent_mip import (
    ReducedFlexibleDirectOptimizer,
    StateDependentCandidateGenerator,
    StateDependentMIPApproach,
)
from intercity_delivery.models.gurobi_results import optional_model_float


class WarmStartedStateDependentMIPApproach:
    """Solve the pruned MILP after seeding it with Algorithm 2's solution."""

    name = "paper_priority_heuristic"

    @staticmethod
    def _apply_start(
        optimizer,
        heuristic: PaperWindowSolution,
        context: PaperWindowContext,
    ) -> None:
        for group, variables in optimizer.decision_variable_groups().items():
            starts = heuristic.decisions.get(group, {})
            committed = context.committed.decisions.get(group, {})
            for key, variable in variables.items():
                variable.Start = float(
                    committed.get(tuple(key), starts.get(tuple(key), 0.0))
                )

        for order_id, variable in optimizer.z_unserved.items():
            variable.Start = float(
                heuristic.unserved_by_order.get(int(order_id), 0.0)
            )
        for order_id, variable in optimizer.q_direct.items():
            variable.Start = sum(
                value
                for source in (
                    context.committed.decisions.get("h_direct", {}),
                    heuristic.decisions.get("h_direct", {}),
                )
                for key, value in source.items()
                if int(key[-1]) == int(order_id)
            )
        for order_id, variable in optimizer.r_transshipment.items():
            variable.Start = sum(
                value
                for source in (
                    context.committed.decisions.get("g_auto", {}),
                    heuristic.decisions.get("g_auto", {}),
                )
                for key, value in source.items()
                if int(key[-1]) == int(order_id)
            )
        optimizer.model.update()

    def solve_window(self, context: PaperWindowContext) -> PaperWindowSolution:
        heuristic_started = time.time()
        heuristic = DynamicBHHPriorityApproach().solve_window(context)
        heuristic_time = time.time() - heuristic_started
        network = StateDependentCandidateGenerator(context).generate()
        optimizer = ReducedFlexibleDirectOptimizer(
            context.config, network.data, network.direct_arcs
        ).build_model()
        optimizer.fix_committed_history(
            context.current_time, context.committed.decisions
        )
        self._apply_start(optimizer, heuristic, context)
        optimizer.model.setParam("OutputFlag", context.output_flag)
        optimizer.model.setParam(
            "TimeLimit",
            max(0.01, context.remaining_time - heuristic_time),
        )
        optimizer.model.optimize()

        diagnostics: Dict[str, object] = {
            **network.diagnostics,
            "heuristic_start_objective": heuristic.objective,
            "heuristic_start_time_sec": round(heuristic_time, 6),
            "heuristic_start_unserved": sum(
                heuristic.unserved_by_order.values()
            ),
            "heuristic_start_diagnostics": heuristic.diagnostics,
            "variables": optimizer.model.NumVars,
            "constraints": optimizer.model.NumConstrs,
            "solution_count": optimizer.model.SolCount,
        }
        if optimizer.model.SolCount <= 0:
            return PaperWindowSolution(
                feasible=False,
                status=optimizer.model.Status,
                objective=None,
                decisions={},
                unserved_by_order={},
                diagnostics=diagnostics,
                message=(
                    "BHH priority start was generated, but the reduced MILP "
                    f"found no feasible solution (status {optimizer.model.Status})."
                ),
            )

        direct_volume = sum(variable.X for variable in optimizer.q_direct.values())
        transshipment_volume = sum(
            variable.X for variable in optimizer.r_transshipment.values()
        )
        return PaperWindowSolution(
            feasible=True,
            status=optimizer.model.Status,
            objective=float(optimizer.model.ObjVal),
            best_bound=optional_model_float(optimizer.model, "ObjBound"),
            mip_gap=optional_model_float(optimizer.model, "MIPGap"),
            decisions=StateDependentMIPApproach._extract_decisions(optimizer),
            unserved_by_order={
                int(order_id): float(variable.X)
                for order_id, variable in optimizer.z_unserved.items()
            },
            direct_volume=direct_volume,
            transshipment_volume=transshipment_volume,
            diagnostics=diagnostics,
            message="BHH priority start + state-dependent reduced MILP solved.",
        )
