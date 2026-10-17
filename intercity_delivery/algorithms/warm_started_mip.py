"""Algorithm 2: use the BHH priority construction as a reduced-MIP start."""

from __future__ import annotations

import math
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

    def __init__(self):
        self._last_mip_time_sec = 0.0

    @staticmethod
    def _apply_start(
        optimizer,
        heuristic: PaperWindowSolution,
        context: PaperWindowContext,
    ) -> int:
        assigned = 0
        for group, variables in optimizer.decision_variable_groups().items():
            starts = heuristic.decisions.get(group, {})
            for key, value in starts.items():
                key = tuple(key)
                if abs(value) > 1e-8 and key in variables:
                    variables[key].Start = float(value)
                    assigned += 1

        for order_id, variable in optimizer.z_unserved.items():
            value = heuristic.unserved_by_order.get(int(order_id), 0.0)
            if value > 1e-8:
                variable.Start = float(value)
                assigned += 1
        for order_id, variable in optimizer.q_direct.items():
            value = sum(
                value
                for source in (
                    context.committed.decisions.get("h_direct", {}),
                    heuristic.decisions.get("h_direct", {}),
                )
                for key, value in source.items()
                if int(key[-1]) == int(order_id)
            )
            if value > 1e-8:
                variable.Start = value
                assigned += 1
        for order_id, variable in optimizer.r_transshipment.items():
            value = sum(
                value
                for source in (
                    context.committed.decisions.get("g_auto", {}),
                    heuristic.decisions.get("g_auto", {}),
                )
                for key, value in source.items()
                if int(key[-1]) == int(order_id)
            )
            if value > 1e-8:
                variable.Start = value
                assigned += 1
        optimizer.model.update()
        return assigned

    def solve_window(self, context: PaperWindowContext) -> PaperWindowSolution:
        preprocess_started = time.time()
        network_started = time.time()
        network = StateDependentCandidateGenerator(context).generate()
        network_time = time.time() - network_started
        optimizer = ReducedFlexibleDirectOptimizer(
            context.config,
            network.data,
            network.direct_arcs,
            first_stage_start_end=context.start_end,
        ).build_model()
        optimizer.fix_committed_history(
            context.current_time, context.committed.decisions
        )

        remaining_windows = max(
            1,
            math.ceil(
                (context.config.T - context.current_time)
                / context.algorithm_config.rolling_step
            ),
        )
        budget_per_window = context.remaining_time / remaining_windows
        trigger_threshold = (
            context.algorithm_config.heuristic_trigger_time_sec
        )
        previous_mip_time = self._last_mip_time_sec
        trigger_reason = None
        if previous_mip_time > trigger_threshold:
            trigger_reason = "previous_window_mip_time"
        elif (
            previous_mip_time > 0.0
            and budget_per_window <= trigger_threshold
            and previous_mip_time >= 0.5 * budget_per_window
        ):
            trigger_reason = "tight_average_window_budget"

        heuristic = None
        heuristic_time = 0.0
        partial_start_values = 0
        heuristic_budget = 0.0
        if trigger_reason is not None:
            elapsed_before_heuristic = time.time() - preprocess_started
            heuristic_budget = min(
                trigger_threshold,
                max(
                    0.0,
                    context.remaining_time - elapsed_before_heuristic - 0.5,
                ),
            )
        if trigger_reason is not None and heuristic_budget > 0.01:
            heuristic_started = time.time()
            heuristic = DynamicBHHPriorityApproach().solve_window(
                context,
                network=network,
                max_time_sec=heuristic_budget,
            )
            heuristic_time = time.time() - heuristic_started
            partial_start_values = self._apply_start(
                optimizer, heuristic, context
            )

        optimizer.model.setParam("OutputFlag", context.output_flag)
        preprocessing_time = time.time() - preprocess_started
        optimizer.model.setParam(
            "TimeLimit",
            max(0.01, context.remaining_time - preprocessing_time),
        )
        mip_started = time.time()
        optimizer.model.optimize()
        history_retry_triggered = False
        history_retry_time = 0.0
        if (
            optimizer.model.SolCount <= 0
            and optimizer.model.Status == 3
        ):
            history_retry_triggered = True
            optimizer.relax_committed_flow_history(
                context.current_time,
                context.committed.decisions,
            )
            optimizer.model.reset()
            optimizer.model.setParam(
                "TimeLimit",
                max(
                    0.01,
                    context.remaining_time
                    - (time.time() - preprocess_started),
                ),
            )
            retry_started = time.time()
            optimizer.model.optimize()
            history_retry_time = time.time() - retry_started
        mip_time = time.time() - mip_started
        self._last_mip_time_sec = mip_time

        diagnostics: Dict[str, object] = {
            **network.diagnostics,
            "history_tolerance_retry_triggered": history_retry_triggered,
            "history_tolerance_retry_time_sec": round(
                history_retry_time, 6
            ),
            "heuristic_triggered": heuristic is not None,
            "heuristic_trigger_reason": trigger_reason or "easy_window",
            "heuristic_trigger_time_sec": trigger_threshold,
            "average_remaining_window_budget_sec": budget_per_window,
            "previous_window_mip_time_sec": previous_mip_time,
            "network_generation_time_sec": round(network_time, 6),
            "preprocessing_time_sec": round(preprocessing_time, 6),
            "mip_solve_time_sec": round(mip_time, 6),
            "partial_start_values": partial_start_values,
            "heuristic_budget_sec": round(heuristic_budget, 6),
            "heuristic_start_objective": (
                heuristic.objective if heuristic is not None else None
            ),
            "heuristic_start_time_sec": round(heuristic_time, 6),
            "heuristic_start_unserved": (
                sum(heuristic.unserved_by_order.values())
                if heuristic is not None else None
            ),
            "heuristic_start_diagnostics": (
                heuristic.diagnostics if heuristic is not None else None
            ),
            "first_stage_restricted_variables": getattr(
                optimizer, "first_stage_restricted_variables", 0
            ),
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
                    "Adaptive reduced MILP found no feasible solution "
                    f"(status {optimizer.model.Status})."
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
            message=(
                "Economic partial start + reduced MILP solved."
                if heuristic is not None
                else "Easy window skipped heuristic; reduced MILP solved."
            ),
        )
