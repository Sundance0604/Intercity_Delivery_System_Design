"""Algorithm 3: adaptive local branching followed by the full reduced MILP."""

from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Tuple

import gurobipy as gp
from gurobipy import GRB

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
from intercity_delivery.algorithms.warm_started_mip import (
    WarmStartedStateDependentMIPApproach,
)
from intercity_delivery.models.gurobi_results import optional_model_float


class LocalBranchingStateDependentMIPApproach:
    """Search near Algorithm 2's vehicle plan, then solve the unrestricted A1 MIP."""

    name = "paper_local_branching"
    _vehicle_groups = ("x_manual", "y_auto", "w_direct")

    def __init__(self):
        self._last_mip_time_sec = 0.0

    @staticmethod
    def _trigger_reason(
        previous_mip_time: float,
        trigger_threshold: float,
        budget_per_window: float,
    ) -> str | None:
        if previous_mip_time > trigger_threshold:
            return "previous_window_mip_time"
        if (
            previous_mip_time > 0.0
            and budget_per_window <= trigger_threshold
            and previous_mip_time >= 0.5 * budget_per_window
        ):
            return "tight_average_window_budget"
        return None

    @staticmethod
    def _center_value(
        group: str,
        key: tuple,
        heuristic: PaperWindowSolution,
        context: PaperWindowContext,
    ) -> int:
        value = sum(
            float(source.get(key, 0.0))
            for source in (
                context.committed.decisions.get(group, {}),
                heuristic.decisions.get(group, {}),
            )
        )
        return max(0, int(round(value)))

    @classmethod
    def _add_integer_l1_neighborhood(
        cls,
        optimizer: ReducedFlexibleDirectOptimizer,
        heuristic: PaperWindowSolution,
        context: PaperWindowContext,
        radius: int,
    ) -> Tuple[List[object], List[object], int, int, int]:
        """Add sum |vehicle - heuristic vehicle| <= radius for future arcs."""

        deviations: List[object] = []
        constraints: List[object] = []
        nonzero_centers = 0
        included_variables = 0
        center_vehicle_units = 0
        for group in cls._vehicle_groups:
            for key, variable in optimizer.decision_variable_groups()[group].items():
                key = tuple(key)
                if int(key[0]) < context.current_time:
                    continue
                center = cls._center_value(group, key, heuristic, context)
                if center <= 0 or variable.UB <= 1e-9:
                    continue
                center_vehicle_units += center
                if center:
                    nonzero_centers += 1
                deviation = optimizer.model.addVar(
                    lb=0.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"A3_dev_{included_variables}",
                )
                constraints.append(
                    optimizer.model.addConstr(
                        deviation >= variable - center,
                        name=f"A3_dev_pos_{included_variables}",
                    )
                )
                constraints.append(
                    optimizer.model.addConstr(
                        deviation >= center - variable,
                        name=f"A3_dev_neg_{included_variables}",
                    )
                )
                deviations.append(deviation)
                included_variables += 1

        effective_radius = min(
            radius,
            max(1, math.ceil(0.25 * center_vehicle_units)),
        )
        neighborhood = optimizer.model.addConstr(
            gp.quicksum(deviations) <= effective_radius,
            name="A3_integer_L1_neighborhood",
        )
        constraints.append(neighborhood)
        optimizer.model.update()
        return (
            deviations,
            constraints,
            nonzero_centers,
            included_variables,
            effective_radius,
        )

    @staticmethod
    def _core_variables(optimizer) -> Iterable[object]:
        for variables in optimizer.decision_variable_groups().values():
            yield from variables.values()
        yield from optimizer.r_transshipment.values()
        yield from optimizer.q_direct.values()
        yield from optimizer.z_unserved.values()

    @classmethod
    def _capture_incumbent(cls, optimizer) -> List[Tuple[object, float]]:
        return [
            (variable, float(variable.X))
            for variable in cls._core_variables(optimizer)
        ]

    @staticmethod
    def _apply_complete_start(
        optimizer,
        incumbent: List[Tuple[object, float]],
    ) -> int:
        for variable, value in incumbent:
            variable.Start = value
        optimizer.model.update()
        return len(incumbent)

    @staticmethod
    def _remove_neighborhood(
        optimizer,
        deviations: List[object],
        constraints: List[object],
    ) -> None:
        optimizer.model.remove(constraints)
        optimizer.model.remove(deviations)
        optimizer.model.update()

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
            context.current_time,
            context.committed.decisions,
        )
        optimizer.model.setParam("OutputFlag", context.output_flag)

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
        trigger_reason = self._trigger_reason(
            previous_mip_time,
            trigger_threshold,
            budget_per_window,
        )

        heuristic = None
        heuristic_time = 0.0
        heuristic_budget = 0.0
        partial_start_values = 0
        local_budget = 0.0
        local_time = 0.0
        local_status = None
        local_solution_count = 0
        local_objective = None
        local_best_bound = None
        local_mip_gap = None
        neighborhood_variables = 0
        neighborhood_nonzero_centers = 0
        effective_radius = 0
        local_incumbent_transferred = False
        transferred_start_values = 0

        if trigger_reason is not None:
            elapsed = time.time() - preprocess_started
            heuristic_budget = min(
                trigger_threshold,
                max(0.0, context.remaining_time - elapsed - 0.5),
            )
        if trigger_reason is not None and heuristic_budget > 0.01:
            heuristic_started = time.time()
            heuristic = DynamicBHHPriorityApproach().solve_window(
                context,
                network=network,
                max_time_sec=heuristic_budget,
            )
            heuristic_time = time.time() - heuristic_started
            partial_start_values = (
                WarmStartedStateDependentMIPApproach._apply_start(
                    optimizer,
                    heuristic,
                    context,
                )
            )

            available = max(
                0.0,
                context.remaining_time
                - (time.time() - preprocess_started)
                - 0.5,
            )
            local_budget = min(
                context.algorithm_config.local_branching_max_time_sec,
                (
                    context.algorithm_config.local_branching_time_fraction
                    * available
                ),
            )

        local_incumbent: List[Tuple[object, float]] = []
        if heuristic is not None and local_budget > 0.01:
            (
                deviations,
                local_constraints,
                neighborhood_nonzero_centers,
                neighborhood_variables,
                effective_radius,
            ) = self._add_integer_l1_neighborhood(
                optimizer,
                heuristic,
                context,
                context.algorithm_config.local_branching_radius,
            )
            optimizer.model.setParam("TimeLimit", max(0.01, local_budget))
            local_started = time.time()
            optimizer.model.optimize()
            local_time = time.time() - local_started
            local_status = int(optimizer.model.Status)
            local_solution_count = int(optimizer.model.SolCount)
            if local_solution_count > 0:
                local_objective = float(optimizer.model.ObjVal)
                local_best_bound = optional_model_float(
                    optimizer.model,
                    "ObjBound",
                )
                local_mip_gap = optional_model_float(
                    optimizer.model,
                    "MIPGap",
                )
                local_incumbent = self._capture_incumbent(optimizer)

            self._remove_neighborhood(
                optimizer,
                deviations,
                local_constraints,
            )
            optimizer.model.reset()
            if local_incumbent:
                transferred_start_values = self._apply_complete_start(
                    optimizer,
                    local_incumbent,
                )
                local_incumbent_transferred = True
            else:
                partial_start_values = (
                    WarmStartedStateDependentMIPApproach._apply_start(
                        optimizer,
                        heuristic,
                        context,
                    )
                )

        preprocessing_time = time.time() - preprocess_started
        optimizer.model.setParam(
            "TimeLimit",
            max(0.01, context.remaining_time - preprocessing_time),
        )
        full_mip_started = time.time()
        optimizer.model.optimize()
        history_retry_triggered = False
        history_retry_time = 0.0
        if optimizer.model.SolCount <= 0 and optimizer.model.Status == 3:
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
        full_mip_time = time.time() - full_mip_started
        self._last_mip_time_sec = full_mip_time

        diagnostics: Dict[str, object] = {
            **network.diagnostics,
            "history_tolerance_retry_triggered": (
                history_retry_triggered
            ),
            "history_tolerance_retry_time_sec": round(
                history_retry_time,
                6,
            ),
            "local_branching_triggered": (
                heuristic is not None and local_budget > 0.01
            ),
            "local_branching_trigger_reason": (
                trigger_reason or "easy_window"
            ),
            "local_branching_radius": (
                context.algorithm_config.local_branching_radius
            ),
            "local_branching_time_fraction": (
                context.algorithm_config.local_branching_time_fraction
            ),
            "local_branching_effective_radius": effective_radius,
            "local_branching_budget_sec": round(local_budget, 6),
            "local_branching_time_sec": round(local_time, 6),
            "local_branching_status": local_status,
            "local_branching_solution_count": local_solution_count,
            "local_branching_objective": local_objective,
            "local_branching_best_bound": local_best_bound,
            "local_branching_mip_gap": local_mip_gap,
            "local_branching_neighborhood_variables": (
                neighborhood_variables
            ),
            "local_branching_nonzero_centers": (
                neighborhood_nonzero_centers
            ),
            "local_incumbent_transferred": (
                local_incumbent_transferred
            ),
            "transferred_start_values": transferred_start_values,
            "heuristic_trigger_time_sec": trigger_threshold,
            "average_remaining_window_budget_sec": budget_per_window,
            "previous_window_mip_time_sec": previous_mip_time,
            "heuristic_budget_sec": round(heuristic_budget, 6),
            "heuristic_start_time_sec": round(heuristic_time, 6),
            "heuristic_start_objective": (
                heuristic.objective if heuristic is not None else None
            ),
            "heuristic_start_diagnostics": (
                heuristic.diagnostics if heuristic is not None else None
            ),
            "partial_start_values": partial_start_values,
            "network_generation_time_sec": round(network_time, 6),
            "preprocessing_time_sec": round(preprocessing_time, 6),
            "full_mip_solve_time_sec": round(full_mip_time, 6),
            "mip_solve_time_sec": round(
                local_time + full_mip_time,
                6,
            ),
            "first_stage_restricted_variables": getattr(
                optimizer,
                "first_stage_restricted_variables",
                0,
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
                    "Algorithm 3 full reduced MILP found no feasible "
                    f"solution (status {optimizer.model.Status})."
                ),
            )

        direct_volume = sum(
            variable.X for variable in optimizer.q_direct.values()
        )
        transshipment_volume = sum(
            variable.X
            for variable in optimizer.r_transshipment.values()
        )
        return PaperWindowSolution(
            feasible=True,
            status=optimizer.model.Status,
            objective=float(optimizer.model.ObjVal),
            best_bound=optional_model_float(optimizer.model, "ObjBound"),
            mip_gap=optional_model_float(optimizer.model, "MIPGap"),
            decisions=StateDependentMIPApproach._extract_decisions(
                optimizer
            ),
            unserved_by_order={
                int(order_id): float(variable.X)
                for order_id, variable in optimizer.z_unserved.items()
            },
            direct_volume=direct_volume,
            transshipment_volume=transshipment_volume,
            diagnostics=diagnostics,
            message=(
                "Algorithm 3 local branching + full reduced MILP solved."
                if diagnostics["local_branching_triggered"]
                else "Easy window skipped A3; full reduced MILP solved."
            ),
        )
