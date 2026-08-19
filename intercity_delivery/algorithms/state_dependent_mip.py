"""论文 Algorithm 1：状态相关候选弧生成与削减 MILP。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from intercity_delivery.data.loader import DataLoader, DeliveryData
from intercity_delivery.models.flexible_direct_optimizer import FlexibleDirectOptimizer
from intercity_delivery.algorithms.paper_rolling_horizon import PaperWindowContext, PaperWindowSolution
from intercity_delivery.models.gurobi_results import optional_model_float


Arc = Tuple[int, int]


@dataclass(frozen=True)
class CandidateNetwork:
    data: DeliveryData
    direct_arcs: Tuple[Arc, ...]
    manual_availability: Dict[int, Dict[int, float]]
    auto_availability: Dict[int, float]
    diagnostics: dict


class StateDependentCandidateGenerator:
    """按已提交车辆状态生成论文的 A_h、Ahat_h 与 Atilde_h。"""

    def __init__(self, context: PaperWindowContext):
        self.ctx = context
        self.cfg = context.config
        self.known_orders = {
            order_id: context.all_orders[order_id]
            for order_id in context.known_order_ids
        }

    @staticmethod
    def _vehicle_arcs(values: Dict[tuple, float]) -> Set[Arc]:
        return {
            (int(key[0]), int(key[1]))
            for key, value in values.items()
            if value > 1e-8
        }

    def _manual_availability(self) -> Dict[int, Dict[int, float]]:
        availability = {
            city: {t: float(self.cfg.N_manual[city]) for t in range(self.cfg.T)}
            for city in (1, 2)
        }
        for key, value in self.ctx.committed.decisions.get("x_manual", {}).items():
            i, j, city = int(key[0]), int(key[1]), int(key[2])
            for t in range(max(0, i), min(self.cfg.T, j)):
                availability[city][t] -= value

        # 直送车辆离开始发城市并在到达后进入目的城市车队。
        for key, value in self.ctx.committed.decisions.get("w_direct", {}).items():
            i, j, flow = int(key[0]), int(key[1]), str(key[2])
            origin, destination = ((1, 2) if flow == "+" else (2, 1))
            for t in range(max(0, i), self.cfg.T):
                availability[origin][t] -= value
            for t in range(max(0, j), self.cfg.T):
                availability[destination][t] += value
        return availability

    def _auto_availability(self) -> Dict[int, float]:
        availability = {
            t: float(sum(self.cfg.N_auto.values())) for t in range(self.cfg.T)
        }
        for key, value in self.ctx.committed.decisions.get("y_auto", {}).items():
            i, j = int(key[0]), int(key[1])
            for t in range(max(0, i), min(self.cfg.T, j)):
                availability[t] -= value
        return availability

    @staticmethod
    def _positive_over_interval(profile: Dict[int, float], i: int, j: int) -> bool:
        return all(profile.get(t, 0.0) > 1e-8 for t in range(i, j))

    def _manual_arc_relevant(self, city: int, i: int, j: int) -> bool:
        for order in self.known_orders.values():
            origin = 1 if order.flow == "+" else 2
            destination = 2 if order.flow == "+" else 1
            if city == origin and i >= order.earliest_start and i < order.latest_completion:
                return True
            if city == destination and j <= order.latest_completion:
                return True
        return False

    def _committed_arcs_by_city(self, city: int) -> Set[Arc]:
        arcs = set()
        for group in ("x_manual", "g_manual"):
            for key, value in self.ctx.committed.decisions.get(group, {}).items():
                if value > 1e-8 and int(key[2]) == city:
                    arcs.add((int(key[0]), int(key[1])))
        return arcs

    def generate(self) -> CandidateNetwork:
        loader = DataLoader(self.cfg)
        manual_availability = self._manual_availability()
        auto_availability = self._auto_availability()

        maximum_manual_duration = {}
        for city, service_function in (
            (1, loader.BHH_function_1),
            (2, loader.BHH_function_2),
        ):
            maximum_manual_duration[city] = max(
                1,
                math.ceil(
                    service_function(self.cfg.capacity_manual) / self.cfg.t_0
                ),
            )

        manual_arcs = {1: set(), 2: set()}
        for city in (1, 2):
            for i in range(self.ctx.current_time, self.ctx.start_end):
                last_j = min(
                    self.ctx.completion_end,
                    i + maximum_manual_duration[city],
                )
                for j in range(i + 1, last_j + 1):
                    if not self._positive_over_interval(
                        manual_availability[city], i, j
                    ):
                        continue
                    if self._manual_arc_relevant(city, i, j):
                        manual_arcs[city].add((i, j))
            manual_arcs[city].update(self._committed_arcs_by_city(city))

        auto_arcs = set()
        tau = self.cfg.travel_time_periods
        for i in range(self.ctx.current_time, self.ctx.start_end):
            j = i + tau
            if j <= self.ctx.completion_end and self._positive_over_interval(
                auto_availability, i, j
            ):
                auto_arcs.add((i, j))
        for group in ("y_auto", "g_auto"):
            auto_arcs.update(
                self._vehicle_arcs(self.ctx.committed.decisions.get(group, {}))
            )

        combined_service_minutes = (
            loader.BHH_function_1(self.cfg.capacity_direct)
            + loader.BHH_function_2(self.cfg.capacity_direct)
        )
        maximum_direct_duration = self.cfg.direct_travel_time_periods + max(
            1, math.ceil(combined_service_minutes / self.cfg.t_0)
        )
        direct_arcs = set()
        for i in range(self.ctx.current_time, self.ctx.start_end):
            for j in range(
                i + self.cfg.direct_travel_time_periods + 1,
                min(self.ctx.completion_end, i + maximum_direct_duration) + 1,
            ):
                for order in self.known_orders.values():
                    if not (order.earliest_start <= i and j <= order.latest_completion):
                        continue
                    origin = 1 if order.flow == "+" else 2
                    if self._positive_over_interval(
                        manual_availability[origin], i, j
                    ):
                        direct_arcs.add((i, j))
                        break
        for group in ("w_direct", "h_direct"):
            direct_arcs.update(
                self._vehicle_arcs(self.ctx.committed.decisions.get(group, {}))
            )

        known_pos = {
            order_id: order
            for order_id, order in self.known_orders.items()
            if order.flow == "+"
        }
        known_neg = {
            order_id: order
            for order_id, order in self.known_orders.items()
            if order.flow == "-"
        }
        arcs_1 = sorted(manual_arcs[1])
        arcs_2 = sorted(manual_arcs[2])
        arcs_auto = sorted(auto_arcs)
        sets_1, sets_2, sets_auto = loader.generate_sets(
            arcs_1, arcs_2, arcs_auto
        )
        coeff_1, coeff_2 = loader.pre_inverse_count(arcs_1, arcs_2)
        epsilon = loader.generate_epsilon_sets(
            known_pos, known_neg, arcs_1, arcs_2
        )
        data = DeliveryData(
            arcs_manual_1=arcs_1,
            arcs_manual_2=arcs_2,
            arcs_auto=arcs_auto,
            sets_manual_1=sets_1,
            sets_manual_2=sets_2,
            sets_auto=sets_auto,
            cap_coeff_1=coeff_1,
            cap_coeff_2=coeff_2,
            pos_orders=known_pos,
            neg_orders=known_neg,
            all_orders=self.known_orders,
            epsilon_sets=epsilon,
        )
        full_count = (
            len(self.ctx.full_data.arcs_manual_1)
            + len(self.ctx.full_data.arcs_manual_2)
            + len(self.ctx.full_data.arcs_auto)
        )
        reduced_count = len(arcs_1) + len(arcs_2) + len(arcs_auto)
        diagnostics = {
            "manual_arcs_city_1": len(arcs_1),
            "manual_arcs_city_2": len(arcs_2),
            "auto_arcs": len(arcs_auto),
            "direct_arcs": len(direct_arcs),
            "baseline_non_direct_arcs": full_count,
            "reduced_non_direct_arcs": reduced_count,
            "non_direct_arc_reduction_rate": (
                1.0 - reduced_count / full_count if full_count else 0.0
            ),
        }
        return CandidateNetwork(
            data=data,
            direct_arcs=tuple(sorted(direct_arcs)),
            manual_availability=manual_availability,
            auto_availability=auto_availability,
            diagnostics=diagnostics,
        )


class ReducedFlexibleDirectOptimizer(FlexibleDirectOptimizer):
    """只在 Algorithm 1 生成的弧上创建变量的协同 MILP。"""

    def __init__(self, config, data, direct_arcs: Sequence[Arc]):
        self._candidate_direct_arcs = tuple(direct_arcs)
        super().__init__(config, data)

    def _prepare_indices(self):
        self.arcs_indices = [
            (i, j, city, flow)
            for city, arcs in (
                (1, self.data.arcs_manual_1),
                (2, self.data.arcs_manual_2),
            )
            for i, j in arcs
            for flow in self.flow
        ]
        self.arcs_direct = list(self._candidate_direct_arcs)
        self.direct_capacity_coeff = {}
        tau = self.cfg.direct_travel_time_periods
        for i, j in self.arcs_direct:
            service_minutes = max(0.0, (j - i - tau) * self.cfg.t_0)
            self.direct_capacity_coeff[(i, j)] = min(
                self.cfg.capacity_direct,
                self._inverse_combined_service_time(service_minutes),
            )

    def fix_committed_history(
        self,
        current_time: int,
        committed: Dict[str, Dict[tuple, float]],
    ) -> None:
        for group, variables in self.decision_variable_groups().items():
            history = committed.get(group, {})
            for key, variable in variables.items():
                if int(key[0]) < current_time:
                    value = float(history.get(tuple(key), 0.0))
                    variable.lb = value
                    variable.ub = value
        self.model.update()


class StateDependentMIPApproach:
    """Algorithm 1 候选网络上的滚动 MILP 窗口解法。"""

    name = "paper_candidate_mip"
    _candidate_generator = StateDependentCandidateGenerator

    @staticmethod
    def _extract_decisions(optimizer) -> Dict[str, Dict[tuple, float]]:
        return {
            group: {
                tuple(key): float(variable.X)
                for key, variable in variables.items()
                if abs(variable.X) > 1e-8
            }
            for group, variables in optimizer.decision_variable_groups().items()
        }

    def solve_window(self, context: PaperWindowContext) -> PaperWindowSolution:
        network = self._candidate_generator(context).generate()
        optimizer = ReducedFlexibleDirectOptimizer(
            context.config, network.data, network.direct_arcs
        ).build_model()
        optimizer.fix_committed_history(
            context.current_time, context.committed.decisions
        )
        optimizer.model.setParam("OutputFlag", context.output_flag)
        optimizer.model.setParam("TimeLimit", max(0.01, context.remaining_time))
        optimizer.model.optimize()
        if optimizer.model.SolCount <= 0:
            return PaperWindowSolution(
                feasible=False,
                status=optimizer.model.Status,
                objective=None,
                decisions={},
                unserved_by_order={},
                diagnostics=network.diagnostics,
                message=f"削减 MILP 未找到可行解，状态 {optimizer.model.Status}。",
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
            decisions=self._extract_decisions(optimizer),
            unserved_by_order={
                int(order_id): float(variable.X)
                for order_id, variable in optimizer.z_unserved.items()
            },
            direct_volume=direct_volume,
            transshipment_volume=transshipment_volume,
            diagnostics={
                **network.diagnostics,
                "variables": optimizer.model.NumVars,
                "constraints": optimizer.model.NumConstrs,
                "solution_count": optimizer.model.SolCount,
            },
            message="状态相关候选弧削减 MILP 获得可行解。",
        )
