"""论文 Algorithm 2：动态 BHH-aware 优先级构造启发式。"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from intercity_delivery.algorithms.paper_rolling_horizon import PaperWindowContext, PaperWindowSolution
from intercity_delivery.algorithms.state_dependent_mip import CandidateNetwork, StateDependentCandidateGenerator


@dataclass(frozen=True)
class ServiceOption:
    kind: str
    arcs: Tuple[tuple, ...]
    completion: int


class DynamicBHHPriorityApproach:
    """在 Algorithm 1 候选网络上直接构造可行车辆与货流决策。"""

    name = "paper_priority_heuristic"

    def __init__(self):
        self.decisions = {
            "x_manual": defaultdict(float),
            "y_auto": defaultdict(float),
            "g_manual": defaultdict(float),
            "g_auto": defaultdict(float),
            "w_direct": defaultdict(float),
            "h_direct": defaultdict(float),
        }
        self.manual_load = defaultdict(float)
        self.auto_load = defaultdict(float)
        self.direct_load = defaultdict(float)

    @staticmethod
    def _sum_order_group(
        values: Dict[tuple, float], order_id: int, order_position: int = -1
    ) -> float:
        return sum(
            value
            for key, value in values.items()
            if int(key[order_position]) == order_id
        )

    @staticmethod
    def _profile_capacity(
        profile: Dict[int, float], i: int, j: int
    ) -> int:
        if i >= j:
            return 0
        return max(0, math.floor(min(profile.get(t, 0.0) for t in range(i, j))))

    @staticmethod
    def _consume_profile(
        profile: Dict[int, float], i: int, j: int, vehicles: int
    ) -> None:
        for t in range(i, j):
            profile[t] = profile.get(t, 0.0) - vehicles

    def _available_on_arc(
        self,
        vehicle_key: tuple,
        load_key: tuple,
        capacity: float,
        profile: Dict[int, float],
        vehicle_group: str,
        load_store: Dict[tuple, float],
    ) -> float:
        i, j = int(vehicle_key[0]), int(vehicle_key[1])
        existing_vehicles = self.decisions[vehicle_group][vehicle_key]
        residual = max(0.0, existing_vehicles * capacity - load_store[load_key])
        addable = self._profile_capacity(profile, i, j)
        return residual + addable * capacity

    def _additional_vehicles(
        self,
        quantity: float,
        vehicle_key: tuple,
        load_key: tuple,
        capacity: float,
        vehicle_group: str,
        load_store: Dict[tuple, float],
    ) -> int:
        existing_vehicles = self.decisions[vehicle_group][vehicle_key]
        residual = max(
            0.0, existing_vehicles * capacity - load_store[load_key]
        )
        return max(
            0,
            math.ceil(max(0.0, quantity - residual) / capacity - 1e-12),
        )

    def _allocate_arc(
        self,
        quantity: float,
        vehicle_key: tuple,
        load_key: tuple,
        load_variable_key: tuple,
        capacity: float,
        profile: Dict[int, float],
        vehicle_group: str,
        flow_group: str,
        load_store: Dict[tuple, float],
    ) -> None:
        i, j = int(vehicle_key[0]), int(vehicle_key[1])
        existing_vehicles = self.decisions[vehicle_group][vehicle_key]
        residual = max(0.0, existing_vehicles * capacity - load_store[load_key])
        new_vehicles = max(0, math.ceil(max(0.0, quantity - residual) / capacity))
        if new_vehicles:
            self.decisions[vehicle_group][vehicle_key] += new_vehicles
            self._consume_profile(profile, i, j, new_vehicles)
        load_store[load_key] += quantity
        self.decisions[flow_group][load_variable_key] += quantity

    @staticmethod
    def _origin_destination(order) -> Tuple[int, int]:
        return (1, 2) if order.flow == "+" else (2, 1)

    def _candidate_options(
        self,
        context: PaperWindowContext,
        network: CandidateNetwork,
        order_id: int,
        stage: str,
        deadline: Optional[float] = None,
    ) -> List[ServiceOption]:
        order = context.all_orders[order_id]
        origin, destination = self._origin_destination(order)
        origin_arcs = (
            network.data.arcs_manual_1
            if origin == 1
            else network.data.arcs_manual_2
        )
        destination_arcs = (
            network.data.arcs_manual_1
            if destination == 1
            else network.data.arcs_manual_2
        )
        theta = context.config.transfer_time_periods
        options: List[ServiceOption] = []

        if stage == "shipped":
            for destination_arc in destination_arcs:
                if deadline is not None and time.perf_counter() >= deadline:
                    return options
                if (
                    destination_arc[0] >= context.current_time
                    and destination_arc[1] <= order.latest_completion
                ):
                    options.append(
                        ServiceOption("destination", (destination_arc,), destination_arc[1])
                    )
            return options

        if stage in ("picked", "untouched"):
            for auto_arc in network.data.arcs_auto:
                if deadline is not None and time.perf_counter() >= deadline:
                    return options
                if auto_arc[0] < context.current_time:
                    continue
                for destination_arc in destination_arcs:
                    if deadline is not None and time.perf_counter() >= deadline:
                        return options
                    if (
                        destination_arc[0] < auto_arc[1] + theta
                        or destination_arc[1] > order.latest_completion
                    ):
                        continue
                    if stage == "picked":
                        options.append(
                            ServiceOption(
                                "auto_destination",
                                (auto_arc, destination_arc),
                                destination_arc[1],
                            )
                        )
                    else:
                        for origin_arc in origin_arcs:
                            if deadline is not None and time.perf_counter() >= deadline:
                                return options
                            if (
                                origin_arc[0] < max(
                                    context.current_time, order.earliest_start
                                )
                                or auto_arc[0] < origin_arc[1] + theta
                            ):
                                continue
                            options.append(
                                ServiceOption(
                                    "transshipment",
                                    (origin_arc, auto_arc, destination_arc),
                                    destination_arc[1],
                                )
                            )
            if stage == "untouched":
                for direct_arc in network.direct_arcs:
                    if deadline is not None and time.perf_counter() >= deadline:
                        return options
                    if (
                        direct_arc[0] >= max(
                            context.current_time, order.earliest_start
                        )
                        and direct_arc[1] <= order.latest_completion
                    ):
                        options.append(
                            ServiceOption("direct", (direct_arc,), direct_arc[1])
                        )
        return options

    def _manual_capacity(self, network: CandidateNetwork, city: int, arc) -> float:
        return (
            network.data.cap_coeff_1[arc]
            if city == 1
            else network.data.cap_coeff_2[arc]
        )

    def _direct_capacity(self, context: PaperWindowContext, arc) -> float:
        from intercity_delivery.models.flexible_direct_optimizer import FlexibleDirectOptimizer

        # 复用模型中联合 BHH 反函数，避免启发式与 MILP 的容量口径漂移。
        helper = object.__new__(FlexibleDirectOptimizer)
        helper.cfg = context.config
        service_minutes = max(
            0.0,
            (arc[1] - arc[0] - context.config.direct_travel_time_periods)
            * context.config.t_0,
        )
        return min(
            context.config.capacity_direct,
            helper._inverse_combined_service_time(service_minutes),
        )

    def _direct_available_capacity(
        self,
        context: PaperWindowContext,
        arc,
        flow: str,
        capacity: float,
        manual_profiles,
    ) -> float:
        origin = 1 if flow == "+" else 2
        vehicle_key = (arc[0], arc[1], flow)
        existing_vehicles = self.decisions["w_direct"][vehicle_key]
        residual = max(
            0.0,
            existing_vehicles * capacity - self.direct_load[vehicle_key],
        )
        addable = self._profile_capacity(
            manual_profiles[origin],
            arc[0],
            context.config.T,
        )
        return residual + addable * capacity

    @staticmethod
    def _scarcity_cost(
        context: PaperWindowContext,
        profile: Dict[int, float],
        i: int,
        j: int,
        vehicles: int,
        hourly_rate: float,
    ) -> float:
        if vehicles <= 0:
            return 0.0
        weight = context.algorithm_config.heuristic_scarcity_weight
        scarcity = sum(
            1.0 / max(1.0, profile.get(t, 0.0))
            for t in range(i, j)
        )
        return (
            weight
            * hourly_rate
            * context.config.period_hours
            * vehicles
            * scarcity
        )

    def _option_capacity(
        self,
        context: PaperWindowContext,
        network: CandidateNetwork,
        order_id: int,
        option: ServiceOption,
        manual_profiles,
        auto_profile,
    ) -> float:
        order = context.all_orders[order_id]
        origin, destination = self._origin_destination(order)
        flow = order.flow
        capacities = []
        if option.kind == "direct":
            arc = option.arcs[0]
            capacity = self._direct_capacity(context, arc)
            capacities.append(
                self._direct_available_capacity(
                    context, arc, flow, capacity, manual_profiles
                )
            )
        else:
            arc_index = 0
            if option.kind == "transshipment":
                arc = option.arcs[arc_index]
                capacity = self._manual_capacity(network, origin, arc)
                capacities.append(
                    self._available_on_arc(
                        (arc[0], arc[1], origin, flow),
                        (arc[0], arc[1], origin, flow),
                        capacity,
                        manual_profiles[origin],
                        "x_manual",
                        self.manual_load,
                    )
                )
                arc_index += 1
            if option.kind in ("transshipment", "auto_destination"):
                arc = option.arcs[arc_index]
                capacities.append(
                    self._available_on_arc(
                        (arc[0], arc[1], flow),
                        (arc[0], arc[1], flow),
                        context.config.capacity_auto,
                        auto_profile,
                        "y_auto",
                        self.auto_load,
                    )
                )
                arc_index += 1
            arc = option.arcs[arc_index]
            capacity = self._manual_capacity(network, destination, arc)
            capacities.append(
                self._available_on_arc(
                    (arc[0], arc[1], destination, flow),
                    (arc[0], arc[1], destination, flow),
                    capacity,
                    manual_profiles[destination],
                    "x_manual",
                    self.manual_load,
                )
            )
        return max(0.0, min(capacities)) if capacities else 0.0

    def _option_economics(
        self,
        context: PaperWindowContext,
        network: CandidateNetwork,
        order_id: int,
        option: ServiceOption,
        amount: float,
        manual_profiles,
        auto_profile,
    ) -> tuple:
        """Return capacity, quantity, net benefit, cost, and scarcity cost."""

        order = context.all_orders[order_id]
        origin, destination = self._origin_destination(order)
        flow = order.flow
        option_capacity = self._option_capacity(
            context,
            network,
            order_id,
            option,
            manual_profiles,
            auto_profile,
        )
        quantity = min(amount, option_capacity)
        if quantity <= 1e-8:
            return option_capacity, 0.0, float("-inf"), 0.0, 0.0

        vehicle_cost = 0.0
        scarcity_cost = 0.0
        if option.kind == "direct":
            arc = option.arcs[0]
            capacity = self._direct_capacity(context, arc)
            vehicle_key = (arc[0], arc[1], flow)
            vehicles = self._additional_vehicles(
                quantity,
                vehicle_key,
                vehicle_key,
                capacity,
                "w_direct",
                self.direct_load,
            )
            vehicle_cost += (
                context.config.cost_direct
                * context.config.period_hours
                * (arc[1] - arc[0])
                * vehicles
            )
            scarcity_cost += self._scarcity_cost(
                context,
                manual_profiles[origin],
                arc[0],
                context.config.T,
                vehicles,
                context.config.cost_manual,
            )
        else:
            arc_index = 0
            if option.kind == "transshipment":
                arc = option.arcs[arc_index]
                capacity = self._manual_capacity(network, origin, arc)
                vehicle_key = (arc[0], arc[1], origin, flow)
                vehicles = self._additional_vehicles(
                    quantity,
                    vehicle_key,
                    vehicle_key,
                    capacity,
                    "x_manual",
                    self.manual_load,
                )
                vehicle_cost += (
                    context.config.cost_manual
                    * context.config.period_hours
                    * (arc[1] - arc[0])
                    * vehicles
                )
                scarcity_cost += self._scarcity_cost(
                    context,
                    manual_profiles[origin],
                    arc[0],
                    arc[1],
                    vehicles,
                    context.config.cost_manual,
                )
                arc_index += 1
            if option.kind in ("transshipment", "auto_destination"):
                arc = option.arcs[arc_index]
                vehicle_key = (arc[0], arc[1], flow)
                vehicles = self._additional_vehicles(
                    quantity,
                    vehicle_key,
                    vehicle_key,
                    context.config.capacity_auto,
                    "y_auto",
                    self.auto_load,
                )
                vehicle_cost += (
                    context.config.cost_auto
                    * context.config.period_hours
                    * (arc[1] - arc[0])
                    * vehicles
                )
                scarcity_cost += self._scarcity_cost(
                    context,
                    auto_profile,
                    arc[0],
                    arc[1],
                    vehicles,
                    context.config.cost_auto,
                )
                arc_index += 1
            arc = option.arcs[arc_index]
            capacity = self._manual_capacity(network, destination, arc)
            vehicle_key = (arc[0], arc[1], destination, flow)
            vehicles = self._additional_vehicles(
                quantity,
                vehicle_key,
                vehicle_key,
                capacity,
                "x_manual",
                self.manual_load,
            )
            vehicle_cost += (
                context.config.cost_manual
                * context.config.period_hours
                * (arc[1] - arc[0])
                * vehicles
            )
            scarcity_cost += self._scarcity_cost(
                context,
                manual_profiles[destination],
                arc[0],
                arc[1],
                vehicles,
                context.config.cost_manual,
            )

        transfer_cost = (
            context.config.transfer_cost_per_unit * quantity
            if option.kind in ("transshipment", "auto_destination")
            else 0.0
        )
        incremental_cost = vehicle_cost + transfer_cost
        net_benefit = (
            order.penalty_lost * quantity
            - incremental_cost
            - scarcity_cost
        )
        return (
            option_capacity,
            quantity,
            net_benefit,
            incremental_cost,
            scarcity_cost,
        )

    def _allocate_option(
        self,
        context: PaperWindowContext,
        network: CandidateNetwork,
        order_id: int,
        option: ServiceOption,
        quantity: float,
        manual_profiles,
        auto_profile,
    ) -> None:
        order = context.all_orders[order_id]
        flow = order.flow
        origin, destination = self._origin_destination(order)
        if option.kind == "direct":
            arc = option.arcs[0]
            vehicle_key = (arc[0], arc[1], flow)
            load_key = vehicle_key
            capacity = self._direct_capacity(context, arc)
            new_vehicles = self._additional_vehicles(
                quantity,
                vehicle_key,
                load_key,
                capacity,
                "w_direct",
                self.direct_load,
            )
            if new_vehicles:
                self.decisions["w_direct"][vehicle_key] += new_vehicles
                for t in range(arc[0], context.config.T):
                    manual_profiles[origin][t] = (
                        manual_profiles[origin].get(t, 0.0) - new_vehicles
                    )
                for t in range(arc[1], context.config.T):
                    manual_profiles[destination][t] = (
                        manual_profiles[destination].get(t, 0.0)
                        + new_vehicles
                    )
            self.direct_load[load_key] += quantity
            self.decisions["h_direct"][
                (arc[0], arc[1], flow, order_id)
            ] += quantity
            return

        arc_index = 0
        if option.kind == "transshipment":
            arc = option.arcs[arc_index]
            self._allocate_arc(
                quantity,
                (arc[0], arc[1], origin, flow),
                (arc[0], arc[1], origin, flow),
                (arc[0], arc[1], origin, flow, order_id),
                self._manual_capacity(network, origin, arc),
                manual_profiles[origin],
                "x_manual",
                "g_manual",
                self.manual_load,
            )
            arc_index += 1
        if option.kind in ("transshipment", "auto_destination"):
            arc = option.arcs[arc_index]
            self._allocate_arc(
                quantity,
                (arc[0], arc[1], flow),
                (arc[0], arc[1], flow),
                (arc[0], arc[1], flow, order_id),
                context.config.capacity_auto,
                auto_profile,
                "y_auto",
                "g_auto",
                self.auto_load,
            )
            arc_index += 1
        arc = option.arcs[arc_index]
        self._allocate_arc(
            quantity,
            (arc[0], arc[1], destination, flow),
            (arc[0], arc[1], destination, flow),
            (arc[0], arc[1], destination, flow, order_id),
            self._manual_capacity(network, destination, arc),
            manual_profiles[destination],
            "x_manual",
            "g_manual",
            self.manual_load,
        )

    def _committed_stage(self, context, order_id: int) -> dict:
        order = context.all_orders[order_id]
        origin, destination = self._origin_destination(order)
        committed = context.committed.decisions
        direct = self._sum_order_group(committed.get("h_direct", {}), order_id)
        auto = self._sum_order_group(committed.get("g_auto", {}), order_id)
        manual_values = committed.get("g_manual", {})
        picked = sum(
            value
            for key, value in manual_values.items()
            if int(key[-1]) == order_id and int(key[2]) == origin
        )
        delivered = sum(
            value
            for key, value in manual_values.items()
            if int(key[-1]) == order_id and int(key[2]) == destination
        )
        return {
            "direct": direct,
            "picked": picked,
            "auto": auto,
            "delivered": delivered,
        }

    def _planned_stage(self, order_id: int, order) -> dict:
        origin, destination = self._origin_destination(order)
        return {
            "direct": self._sum_order_group(
                self.decisions["h_direct"], order_id
            ),
            "auto": self._sum_order_group(self.decisions["g_auto"], order_id),
            "picked": sum(
                value
                for key, value in self.decisions["g_manual"].items()
                if int(key[-1]) == order_id and int(key[2]) == origin
            ),
            "delivered": sum(
                value
                for key, value in self.decisions["g_manual"].items()
                if int(key[-1]) == order_id and int(key[2]) == destination
            ),
        }

    @staticmethod
    def _stage_and_quantity(order, totals: dict) -> Tuple[str, float]:
        shipped_waiting = max(0.0, totals["auto"] - totals["delivered"])
        if shipped_waiting > 1e-8:
            return "shipped", shipped_waiting
        picked_waiting = max(0.0, totals["picked"] - totals["auto"])
        if picked_waiting > 1e-8:
            return "picked", picked_waiting
        untouched = max(
            0.0, order.quantity - totals["direct"] - totals["picked"]
        )
        return "untouched", untouched

    def _total_cost(self, context, unserved_by_order) -> float:
        combined = {}
        for group in self.decisions:
            combined[group] = defaultdict(float)
            for source in (
                context.committed.decisions.get(group, {}),
                self.decisions[group],
            ):
                for key, value in source.items():
                    combined[group][tuple(key)] += value
        cost = sum(
            context.all_orders[order_id].penalty_lost * amount
            for order_id, amount in unserved_by_order.items()
        )
        cost += sum(
            context.config.cost_manual
            * context.config.period_hours
            * (key[1] - key[0])
            * value
            for key, value in combined["x_manual"].items()
        )
        cost += sum(
            context.config.cost_auto
            * context.config.period_hours
            * context.config.travel_time_periods
            * value
            for value in combined["y_auto"].values()
        )
        cost += sum(
            context.config.cost_direct
            * context.config.period_hours
            * (key[1] - key[0])
            * value
            for key, value in combined["w_direct"].items()
        )
        cost += context.config.transfer_cost_per_unit * sum(
            combined["g_auto"].values()
        )
        return float(cost)

    def solve_window(
        self,
        context: PaperWindowContext,
        network: Optional[CandidateNetwork] = None,
        max_time_sec: Optional[float] = None,
    ) -> PaperWindowSolution:
        self.__init__()
        started = time.perf_counter()
        deadline = (
            started + max(0.0, max_time_sec)
            if max_time_sec is not None
            else None
        )
        network = network or StateDependentCandidateGenerator(context).generate()
        manual_profiles = {
            city: dict(profile)
            for city, profile in network.manual_availability.items()
        }
        auto_profile = dict(network.auto_availability)
        known_ids = list(context.known_order_ids)
        option_cache = {}
        economically_rejected = set()
        evaluated_options = 0
        selected_net_benefit = 0.0
        selected_incremental_cost = 0.0
        selected_scarcity_cost = 0.0
        iterations = 0
        ranked_orders = 0
        time_limit_reached = False
        max_iterations = max(1, len(known_ids) * 20)

        while iterations < max_iterations:
            if deadline is not None and time.perf_counter() >= deadline:
                time_limit_reached = True
                break
            ranked = []
            for order_id in known_ids:
                if deadline is not None and time.perf_counter() >= deadline:
                    time_limit_reached = True
                    break
                order = context.all_orders[order_id]
                committed = self._committed_stage(context, order_id)
                planned = self._planned_stage(order_id, order)
                totals = {
                    key: committed[key] + planned[key] for key in committed
                }
                stage, amount = self._stage_and_quantity(order, totals)
                delivered = totals["direct"] + totals["delivered"]
                if delivered >= order.quantity - 1e-8 or amount <= 1e-8:
                    continue
                cache_key = (order_id, stage)
                if cache_key not in option_cache:
                    option_cache[cache_key] = self._candidate_options(
                        context, network, order_id, stage, deadline=deadline
                    )
                if deadline is not None and time.perf_counter() >= deadline:
                    time_limit_reached = True
                    break
                options = option_cache[cache_key]
                evaluated = [
                    (option, *self._option_economics(
                        context,
                        network,
                        order_id,
                        option,
                        amount,
                        manual_profiles,
                        auto_profile,
                    ))
                    for option in options
                ]
                evaluated_options += len(evaluated)
                capacitated = [item for item in evaluated if item[2] > 1e-8]
                feasible = [
                    item
                    for item in capacitated
                    if item[3]
                    > context.algorithm_config.heuristic_min_net_benefit
                ]
                if not feasible:
                    if capacitated:
                        economically_rejected.add(order_id)
                    continue
                ordered = sorted(
                    feasible,
                    key=lambda item: (-item[3], item[0].completion),
                )
                best = ordered[0]
                second_net_benefit = (
                    ordered[1][3] if len(ordered) > 1 else 0.0
                )
                slack = max(0, order.latest_completion - best[0].completion)
                regret = (
                    best[3] - max(0.0, second_net_benefit)
                ) / (slack + 1.0)
                ranked.append(
                    (
                        regret,
                        best[3],
                        -best[0].completion,
                        order_id,
                        stage,
                        amount,
                        feasible,
                    )
                )
                ranked_orders += 1

            if not ranked:
                break
            (
                _regret,
                _best_net_benefit,
                _negative_completion,
                order_id,
                stage,
                amount,
                feasible,
            ) = max(ranked, key=lambda item: item[:3])

            # 默认遵循最短完成时间；比例下界未满足时优先直送，上界达到时优先换装。
            current_direct = sum(self.decisions["h_direct"].values()) + sum(
                context.committed.decisions.get("h_direct", {}).values()
            )
            current_trans = sum(self.decisions["g_auto"].values()) + sum(
                context.committed.decisions.get("g_auto", {}).values()
            )
            current_ratio = (
                current_direct / (current_direct + current_trans)
                if current_direct + current_trans > 1e-8
                else 0.0
            )
            direct_options = [item for item in feasible if item[0].kind == "direct"]
            trans_options = [item for item in feasible if item[0].kind != "direct"]
            if (
                stage == "untouched"
                and current_ratio < context.config.direct_ratio_min
                and direct_options
            ):
                pool = direct_options
            elif current_ratio >= context.config.direct_ratio_max and trans_options:
                pool = trans_options
            else:
                pool = feasible
            (
                option,
                _capacity,
                quantity,
                net_benefit,
                incremental_cost,
                scarcity_cost,
            ) = max(
                pool,
                key=lambda item: (item[3], -item[0].completion),
            )
            self._allocate_option(
                context,
                network,
                order_id,
                option,
                quantity,
                manual_profiles,
                auto_profile,
            )
            economically_rejected.discard(order_id)
            selected_net_benefit += net_benefit
            selected_incremental_cost += incremental_cost
            selected_scarcity_cost += scarcity_cost
            iterations += 1

        unserved_by_order = {}
        direct_volume = 0.0
        transshipment_volume = 0.0
        for order_id in known_ids:
            order = context.all_orders[order_id]
            committed = self._committed_stage(context, order_id)
            planned = self._planned_stage(order_id, order)
            direct = committed["direct"] + planned["direct"]
            delivered = committed["delivered"] + planned["delivered"]
            direct_volume += direct
            transshipment_volume += delivered
            unserved_by_order[order_id] = max(
                0.0, order.quantity - direct - delivered
            )

        served = direct_volume + transshipment_volume
        direct_ratio = direct_volume / served if served > 1e-8 else 0.0
        ratio_violation = not (
            context.config.direct_ratio_min - 1e-7
            <= direct_ratio
            <= context.config.direct_ratio_max + 1e-7
        ) if served > 1e-8 else False
        decisions = {
            group: dict(values) for group, values in self.decisions.items()
        }
        diagnostics = {
            **network.diagnostics,
            "priority_iterations": iterations,
            "direct_ratio": direct_ratio,
            "direct_ratio_violation": ratio_violation,
            "constructed_nonzero_decisions": sum(
                len(values) for values in decisions.values()
            ),
            "economically_rejected_orders": len(economically_rejected),
            "economically_rejected_order_ids": sorted(economically_rejected),
            "evaluated_service_options": evaluated_options,
            "cached_option_sets": len(option_cache),
            "selected_net_benefit": selected_net_benefit,
            "selected_incremental_cost": selected_incremental_cost,
            "selected_scarcity_cost": selected_scarcity_cost,
            "heuristic_time_limit_sec": max_time_sec,
            "heuristic_time_limit_reached": time_limit_reached,
            "ranked_orders_before_limit": ranked_orders,
            "heuristic_elapsed_sec": time.perf_counter() - started,
        }
        return PaperWindowSolution(
            feasible=True,
            status=0,
            objective=self._total_cost(context, unserved_by_order),
            decisions=decisions,
            unserved_by_order=unserved_by_order,
            direct_volume=direct_volume,
            transshipment_volume=transshipment_volume,
            diagnostics=diagnostics,
            message=(
                "动态 BHH 优先级启发式完成构造。"
                if not ratio_violation
                else "启发式完成构造，但当前窗口直送比例未满足边界。"
            ),
        )
