"""论文 Solution Approach 专用 Rolling Horizon 控制器。

该控制器与 rolling_horizon.py 完全独立，支持扩展完成窗口、订单逐期可见以及
可插拔的窗口解法。候选弧 MIP 与 BHH 优先级启发式都通过 PaperWindowApproach
接口接入。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from intercity_delivery.configuration import DeliveryConfig, RollingHorizonConfig
from intercity_delivery.data.loader import DeliveryData


@dataclass
class PaperCommittedState:
    """论文滚动流程中已经执行的弧变量。"""

    decisions: Dict[str, Dict[tuple, float]] = field(
        default_factory=lambda: {
            "x_manual": {},
            "y_auto": {},
            "g_manual": {},
            "g_auto": {},
            "w_direct": {},
            "h_direct": {},
        }
    )

    def update(self, additions: Dict[str, Dict[tuple, float]]) -> None:
        for group, values in additions.items():
            target = self.decisions.setdefault(group, {})
            for key, value in values.items():
                target[tuple(key)] = float(value)

    def committed_counts(self) -> Dict[str, int]:
        return {name: len(values) for name, values in self.decisions.items()}


@dataclass(frozen=True)
class PaperWindowContext:
    config: DeliveryConfig
    algorithm_config: RollingHorizonConfig
    full_data: DeliveryData
    all_orders: dict
    known_order_ids: frozenset
    current_time: int
    start_end: int
    completion_end: int
    control_end: int
    committed: PaperCommittedState
    remaining_time: float
    output_flag: int = 0


@dataclass
class PaperWindowSolution:
    feasible: bool
    status: Optional[int]
    objective: Optional[float]
    decisions: Dict[str, Dict[tuple, float]]
    unserved_by_order: Dict[int, float]
    direct_volume: float = 0.0
    transshipment_volume: float = 0.0
    best_bound: Optional[float] = None
    mip_gap: Optional[float] = None
    diagnostics: dict = field(default_factory=dict)
    message: str = ""


class PaperWindowApproach(Protocol):
    name: str

    def solve_window(self, context: PaperWindowContext) -> PaperWindowSolution:
        ...


@dataclass
class PaperRollingOutcome:
    status: Optional[int]
    solve_time_sec: float
    total_cost: Optional[float]
    unserved_amount: Optional[float]
    auto_usage: float
    manual_usage: float
    direct_volume: float
    transshipment_volume: float
    detail: dict
    message: str


class PaperRollingHorizonController:
    """实现论文 3.1 的控制窗、起始窗和扩展完成窗。"""

    def __init__(
        self,
        config: DeliveryConfig,
        data: DeliveryData,
        orders_tuple,
        algorithm_config: RollingHorizonConfig,
        approach: PaperWindowApproach,
    ):
        self.config = config
        self.data = data
        self.orders_tuple = orders_tuple
        self.algorithm_config = algorithm_config
        self.approach = approach
        algorithm_config.validate()

    @staticmethod
    def _commit_control_window(
        solution: PaperWindowSolution,
        start: int,
        end: int,
    ) -> Dict[str, Dict[tuple, float]]:
        committed = {}
        for group, values in solution.decisions.items():
            committed[group] = {
                tuple(key): float(value)
                for key, value in values.items()
                if start <= int(key[0]) < end and abs(value) > 1e-8
            }
        return committed

    def run(self, time_limit: int, output_flag: int = 0) -> PaperRollingOutcome:
        if time_limit <= 0:
            raise ValueError("time_limit 必须大于 0。")

        started = time.time()
        current_time = 0
        committed = PaperCommittedState()
        windows: List[dict] = []
        last_solution: Optional[PaperWindowSolution] = None
        all_orders = self.orders_tuple[2]

        while current_time < self.config.T:
            remaining = time_limit - (time.time() - started)
            if remaining <= 0:
                break
            start_end = min(
                self.config.T,
                current_time + self.algorithm_config.prediction_horizon,
            )
            extension = getattr(self.algorithm_config, "extension_horizon", 0)
            completion_end = min(self.config.T, start_end + extension)
            control_end = min(
                self.config.T,
                current_time + self.algorithm_config.rolling_step,
            )
            known_order_ids = frozenset(
                order_id
                for order_id, order in all_orders.items()
                if order.earliest_start <= current_time
            )
            context = PaperWindowContext(
                config=self.config,
                algorithm_config=self.algorithm_config,
                full_data=self.data,
                all_orders=all_orders,
                known_order_ids=known_order_ids,
                current_time=current_time,
                start_end=start_end,
                completion_end=completion_end,
                control_end=control_end,
                committed=committed,
                remaining_time=remaining,
                output_flag=output_flag,
            )
            window_started = time.time()
            solution = self.approach.solve_window(context)
            record = {
                "window_start": current_time,
                "start_window_end": start_end,
                "completion_window_end": completion_end,
                "control_end": control_end,
                "known_orders": len(known_order_ids),
                "status": solution.status,
                "feasible": solution.feasible,
                "objective": solution.objective,
                "best_bound": solution.best_bound,
                "mip_gap": solution.mip_gap,
                "solve_time_sec": round(time.time() - window_started, 4),
                "diagnostics": solution.diagnostics,
                "message": solution.message,
            }
            windows.append(record)
            if not solution.feasible:
                return PaperRollingOutcome(
                    status=solution.status,
                    solve_time_sec=round(time.time() - started, 2),
                    total_cost=None,
                    unserved_amount=None,
                    auto_usage=0.0,
                    manual_usage=0.0,
                    direct_volume=0.0,
                    transshipment_volume=0.0,
                    detail={"windows": windows},
                    message=solution.message or "论文滚动窗口未获得可行解。",
                )
            committed.update(
                self._commit_control_window(solution, current_time, control_end)
            )
            last_solution = solution
            current_time = control_end

        elapsed = time.time() - started
        if last_solution is None:
            return PaperRollingOutcome(
                status=None,
                solve_time_sec=round(elapsed, 2),
                total_cost=None,
                unserved_amount=None,
                auto_usage=0.0,
                manual_usage=0.0,
                direct_volume=0.0,
                transshipment_volume=0.0,
                detail={"windows": windows},
                message="论文 Rolling Horizon 未完成首个窗口。",
            )

        unserved_amount = sum(last_solution.unserved_by_order.values())
        auto_usage = sum(committed.decisions.get("y_auto", {}).values())
        manual_usage = sum(committed.decisions.get("x_manual", {}).values()) + sum(
            committed.decisions.get("w_direct", {}).values()
        )
        completed = current_time >= self.config.T
        detail = {
            "algorithm": {
                "approach": self.approach.name,
                "prediction_horizon": self.algorithm_config.prediction_horizon,
                "rolling_step": self.algorithm_config.rolling_step,
                "extension_horizon": getattr(
                    self.algorithm_config, "extension_horizon", 0
                ),
                "completed_horizon": current_time,
                "completed": completed,
                "committed_decision_counts": committed.committed_counts(),
            },
            "windows": windows,
            "solution": {
                "unserved_by_order": last_solution.unserved_by_order,
                "direct_volume": last_solution.direct_volume,
                "transshipment_volume": last_solution.transshipment_volume,
                "committed_decisions": {
                    name: {str(key): value for key, value in values.items()}
                    for name, values in committed.decisions.items()
                },
            },
        }
        return PaperRollingOutcome(
            status=last_solution.status,
            solve_time_sec=round(elapsed, 2),
            total_cost=last_solution.objective,
            unserved_amount=unserved_amount,
            auto_usage=auto_usage,
            manual_usage=manual_usage,
            direct_volume=last_solution.direct_volume,
            transshipment_volume=last_solution.transshipment_volume,
            detail=detail,
            message=(
                f"{self.approach.name} 完成 {len(windows)} 个论文滚动窗口。"
                if completed
                else f"{self.approach.name} 在 t={current_time} 达到总时间限制。"
            ),
        )
