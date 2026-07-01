"""Rolling Horizon 求解控制器。

对外只接收一次完整算例；内部按预测区间重复构建并求解原始 Optimizer。
每轮只提交 rolling_step 覆盖的决策，预测区间后段会在下一轮重新优化。
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import gurobipy as gp

from config import DeliveryConfig, RollingHorizonConfig
from data_loader import DeliveryData
from optimizer import Optimizer


@dataclass
class RollingHorizonOutcome:
    """控制器返回给求解器适配层的内部结果。"""

    status: Optional[int]
    solve_time_sec: float
    total_cost: Optional[float]
    unserved_amount: Optional[float]
    auto_usage: float
    manual_usage: float
    detail: Optional[dict]
    message: str


@dataclass
class CommittedDecisions:
    """所有已经执行的弧决策；键与 Optimizer 中的变量键完全一致。"""

    x_manual: Dict[tuple, float] = field(default_factory=dict)
    y_auto: Dict[tuple, float] = field(default_factory=dict)
    g_manual: Dict[tuple, float] = field(default_factory=dict)
    g_auto: Dict[tuple, float] = field(default_factory=dict)
    w_direct: Dict[tuple, float] = field(default_factory=dict)
    h_direct: Dict[tuple, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Dict[tuple, float]]:
        return {
            "x_manual": self.x_manual,
            "y_auto": self.y_auto,
            "g_manual": self.g_manual,
            "g_auto": self.g_auto,
            "w_direct": self.w_direct,
            "h_direct": self.h_direct,
        }

    def update(self, decisions: Dict[str, Dict[tuple, float]]) -> None:
        for group_name, values in decisions.items():
            getattr(self, group_name).update(values)


class RollingHorizonController:
    """按固定预测区间和滚动步长重复求解配送模型。"""

    def __init__(
        self,
        config: DeliveryConfig,
        data: DeliveryData,
        algorithm_config: Optional[RollingHorizonConfig] = None,
        optimizer_class=Optimizer,
    ):
        self.config = config
        self.data = data
        self.algorithm_config = algorithm_config or RollingHorizonConfig()
        self.optimizer_class = optimizer_class
        self.algorithm_config.validate()

    def run(self, time_limit: int, output_flag: int = 0) -> RollingHorizonOutcome:
        """运行完整滚动过程。

        time_limit 是整个滚动过程的求解预算，而不是每个窗口各自的预算。
        """

        if time_limit <= 0:
            raise ValueError("time_limit 必须大于 0。")

        started_at = time.time()
        current_time = 0
        committed = CommittedDecisions()
        window_records: List[dict] = []
        last_optimizer: Optional[Optimizer] = None
        stopped_by_budget = False

        while current_time < self.config.T:
            elapsed = time.time() - started_at
            remaining_time = time_limit - elapsed
            if remaining_time <= 0:
                stopped_by_budget = True
                break

            window_end = min(
                current_time + self.algorithm_config.prediction_horizon,
                self.config.T,
            )
            commit_end = min(
                current_time + self.algorithm_config.rolling_step,
                self.config.T,
            )

            optimizer = self.optimizer_class(self.config, self.data)
            optimizer.setup_variables()
            optimizer.set_objective()
            optimizer.set_constraints()
            optimizer.configure_rolling_window(
                current_time=current_time,
                window_end=window_end,
                committed_decisions=committed.as_dict(),
            )
            optimizer.model.setParam("OutputFlag", output_flag)
            optimizer.model.setParam("TimeLimit", max(0.01, remaining_time))

            window_started_at = time.time()
            optimizer.model.optimize()
            window_solve_time = time.time() - window_started_at

            record = {
                "window_start": current_time,
                "window_end": window_end,
                "commit_end": commit_end,
                "status": optimizer.model.Status,
                "solve_time_sec": round(window_solve_time, 4),
                "solution_count": optimizer.model.SolCount,
            }
            if optimizer.model.SolCount > 0:
                record["objective"] = optimizer.model.ObjVal
                record["mip_gap"] = optimizer.model.MIPGap
            window_records.append(record)

            if optimizer.model.SolCount <= 0:
                return RollingHorizonOutcome(
                    status=optimizer.model.Status,
                    solve_time_sec=round(time.time() - started_at, 2),
                    total_cost=None,
                    unserved_amount=None,
                    auto_usage=0.0,
                    manual_usage=0.0,
                    detail={"windows": window_records},
                    message=(
                        f"窗口 [{current_time}, {window_end}] 未找到可行解，"
                        f"Gurobi 状态码：{optimizer.model.Status}"
                    ),
                )

            committed.update(
                optimizer.extract_committed_decisions(current_time, commit_end)
            )
            last_optimizer = optimizer
            current_time = commit_end

            if optimizer.model.Status == gp.GRB.TIME_LIMIT:
                stopped_by_budget = True
                break

        solve_time = time.time() - started_at
        if last_optimizer is None:
            return RollingHorizonOutcome(
                status=gp.GRB.TIME_LIMIT,
                solve_time_sec=round(solve_time, 2),
                total_cost=None,
                unserved_amount=None,
                auto_usage=0.0,
                manual_usage=0.0,
                detail={"windows": window_records},
                message="Rolling Horizon 在建立第一个可行窗口解之前已耗尽时间预算。",
            )

        unserved_amount = sum(
            variable.X for variable in last_optimizer.z_unserved.values()
        )
        completed = current_time >= self.config.T
        if completed:
            message = (
                f"Rolling Horizon 完成，共求解 {len(window_records)} 个窗口；"
                "返回值是滚动策略解，不代表全局最优解。"
            )
        else:
            message = (
                f"Rolling Horizon 在 t={current_time} 停止，"
                f"已完成 {len(window_records)} 个窗口并耗尽总时间预算。"
            )

        detail = {
            "algorithm": {
                "prediction_horizon": self.algorithm_config.prediction_horizon,
                "rolling_step": self.algorithm_config.rolling_step,
                "completed_horizon": current_time,
                "stopped_by_budget": stopped_by_budget,
                "committed_decision_counts": {
                    group_name: len(values)
                    for group_name, values in committed.as_dict().items()
                },
            },
            "windows": window_records,
            "solution": {
                "y_auto": {
                    str(key): variable.X
                    for key, variable in last_optimizer.y_auto.items()
                    if variable.X > 0.1
                },
                "z_unserved": {
                    key: variable.X
                    for key, variable in last_optimizer.z_unserved.items()
                    if variable.X > 0.1
                },
            },
        }
        if hasattr(last_optimizer, "w_direct"):
            detail["solution"]["w_direct"] = {
                str(key): variable.X
                for key, variable in last_optimizer.w_direct.items()
                if variable.X > 0.1
            }
        if hasattr(last_optimizer, "q_direct"):
            direct_volume = sum(
                variable.X for variable in last_optimizer.q_direct.values()
            )
            transshipment_volume = sum(
                variable.X
                for variable in last_optimizer.r_transshipment.values()
            )
            served_volume = direct_volume + transshipment_volume
            detail["solution"]["direct_volume"] = direct_volume
            detail["solution"]["transshipment_volume"] = transshipment_volume
            detail["solution"]["direct_ratio"] = (
                direct_volume / served_volume if served_volume > 0 else 0.0
            )

        return RollingHorizonOutcome(
            status=last_optimizer.model.Status,
            solve_time_sec=round(solve_time, 2),
            total_cost=last_optimizer.model.ObjVal,
            unserved_amount=unserved_amount,
            auto_usage=sum(variable.X for variable in last_optimizer.y_auto.values()),
            manual_usage=(
                sum(variable.X for variable in last_optimizer.x_manual.values())
                + (
                    sum(
                        variable.X
                        for variable in last_optimizer.w_direct.values()
                    )
                    if hasattr(last_optimizer, "w_direct")
                    else 0.0
                )
            ),
            detail=detail,
            message=message,
        )
