"""求解器接口层。

本文件专门负责“怎么求解一个已经生成好的算例”。
后续如果要加入 rolling horizon、启发式算法、ALNS 等方法，优先在这里新增
一个 Solver 类，然后注册到 SOLVER_REGISTRY 中。这样 GUI、数据生成和实验记录
都不需要跟着大改，方便保证不同算法使用同一批输入订单进行公平比较。
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional

from config import DeliveryConfig, RollingHorizonConfig
from data_loader import DeliveryData


@dataclass
class SolverResult:
    """统一的求解结果格式。

    所有求解器都应该返回这个结构。这样实验汇总 CSV 可以用同一套字段记录
    精确 MIP、rolling horizon、启发式算法等不同方法的结果。
    """

    solver_name: str
    status: Optional[int]
    solve_time_sec: float
    total_cost: Optional[float]
    best_bound: Optional[float]
    mip_gap: Optional[float]
    unserved_rate: Optional[float]
    auto_usage: float
    manual_usage: float
    detail: Optional[dict]
    direct_ratio: Optional[float] = None
    direct_volume: Optional[float] = None
    transshipment_volume: Optional[float] = None
    message: str = ""


class BaseSolver:
    """所有求解器的基类。

    子类只需要实现 solve 方法。solve 的输入是同一个 config、data、orders，
    因此可以确保多个算法在完全相同的订单数据上进行对比。
    """

    name = "base"
    display_name = "基础求解器"
    sensitivity_sources = frozenset({"model", "order"})

    def solve(
        self,
        config: DeliveryConfig,
        data: DeliveryData,
        orders_tuple,
        time_limit: int,
        algorithm_config: RollingHorizonConfig,
    ) -> SolverResult:
        raise NotImplementedError


class ExactMIPSolver(BaseSolver):
    """Gurobi 精确 MIP 求解器。

    这是当前论文模型的基准求解方式。小规模算例可用于得到最优解；中大规模
    算例则可配合 time_limit 得到限时可行解和 MIP Gap。
    """

    name = "exact_mip"
    display_name = "精确 MIP (Gurobi)"

    def solve(
        self,
        config: DeliveryConfig,
        data: DeliveryData,
        orders_tuple,
        time_limit: int,
        algorithm_config: RollingHorizonConfig,
    ) -> SolverResult:
        import gurobipy as gp
        from optimizer import Optimizer

        start_time = time.time()
        _, _, all_orders = orders_tuple
        total_demand = sum(order.quantity for order in all_orders.values())

        opt = Optimizer(config, data)
        opt.setup_variables()
        opt.set_objective()
        opt.set_constraints()
        opt.model.setParam("TimeLimit", time_limit)
        opt.model.setParam("OutputFlag", 0)
        opt.model.optimize()

        solve_time = time.time() - start_time

        if opt.model.SolCount <= 0:
            return SolverResult(
                solver_name=self.name,
                status=opt.model.Status,
                solve_time_sec=round(solve_time, 2),
                total_cost=None,
                best_bound=None,
                mip_gap=None,
                unserved_rate=None,
                auto_usage=0,
                manual_usage=0,
                detail=None,
                message=f"未找到可行解，Gurobi 状态码：{opt.model.Status}",
            )

        unserved_amount = sum(v.X for v in opt.z_unserved.values())
        detail = {
            "solution": {
                "y_auto": {str(k): v.X for k, v in opt.y_auto.items() if v.X > 0.1},
                "z_unserved": {k: v.X for k, v in opt.z_unserved.items() if v.X > 0.1},
            }
        }

        if opt.model.Status == gp.GRB.OPTIMAL:
            message = "已找到全局最优解"
        elif opt.model.Status == gp.GRB.TIME_LIMIT:
            message = f"达到时间限制，当前 MIP Gap 为 {opt.model.MIPGap * 100:.2f}%"
        else:
            message = f"已找到可行解，Gurobi 状态码：{opt.model.Status}"

        return SolverResult(
            solver_name=self.name,
            status=opt.model.Status,
            solve_time_sec=round(solve_time, 2),
            total_cost=opt.model.ObjVal,
            best_bound=opt.model.ObjBound,
            mip_gap=opt.model.MIPGap,
            unserved_rate=round(unserved_amount / total_demand, 4) if total_demand > 0 else 0,
            auto_usage=sum(v.X for v in opt.y_auto.values()),
            manual_usage=sum(v.X for v in opt.x_manual.values()),
            detail=detail,
            message=message,
        )


class RollingHorizonSolver(BaseSolver):
    """重复求解预测区间，并只提交窗口前段决策的 Rolling Horizon 求解器。"""

    name = "rolling_horizon"
    display_name = "Rolling Horizon"
    sensitivity_sources = frozenset({"model", "algorithm", "order"})

    def solve(
        self,
        config: DeliveryConfig,
        data: DeliveryData,
        orders_tuple,
        time_limit: int,
        algorithm_config: RollingHorizonConfig,
    ) -> SolverResult:
        from rolling_horizon import RollingHorizonController

        _, _, all_orders = orders_tuple
        total_demand = sum(order.quantity for order in all_orders.values())
        outcome = RollingHorizonController(
            config,
            data,
            algorithm_config=algorithm_config,
        ).run(time_limit=time_limit)

        return SolverResult(
            solver_name=self.name,
            status=outcome.status,
            solve_time_sec=outcome.solve_time_sec,
            total_cost=outcome.total_cost,
            best_bound=None,
            mip_gap=None,
            unserved_rate=(
                round(outcome.unserved_amount / total_demand, 4)
                if outcome.unserved_amount is not None and total_demand > 0
                else None
            ),
            auto_usage=outcome.auto_usage,
            manual_usage=outcome.manual_usage,
            detail=outcome.detail,
            message=outcome.message,
        )


class FlexibleDirectMIPSolver(BaseSolver):
    """直送与换装运输共存模型的完整 MIP 求解器。"""

    name = "flexible_direct_mip"
    display_name = "直送-换装协同 MIP"
    sensitivity_sources = frozenset({"model", "order"})

    def solve(
        self,
        config: DeliveryConfig,
        data: DeliveryData,
        orders_tuple,
        time_limit: int,
        algorithm_config: RollingHorizonConfig,
    ) -> SolverResult:
        import gurobipy as gp
        from flexible_direct_optimizer import FlexibleDirectOptimizer

        start_time = time.time()
        _, _, all_orders = orders_tuple
        total_demand = sum(order.quantity for order in all_orders.values())

        optimizer = FlexibleDirectOptimizer(config, data).build_model()
        optimizer.model.setParam("OutputFlag", 0)
        optimizer.model.setParam("TimeLimit", time_limit)
        optimizer.model.optimize()
        solve_time = time.time() - start_time

        if optimizer.model.SolCount <= 0:
            return SolverResult(
                solver_name=self.name,
                status=optimizer.model.Status,
                solve_time_sec=round(solve_time, 2),
                total_cost=None,
                best_bound=None,
                mip_gap=None,
                unserved_rate=None,
                auto_usage=0.0,
                manual_usage=0.0,
                detail=None,
                message=(
                    "直送-换装协同模型未找到可行解，"
                    f"Gurobi 状态码：{optimizer.model.Status}"
                ),
            )

        unserved_amount = sum(v.X for v in optimizer.z_unserved.values())
        direct_volume = sum(v.X for v in optimizer.q_direct.values())
        transshipment_volume = sum(
            v.X for v in optimizer.r_transshipment.values()
        )
        served_volume = direct_volume + transshipment_volume
        direct_ratio = direct_volume / served_volume if served_volume > 0 else 0.0

        if optimizer.model.Status == gp.GRB.OPTIMAL:
            message = (
                f"直送-换装协同模型达到最优；直送比例 {direct_ratio:.2%}"
            )
        elif optimizer.model.Status == gp.GRB.TIME_LIMIT:
            message = (
                "直送-换装协同模型达到时间限制；"
                f"当前 Gap {optimizer.model.MIPGap * 100:.2f}%，"
                f"直送比例 {direct_ratio:.2%}"
            )
        else:
            message = (
                f"直送-换装协同模型获得可行解；直送比例 {direct_ratio:.2%}"
            )

        detail = {
            "solution": {
                "w_direct": {
                    str(key): var.X
                    for key, var in optimizer.w_direct.items()
                    if var.X > 0.1
                },
                "q_direct": {
                    key: var.X
                    for key, var in optimizer.q_direct.items()
                    if var.X > 1e-7
                },
                "r_transshipment": {
                    key: var.X
                    for key, var in optimizer.r_transshipment.items()
                    if var.X > 1e-7
                },
                "z_unserved": {
                    key: var.X
                    for key, var in optimizer.z_unserved.items()
                    if var.X > 1e-7
                },
                "direct_volume": direct_volume,
                "transshipment_volume": transshipment_volume,
                "direct_ratio": direct_ratio,
            }
        }

        return SolverResult(
            solver_name=self.name,
            status=optimizer.model.Status,
            solve_time_sec=round(solve_time, 2),
            total_cost=optimizer.model.ObjVal,
            best_bound=optimizer.model.ObjBound,
            mip_gap=optimizer.model.MIPGap,
            unserved_rate=(
                round(unserved_amount / total_demand, 4)
                if total_demand > 0
                else 0
            ),
            auto_usage=sum(v.X for v in optimizer.y_auto.values()),
            manual_usage=(
                sum(v.X for v in optimizer.x_manual.values())
                + sum(v.X for v in optimizer.w_direct.values())
            ),
            detail=detail,
            direct_ratio=direct_ratio,
            direct_volume=direct_volume,
            transshipment_volume=transshipment_volume,
            message=message,
        )


class FlexibleDirectRollingSolver(BaseSolver):
    """使用通用 Rolling Horizon 控制器求解直送/换装共存模型。"""

    name = "flexible_direct_rolling"
    display_name = "直送-换装协同 Rolling Horizon"
    sensitivity_sources = frozenset({"model", "algorithm", "order"})

    def solve(
        self,
        config: DeliveryConfig,
        data: DeliveryData,
        orders_tuple,
        time_limit: int,
        algorithm_config: RollingHorizonConfig,
    ) -> SolverResult:
        from flexible_direct_optimizer import FlexibleDirectOptimizer
        from rolling_horizon import RollingHorizonController

        _, _, all_orders = orders_tuple
        total_demand = sum(order.quantity for order in all_orders.values())
        outcome = RollingHorizonController(
            config,
            data,
            algorithm_config=algorithm_config,
            optimizer_class=FlexibleDirectOptimizer,
        ).run(time_limit=time_limit)
        solution_detail = (
            outcome.detail.get("solution", {}) if outcome.detail else {}
        )

        return SolverResult(
            solver_name=self.name,
            status=outcome.status,
            solve_time_sec=outcome.solve_time_sec,
            total_cost=outcome.total_cost,
            best_bound=None,
            mip_gap=None,
            unserved_rate=(
                round(outcome.unserved_amount / total_demand, 4)
                if outcome.unserved_amount is not None and total_demand > 0
                else None
            ),
            auto_usage=outcome.auto_usage,
            manual_usage=outcome.manual_usage,
            detail=outcome.detail,
            direct_ratio=solution_detail.get("direct_ratio"),
            direct_volume=solution_detail.get("direct_volume"),
            transshipment_volume=solution_detail.get("transshipment_volume"),
            message=outcome.message,
        )


# 所有可选求解器都在这里注册。GUI 和命令行都会读取这个注册表。
SOLVER_REGISTRY: Dict[str, BaseSolver] = {
    ExactMIPSolver.name: ExactMIPSolver(),
    RollingHorizonSolver.name: RollingHorizonSolver(),
    FlexibleDirectMIPSolver.name: FlexibleDirectMIPSolver(),
    FlexibleDirectRollingSolver.name: FlexibleDirectRollingSolver(),
}


def get_solver_names():
    """返回所有求解器的内部名称。"""

    return list(SOLVER_REGISTRY.keys())


def get_solver_display_name(name: str) -> str:
    """返回 GUI 中显示的求解器名称。"""

    return SOLVER_REGISTRY[name].display_name
