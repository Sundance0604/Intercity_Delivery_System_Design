"""求解器接口层。

本文件专门负责“怎么求解一个已经生成好的算例”。
后续如果要加入 rolling horizon、启发式算法、ALNS 等方法，优先在这里新增
一个 Solver 类，然后注册到 SOLVER_REGISTRY 中。这样 GUI、数据生成和实验记录
都不需要跟着大改，方便保证不同算法使用同一批输入订单进行公平比较。
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional

from config import DeliveryConfig
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
    message: str = ""


class BaseSolver:
    """所有求解器的基类。

    子类只需要实现 solve 方法。solve 的输入是同一个 config、data、orders，
    因此可以确保多个算法在完全相同的订单数据上进行对比。
    """

    name = "base"
    display_name = "基础求解器"

    def solve(
        self,
        config: DeliveryConfig,
        data: DeliveryData,
        orders_tuple,
        time_limit: int,
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


class RollingHorizonPlaceholderSolver(BaseSolver):
    """rolling horizon 算法预留接口。

    目前这里只放接口占位，目的是先把实验平台搭好。后续正式实现 rolling horizon
    时，建议保持 solve 的输入输出不变，这样同一批订单可直接同时跑 exact_mip
    和 rolling_horizon，结果表通过 Solver 字段即可区分。
    """

    name = "rolling_horizon"
    display_name = "Rolling Horizon (待实现)"

    def solve(
        self,
        config: DeliveryConfig,
        data: DeliveryData,
        orders_tuple,
        time_limit: int,
    ) -> SolverResult:
        return SolverResult(
            solver_name=self.name,
            status=None,
            solve_time_sec=0,
            total_cost=None,
            best_bound=None,
            mip_gap=None,
            unserved_rate=None,
            auto_usage=0,
            manual_usage=0,
            detail=None,
            message="rolling horizon 求解器尚未实现；这是为后续算法研究保留的接口。",
        )


# 所有可选求解器都在这里注册。GUI 和命令行都会读取这个注册表。
SOLVER_REGISTRY: Dict[str, BaseSolver] = {
    ExactMIPSolver.name: ExactMIPSolver(),
    RollingHorizonPlaceholderSolver.name: RollingHorizonPlaceholderSolver(),
}


def get_solver_names():
    """返回所有求解器的内部名称。"""

    return list(SOLVER_REGISTRY.keys())


def get_solver_display_name(name: str) -> str:
    """返回 GUI 中显示的求解器名称。"""

    return SOLVER_REGISTRY[name].display_name
