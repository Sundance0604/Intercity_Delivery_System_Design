import gurobipy as gp
from gurobipy import GRB
from intercity_delivery.configuration import DeliveryConfig
from intercity_delivery.data.loader import DeliveryData, DataLoader, OrderBatch
from dataclasses import dataclass
import pandas as pd
import math
import numpy as np
from typing import List, Tuple, Dict

class Optimizer:
    def __init__(self, config: DeliveryConfig, data: DeliveryData):
        self.cfg = config
        self.data = data
        self.model = gp.Model("Intercity_Delivery_Optimization")

    # 设置变量
    def setup_variables(self):
        self.flow = ["+", "-"] # 流量方向：+表示城市1到城市2，-表示城市2到城市1
        self.arcs_indices = []
        # 城市 1 的弧
        for (i, j) in self.data.arcs_manual_1:
            for flow in self.flow: 
                self.arcs_indices.append((i, j, 1, flow))
                     
        # 城市 2 的弧
        for (i, j) in self.data.arcs_manual_2:
            for flow in self.flow:
                self.arcs_indices.append((i, j, 2, flow))
                
        # 创建变量 x
        self.x_manual = self.model.addVars(self.arcs_indices, vtype=GRB.INTEGER, name="x_manual")
        # 创建变量 y 
        self.y_auto = self.model.addVars(
            self.data.arcs_auto, self.flow, vtype=GRB.INTEGER, name="y_auto"
        )
        # 创建变量 g
        # 论文将人工车和自动车承运货量定义为非负连续变量 R+。
        # 即便当前随机算例中的订单需求量是整数，车辆也允许承运部分货量。
        manual_load_indices = [
            (*arc, order_id)
            for arc in self.arcs_indices
            for order_id, order in self.data.all_orders.items()
            if order.flow == arc[3]
        ]
        self.g_manual = self.model.addVars(
            manual_load_indices,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="g_manual",
        )
        auto_load_indices = [
            (i, j, flow, order_id)
            for i, j in self.data.arcs_auto
            for flow, orders in (
                ("+", self.data.pos_orders),
                ("-", self.data.neg_orders),
            )
            for order_id in orders
        ]
        self.g_auto = self.model.addVars(
            auto_load_indices,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="g_auto",
        )
        # 创建变量 z
        self.z_unserved = self.model.addVars(
            self.data.all_orders.keys(), vtype=GRB.INTEGER, name="z_unserved"
        )

    def set_objective(self):
        # 未服务惩罚
        self.penalty_unserved = gp.quicksum(
            order.penalty_lost * self.z_unserved[l] for l, order in self.data.all_orders.items()
        )    
        # 人工车辆成本
        self.cost_manual = gp.quicksum(
            self.cfg.cost_manual * self.cfg.period_hours * (j-i) * self.x_manual[i, j, city, flow]
            for (i, j, city, flow) in self.arcs_indices
        )
        # 自动驾驶车辆成本
        self.cost_auto = self.cfg.cost_auto * self.cfg.travel_time_periods * self.cfg.period_hours * self.y_auto.sum()
        # 总目标函数
        self.model.setObjective(
            self.penalty_unserved + self.cost_manual + self.cost_auto,
            GRB.MINIMIZE
        )
        
    def set_constraints(self):
        # 约束 (2)：城市内部人工车队规模限制。
        # 对任意城市 k 和时间段 t，统计所有覆盖 t 的人工服务弧 (i,j)
        # 上正在执行任务的车辆数，并同时计入正、反两个运输方向。该总数不能
        # 超过城市 k 配置的人工车辆数 N^k，防止同一辆人工车在同一时段被重复使用。
        for t in range(self.cfg.T):
            for city in [1, 2]:
                # 1. 获取该城市在时间 t 的活跃弧集合 S^k(t)
                active_arcs = (
                    self.data.sets_manual_1[t] if city == 1 
                    else self.data.sets_manual_2[t]
                )
                
                # 2. 计算当前活跃车辆总数
                active_vehicles = gp.quicksum(
                    self.x_manual[i, j, city, flow]
                    for (i, j) in active_arcs
                    for flow in self.flow
                )
                
                # 3. 添加约束: 活跃车辆数 <= 该城市的车队上限
                self.model.addConstr(
                    active_vehicles <= self.cfg.N_manual[city],
                    name=f"(2)Fleet_Capacity_InnerCity{city}_Time{t}"
                )
        # 约束 (3)：城际自动驾驶车队的同时在途规模限制。
        # 对任意时间段 t，统计所有覆盖 t 的城际弧上、两个方向的自动驾驶车辆数。
        # 同时在途车辆总数不得超过两个城市初始自动驾驶车辆数之和 N_hat^1+N_hat^2。
        for t in range(self.cfg.T):        
            # 1. 计算当前活跃车辆总数
            active_vehicles = gp.quicksum(
                self.y_auto[i, j, flow]
                for (i, j) in self.data.sets_auto[t]
                for flow in self.flow
            )
            
            # 2. 添加约束: 活跃车辆数 <= 该城市的车队上限
            self.model.addConstr(
                active_vehicles <= sum(self.cfg.N_auto.values()),
                name=f"(3)Fleet_Capacity_InterCity_Time{t}"
            )
        # 约束 (4) 与 (5)：两个城市的自动驾驶车辆时序平衡限制。
        # 在时间 t 前，i<=t 表示已经从始发城市发出的车辆，j<=t 表示已经到达
        # 目的城市的车辆。任一城市的累计出发量不能超过该城市的初始车辆数与
        # 从另一城市累计到达的车辆数之和。
        for t in range(self.cfg.T):
            # ㊣流计算
            positive_departures = gp.quicksum(
                self.y_auto[i, j, "+"]
                for (i, j) in self.data.arcs_auto if i <= t
            )
            # 逆流计算
            negative_departures = gp.quicksum(
                self.y_auto[i, j, "-"]
                for (i, j) in self.data.arcs_auto if i <= t
            )
            positive_arrivals = gp.quicksum(
                self.y_auto[i, j, "+"]
                for (i, j) in self.data.arcs_auto if j <= t
            )
            negative_arrivals = gp.quicksum(
                self.y_auto[i, j, "-"]
                for (i, j) in self.data.arcs_auto if j <= t
            )
            # 约束 (4)：N_hat^1 + 累计反向到达 - 累计正向出发 >= 0。
            self.model.addConstr(
                self.cfg.N_auto[1] + negative_arrivals - positive_departures >= 0,
                name=f"(4)Intercity_Postive_Flow_Balance_Time{t}"
            )
            # 约束 (5)：N_hat^2 + 累计正向到达 - 累计反向出发 >= 0。
            self.model.addConstr(
                self.cfg.N_auto[2] + positive_arrivals - negative_departures >= 0,
                name=f"(5)Intercity_Negtive_Flow_Balance_Time{t}"
            )
        # 约束 (6)：人工车辆在单条城市服务弧上的总承运能力限制。
        # 对城市 k、方向 flow 和服务弧 (i,j)，先汇总该方向全部订单在该弧上的
        # 人工承运货量。单车在持续时间 (j-i)*t0 内可完成的最大服务量由
        # (f^k)^(-1)[(j-i)*t0] 给出，再乘该弧投入的人工车辆数 x，得到总容量。
        for (i, j, city, flow) in self.arcs_indices:
            coeff_dict = (self.data.cap_coeff_1 if city == 1 else self.data.cap_coeff_2)
            orders = (self.data.pos_orders if flow == "+" else self.data.neg_orders)
            
            lhs = gp.quicksum(
                self.g_manual[i, j, city, flow, l] 
                for l in orders.keys()
            )
            rhs = self.x_manual[i, j, city, flow] * coeff_dict[(i, j)]
            
            self.model.addConstr(lhs <= rhs, name=f"(6)Manual_Cap_{city}_{flow}_{i}_{j}")

        # 约束 (7)：自动驾驶车辆对每一订单的城际承运能力限制。
        # 严格按照论文原式，对每个方向、城际弧和订单 l 分别建立
        # g_hat^l_ij <= y^flow_ij * M_hat。这里不再对同一弧上的订单求和；
        # 因而每个订单在该弧上的承运量分别受投入车辆总容量约束。
        self.model.addConstrs(
            (self.g_auto[i, j, flow, l]
             <= self.y_auto[i, j, flow] * self.cfg.capacity_auto
            for (i, j) in self.data.arcs_auto
            for flow in self.flow
            for l in (
                self.data.pos_orders if flow == "+" else self.data.neg_orders
            ).keys()),
            name="(7)Auto_Capacity_Per_Order"
        )

        # 约束 (8)：订单服务时间窗限制。
        # epsilon_sets 预先收集所有违反时间窗的“人工服务弧-城市-方向-订单”组合：
        # 始发城市服务开始早于 s_l，或目的城市服务完成晚于 e_l。将这些组合上的
        # 人工承运货量固定为 0，确保任何实际服务都落在订单允许的时间窗内。
        self.model.addConstrs(
            (self.g_manual[i, j, city, flow, l] == 0  
            for (i, j, city, flow, l) in self.data.epsilon_sets 
            ),
            name=f"(8)Time_Window_Violation"
        )
        # 约束 (9) 与 (10)：每个订单在两个换装节点上的时序流量守恒。
        # 约束按订单分别建立，避免不同订单之间的货量相互抵消或借用。
        for t in range(self.cfg.T):  
            for flow in self.flow:
                if flow == "+":
                    # 正向 (+): City 1 (Origin) -> Auto -> City 2 (Dest)
                    orders = self.data.pos_orders
                    origin_city = 1
                    dest_city = 2
                    arcs_manual_origin = self.data.arcs_manual_1 
                    arcs_manual_dest   = self.data.arcs_manual_2  
                else:
                    # 反向 (-): City 2 (Origin) -> Auto -> City 1 (Dest)
                    orders = self.data.neg_orders
                    origin_city = 2
                    dest_city = 1
                    arcs_manual_origin = self.data.arcs_manual_2
                    arcs_manual_dest   = self.data.arcs_manual_1

                for l in orders.keys():
                    # 约束 (9) 的左侧：截至 t 已从始发城市发出的城际货量。
                    auto_departure_origin = gp.quicksum(
                        self.g_auto[i, j, flow, l]
                        for (i, j) in self.data.arcs_auto
                        if i <= t
                    )

                    # 约束 (9) 的右侧：截至 t 已完成人工揽收并到达始发换装点的货量。
                    # 城际累计发出量不能超过已经完成始发端人工服务的累计货量。
                    manual_arrival_origin = gp.quicksum(
                        self.g_manual[i, j, origin_city, flow, l]
                        for (i, j) in arcs_manual_origin
                        if j <= t
                    )

                    self.model.addConstr(
                        auto_departure_origin <= manual_arrival_origin,
                        name=f"(9)transfer_origin_dir{flow}_order{l}_t{t}"
                    )
                    # 约束 (10) 的左侧：截至 t 已到达目的城市的城际货量。
                    auto_arrival_dest = gp.quicksum(
                        self.g_auto[i, j, flow, l]
                        for (i, j) in self.data.arcs_auto
                        if j <= t
                    )

                    # 约束 (10) 的右侧：截至 t 已从目的换装点开始末端人工服务的货量。
                    # 末端人工累计发出量不能超过已经完成城际运输的累计到达量。
                    manual_departure_dest = gp.quicksum(
                        self.g_manual[i, j, dest_city, flow, l]
                        for (i, j) in arcs_manual_dest
                        if i <= t
                    )

                    self.model.addConstr(
                        auto_arrival_dest >= manual_departure_dest,
                        name=f"(10)transfer_dest_dir{flow}_order{l}_t{t}"
                    )
        # 约束 (11)：订单需求量与未服务量守恒。
        # 对每个订单 l 和每个城市 k，该订单在城市 k 所有人工服务弧上的承运量
        # 必须等于 d_l-z_l。由于两个城市分别使用同一个 z_l，这也保证一笔订单
        # 在始发端和目的端具有相同的最终服务量；未被完整链路服务的部分计入 z_l。
        for k in [1, 2]: 
            self.model.addConstrs(
                (gp.quicksum(
    
                    self.g_manual[i, j, k, self.data.all_orders[l].flow, l]
                    for (i, j) in (self.data.arcs_manual_1 if k == 1 else self.data.arcs_manual_2)
                 ) == self.data.all_orders[l].quantity - self.z_unserved[l]
                 
                 for l in self.data.all_orders.keys()),
                
                name=f"Demand_Conservation_City{k}"
            )

    def decision_variable_groups(self):
        """返回全部带时间弧的决策变量，供滚动时域统一固定和截断。"""

        return {
            "x_manual": self.x_manual,
            "y_auto": self.y_auto,
            "g_manual": self.g_manual,
            "g_auto": self.g_auto,
        }

    def configure_rolling_window(
        self,
        current_time: int,
        window_end: int,
        committed_decisions: Dict[str, Dict[tuple, float]],
    ):
        """固定历史决策，并关闭预测区间之外的弧。

        模型仍保留全局时间索引。这样历史弧、跨窗口在途车辆和已经承运的货量
        会继续参与原有累计平衡约束，不需要在每轮人工重建一套边界状态约束。
        """

        if not 0 <= current_time < window_end <= self.cfg.T:
            raise ValueError(
                f"非法滚动窗口：current_time={current_time}, window_end={window_end}"
            )

        for group_name, variables in self.decision_variable_groups().items():
            committed = committed_decisions.get(group_name, {})
            for key, variable in variables.items():
                departure_time = key[0]
                arrival_time = key[1]

                if departure_time < current_time:
                    value = float(committed.get(tuple(key), 0.0))
                    variable.lb = value
                    variable.ub = value
                elif departure_time >= window_end or arrival_time > window_end:
                    variable.ub = 0.0

        self.model.update()

    def extract_committed_decisions(
        self,
        start_time: int,
        commit_end: int,
        tolerance: float = 1e-7,
    ) -> Dict[str, Dict[tuple, float]]:
        """提取本轮真正执行的决策；预测区间后段的解不会被固定。"""

        decisions: Dict[str, Dict[tuple, float]] = {}
        for group_name, variables in self.decision_variable_groups().items():
            decisions[group_name] = {
                tuple(key): float(variable.X)
                for key, variable in variables.items()
                if start_time <= key[0] < commit_end
                and abs(variable.X) > tolerance
            }
        return decisions
