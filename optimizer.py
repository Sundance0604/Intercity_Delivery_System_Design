import gurobipy as gp
from gurobipy import GRB
from config import DeliveryConfig
from data_loader import DeliveryData, DataLoader, OrderBatch
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
        self.orders_indices = []
        # 城市 1 的弧
        for (i, j) in self.data.arcs_manual_1:
            for flow in self.flow: 
                self.arcs_indices.append((i, j, 1, flow))
                self.orders_indices.append((i, j, flow))        
        # 城市 2 的弧
        for (i, j) in self.data.arcs_manual_2:
            for flow in self.flow:
                self.arcs_indices.append((i, j, 2, flow))
                self.orders_indices.append((i, j, flow))      
        # 创建变量 x
        self.x_manual = self.model.addVars(self.arcs_indices, vtype=GRB.INTEGER, name="x_manual")
        # 创建变量 y 
        self.y_auto = self.model.addVars(
            self.data.arcs_auto, self.flow, vtype=GRB.INTEGER, name="y_auto"
        )
        # 创建变量 g
        self.g_manual = self.model.addVars(
            self.orders_indices, self.data.all_orders.keys(), vtype=GRB.INTEGER, name= "g_manual"
        )
        self.g_auto = self.model.addVars(
            self.data.arcs_auto, self.flow, self.data.all_orders.keys(), vtype=GRB.INTEGER, name = "g_auto"
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
            self.cfg.cost_manual * self.cfg.t_0 * (j-i) * self.x_manual[i, j, city, flow] 
            for (i, j, city, flow) in self.arcs_indices
        )
        # 自动驾驶车辆成本
        self.cost_auto = self.cfg.cost_auto * self.cfg.travel_time_periods * self.cfg.t_0 * self.y_auto.sum()
        # 总目标函数
        self.model.setObjective(
            self.penalty_unserved + self.cost_manual + self.cost_auto,
            GRB.MINIMIZE
        )
        
    def set_constraints(self):
        # 建立第一个约束(2)
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
        # 建立第二个约束(3)
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
        # 建立第三、四个约束(4)(5)
        for t in range(self.cfg.T):
            # ㊣流计算
            postive_flow = gp.quicksum(
                self.y_auto[i, j, "+"]
                for (i, j) in self.data.arcs_auto[t] if i < t
            )
            # 逆流计算
            negative_flow = gp.quicksum(
                self.y_auto[i, j, "-"]
                for (i, j) in self.data.arcs_auto[t] if i < t
            )
            # 添加约束：㊣流 - 逆流 + \hat{N}^1 \geq 0
            self.model.addConstr(
                postive_flow - negative_flow + self.cfg.N_auto[1] >= 0,
                name=f"(4)Intercity_Postive_Flow_Balance_Time{t}"
            )
            # 添加约束：㊣流 - 逆流 + \hat{N}^2 \geq 0
            self.model.addConstr(
                negative_flow - postive_flow + self.cfg.N_auto[2] >= 0,
                name=f"(5)Intercity_Negtive_Flow_Balance_Time{t}"
            )
        # 建立第五个约束(6)
        for (i, j, city, flow) in self.arcs_indices:
            coeff_dict = (
                self.data.cap_coeff_1 if city == 1
                else self.data.cap_coeff_2
            )
            orders = (
                self.data.pos_orders if flow == "+"
                else self.data.neg_orders
            )
            self.model.addConstr(
                gp.quicksum(
                    self.g_manual[i, j, flow, l] <= self.x_manual[i, j, city, flow] * coeff_dict[(i, j)]
                    for l in orders.keys()
                ),
                name=f"(6)Manual_intercity_volume_city{city}_dir{flow}"
            )
        # 建立第六个约束(7)
        self.model.addConstrs(
            (self.g_auto[i, j, flow, l] <= self.y_auto[i, j, flow] * self.cfg.capacity_auto
            for (i, j) in self.data.arcs_auto
            for flow in self.flow
            for l in (self.data.pos_orders if flow == "+" else self.data.neg_orders)),
            name="(7)Auto_intercity_volume"
        )
        # 建立第七个约束(8)
        self.model.addConstrs(
            (gp.quicksum(self.g_auto[i, j, flow, l] for l in (self.data.pos_orders if flow == "+" else self.data.neg_orders))
             <= self.y_auto[i, j, flow] * self.cfg.capacity_auto
            for (i, j) in self.data.arcs_auto
            for flow in self.flow),
            name="(7)Auto_Capacity_Total"
        )
        # 建立第八、九个约束(9)(10)
        for t in range(self.cfg.T):  
            for flow in self.flow:
                if flow == "+":
                    # 正向 (+): City 1 (Origin) -> Auto -> City 2 (Dest)
                    orders = self.data.pos_orders
                    arcs_manual_origin = self.data.arcs_manual_1 
                    arcs_manual_dest   = self.data.arcs_manual_2  
                else:
                    # 反向 (-): City 2 (Origin) -> Auto -> City 1 (Dest)
                    orders = self.data.neg_orders
                    arcs_manual_origin = self.data.arcs_manual_2
                    arcs_manual_dest   = self.data.arcs_manual_1

                auto_departure_origin = gp.quicksum(
                    self.g_auto[i, j, flow, l]
                    for (i, j) in self.data.arcs_auto
                    for l in orders.keys()
                    if i <= t
                )

                manual_arrival_origin = gp.quicksum(
                    self.g_manual[i, j, flow, l]
                    for (i, j) in arcs_manual_origin
                    for l in orders.keys()
                    if j <= t
                )

                self.model.addConstr(
                    auto_departure_origin <= manual_arrival_origin,
                    name=f"(9)transfer_origin_dir{flow}_t{t}"
                )
                # 这里是约束(10)
                auto_arrival_dest = gp.quicksum(
                    self.g_auto[i, j, flow, l]
                    for (i, j) in self.data.arcs_auto
                    for l in orders.keys()
                    if j <= t
                )

                manual_departure_dest = gp.quicksum(
                    self.g_manual[i, j, flow, l]
                    for (i, j) in arcs_manual_dest
                    for l in orders.keys()
                    if i <= t
                )

                self.model.addConstr(
                    auto_arrival_dest >= manual_departure_dest,
                    name=f"(10)transfer_dest_dir{flow}_t{t}"
                )
        # 建立第十个约束(11)
        self.model.addConstrs(
        (gp.quicksum(
            self.g_manual[i, j, order.flow, l]
            for (i, j) in (self.data.arcs_manual_1 if order.flow == "+" else self.data.arcs_manual_2)
         ) == order.quantity - self.z_unserved[l]
         for l, order in self.data.all_orders.items()),
            name="unserved_passenger_volume"
        )
                      
