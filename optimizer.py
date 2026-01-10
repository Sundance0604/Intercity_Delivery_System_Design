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
        # 城市 1 的弧
        for (i, j) in self.data.arcs_manual_1:
            for direction in self.flow: 
                self.arcs_indices.append((i, j, 1, direction))        
        # 城市 2 的弧
        for (i, j) in self.data.arcs_manual_2:
            for direction in self.flow:
                self.arcs_indices.append((i, j, 2, direction))
        
        # 创建变量 x
        self.x_manual = self.model.addVars(self.arcs_indices, vtype=GRB.INTEGER, name="x_manual")
        # 创建变量 y 
        self.y_auto = self.model.addVars(
            self.data.arcs_auto, self.flow, vtype=GRB.INTEGER, name="y_auto"
        )
        # 创建变量 g
        self.g_manual = self.model.addVars(
            self.arcs_indices, self.data.all_orders.keys(), vtype=GRB.INTEGER, name= "g_manual"
        )
        self.g_auto = self.model.addVars(
            self.data.arcs_auto, self.data.all_orders.keys(), vtype=GRB.INTEGER, name = "g_auto"
        )
        # 创建变量 z
        self.z_unserved = self.model.addVars(
            self.data.all_orders.keys(), vtype=GRB.INTEGER, name="z_unserved"
        )

    def set_objective(self):
        # 未服务惩罚
        self.penalty_unserved = gp.quicksum(
            self.cfg.penalty_lost[k] * self.z_unserved[k] for k in self.orders.keys()
        )
        # 人工车辆成本
        self.cost_manual = gp.quicksum(
            self.cfg.cost_manual * self.cfg.t_0 * (j-i) * self.x_manual[i, j, city, direction] 
            for (i, j, city, direction) in self.arcs_indices
        )
        # 自动驾驶车辆成本
        self.cost_auto = self.cfg.capacity_auto * self.cfg.travel_time_periods * self.cfg.t_0 * self.y_auto.sum()
        # 总目标函数
        self.model.setObejctive(
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
                    self.x_manual[i, j, city, direction]
                    for (i, j) in active_arcs
                    for direction in self.flow
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
                self.y_auto[i, j, direction]
                for (i, j) in self.data.sets_auto[t]
                for direction in self.flow
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
                for (i, j) in self.data.sets_auto[t] if i < t
            )
            # 逆流计算
            negative_flow = gp.quicksum(
                self.y_auto[i, j, "-"]
                for (i, j) in self.data.sets_auto[t] if i < t
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
        for (i, j, city, direction) in self.arcs_indices:
            coeff_dict = (
                self.data.cap_coeff_1 if city == 1
                else self.data.cap_coeff_2
            )
            orders = (
                self.data.pos_orders if direction == "+"
                else self.data.neg_orders
            )
            self.model.addConstr(
                gp.quicksum(
                    self.g_manual[i, j, city, direction, l] <= self.x_manual[i, j, city, direction] * coeff_dict[(i, j)]
                    for l in orders.keys()
                ),
                name=f"(6)Manual_intercity_volume_time{t}_city{city}_dir{direction}"
            )
        
