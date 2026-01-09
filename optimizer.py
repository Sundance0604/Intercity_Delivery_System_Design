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
    def __init__(self, config: DeliveryConfig, data: DeliveryData, orders: Dict[OrderBatch]):
        self.cfg = config
        self.data = data
        self.orders = orders
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
        self.x_manual = self.model.addVars(self.arcs_indices, self.flow, vtype=GRB.INTEGER, name="x_manual")
        # 创建变量 y 
        self.y_auto = self.model.addVars(
            self.data.arcs_auto, self.flow, vtype=GRB.INTEGER, name="y_auto"
        )
        # 创建变量 g
        self.g_manual = self.model.addVars(
            self.arcs_indices, self.orders.keys(), vtype=GRB.INTEGER, name= "g_manual"
        )
        self.g_auto = self.model.addVars(
            self.data.arcs_auto, self.orders.keys(), vtype=GRB.INTEGER, name = "g_auto"
        )
        # 创建变量 z
        self.z_unserved = self.model.addVars(
            self.orders.keys(), vtype=GRB.INTEGER, name="z_unserved"
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
        
        