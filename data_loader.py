# data_loader.py
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
from config import DeliveryConfig
from functools import lru_cache
from typing import Iterator


@dataclass
class DeliveryData:
   
    # 集合定义
    arcs_manual_1: List[Tuple[int, int]]       # A^1: 城市1中人工车辆所有可能的时间弧 (i, j)
    arcs_manual_2: List[Tuple[int, int]]       # A^2: 城市2中人工车辆所有可能的时间弧 (i, j)
    arcs_auto: List[Tuple[int, int]]         # hat{A}: 自动驾驶车辆所有可能的时间弧 (i, j)
    
    # 状态映射 S^k(t) 和 hat{S}(t)
    # key: 时间点 t, value: 覆盖该时间点的路径列表 [(i, j), ...]
    sets_manual_1: Dict[int, List[Tuple[int, int]]]
    sets_manual_2: Dict[int, List[Tuple[int, int]]]
    sets_auto: Dict[int, List[Tuple[int, int]]]
    
    # 预计算参数
    # key: (i, j), value: 对应路径的最大载重量 (公式6右侧部分)
    cap_coeff_1: Dict[Tuple[int, int], float]
    cap_coeff_2: Dict[Tuple[int, int], float]

    # order categorization set
    pos_orders: Dict[int, List[int]]  # 正向订单集合
    neg_orders: Dict[int, List[int]]  # 反向订单集合
    all_orders: Dict[int, List[int]]  # 所有订单集合

    # the set of time pairs that violate the time-window constraints
    epsilon_sets: List[Tuple[int, int, str, int]]
    
class DataLoader:
    def __init__(self, config: DeliveryConfig):
        self.cfg = config
    # 为城市1的BHH近似
    @lru_cache(maxsize=1024)
    def BHH_function_1(self, load: float) -> float:
        return self.cfg.service_a_1 * load + self.cfg.service_b_1 * math.sqrt(load)
    # 为城市2的BHH近似
    @lru_cache(maxsize=1024)
    def BHH_function_2(self, load: float) -> float:
        return self.cfg.service_a_2 * load + self.cfg.service_b_2 * math.sqrt(load)

    # 为城市1的反函数
    @lru_cache(maxsize=1024)
    def inverse_function_1(self, duration_minutes: float) -> float:
        """
        求解反函数 (f^1)^(-1)(T)。
        已知: T = a * lambda + b * sqrt(lambda)
        求解: lambda
        由 u = sqrt(lambda), 则 a*u^2 + b*u - T = 0
        """
        a = self.cfg.service_a_1
        b = self.cfg.service_b_1
        T_val = duration_minutes
        
        # 二次方程求根公式: u = (-b + sqrt(b^2 + 4aT)) / 2a
        delta = b**2 + 4 * a * T_val
        u = (-b + math.sqrt(delta)) / (2 * a)
        return u**2
    
    # 为城市2的反函数
    @lru_cache(maxsize=1024)
    def inverse_function_2(self, duration_minutes: float) -> float:
        """
        求解反函数 (f^2)^(-1)(T)。
        已知: T = a * lambda + b * sqrt(lambda)
        求解: lambda 
        由 u = sqrt(lambda), 则 a*u^2 + b*u - T = 0
        """
        a = self.cfg.service_a_2
        b = self.cfg.service_b_2
        T_val = duration_minutes
        
        # 二次方程求根公式: u = (-b + sqrt(b^2 + 4aT)) / 2a
        delta = b**2 + 4 * a * T_val
        u = (-b + math.sqrt(delta)) / (2 * a)
        return u**2
     # 人工车辆:  j > i 且 j <= i+f^k(M) 即可
    def _generate_single_city_arcs(self, service_func) -> Iterator[Tuple[int, int]]:
        for i in range(self.cfg.T):
            limit = i + int(service_func(self.cfg.capacity_manual))
            for j in range(i + 1, limit):
                yield (i, j)  

    def generate_arcs_manual(self):
        iter_1 = list(self._generate_single_city_arcs(self.BHH_function_1))
        iter_2 = list(self._generate_single_city_arcs(self.BHH_function_2))
        return iter_1, iter_2
    
     # 自动驾驶车辆: 固定行驶时间 tau
    def generate_arcs_auto(self) -> List[Tuple[int, int]]:
        # j = i + tau
        arcs_auto = []
        tau = self.cfg.travel_time_periods
        for i in range(self.cfg.T - tau + 1):
            j = i + tau
            arcs_auto.append((i, j))
        return arcs_auto
    # 生成S^k(t)和\hat{S}(t)
    def generate_sets(self, arcs_manual_1, arcs_manual_2, arcs_auto):
      
        sets_manual_1 = {t: [] for t in range(self.cfg.T)}    
        for (i, j) in arcs_manual_1:
            for t in range(i, j):
                if t < self.cfg.T:
                    sets_manual_1[t].append((i, j))

        sets_manual_2 = {t: [] for t in range(self.cfg.T)}
        for (i, j) in arcs_manual_2:
            for t in range(i, j):
                if t < self.cfg.T:
                    sets_manual_2[t].append((i, j))

        sets_auto = {t: [] for t in range(self.cfg.T)}
        for (i, j) in arcs_auto:
            for t in range(i, j):
                if t < self.cfg.T:
                    sets_auto[t].append((i, j))

        return sets_manual_1, sets_manual_2, sets_auto
    # 生成约束七中所出现的集合epsilon
    def generate_epsilon_sets(self, pos_orders, neg_orders, arcs_manual_1, arcs_manual_2):
        
        infeasible_sets = []

        # 1. 处理正向订单 (City 1 -> City 2)
        for l, order in pos_orders.items():
            # 出发太早 (i < s_l)
           
            infeasible_sets.extend([
                (i, j, 1, "+", l)  
                for (i, j) in arcs_manual_1
                if i < order.earliest_start  
            ])
            # 到达太晚 (j > e_l)
            infeasible_sets.extend([
                (i, j, 2, "+", l)  
                for (i, j) in arcs_manual_2
                if j > order.latest_completion 
            ])

        # 2. 处理反向订单 (City 2 -> City 1)
        for l, order in neg_orders.items():
            # 出发太早
            infeasible_sets.extend([
                (i, j, 2, "-", l)  
                for (i, j) in arcs_manual_2   
                if i < order.earliest_start   
            ])
            # 到达太晚
            infeasible_sets.extend([
                (i, j, 1, "-", l) 
                for (i, j) in arcs_manual_1   
                if j > order.latest_completion 
            ])
                
        return infeasible_sets
    # 预计算公式 6所需的参数
    # lambda = (f)^-1( (j-i)*t0 )
    def pre_inverse_count(self, arcs_manual_1:List, arcs_manual_2:List) -> Dict[Tuple[int, int], float]:
        cap_coeff_1 = {}
        cap_coeff_2 = {}
        # 对第一个城市计算
        for (i, j) in arcs_manual_1:
            duration = (j - i) * self.cfg.t_0
            # 这里调用反函数求解
            max_load = self.inverse_function_1(duration)
            cap_coeff_1[(i, j)] = max_load
        # 对第二个城市计算
        for (i, j) in arcs_manual_2:
            duration = (j - i) * self.cfg.t_0
            # 这里调用反函数求解
            max_load = self.inverse_function_2(duration)
            cap_coeff_2[(i, j)] = max_load

        return cap_coeff_1, cap_coeff_2
    
@dataclass
class OrderBatch:
    batch_id: int            # 对应 l
    flow: str           # 对应 方向,记为+或-
    quantity: int            # 对应 d_l 
    earliest_start: int      # 对应 s_l 
    latest_completion: int   # 对应 e_l
    penalty_lost: float      # 对应 delta_l