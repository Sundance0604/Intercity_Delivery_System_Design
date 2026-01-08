# data_loader.py
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
from config import DeliveryConfig

@dataclass
class DeliveryData:
   
    # 集合定义
    arcs_manual_1: List[Tuple[int, int]]       # A^1: 城市1中人工车辆所有可能的时间弧 (i, j)
    arcs_manual_2: List[Tuple[int, int]]       # A^2: 城市2中人工车辆所有可能的时间弧 (i, j)
    arcs_auto: List[Tuple[int, int]]         # hat{A}: 自动驾驶车辆所有可能的时间弧 (i, j)
    
    # 状态映射 S^k(t) 和 hat{S}(t)
    # key: 时间点 t, value: 覆盖该时间点的路径列表 [(i, j), ...]
    active_manual_sets_1: Dict[int, List[Tuple[int, int]]]
    active_manual_sets_2: Dict[int, List[Tuple[int, int]]]
    active_auto_sets: Dict[int, List[Tuple[int, int]]]
    
    # 预计算参数
    # key: (i, j), value: 对应路径的最大载重量 (公式6右侧部分)
    manual_capacity_coeff: Dict[Tuple[int, int], float] 
    
class DataLoader:
    def __init__(self, config: DeliveryConfig):
        self.cfg = config
    # 为城市1的BHH近似
    def BHH_function_1(self, load: float) -> float:
        a = self.cfg.service_a_1
        b = self.cfg.service_b_1
        return a * load + b * math.sqrt(load)
    # 为城市2的BHH近似
    def BHH_function_2(self, load: float) -> float:
        a = self.cfg.service_a_2
        b = self.cfg.service_b_2
        return a * load + b * math.sqrt(load)

    # 为城市1的反函数
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
    def generate_arcs_manual(self) -> List[Tuple[int, int]]:
        arcs_manual_1 = []
        arcs_manual_2 = []
        for i in range(self.cfg.T):
            for j in range(i+1, i+self._BHH_function_1(self.cfg.N_manual[1])):
                arcs_manual_1.append((i, j))
            for j in range(i+1, i+self._BHH_function_2(self.cfg.N_manual[2])):
                arcs_manual_2.append((i, j))
        return arcs_manual_1, arcs_manual_2
     # 自动驾驶车辆: 固定行驶时间 tau
    def generate_arcs_auto(self) -> List[Tuple[int, int]]:
        # j = i + tau
        arcs_auto = []
        tau = self.cfg.travel_time_periods
        for i in range(self.cfg.T - tau + 1):
            j = i + tau
            arcs_auto.append((i, j))
        return arcs_auto
    def generate_arcs_for_t(self, t: int, arcs:List) -> List[Tuple[int, int]]:
        arcs_t = []
        # 计算活跃集合 S(t)
        # S(t) = {(i, j) | i <= t < j}
        for (i, j) in arcs:
            for t in range(i, j):
                if t < self.cfg.T:
                    arcs_t[t].append((i, j))
        return arcs_t
           
    # 预计算公式 6所需的参数
    # lambda = (f)^-1( (j-i)*t0 )
    def pre_inverse_count(self, arcs_manual_1:List, arcs_manual_2:List) -> Dict[Tuple[int, int], float]:
        cap_coeff_1 = {}
        cap_coeff_2 = {}
        # 对第一个城市计算
        for (i, j) in arcs_manual_1:
            duration = (j - i) * self.cfg.t0
            # 这里调用反函数求解
            max_load = self.inverse_function_1(duration)
            cap_coeff_1[(i, j)] = max_load
        # 对第二个城市计算
        for (i, j) in arcs_manual_2:
            duration = (j - i) * self.cfg.t0
            # 这里调用反函数求解
            max_load = self.inverse_function_2(duration)
            cap_coeff_2[(i, j)] = max_load

        return cap_coeff_1, cap_coeff_2
    
@dataclass
class OrderBatch:
    batch_id: int            # 对应 l
    direction: int           # 对应 方向 (1或2)
    quantity: int            # 对应 d_l 
    earliest_start: int      # 对应 s_l 
    latest_completion: int   # 对应 e_l