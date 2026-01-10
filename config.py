# config.py
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class DeliveryConfig:
    """
    城际物流系统参数配置类
    对应Table 1: Notation List
    """
    # --- 1. 时间参数 ---
    T: int = 24                # T: discretized time periods
    t_0: float = 60.0           # t_0: duration of single period
    travel_time_periods: int = 4  # tau:driving time between cities 1 and 2

    # --- 2. 载荷参数 ---
    # N^i: number of available vehicles for city i\in{1,2}
    N_manual: Dict[int, int] = field(default_factory=lambda: {1: 30, 2: 30}) 
    # hat{N}^i: number of availabe automated vehicles
    N_auto: Dict[int, int] = field(default_factory=lambda: {1: 15, 2: 15})
    
    # M: capacity of manually driven vehicles
    capacity_manual: float = 1000.0  # 如果是乘客数目应该是int
    # hat{M}: capacity of automated vehicles
    capacity_auto: float = 2000.0   # 此应同上      

    # --- 3. 成本参数 ---
    cost_manual: float = 20.0     # c: unit driving cost for manually driven vehicles
    cost_auto: float = 15.0       # hat{c}: unit driving cost for automated vehicles
    penalty_lost: Dict[float] = 500.0   # delta_l: unity penalty cost for lost demand

    # --- 4. 服务效率函数参数 ---
    # 假设 f(lambda) = a*lambda + b*sqrt(lambda)
    # 用于计算人工车辆在特定时间内的最大载货量
    service_a_1: float = 0.05
    service_b_1: float = 0.1

    service_a_2: float = 0.05
    service_b_2: float = 0.1
