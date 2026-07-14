# config.py
from dataclasses import dataclass, field
from typing import Dict, Tuple

DIRECT_MODEL_SOLVERS = (
    "flexible_direct_mip",
    "flexible_direct_rolling",
    "paper_candidate_mip",
    "paper_priority_heuristic",
)


@dataclass(frozen=True)
class RollingHorizonConfig:
    """Rolling Horizon 算法参数。

    这些参数属于求解策略，而不是配送数学模型，因此与 DeliveryConfig 分开，
    避免被现有的模型参数灵敏度分析自动收集。
    """

    prediction_horizon: int = 8
    rolling_step: int = 2
    extension_horizon: int = field(
        default=6,
        metadata={
            "solvers": ("paper_candidate_mip", "paper_priority_heuristic"),
            "sensitivity_levels": [4, 6, 8],
        },
    )
    priority_epsilon: float = field(
        default=1e-6,
        metadata={
            "solvers": ("paper_priority_heuristic",),
            "sensitivity_levels": [1e-8, 1e-6, 1e-4],
        },
    )

    def validate(self) -> None:
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon 必须大于 0。")
        if self.rolling_step <= 0:
            raise ValueError("rolling_step 必须大于 0。")
        if self.rolling_step > self.prediction_horizon:
            raise ValueError("rolling_step 不能大于 prediction_horizon。")
        if self.extension_horizon < 0:
            raise ValueError("extension_horizon 不能小于 0。")
        if self.priority_epsilon <= 0:
            raise ValueError("priority_epsilon 必须大于 0。")


@dataclass(frozen=True)
class OrderGenerationConfig:
    """随机订单生成参数。

    与数学模型和求解算法分离；新增字段后，实验核心和 GUI 会自动发现它。
    """

    num_orders: int = 20
    buffer_range: Tuple[int, int] = (0, 5)
    large_order_prob: float = 0.3
    small_quantity_range: Tuple[int, int] = (10, 50)
    large_quantity_range: Tuple[int, int] = (100, 300)

    def validate(self) -> None:
        if self.num_orders <= 0:
            raise ValueError("num_orders 必须大于 0。")
        if not 0 <= self.large_order_prob <= 1:
            raise ValueError("large_order_prob 必须位于 [0, 1]。")
        for name in (
            "buffer_range",
            "small_quantity_range",
            "large_quantity_range",
        ):
            lower, upper = getattr(self, name)
            if lower < 0 or lower > upper:
                raise ValueError(f"{name} 必须满足 0 <= 下限 <= 上限。")


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
    # delta_l: unity penalty cost for lost demand is in class OrderBatch
    penalty_lost: float = 10
    # --- 4. 服务效率函数参数 ---
    # 假设 f(lambda) = a*lambda + b*sqrt(lambda)
    # 用于计算人工车辆在特定时间内的最大载货量
    service_a_1: float = 0.05
    service_b_1: float = 0.1

    service_a_2: float = 0.05
    service_b_2: float = 0.1

    # --- 5. 直送变种模型参数 ---
    # 以下参数仅供 flexible_direct_mip / flexible_direct_rolling 使用。
    # solvers 元数据使灵敏度系统不会把这些参数交给不使用直送机制的基准求解器。
    direct_travel_time_periods: int = field(
        default=4,
        metadata={
            "solvers": DIRECT_MODEL_SOLVERS,
            "sensitivity_levels": [3, 4, 5],
        },
    )
    capacity_direct: float = field(
        default=1000.0,
        metadata={
            "solvers": DIRECT_MODEL_SOLVERS,
            "sensitivity_levels": [500.0, 1000.0, 1500.0],
        },
    )
    cost_direct: float = field(
        default=25.0,
        metadata={
            "solvers": DIRECT_MODEL_SOLVERS,
            "sensitivity_levels": [15.0, 25.0, 35.0],
        },
    )
    transfer_time_periods: int = field(
        default=0,
        metadata={
            "solvers": DIRECT_MODEL_SOLVERS,
            "sensitivity_levels": [0, 1, 2, 4],
        },
    )
    transfer_cost_per_unit: float = field(
        default=0.0,
        metadata={
            "solvers": DIRECT_MODEL_SOLVERS,
            "sensitivity_levels": [0.0, 5.0, 10.0, 20.0],
        },
    )
    direct_ratio_min: float = field(
        default=0.0,
        metadata={
            "solvers": DIRECT_MODEL_SOLVERS,
            "sensitivity_levels": [0.0, 0.25, 0.5, 0.75],
        },
    )
    direct_ratio_max: float = field(
        default=1.0,
        metadata={
            "solvers": DIRECT_MODEL_SOLVERS,
            "sensitivity_levels": [0.25, 0.5, 0.75, 1.0],
        },
    )
