"""允许“换装运输 + 人工车辆跨城直送”共存的城际配送优化模型。

模型假设
--------
1. 直送车辆来自两个城市原有的人工驾驶车队，而不是额外无限车源。
2. 直送车辆从一个城市出发后停留在目的城市，后续可以执行当地任务或反向直送。
3. OrderBatch 表示可拆分的货量批次，因此同一批次可以部分直送、部分换装运输。
4. direct_ratio_min/max 是已服务货量口径的外生政策边界；默认 [0, 1] 表示让模型
   自主选择最优直送比例。
"""

import math
from typing import Dict

import gurobipy as gp
from gurobipy import GRB

from config import DeliveryConfig
from data_loader import DeliveryData


class FlexibleDirectOptimizer:
    """完整构建并求解直送/换装共存的混合整数规划。"""

    def __init__(self, config: DeliveryConfig, data: DeliveryData):
        self.cfg = config
        self.data = data
        self.model = gp.Model("Flexible_Direct_Intercity_Delivery")
        self.flow = ["+", "-"]
        self._validate_config()
        self._prepare_indices()

    def _validate_config(self):
        """在建模前阻止没有业务含义的直送参数组合。"""

        if self.cfg.direct_travel_time_periods <= 0:
            raise ValueError("direct_travel_time_periods 必须大于 0。")
        if self.cfg.capacity_direct <= 0:
            raise ValueError("capacity_direct 必须大于 0。")
        if self.cfg.cost_direct < 0 or self.cfg.transfer_cost_per_unit < 0:
            raise ValueError("直送成本和换装成本不能为负。")
        if self.cfg.transfer_time_periods < 0:
            raise ValueError("transfer_time_periods 不能为负。")
        if not 0 <= self.cfg.direct_ratio_min <= self.cfg.direct_ratio_max <= 1:
            raise ValueError(
                "直送比例必须满足 0 <= direct_ratio_min <= "
                "direct_ratio_max <= 1。"
            )

    def _prepare_indices(self):
        """生成城市人工弧索引和端到端直送弧及其单车容量。"""

        self.arcs_indices = []
        for i, j in self.data.arcs_manual_1:
            for flow in self.flow:
                self.arcs_indices.append((i, j, 1, flow))
        for i, j in self.data.arcs_manual_2:
            for flow in self.flow:
                self.arcs_indices.append((i, j, 2, flow))

        self.arcs_direct = []
        self.direct_capacity_coeff = {}
        tau = self.cfg.direct_travel_time_periods

        # 一辆直送车需要先在始发城市服务，再跨城行驶，最后在目的城市服务。
        # 两端服务总时间采用 f1(q)+f2(q)；对每个候选持续时间反求最大可服务货量。
        for i in range(self.cfg.T):
            for j in range(i + tau + 1, self.cfg.T + 1):
                service_minutes = (j - i - tau) * self.cfg.t_0
                capacity = min(
                    self.cfg.capacity_direct,
                    self._inverse_combined_service_time(service_minutes),
                )
                if capacity > 1e-9:
                    self.arcs_direct.append((i, j))
                    self.direct_capacity_coeff[(i, j)] = capacity

    def _inverse_combined_service_time(self, duration_minutes: float) -> float:
        """求 f1(q)+f2(q)=duration 的非负反函数。"""

        a = self.cfg.service_a_1 + self.cfg.service_a_2
        b = self.cfg.service_b_1 + self.cfg.service_b_2
        if duration_minutes <= 0:
            return 0.0
        if a <= 0:
            return (duration_minutes / b) ** 2 if b > 0 else 0.0
        root_q = (-b + math.sqrt(b * b + 4 * a * duration_minutes)) / (2 * a)
        return max(0.0, root_q * root_q)

    def setup_variables(self):
        """创建换装链变量、直送变量和需求拆分变量。"""

        # x：仅执行城市内集货或末端配送的人工车辆。
        self.x_manual = self.model.addVars(
            self.arcs_indices, vtype=GRB.INTEGER, lb=0, name="x_manual"
        )

        # y：在两个换装中心之间执行自动驾驶干线运输的车辆。
        self.y_auto = self.model.addVars(
            self.data.arcs_auto,
            self.flow,
            vtype=GRB.INTEGER,
            lb=0,
            name="y_auto",
        )

        # w：同一人工车辆从始发城市取货、跨城并在目的城市完成配送。
        self.w_direct = self.model.addVars(
            self.arcs_direct,
            self.flow,
            vtype=GRB.INTEGER,
            lb=0,
            name="w_direct",
        )

        # g_manual：选择换装机制的订单在两个城市人工服务弧上的货量。
        self.g_manual = self.model.addVars(
            self.arcs_indices,
            self.data.all_orders.keys(),
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="g_manual",
        )

        # g_auto：选择换装机制的订单在自动驾驶干线弧上的货量。
        self.g_auto = self.model.addVars(
            self.data.arcs_auto,
            self.flow,
            self.data.all_orders.keys(),
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="g_auto",
        )

        # h_direct：订单在端到端直送弧上的货量。只创建订单真实方向上的变量。
        direct_load_indices = [
            (i, j, flow, order_id)
            for i, j in self.arcs_direct
            for flow, orders in (
                ("+", self.data.pos_orders),
                ("-", self.data.neg_orders),
            )
            for order_id in orders
        ]
        self.h_direct = self.model.addVars(
            direct_load_indices,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="h_direct",
        )

        # r、q、z 分别表示每笔订单的换装货量、直送货量和未服务货量。
        orders = self.data.all_orders.keys()
        self.r_transshipment = self.model.addVars(
            orders, vtype=GRB.CONTINUOUS, lb=0, name="r_transshipment"
        )
        self.q_direct = self.model.addVars(
            orders, vtype=GRB.CONTINUOUS, lb=0, name="q_direct"
        )
        self.z_unserved = self.model.addVars(
            orders, vtype=GRB.CONTINUOUS, lb=0, name="z_unserved"
        )

    def set_objective(self):
        """最小化未服务、城市人工、自动干线、直送和换装处理成本。"""

        self.penalty_unserved = gp.quicksum(
            order.penalty_lost * self.z_unserved[order_id]
            for order_id, order in self.data.all_orders.items()
        )
        self.cost_manual = gp.quicksum(
            self.cfg.cost_manual
            * self.cfg.t_0
            * (j - i)
            * self.x_manual[i, j, city, flow]
            for i, j, city, flow in self.arcs_indices
        )
        self.cost_auto = (
            self.cfg.cost_auto
            * self.cfg.travel_time_periods
            * self.cfg.t_0
            * self.y_auto.sum()
        )
        self.cost_direct = gp.quicksum(
            self.cfg.cost_direct
            * self.cfg.t_0
            * (j - i)
            * self.w_direct[i, j, flow]
            for i, j in self.arcs_direct
            for flow in self.flow
        )
        self.cost_transfer = self.cfg.transfer_cost_per_unit * gp.quicksum(
            self.r_transshipment[order_id]
            for order_id in self.data.all_orders
        )

        self.model.setObjective(
            self.penalty_unserved
            + self.cost_manual
            + self.cost_auto
            + self.cost_direct
            + self.cost_transfer,
            GRB.MINIMIZE,
        )

    def set_constraints(self):
        """建立全部车辆、容量、时间窗、流量和比例约束。"""

        self._add_manual_fleet_constraints()
        self._add_auto_fleet_constraints()
        self._add_capacity_constraints()
        self._add_time_window_constraints()
        self._add_transfer_sequence_constraints()
        self._add_demand_split_constraints()
        self._add_direct_ratio_constraints()

    def _add_manual_fleet_constraints(self):
        """约束 FD-(2)：人工车队同时承担城市内任务与跨城直送。

        直送会改变车辆所在城市。以城市1为例：
        城市内正在服务的车辆 + 累计从城市1直送出发的车辆
        - 累计反向直送到达城市1的车辆 <= 城市1初始人工车数。
        城市2采用完全对称的表达式。该写法也会自动占用尚在途的直送车辆。
        """

        for t in range(self.cfg.T):
            for city in (1, 2):
                active_arcs = (
                    self.data.sets_manual_1[t]
                    if city == 1
                    else self.data.sets_manual_2[t]
                )
                local_active = gp.quicksum(
                    self.x_manual[i, j, city, flow]
                    for i, j in active_arcs
                    for flow in self.flow
                )

                outbound_flow = "+" if city == 1 else "-"
                inbound_flow = "-" if city == 1 else "+"
                cumulative_outbound = gp.quicksum(
                    self.w_direct[i, j, outbound_flow]
                    for i, j in self.arcs_direct
                    if i <= t
                )
                cumulative_inbound = gp.quicksum(
                    self.w_direct[i, j, inbound_flow]
                    for i, j in self.arcs_direct
                    if j <= t
                )

                self.model.addConstr(
                    local_active
                    + cumulative_outbound
                    - cumulative_inbound
                    <= self.cfg.N_manual[city],
                    name=f"FD_(2)_ManualFleet_City{city}_Time{t}",
                )

    def _add_auto_fleet_constraints(self):
        """约束 FD-(3)–FD-(5)：自动车在途规模和城市库存平衡。

        FD-(3)限制两个方向同时在途的自动车总数。
        FD-(4)/(5)使用“累计出发 <= 初始库存 + 累计反向到达”，避免车辆在尚未
        到达另一城市前被重复派遣。
        """

        for t in range(self.cfg.T):
            auto_active = gp.quicksum(
                self.y_auto[i, j, flow]
                for i, j in self.data.sets_auto[t]
                for flow in self.flow
            )
            self.model.addConstr(
                auto_active <= sum(self.cfg.N_auto.values()),
                name=f"FD_(3)_AutoFleet_Time{t}",
            )

            pos_departures = gp.quicksum(
                self.y_auto[i, j, "+"]
                for i, j in self.data.arcs_auto
                if i <= t
            )
            neg_departures = gp.quicksum(
                self.y_auto[i, j, "-"]
                for i, j in self.data.arcs_auto
                if i <= t
            )
            pos_arrivals = gp.quicksum(
                self.y_auto[i, j, "+"]
                for i, j in self.data.arcs_auto
                if j <= t
            )
            neg_arrivals = gp.quicksum(
                self.y_auto[i, j, "-"]
                for i, j in self.data.arcs_auto
                if j <= t
            )

            self.model.addConstr(
                pos_departures <= self.cfg.N_auto[1] + neg_arrivals,
                name=f"FD_(4)_AutoBalance_City1_Time{t}",
            )
            self.model.addConstr(
                neg_departures <= self.cfg.N_auto[2] + pos_arrivals,
                name=f"FD_(5)_AutoBalance_City2_Time{t}",
            )

    def _add_capacity_constraints(self):
        """约束 FD-(6)–FD-(8)：三类运输任务的共享容量。

        每一条弧都先对该方向全部订单货量求和，再与车辆数乘单车容量比较。
        这样一辆车的容量不会被多笔订单分别、重复使用。
        """

        # FD-(6)：城市人工服务弧容量。
        for i, j, city, flow in self.arcs_indices:
            coeff = (
                self.data.cap_coeff_1[(i, j)]
                if city == 1
                else self.data.cap_coeff_2[(i, j)]
            )
            orders = (
                self.data.pos_orders if flow == "+" else self.data.neg_orders
            )
            self.model.addConstr(
                gp.quicksum(
                    self.g_manual[i, j, city, flow, order_id]
                    for order_id in orders
                )
                <= coeff * self.x_manual[i, j, city, flow],
                name=f"FD_(6)_ManualCapacity_{city}_{flow}_{i}_{j}",
            )

        # FD-(7)：自动驾驶干线弧容量。这里采用总货量容量，而非逐订单容量。
        for i, j in self.data.arcs_auto:
            for flow in self.flow:
                orders = (
                    self.data.pos_orders if flow == "+" else self.data.neg_orders
                )
                self.model.addConstr(
                    gp.quicksum(
                        self.g_auto[i, j, flow, order_id]
                        for order_id in orders
                    )
                    <= self.cfg.capacity_auto * self.y_auto[i, j, flow],
                    name=f"FD_(7)_AutoCapacity_{flow}_{i}_{j}",
                )

        # FD-(8)：端到端直送弧容量。
        for i, j in self.arcs_direct:
            for flow in self.flow:
                orders = (
                    self.data.pos_orders if flow == "+" else self.data.neg_orders
                )
                self.model.addConstr(
                    gp.quicksum(
                        self.h_direct[i, j, flow, order_id]
                        for order_id in orders
                    )
                    <= self.direct_capacity_coeff[(i, j)]
                    * self.w_direct[i, j, flow],
                    name=f"FD_(8)_DirectCapacity_{flow}_{i}_{j}",
                )

    def _add_time_window_constraints(self):
        """约束 FD-(9)–FD-(10)：换装链和直送链都必须满足订单时间窗。"""

        # FD-(9)：沿用基础模型的城市人工服务时间窗。
        self.model.addConstrs(
            (
                self.g_manual[i, j, city, flow, order_id] == 0
                for i, j, city, flow, order_id in self.data.epsilon_sets
            ),
            name="FD_(9)_TransshipmentTimeWindow",
        )

        # FD-(10)：直送弧从取货开始到最终送达均需落在 [s_l, e_l] 内。
        for order_id, order in self.data.all_orders.items():
            for i, j in self.arcs_direct:
                if i < order.earliest_start or j > order.latest_completion:
                    self.model.addConstr(
                        self.h_direct[i, j, order.flow, order_id] == 0,
                        name=f"FD_(10)_DirectTimeWindow_{order_id}_{i}_{j}",
                    )

    def _add_transfer_sequence_constraints(self):
        """约束 FD-(11)–FD-(13)：换装货量按取货、干线、末端配送依次流动。

        transfer_time_periods 同时作为始发换装和目的换装的处理时间。累计干线出发
        不能超过足够早已到达始发换装点的货量；累计末端配送出发不能超过足够早
        已到达目的换装点的干线货量。最后令每笔订单的干线总货量等于 r_l。
        """

        theta = self.cfg.transfer_time_periods
        for flow, orders, origin_city, dest_city, origin_arcs, dest_arcs in (
            (
                "+",
                self.data.pos_orders,
                1,
                2,
                self.data.arcs_manual_1,
                self.data.arcs_manual_2,
            ),
            (
                "-",
                self.data.neg_orders,
                2,
                1,
                self.data.arcs_manual_2,
                self.data.arcs_manual_1,
            ),
        ):
            for order_id in orders:
                for t in range(self.cfg.T):
                    auto_departed = gp.quicksum(
                        self.g_auto[i, j, flow, order_id]
                        for i, j in self.data.arcs_auto
                        if i <= t
                    )
                    origin_ready = gp.quicksum(
                        self.g_manual[i, j, origin_city, flow, order_id]
                        for i, j in origin_arcs
                        if j <= t - theta
                    )
                    self.model.addConstr(
                        auto_departed <= origin_ready,
                        name=f"FD_(11)_OriginTransfer_{flow}_{order_id}_{t}",
                    )

                    auto_ready_at_destination = gp.quicksum(
                        self.g_auto[i, j, flow, order_id]
                        for i, j in self.data.arcs_auto
                        if j <= t - theta
                    )
                    destination_departed = gp.quicksum(
                        self.g_manual[i, j, dest_city, flow, order_id]
                        for i, j in dest_arcs
                        if i <= t
                    )
                    self.model.addConstr(
                        destination_departed <= auto_ready_at_destination,
                        name=f"FD_(12)_DestinationTransfer_{flow}_{order_id}_{t}",
                    )

                self.model.addConstr(
                    gp.quicksum(
                        self.g_auto[i, j, flow, order_id]
                        for i, j in self.data.arcs_auto
                    )
                    == self.r_transshipment[order_id],
                    name=f"FD_(13)_AutoVolume_{order_id}",
                )

    def _add_demand_split_constraints(self):
        """约束 FD-(14)–FD-(17)：每笔订单拆成换装、直送和未服务三部分。"""

        for order_id, order in self.data.all_orders.items():
            flow = order.flow

            # FD-(14)：始发端人工集货总量必须等于该订单的换装货量 r_l。
            origin_city = 1 if flow == "+" else 2
            origin_arcs = (
                self.data.arcs_manual_1
                if origin_city == 1
                else self.data.arcs_manual_2
            )
            self.model.addConstr(
                gp.quicksum(
                    self.g_manual[i, j, origin_city, flow, order_id]
                    for i, j in origin_arcs
                )
                == self.r_transshipment[order_id],
                name=f"FD_(14)_OriginManualVolume_{order_id}",
            )

            # FD-(15)：目的端人工配送总量也必须等于同一个换装货量 r_l。
            dest_city = 2 if flow == "+" else 1
            dest_arcs = (
                self.data.arcs_manual_2
                if dest_city == 2
                else self.data.arcs_manual_1
            )
            self.model.addConstr(
                gp.quicksum(
                    self.g_manual[i, j, dest_city, flow, order_id]
                    for i, j in dest_arcs
                )
                == self.r_transshipment[order_id],
                name=f"FD_(15)_DestinationManualVolume_{order_id}",
            )

            # FD-(16)：所有直送弧货量之和定义为该订单的直送货量 q_l。
            self.model.addConstr(
                gp.quicksum(
                    self.h_direct[i, j, flow, order_id]
                    for i, j in self.arcs_direct
                )
                == self.q_direct[order_id],
                name=f"FD_(16)_DirectVolume_{order_id}",
            )

            # FD-(17)：需求守恒。每单位需求只能属于换装、直送或未服务之一。
            self.model.addConstr(
                self.r_transshipment[order_id]
                + self.q_direct[order_id]
                + self.z_unserved[order_id]
                == order.quantity,
                name=f"FD_(17)_DemandSplit_{order_id}",
            )

    def _add_direct_ratio_constraints(self):
        """约束 FD-(18)–FD-(19)：控制已服务货量中的总体直送比例。

        alpha 是给定常数，所以 q >= alpha*(d-z) 仍然是线性约束。
        默认上下界 [0,1] 不限制机制选择，求解结果中的比例是模型内生结果。
        """

        direct_volume = gp.quicksum(
            self.q_direct[order_id] for order_id in self.data.all_orders
        )
        served_volume = gp.quicksum(
            order.quantity - self.z_unserved[order_id]
            for order_id, order in self.data.all_orders.items()
        )
        self.model.addConstr(
            direct_volume >= self.cfg.direct_ratio_min * served_volume,
            name="FD_(18)_MinimumDirectRatio",
        )
        self.model.addConstr(
            direct_volume <= self.cfg.direct_ratio_max * served_volume,
            name="FD_(19)_MaximumDirectRatio",
        )

    def decision_variable_groups(self):
        """返回全部带时间弧的变量，供通用 Rolling Horizon 固定历史。"""

        return {
            "x_manual": self.x_manual,
            "y_auto": self.y_auto,
            "g_manual": self.g_manual,
            "g_auto": self.g_auto,
            "w_direct": self.w_direct,
            "h_direct": self.h_direct,
        }

    def configure_rolling_window(
        self,
        current_time: int,
        window_end: int,
        committed_decisions: Dict[str, Dict[tuple, float]],
    ):
        """固定已执行弧，并关闭预测区间外或跨越窗口末端的候选弧。"""

        if not 0 <= current_time < window_end <= self.cfg.T:
            raise ValueError(
                f"非法滚动窗口：current_time={current_time}, window_end={window_end}"
            )
        for group_name, variables in self.decision_variable_groups().items():
            committed = committed_decisions.get(group_name, {})
            for key, variable in variables.items():
                departure_time, arrival_time = key[0], key[1]
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
        """仅提交当前控制区间内开始的非零弧决策。"""

        return {
            group_name: {
                tuple(key): float(variable.X)
                for key, variable in variables.items()
                if start_time <= key[0] < commit_end
                and abs(variable.X) > tolerance
            }
            for group_name, variables in self.decision_variable_groups().items()
        }

    def build_model(self):
        """便捷接口：按正确顺序一次性完成变量、目标和约束构建。"""

        self.setup_variables()
        self.set_objective()
        self.set_constraints()
        return self
