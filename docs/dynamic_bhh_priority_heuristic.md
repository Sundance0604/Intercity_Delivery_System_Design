# 解法 2：剪枝 + BHH 优先级生成初解

## 对应论文内容

本实现对应 `Intercity_Operation.pdf` 的 Solution Approach / Algorithm 2。论文中的动态 BHH-aware 优先级算法用于为削减后的弧模型生成可行初解，而不是替代 MILP 成为最终求解器。因此当前流程为：状态相关候选弧剪枝 → 动态优先级构造解 → 写入 Gurobi MIP Start → 在剩余时间内求解同一个削减 MILP。

统一注册名仍为 `paper_priority_heuristic`，GUI 显示为“Rolling Horizon：剪枝 + 生成解”。入口实现位于 `intercity_delivery/algorithms/warm_started_mip.py`；构造器位于 `intercity_delivery/algorithms/bhh_priority_heuristic.py`；两者都通过论文专用 `paper_rolling_horizon.py` 运行。

## 动态优先级初解

每轮按当前状态重新计算优先级。排序指标是剩余松弛时间除以 `penalty_lost × remaining_quantity + priority_epsilon`。每次分配后立即更新人工车、自动车、直送车辆和弧容量，再重新排序。

构造器同时检查：

- 人工弧的 BHH 货量上限及车辆占用；
- 自动驾驶弧的固定行驶时间、车辆数和容量；
- 直送任务两端的 BHH 服务时间、车辆容量和跨城车辆转移；
- 订单最早开始、最晚完成及已提交运输阶段。

## MIP Start 与最终解

构造结果会为 `x_manual`、`y_auto`、`g_manual`、`g_auto`、`w_direct`、`h_direct`、`q_direct`、`r_transshipment` 和 `z_unserved` 设置初值。历史已提交变量使用固定历史值。构造时间从当前窗口总预算中扣除，MILP 只使用剩余时间。

最终返回值来自削减 MILP，所以它严格执行模型中的直送比例和需求守恒约束，并提供最优界与 MIP Gap。JSON 的每窗 `diagnostics` 额外保存：

- `heuristic_start_objective`；
- `heuristic_start_unserved`；
- `heuristic_start_time_sec`；
- `heuristic_start_diagnostics`；
- 最终模型变量数、约束数和解数量。

在时间充足的小算例中，解法 1 与解法 2可能得到相同最优解和近似时间。Algorithm 2 的价值应在更大实例或严格限时下，用“首个可行解时间、限时目标值、Gap”检验，不能把构造器单独返回的时间作为最终算法速度。

## GUI 使用

在 Solution Approach 区域选择“Rolling Horizon：剪枝 + 生成解”。可以单独选择，也可与“Rolling Horizon：剪枝”同时运行；同一实验规格和种子复用完全相同的订单。