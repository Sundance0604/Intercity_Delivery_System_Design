# 解法 1：状态相关候选弧削减 MIP

## 对应论文内容

本实现对应 `Intercity_Operation.pdf` 的 Solution Approach / Algorithm 1。核心思想不是在每个滚动窗口重复建立完整时空网络，而是根据当前已经提交的车辆状态、当前可见订单和时间窗，只保留仍可能进入可行解的人工配送弧、自动驾驶干线弧和直送弧，再求解削减后的混合整数规划。

实现文件为 `intercity_delivery/algorithms/state_dependent_mip.py`，统一注册名为 `paper_candidate_mip`。它只通过论文专用 `paper_rolling_horizon.py` 运行。

## 窗口和状态

每次迭代使用三个边界：

- 控制窗口：`[current_time, current_time + rolling_step)`，只有这里开始的决策会被提交。
- 预测起始窗口：截至 `prediction_horizon`，允许新车辆任务开始。
- 扩展完成窗口：再延长 `extension_horizon`，允许窗口末端开始的任务在更晚时刻完成。

已提交状态包含人工车、自动车、直送车以及各类货流。候选弧生成器先扣除这些任务占用的车辆，并把完成直送后的车辆计入目的城市车队，然后筛选满足剩余运力、订单方向与时间窗的弧。

## 求解流程

1. 逐期揭示 `earliest_start <= current_time` 的订单。
2. 生成状态相关候选网络，并保留所有历史已提交弧。
3. 使用 `ReducedFlexibleDirectOptimizer` 建立直送/换装共存 MIP。
4. 固定历史决策，只优化当前及未来候选决策。
5. 提交控制窗口内开始的决策，滚动到下一窗口。

每个窗口的 JSON 明细包含候选弧数量、相对完整网络的削减率、变量数、约束数、目标值、最优界和 MIP Gap，便于论文报告计算规模变化。

## 适用性与参数

该解法适合需要可解释优化界、但完整时空网络规模过大的实验。主要算法参数为：

- `prediction_horizon`：新任务的预测起始范围。
- `rolling_step`：每次实际提交的时间跨度。
- `extension_horizon`：窗口末端任务可使用的额外完成范围。

由于拒单罚金和车辆成本量纲会直接决定是否服务订单，实验前应校准 `penalty_lost` 与车辆成本。若罚金远低于完成配送的成本，MIP 合理地会选择拒单。

## GUI 使用

在“论文 Solution Approach”中勾选“论文解法 1：状态相关候选弧 MIP”。它可以单独运行，也可以与解法 2 同时运行；相同实验规格和种子会复用同一组订单。
