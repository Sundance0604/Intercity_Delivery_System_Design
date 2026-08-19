# 仿真实验运行结果说明

## 1. 输出文件

每次完整运行会在 `results/` 中生成：

```text
full_experiment_summary_<城市对或generated>__<算法>__YYYYMMDD_HHMMSS.csv
full_experiment_results_<城市对或generated>__<算法>__YYYYMMDD_HHMMSS.json
detail_<Exp_ID>_<城市对或generated>__<算法>__YYYYMMDD_HHMMSS.json
```

文件名中的算法标签：RH_pruning 表示“Rolling Horizon：剪枝”，RH_pruning_solution 表示“Rolling Horizon：剪枝 + 生成解”。同时选择两种时用加号连接。

CSV 适合统计和制图；完整批次 JSON 用于复现输入、检查订单与解；`detail` 文件目前
只为快速测试等标记了 `save_detail` 的算例生成。

## 2. CSV 字段

CSV 每行代表“一个算例 + 一个求解器”。同一个 `Exp_ID` 交给两个求解器时会有两行，
输入字段完全相同，只有求解状态和结果字段不同。

### 2.1 实验识别与灵敏度字段

| 字段 | 含义 |
|---|---|
| `Scenario` | `quick` 或 `sensitivity` |
| `Exp_ID` | 唯一算例编号 |
| `Solver` | 求解器内部名称 |
| City_1_CFS_Area、City_2_CFS_Area | 真实数据的两个 CFS Area 代码 |
| City_1_Name、City_2_Name | 官方 CFS Area 英文名称 |
| `Seed` | 订单随机种子 |
| `Sensitivity_Parameter` | 本算例唯一变化的参数，如 `config.cost_auto` |
| `Sensitivity_Value` | 该参数本算例的实际值；字典和区间保存为 JSON 文本 |
| `Sensitivity_Level` | 该参数在输入水平列表中的序号，从 1 开始 |

灵敏度算例编号格式：

```text
SENS_<参数来源>_<参数名>_L<水平序号>_S<随机种子>
```

例如 `SENS_MODEL_COST_AUTO_L2_S3001` 表示：测试 `model.cost_auto` 的第 2 个
水平，随机种子为 3001。

### 2.2 求解状态

| 字段 | 含义 |
|---|---|
| `Status` | 求解器状态码；Gurobi 中 2 为最优，9 为达到时间限制 |
| `Solve_Time_Sec` | 建模与求解用时，单位为秒 |
| `Time_Limit_Sec` | 用户设置的单算例时间限制 |
| `Message` | 求解器返回的文字状态 |

### 2.3 订单规模与总需求

| 字段 | 含义 |
|---|---|
| `Num_Orders` | 订单批次数 |
| `Total_Demand` | 本算例全部订单货量之和 |

完整订单配置由后续的 `Order_*` 动态列记录。

### 2.4 三类动态配置参数

三个配置 dataclass 的每个字段会自动写成一列：

```text
Model_<DeliveryConfig 字段名>
Algorithm_<RollingHorizonConfig 字段名>
Order_<OrderGenerationConfig 字段名>
```

例如：

| 字段 | 含义 |
|---|---|
| `Model_T` | 离散规划期长度 |
| `Model_t_0` | 单个时间段长度 |
| `Model_travel_time_periods` | 城际行驶时间段数 |
| `Model_N_manual`、`Model_N_auto` | 两类车辆数字典 |
| `Model_capacity_manual`、`Model_capacity_auto` | 两类车辆容量 |
| `Model_cost_manual`、`Model_cost_auto` | 两类车辆每车小时成本 |
| `Model_penalty_lost` | 单位未服务惩罚 |
| `Algorithm_prediction_horizon` | Rolling Horizon 预测区间 |
| `Algorithm_rolling_step` | Rolling Horizon 滚动步长 |
| `Order_num_orders` | 订单数量 |
| `Order_buffer_range` | 时间窗随机缓冲区间 |

字典、列表和区间在 CSV 单元格中保存为 JSON 文本。用户在 `intercity_delivery/configuration.py` 新增字段后，
CSV 会自动新增对应列，不需要修改输出代码。

### 2.5 目标值与算法质量

| 字段 | 含义 |
|---|---|
| `Total_Cost` | 当前解的目标函数值 |
| `Best_Bound` | Gurobi 当前最优界 |
| `MIP_Gap` | 当前可行解与最优界的相对差距 |
| `Unserved_Rate` | 未服务货量除以总需求量 |
| `Auto_Usage` | 自动驾驶车辆弧变量取值之和 |
| `Manual_Usage` | 人工车辆弧变量取值之和 |

`Auto_Usage` 和 `Manual_Usage` 是时间弧上的车辆使用量，不等于物理车辆总数。

## 3. 完整批次 JSON

JSON 顶层结构：

```json
{
  "format_version": 4,
  "generated_at": "20260627_181023",
  "experiment_count": 1,
  "solver_names": ["exact_mip"],
  "experiments": []
}
```

| 字段 | 含义 |
|---|---|
| format_version | 输出结构版本，当前为 4 |
| `solver_run_count` | 实际执行的求解次数 |
| `generated_at` | 本批次时间戳 |
| `experiment_count` | 算例数量，不乘求解器数量 |
| `solver_names` | 本批次选择的求解器 |
| city_pair、city_names | 真实数据的城市代码和官方名称；生成数据为 null |
| result_context | 写入结果文件名的城市对和算法标签 |
| `experiments` | 按算例组织的完整数据 |

### 3.1 experiments 中的单个算例

| 字段 | 含义 |
|---|---|
| `scenario`、`experiment_id` | 场景和算例编号 |
| `sensitivity_parameter` | 被测试参数；快速测试为 `null` |
| `sensitivity_value` | 被测试参数的原始结构化值 |
| `sensitivity_level` | 水平序号 |
| `time_limit_sec` | 单算例时间限制 |
| `model_parameters` | 完整模型参数 |
| `algorithm_parameters` | 完整算法参数 |
| `order_parameters` | 完整订单参数 |
| `generation_parameters` | 订单参数与随机种子 |
| real_data_metadata | SQLite 抽样统计、城市名称和模型时间建议；生成数据为 null |
| `orders` | 该算例实际生成的全部订单 |
| `solver_results` | 各求解器在同一输入上的结果列表 |

同一算例的订单只保存一次，多个求解器结果放入同一个 `solver_results` 数组，既保证
输入可比，又避免重复保存大量订单。

### 3.2 generation_parameters

| 字段 | 含义 |
|---|---|
| `num_orders` | 订单数 |
| `seed` | 随机种子 |
| `buffer_range` | 时间窗缓冲区间 |
| `large_order_prob` | 大订单概率 |
| `small_quantity_range` | 小订单货量区间 |
| `large_quantity_range` | 大订单货量区间 |

### 3.3 orders

| 字段 | 含义 |
|---|---|
| `batch_id` | 订单批次编号 |
| `flow` | `+` 为城市 1 到 2，`-` 为城市 2 到 1 |
| `quantity` | 订单货量 |
| `earliest_start` | 最早开始时段 |
| `latest_completion` | 最晚完成时段 |
| `penalty_lost` | 单位未服务惩罚 |

### 3.4 solver_results

每个元素保存 `solver`、`status`、`solve_time_sec`、`total_cost`、`best_bound`、
`mip_gap`、`unserved_rate`、两类车辆使用量、状态信息和求解器专属 `detail`。

精确 MIP 当前在 `detail.solution` 中保存非零的：

| 字段 | 含义 |
|---|---|
| `y_auto` | 自动驾驶车辆跨城运输变量 |
| `z_unserved` | 各订单未服务货量 |

### 3.5 论文 Rolling Horizon 窗口诊断

`detail.windows` 对每个窗口保存起始窗、扩展完成窗、控制窗、可见订单数、目标值、最优界、MIP Gap 和 `diagnostics`。对 `paper_priority_heuristic`，`diagnostics` 还包含：

| 字段 | 含义 |
|---|---|
| `heuristic_start_objective` | 动态优先级构造初解的目标值 |
| `heuristic_start_unserved` | 构造初解的未服务货量 |
| `heuristic_start_time_sec` | 初解构造时间；已计入窗口总预算 |
| `heuristic_start_diagnostics` | 初解迭代数、直送比例和候选弧信息 |
| `solution_count` | 最终剪枝 MIP 得到的可行解数量 |

最终窗口的 `objective`、`best_bound` 和 `mip_gap` 属于剪枝 MIP，不是构造器单独结果。
## 4. 论文分析建议

灵敏度分析不再依赖不同 `Scenario` 名称，而是按下列字段分组：

```text
Sensitivity_Parameter + Sensitivity_Level
```

每组包含多个 seed 时，建议至少统计：

| 指标 | 建议统计量 |
|---|---|
| `Total_Cost` | 均值、标准差、最小值、最大值 |
| `Unserved_Rate` | 均值、标准差 |
| `Solve_Time_Sec` | 中位数、均值 |
| `MIP_Gap` | 均值及达到时间限制的实例比例 |
| `Auto_Usage`、`Manual_Usage` | 均值及结构占比 |

比较算法时按 `Exp_ID` 配对，不要把不同 seed 或不同参数水平直接比较。若精确 MIP
已证明最优，可计算：

```text
Algorithm_Gap = (Algorithm_Cost - Exact_MIP_Cost) / Exact_MIP_Cost
```

## 5. 完整性检查

正式分析前建议确认：

1. CSV 中每个 `Sensitivity_Parameter` 的水平数与 GUI 输入一致；
2. 每个水平的行数等于 seed 数乘求解器数；
3. `Model_*`、`Algorithm_*`、`Order_*` 分别动态覆盖三类配置字段；
4. JSON 的 `experiment_count` 等于 `experiments` 数组长度；
5. 相同 `Exp_ID` 的不同求解器使用同一份 `orders`；
6. `Status=9` 时结合 `MIP_Gap` 判断限时解质量；
7. 全部未服务时检查成本、惩罚、运力和时间窗的量纲。
