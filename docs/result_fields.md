# 仿真实验运行结果说明

本文档说明 `results/` 目录中输出文件的类型、命名规则、CSV 汇总字段含义、JSON 详细结果含义，以及如何根据这些结果进行论文分析。

## 1. 输出文件位置

所有实验结果默认保存在项目根目录下的 `results/` 文件夹中。

常见输出文件包括：

```text
results/full_experiment_summary_YYYYMMDD_HHMMSS.csv
results/detail_<Exp_ID>_<Solver>_YYYYMMDD_HHMMSS.json
```

其中：

| 片段 | 含义 |
|---|---|
| `YYYYMMDD_HHMMSS` | 实验开始运行时的时间戳 |
| `full_experiment_summary` | 批量实验汇总表 |
| `detail` | 单个算例的详细订单和解信息 |
| `<Exp_ID>` | 算例编号 |
| `<Solver>` | 求解器名称，例如 `exact_mip` |

## 2. CSV 汇总结果说明

CSV 文件是论文制表和统计分析的主要数据来源。每一行表示：

```text
一个算例 + 一个求解器
```

如果同一个算例同时使用 `exact_mip` 和 `rolling_horizon` 两种求解器，则 CSV 中会出现两行，它们的 `Exp_ID` 相同，但 `Solver` 不同。

### 2.1 实验识别字段

| 字段 | 含义 | 示例 |
|---|---|---|
| `Scenario` | 实验场景名称 | `baseline`、`scale`、`sens_auto_fleet` |
| `Exp_ID` | 算例编号 | `BASE_N20_S1001` |
| `Solver` | 求解方式 | `exact_mip` |
| `Seed` | 随机种子 | `1001` |

#### Scenario 的可能取值

| 取值 | 含义 |
|---|---|
| `quick` | 快速测试 |
| `baseline` | 小规模基准实验 |
| `scale` | 规模扩展实验 |
| `sens_auto_fleet` | 自动车数量灵敏度分析 |
| `sens_auto_cost` | 自动车单位成本灵敏度分析 |
| `sens_manual_fleet` | 人工车数量灵敏度分析 |
| `sens_time_window` | 时间窗紧迫程度灵敏度分析 |
| `sens_demand_mix` | 大订单比例灵敏度分析 |

#### Exp_ID 命名规则

| 示例 | 含义 |
|---|---|
| `QUICK_N20_S42` | quick 场景，20 个订单，随机种子 42 |
| `BASE_N50_S1002` | baseline 场景，50 个订单，随机种子 1002 |
| `SCALE_N500_S2001` | scale 场景，500 个订单，随机种子 2001 |
| `SENS_AUTO_20_S3001` | 自动车数量灵敏度，自动车数量为 20，随机种子 3001 |
| `SENS_AUTO_COST_10_S3001` | 自动车成本灵敏度，自动车单位成本为 10 |
| `SENS_WINDOW_3_S3001` | 时间窗缓冲上限为 3 |
| `SENS_DEMAND_0.5_S3001` | 大订单比例为 0.5 |

### 2.2 求解状态字段

| 字段 | 含义 | 说明 |
|---|---|---|
| `Status` | 求解器状态码 | 对 Gurobi 来说，`2` 表示最优，`9` 表示达到时间限制 |
| `Solve_Time_Sec` | 实际求解时间，单位秒 | 包含建模和求解时间 |
| `Time_Limit_Sec` | 用户设置的单算例时间限制 | 来自界面或命令行参数 |
| `Message` | 求解器返回的文字说明 | 例如“已找到全局最优解” |

常见 Gurobi 状态码：

| 状态码 | 含义 |
|---:|---|
| `2` | 已找到全局最优解 |
| `3` | 模型不可行 |
| `4` | 模型不可行或无界 |
| `5` | 模型无界 |
| `9` | 达到时间限制 |

如果 `Status = 9` 且 `Total_Cost` 非空，说明 Gurobi 虽然没有证明最优，但已经找到了可行解。

### 2.3 算例规模与订单结构字段

| 字段 | 含义 | 示例 |
|---|---|---|
| `Num_Orders` | 订单批次数量 | `20`、`100`、`500` |
| `Total_Demand` | 所有订单需求量总和 | `1767` |
| `Buffer_Min` | 时间窗随机缓冲下限 | `0` |
| `Buffer_Max` | 时间窗随机缓冲上限 | `5` |
| `Large_Order_Prob` | 大订单生成概率 | `0.3` |

解释：

```text
latest_completion = earliest_start + travel_time_periods + 1 + random_buffer
```

其中 `random_buffer` 从 `[Buffer_Min, Buffer_Max]` 中随机抽取。`Buffer_Max` 越小，时间窗通常越紧，模型越难满足所有订单。

当前订单生成逻辑中：

| 类型 | 数量范围 |
|---|---|
| 小订单 | 10-50 |
| 大订单 | 100-300 |

`Large_Order_Prob` 越大，大订单越多，总需求量和容量压力通常越高。

### 2.4 参数字段

| 字段 | 含义 | 对应模型参数 |
|---|---|---|
| `Param_N_Auto` | 每个城市自动驾驶车辆数量 | `hat{N}^i` |
| `Param_N_Manual` | 每个城市人工驾驶车辆数量 | `N^i` |
| `Param_Cost_Auto` | 自动驾驶车辆单位成本 | `hat{c}` |
| `Penalty_Lost` | 单位未服务需求惩罚成本 | `delta_l` |

注意：

`Param_N_Auto` 和 `Param_N_Manual` 目前记录的是城市 1 的取值。当前实验设计中两个城市通常设置为相同车辆数，因此该字段可以代表每城市车辆数。

### 2.5 目标值与求解质量字段

| 字段 | 含义 | 说明 |
|---|---|---|
| `Total_Cost` | 当前解的总目标函数值 | 包括运输成本和未服务惩罚 |
| `Best_Bound` | Gurobi 当前最优界 | 用于判断最优性差距 |
| `MIP_Gap` | MIP 最优性差距 | 越接近 0 越好 |

MIP Gap 的含义：

```text
MIP_Gap = |Total_Cost - Best_Bound| / |Total_Cost|
```

如果 `MIP_Gap = 0`，并且状态码为 `2`，说明已经证明全局最优。

如果 `MIP_Gap > 0`，说明当前解和理论最优界之间仍有差距，通常发生在达到时间限制的大规模算例中。

### 2.6 服务质量与车辆使用字段

| 字段 | 含义 | 说明 |
|---|---|---|
| `Unserved_Rate` | 未服务需求比例 | `未服务需求量 / 总需求量` |
| `Auto_Usage` | 自动驾驶车辆弧使用量总和 | 所有 `y_auto` 变量取值之和 |
| `Manual_Usage` | 人工车辆弧使用量总和 | 所有 `x_manual` 变量取值之和 |

解释：

`Unserved_Rate = 1.0` 表示全部需求未服务；`Unserved_Rate = 0.0` 表示全部需求都被服务。

`Auto_Usage` 和 `Manual_Usage` 不是车辆数本身，而是车辆在时间弧上的使用次数总和。例如同一辆车在多个时间弧上运行，会贡献多个使用量。

## 3. JSON 详细结果说明

详细 JSON 文件只会对部分算例保存，通常是较小规模的算例。其目的是方便检查订单数据和具体解。

典型结构如下：

```json
{
  "scenario": "baseline",
  "experiment_id": "BASE_N20_S1001",
  "solver": "exact_mip",
  "seed": 1001,
  "buffer_range": [0, 5],
  "large_order_prob": 0.3,
  "config": {},
  "orders": {},
  "solution": {}
}
```

### 3.1 顶层字段

| 字段 | 含义 |
|---|---|
| `scenario` | 实验场景 |
| `experiment_id` | 算例编号 |
| `solver` | 求解器名称 |
| `seed` | 随机种子 |
| `buffer_range` | 时间窗缓冲区间 |
| `large_order_prob` | 大订单比例 |
| `config` | 本算例使用的完整模型参数 |
| `orders` | 本算例生成的全部订单 |
| `solution` | 求解器输出的关键解变量 |

### 3.2 orders 字段

`orders` 中每个键是订单编号，每个值是一批订单的信息。

| 字段 | 含义 |
|---|---|
| `batch_id` | 订单批次编号 |
| `flow` | 运输方向，`+` 表示城市 1 到城市 2，`-` 表示城市 2 到城市 1 |
| `quantity` | 该批订单需求量 |
| `earliest_start` | 最早开始时间 |
| `latest_completion` | 最晚完成时间 |
| `penalty_lost` | 单位未服务惩罚成本 |

### 3.3 solution 字段

当前 `exact_mip` 求解器主要保存两个变量：

| 字段 | 含义 |
|---|---|
| `y_auto` | 非零自动驾驶车辆跨城运输变量 |
| `z_unserved` | 非零未服务需求变量 |

`y_auto` 的键是 Gurobi 变量索引的字符串形式，例如：

```text
(0, 4, '+')
```

含义是：

| 位置 | 含义 |
|---|---|
| `0` | 自动车出发时间段 |
| `4` | 自动车到达时间段 |
| `+` | 正向运输，城市 1 到城市 2 |

`z_unserved` 的键是订单编号，值是该订单未服务的货量。

## 4. 如何解读结果

### 4.1 判断模型是否正常运行

可以先看以下字段：

| 检查项 | 理想情况 |
|---|---|
| `Status` | 小规模算例通常应为 `2` |
| `MIP_Gap` | 小规模算例应接近 `0` |
| `Solve_Time_Sec` | quick 或 baseline 不应过长 |
| `Total_Cost` | 不应为空 |

### 4.2 判断是否存在参数量纲问题

如果出现：

```text
Unserved_Rate = 1.0
Auto_Usage = 0
Manual_Usage = 0
```

说明模型选择“不派车、全部放弃服务”。这通常不是程序错误，而是当前成本参数下，未服务惩罚比运输成本更低。

可考虑：

1. 提高 `config.py` 中的 `penalty_lost`；
2. 降低 `cost_manual` 或 `cost_auto`；
3. 调整订单需求量范围；
4. 检查时间窗是否过紧。

### 4.3 比较不同求解器

未来实现 rolling horizon 后，可以按 `Exp_ID` 分组，对比不同 `Solver` 的结果。

常用指标：

| 指标 | 用途 |
|---|---|
| `Total_Cost` | 比较解质量 |
| `Solve_Time_Sec` | 比较求解效率 |
| `Unserved_Rate` | 比较服务质量 |
| `Auto_Usage` / `Manual_Usage` | 比较车辆调度结构 |

算法 Gap 可按如下方式计算：

```text
Algorithm_Gap = (Algorithm_Cost - Exact_MIP_Cost) / Exact_MIP_Cost
```

注意：只有当 `Exact_MIP_Cost` 是最优解或足够可靠的限时解时，该 Gap 才有较强解释意义。

### 4.4 做灵敏度分析

建议按 `Scenario` 分组分析：

| Scenario | 推荐横轴 | 推荐纵轴 |
|---|---|---|
| `sens_auto_fleet` | `Param_N_Auto` | `Total_Cost`、`Unserved_Rate`、`Auto_Usage` |
| `sens_auto_cost` | `Param_Cost_Auto` | `Total_Cost`、`Auto_Usage` |
| `sens_manual_fleet` | `Param_N_Manual` | `Total_Cost`、`Unserved_Rate`、`Manual_Usage` |
| `sens_time_window` | `Buffer_Max` | `Total_Cost`、`Unserved_Rate` |
| `sens_demand_mix` | `Large_Order_Prob` | `Total_Cost`、`Unserved_Rate` |

如果每个水平有多个随机种子，建议对同一参数水平取平均值，并同时报告标准差。

## 5. 结果文件是否应提交到 GitHub

一般建议：

| 文件类型 | 是否提交 | 说明 |
|---|---|---|
| 源代码 | 是 | 程序主体 |
| 文档 | 是 | 操作说明和结果说明 |
| 小型示例结果 | 可选 | 可用于演示 |
| 大规模实验结果 | 不建议 | 文件可能很大，且容易频繁变化 |
| `__pycache__` | 不建议 | Python 自动生成缓存 |

如果需要保存正式论文实验结果，建议单独建立 `paper_results/` 或 `analysis/` 目录，并在 README 中说明实验日期和参数配置。
