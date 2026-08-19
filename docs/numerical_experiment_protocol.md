# 城际配送系统数值仿真实验方案

## 1. 实验目标

本方案用于评价城际配送系统中“人工城市内配送 + 自动驾驶城际干线 + 人工跨城直送”的协同运营机制，以及两种基于 Rolling Horizon 的 Solution Approach：

1. `paper_candidate_mip`：Rolling Horizon + 状态相关候选网络剪枝 + 窗口 MIP；
2. `paper_priority_heuristic`：Rolling Horizon + 同样的候选网络剪枝 + 动态优先级 MIP Start + 剪枝 MIP 改进。

实验结构参考 Li and Liu (2021) 在 *Optimizing flexible one-to-two matching in ride-hailing systems with boundedly rational users* 中采用的数值实验顺序：真实数据校准、基准场景、关键参数敏感性、运营机制对照、系统效益和计算效率分析。由于本项目研究的是城际货运，而不是乘客拼车，论文中的乘客绕行、折扣和司机福利指标不直接照搬，而是替换为订单时间窗、服务成本、服务率和计算效率。

本实验回答以下研究问题：

- RQ1：状态相关候选网络剪枝能否在不显著损失解质量的前提下缩小模型规模？
- RQ2：动态优先级 MIP Start 能否改善严格限时下的首个可行解、目标值和 MIP Gap？
- RQ3：允许直送与自动干线换装灵活协同，是否优于仅换装或仅直送？
- RQ4：订单规模、时间窗、车队规模、成本和城际距离如何影响系统成本与服务率？
- RQ5：两种方法能否在 Rolling Horizon 的在线决策时间预算内稳定完成求解？

## 2. 总体实验流程

```mermaid
flowchart LR
    A[检查 CFS SQLite] --> B[筛选并分层城市对]
    B --> C[校准时间与成本参数]
    C --> D[冻结订单实例]
    D --> E[小规模正确性验证]
    E --> F[运营机制对照]
    F --> G[两种算法正式比较]
    G --> H[规模扩展与参数敏感性]
    H --> I[Rolling Horizon 实时性检验]
    I --> J[统计分析与论文图表]
```

正式实验以官方 2022 Commodity Flow Survey Public Use Microdata（CFS PUMS）为主要数据。程序生成数据只用于可控压力测试和边界情况，不作为主要实证结论的依据。

## 3. 数据准备与城市对选择

### 3.1 数据来源

正式实验从以下 SQLite 数据库读取：

```text
data/cfs_2022_pums.sqlite
```

SQLite 应由项目中的 `intercity_delivery.data.sqlite_store` 从 CFS CSV 构建，并通过索引支持城市对筛选。每次正式实验应记录：

- CFS 数据版本；
- SQLite 文件大小和校验值；
- 城市 1、城市 2 的 CFS Area 代码和官方名称；
- 两个方向的原始记录数；
- 过滤规则；
- 抽样种子；
- 处理脚本的 Git commit。

### 3.2 城市对分层

为了避免只使用单一城市对得出偶然结论，候选城市对必须满足：

1. 两个运输方向都有有效记录；
2. 每个方向的有效记录数足够支持目标订单规模；
3. 距离、货量和方向字段完整；
4. 按当前规划期生成的订单至少存在可行的最短完成时间。

对每个候选城市对计算：

```text
Distance       = 加权中位运输距离
Volume         = 两个方向的加权总货量
Balance        = min(Volume_AB, Volume_BA) / max(Volume_AB, Volume_BA)
Record_Count   = 两个方向的有效记录数
```

将距离划分为短、中、长三个档次，将方向平衡度划分为平衡和不平衡两个档次，从以下六个分层单元中各选择一个代表城市对：

| 距离档次 | 平衡流 | 不平衡流 |
|---|---:|---:|
| 短距离 | 1 个 | 1 个 |
| 中距离 | 1 个 | 1 个 |
| 长距离 | 1 个 | 1 个 |

代表城市对优先选择接近本分层中位距离和中位货量的对象，不以求解结果好坏作为筛选条件。洛杉矶—旧金山（`06-348`—`06-488`）可作为已有的中距离基准城市对。

### 3.3 校准城市对与留出城市对

六个城市对分为两组：

- 校准组：3 个城市对，用于确定基准成本、车队和 Rolling Horizon 参数；
- 留出组：3 个城市对，只用于最终外部有效性检验，不参与参数调优。

正式结果必须分别报告校准组和留出组，防止根据全部测试结果反向调整参数。

### 3.4 订单时间字段构造

CFS 提供货源、目的地、货量和运输距离，但不提供本模型需要的真实下单时刻和承诺完成时刻。因此，实验中必须把以下字段标记为“基于真实 CFS 记录生成”，不能声称它们是 CFS 原始观测：

- `earliest_start`；
- 最短完成时刻；
- `latest_completion`。

城市对的城际运输时段数为：

```text
tau_AB = ceil((Distance_AB / Nominal_Speed) / t_0)
```

订单的最短完成时刻定义为：

```text
minimum_completion_l
    = earliest_start_l
    + origin_service_time_l
    + tau_AB
    + destination_service_time_l
```

订单截止时刻定义为：

```text
latest_completion_l = minimum_completion_l + slack_l
```

其中 `slack_l` 是实验控制的时间窗松弛量。每个城市对必须使用该城市对全部合格记录的加权中位距离预先固定 `travel_time_periods` 和 `direct_travel_time_periods`，不能让运输时间随订单抽样种子变化。洛杉矶—旧金山也必须先完成这一城市对级校准，不能继续使用默认值 4 或某一次小样本给出的偶然建议值。

### 3.5 配对订单实例

同一个“城市对 + 参数水平 + 随机种子”只生成一次订单，并保存为 JSON。所有待比较求解器读取同一份订单，禁止分别随机生成。建议使用：

- 预实验：3～5 个种子；
- 正式算法比较：至少 10 个种子；
- 重点敏感性和最终报告：20 个种子。

## 4. 基准参数

以下是初始基准值。城市相关的运输时间必须由真实数据覆盖。

| 类别 | 参数 | 基准值 |
|---|---|---:|
| 规划期 | `T` | 24 |
| 单时段长度 | `t_0` | 60 分钟 |
| 城际时间 | `travel_time_periods` | 按城市对校准 |
| 直送城际时间 | `direct_travel_time_periods` | 按城市对校准 |
| 人工车辆 | `N_manual` | 每城 30 |
| 自动车辆 | `N_auto` | 每城 15 |
| 人工车容量 | `capacity_manual` | 1000 |
| 自动车容量 | `capacity_auto` | 2000 |
| 直送容量 | `capacity_direct` | 1000 |
| 预测区间 | `prediction_horizon` | 8 |
| 滚动步长 | `rolling_step` | 2 |
| 扩展完成区间 | `extension_horizon` | 6 |
| 基准订单数 | `num_orders` | 100 |
| 时间窗缓冲 | `buffer_range` | `[0,5]` |

在正式求解前，应进行成本量纲校准，使基准实例既不会全部拒单，也不会在任何条件下都服务全部订单。建议把基准未服务率控制在 5%～30%，以便敏感性实验能够观察到系统行为变化。

## 5. 实验阶段

### 5.1 阶段 A：数据与程序预检

目标是排除数据和实现错误，不形成论文结论。

检查项目：

1. SQLite 能正确显示列名和城市对；
2. 两个方向都能抽取订单；
3. 同一随机种子重复运行得到相同订单；
4. `minimum_completion <= latest_completion <= T`；
5. 城际时间使用城市对元数据，而不是默认值；
6. 两种算法都能完成至少一个 Rolling Horizon 窗口；
7. 结果文件名包含城市对和算法名称；
8. JSON 中保存完整订单、配置和窗口明细。

推荐设置：1 个城市对、20 个订单、2 个随机种子、每种算法各运行一次。

### 5.2 阶段 B：小规模正确性验证

目标是验证剪枝和启发式的可行性及解质量。

| 维度 | 水平 |
|---|---|
| 城市对 | 1 个基准城市对 |
| 订单数 | 10、20、40 |
| 随机种子 | 10 个 |
| 求解器 | 全时域 MIP、未剪枝 RH、剪枝 RH、剪枝+生成解 RH |

共计：

```text
3 × 10 × 4 = 120 次求解
```

主要比较：

- 剪枝 MIP与未剪枝 RH 的目标值差异；
- 剪枝前后变量数、约束数和弧数量；
- 启发式相对剪枝 MIP 的成本差距；
- 所有方法的可行率和时间窗满足率。

如果剪枝理论上应当无损，则在双方都求至最优时应满足：

```text
abs(Cost_pruned - Cost_full) <= 1e-6 × max(1, abs(Cost_full))
```

若不满足，应把当前剪枝明确标记为近似剪枝，并报告经验最优性损失。

### 5.3 阶段 C：运营机制对照

本阶段仿照参考论文的机制对照实验，但将 one-to-one/one-to-two 替换为本项目的运输机制。

| 机制 | 参数设置 |
|---|---|
| 仅换装 | `direct_ratio_min=0`，`direct_ratio_max=0` |
| 灵活直送+换装 | `direct_ratio_min=0`，`direct_ratio_max=1` |
| 仅直送 | `direct_ratio_min=1`，`direct_ratio_max=1` |

车队设置为基准规模的：

```text
0.75、1.00、1.25
```

建议先在 3 个代表城市对、100 个订单、10 个种子上运行。核心问题是：灵活协同是否降低单位已服务货量成本、提高服务率，或者以更小车队达到相同服务水平。

Algorithm 2 的构造初解把直送比例作为偏好；最终结果仍由剪枝 MIP 给出并严格满足比例约束。机制分析应区分 `heuristic_start_diagnostics` 与最终 MIP 指标。

### 5.4 阶段 D：两种 Solution Approach 正式比较

| 维度 | 水平 |
|---|---|
| 城市对 | 6 个分层城市对 |
| 订单数 | 50、100、200 |
| 随机种子 | 每组 10 个 |
| 算法 | `paper_candidate_mip`、`paper_priority_heuristic` |

总运行数：

```text
6 × 3 × 10 × 2 = 360 次求解
```

每次算法比较都按相同的城市对、订单规模和随机种子配对。主要输出：

- 总成本及配对差值；
- 未服务率及配对差值；
- 总求解时间、首个可行解质量和限时目标改进；
- 直送/换装货量比例；
- 候选弧削减率；
- MIP 窗口 Gap；
- 初解构造时间、初解目标值与最终目标改进。

### 5.5 阶段 E：规模扩展实验

在短、中、长距离各选择一个城市对，设置订单数：

```text
20、50、100、200、400、800
```

所有规模同时运行无初解的剪枝 MIP和带动态初解的剪枝 MIP；若模型达到时间限制，仍保留限时结果，但必须报告：

- 达到时间限制的比例；
- 已得可行解的窗口 MIP Gap；
- 完成的 Rolling Horizon 窗口数；
- 最大窗口计算时间；
- 动态初解相对无初解 MIP 的限时目标与 Gap 改进。

不得把达到时间限制的 MIP 结果写成“最优解”。

### 5.6 阶段 F：单因素敏感性

每次只改变一个参数，其余参数保持基准值。

#### 运营参数

| 参数 | 水平 |
|---|---|
| `buffer_range` | `[0,2]`、`[0,5]`、`[0,8]` |
| `num_orders` | 50、100、200、400 |
| 车队规模倍率 | 0.75、1.00、1.25 |
| `penalty_lost` 倍率 | 0.5、1.0、2.0 |
| `cost_direct` 倍率 | 0.75、1.00、1.25 |
| `transfer_time_periods` | 0、1、2、4 |
| `transfer_cost_per_unit` | 0、5、10、20 |
| 自动车/人工车容量比 | 1、2、3 |

#### Rolling Horizon 参数

| 参数 | 水平 |
|---|---|
| `prediction_horizon` | 4、8、12 |
| `rolling_step` | 1、2、4 |
| `extension_horizon` | `ceil(0.5*tau)`、`tau`、`ceil(1.5*tau)` |
| `priority_epsilon` | `1e-8`、`1e-6`、`1e-4` |

先在校准城市对上以 3～5 个种子筛选出敏感参数，再用 20 个种子运行重点参数。留出城市对只验证已经确定的参数组合。

### 5.7 阶段 G：计算效率与在线可用性

仿照参考论文，以单窗口最大计算时间评价 Rolling Horizon 是否可在线执行。

定义：

```text
Online_Ratio
    = max(Window_Solve_Time)
    / (rolling_step × t_0 × 60)
```

基本在线判据：

```text
Online_Ratio <= 1
```

由于本项目单时段可能长达 60 分钟，上述标准可能过于宽松，因此额外采用工程判据：

```text
max(Window_Solve_Time) <= 60 秒
```

计算效率实验至少报告：

- 总求解时间；
- 窗口时间的均值、中位数、P95 和最大值；
- 窗口超时比例；
- 候选弧削减率；
- 变量数和约束数；
- MIP 窗口 Gap；
- 完成窗口数；
- 动态初解对总时间、限时目标和 Gap 的影响。

## 6. 评价指标

### 6.1 系统绩效

```text
Service_Rate = 1 - Unserved_Rate

Cost_Per_Served_Unit
    = Total_Cost / max(Served_Quantity, epsilon)

Direct_Share
    = Direct_Volume / max(Served_Quantity, epsilon)

Transshipment_Share
    = Transshipment_Volume / max(Served_Quantity, epsilon)
```

建议新增成本分解：

- 人工城市内配送成本；
- 自动驾驶干线成本；
- 人工跨城直送成本；
- 换装成本；
- 未服务惩罚。

### 6.2 服务质量

- 未服务率；
- 按时完成率；
- 平均完成时长；
- 最短完成时间以上的额外等待；
- 时间窗松弛消耗比例；
- 按城市方向分别统计的服务率。

### 6.3 算法质量

对小规模精确基准定义：

```text
Algorithm_Gap
    = (Algorithm_Cost - Benchmark_Cost)
    / max(abs(Benchmark_Cost), epsilon)
```

同时记录：

- 可行率；
- 最优率；
- 达到时间限制的比例；
- MIP Gap；
- 动态初解目标值及其被最终 MIP 改进的幅度；
- 同一订单实例上的配对成本差值。

### 6.4 剪枝效果

```text
Arc_Reduction
    = 1 - Reduced_Arc_Count / Baseline_Arc_Count

Variable_Reduction
    = 1 - Reduced_Variable_Count / Full_Variable_Count
```

报告剪枝率时必须同时报告目标值差异，不能只以模型规模下降证明算法有效。

## 7. 统计分析方法

### 7.1 配对设计

两种算法必须使用完全相同的：

- 城市对；
- 参数水平；
- 订单数；
- 随机种子；
- 实际订单集合；
- 总时间限制。

以 `Exp_ID` 为配对键计算差值，不能把不同种子或不同参数水平直接混合比较。

### 7.2 描述统计

| 指标 | 统计量 |
|---|---|
| 总成本、服务率 | 均值、标准差、95% 置信区间 |
| 求解时间 | 中位数、P95、最大值、IQR |
| MIP Gap | 均值、中位数、超时比例 |
| 剪枝率 | 均值、最小值、最大值 |
| 算法差值 | 配对均值、配对中位数、bootstrap 95% CI |

求解时间通常右偏，算法比较优先采用 Wilcoxon 配对检验；成本差值可同时报告配对 t 检验和 bootstrap 置信区间。多个参数同时检验时使用 Holm 方法校正显著性水平。

统计显著性不能替代实际意义。算法结论必须同时报告成本变化百分比、服务率变化百分点和时间加速比。

## 8. 结果文件与可复现性

现有输出文件命名为：

```text
full_experiment_summary_<城市对>__<算法>__<时间戳>.csv
full_experiment_results_<城市对>__<算法>__<时间戳>.json
detail_<Exp_ID>_<城市对>__<算法>__<时间戳>.json
```

每个正式实验批次应另保存一个 manifest，至少包含：

```text
experiment_name
git_commit
data_file_hash
city_pairs
seed_list
solver_names
time_limit
model_parameters
algorithm_parameters
order_parameters
start_time
end_time
machine_information
gurobi_version
```

正式结果不得手工修改。若发现异常，应修复后生成新的批次，并保留旧批次及问题说明。

## 9. GUI 与批处理运行建议

GUI 用于：

- 检查 SQLite 列名；
- 选择单个城市对；
- 同时勾选两种 Solution Approach；
- 运行 1～2 个种子的基准测试；
- 检查日志和输出文件名。

正式的 360 次以上实验应使用 CLI 或专门的批处理脚本。使用 `pavane` 环境：

```powershell
conda activate pavane
```

单算法真实数据预检示例：

```powershell
python main.py --cli --scenario quick `
  --solver paper_candidate_mip `
  --data-source real `
  --real-data-path data\cfs_2022_pums.sqlite `
  --city-a 06-348 --city-b 06-488 `
  --time-limit 300

python main.py --cli --scenario quick `
  --solver paper_priority_heuristic `
  --data-source real `
  --real-data-path data\cfs_2022_pums.sqlite `
  --city-a 06-348 --city-b 06-488 `
  --time-limit 300
```

两次运行必须使用相同参数和种子。正式运行前先执行 `--dry-run` 检查实验数量和参数水平。

## 10. 实验停止与验收标准

### 10.1 数据验收

- 六个城市对均有双向记录；
- 城市对分层规则可复现；
- 订单字段和单位完整；
- 最短完成时间计算有明确公式；
- 同一随机种子能够重复生成相同订单。

### 10.2 算法验收

- 小规模剪枝 MIP 与未剪枝基准的差异得到解释；
- 两种算法可行率达到 100%，或完整报告失败实例；
- MIP 超时实例保留 Gap，且不被标为最优；
- 动态初解与最终 MIP 指标被分开报告；
- 不出现 `MIPGap` 等不可用属性读取异常。

### 10.3 实验完整性

- 每个参数水平的结果行数等于种子数乘适用求解器数；
- 同一 `Exp_ID` 的不同算法使用相同订单；
- 所有正式结果带城市对、算法、种子和配置；
- 校准城市对和留出城市对分开报告；
- 失败、超时和不可行结果没有从样本中静默删除。

## 11. 论文建议图表

建议最终论文按以下顺序展示：

1. 城市对运输距离和最短运输时间分布；
2. 基准参数表；
3. 小规模算法正确性和目标差距表；
4. 仅换装、灵活协同、仅直送的机制对照表；
5. 时间窗松弛对服务率和路径结构的影响；
6. 成本/罚金参数对总成本和未服务率的影响；
7. 车队规模对服务率和单位货量成本的影响；
8. 订单规模对两种算法求解时间的影响；
9. 候选弧削减率与变量数量关系；
10. 两种算法的目标差距与求解时间加速比；
11. 最大窗口计算时间与在线时间预算对照；
12. 校准城市对和留出城市对的结果对比。

## 12. 正式实验前建议补充的结果字段

当前实现已经能够记录总成本、未服务率、车辆使用量、候选弧数量、弧削减率、变量数、约束数、窗口时间和窗口 MIP Gap。正式实验前建议进一步增加：

1. 成本分解；
2. 每个订单的实际完成时刻；
3. 时间窗松弛消耗；
4. 窗口时间的汇总字段：均值、P95、最大值；
5. 窗口超时标志；
6. 城市方向服务率；
7. 实验 manifest 和数据文件哈希；
8. CLI 同时选择两个指定 Solution Approach 的批处理接口。

这些字段补充后，现有 GUI、SQLite 城市对选择、Rolling Horizon 窗口明细和结果命名机制即可支持完整的论文数值实验。
