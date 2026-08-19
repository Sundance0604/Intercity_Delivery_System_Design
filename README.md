# Intercity Delivery System Design

基于混合整数规划（MIP）的城际配送系统优化与仿真实验平台。系统协同调度城市内
人工驾驶车辆和城际自动驾驶车辆，在车辆、容量、换装时序和订单时间窗约束下最小化
车辆成本与未服务惩罚。

## 论文 Solution Approach（2026-07-14）

`Intercity_Operation.pdf` 的两个解法已经实现，并作为两个独立求解器接入实验框架：

- paper_candidate_mip：Rolling Horizon + 状态相关候选网络剪枝；每个窗口求解剪枝后的 MIP。
- paper_priority_heuristic：Rolling Horizon + 同样的候选网络剪枝 + 动态优先级构造 MIP Start + 剪枝 MIP 改进。

两者使用 `intercity_delivery/algorithms/paper_rolling_horizon.py`。该控制器支持控制窗口、预测起始窗口、
扩展完成窗口、订单逐期揭示和已提交车辆/货流状态；原有算法位于 `intercity_delivery/algorithms/rolling_horizon.py`。
GUI 中两个论文解法是独立复选框，可全选或只选其中一个，旧求解器保留为可选基准。

因此两个 Solution Approach 的共同前提都是 Rolling Horizon。解法 1 直接求解剪枝 MIP；解法 2 先生成动态 BHH 优先级初解，再用它启动同一剪枝 MIP。GUI 分别显示为“Rolling Horizon：剪枝”和“Rolling Horizon：剪枝 + 生成解”。

对应说明：

- [解法 1：状态相关候选弧削减 MIP](docs/state_dependent_candidate_mip.md)
- [解法 2：剪枝 + BHH 优先级生成初解](docs/dynamic_bhh_priority_heuristic.md)

### 生成数据与真实数据

GUI 的“测试数据”区域必须明确选择数据来源：

- **生成数据**：使用 `OrderGenerationConfig` 和随机种子构造订单。
- **真实数据**：直接加载带索引的 CFS SQLite。GUI 显示 shipments 列名，并提供双向城市 1/城市 2 联动选择。

真实模式从 SQLite 中筛选用户选择的两个 CFS Area，按 WGT_FACTOR 加权、双向平衡地抽样订单；城市对运输时间由全部合格 OD 记录加权校准并自动用于模型，随机种子只影响订单抽样。若合格记录不足或规划期短于运输最短完成时间，程序会在求解前明确报错。`cost_manual`、`cost_auto`、`cost_direct` 均为每车小时费率，目标函数用 `t_0 / 60` 把分钟换算为小时。

官方 2022 CFS PUMS 的处理方法见：

- [CFS 2022 数据内容与处理](docs/cfs_2022_data_processing.md)

对于 2GB 以上的官方 CSV，推荐先建立一次带索引的 SQLite 缓存，再从缓存生成模型订单：

```powershell
conda activate pavane
python -m intercity_delivery.data.sqlite_store `
  --input data\cfs_2022_pums.csv `
  --output data\cfs_2022_pums.sqlite

python -m intercity_delivery.data.cfs_processor `
  --input data\cfs_2022_pums.sqlite `
  --output-dir data\cfs_processed --num-orders 100 --seed 42
```

首次建库仍会顺序扫描一次 CSV；随后按 OD、方式和距离的重复筛选直接走 SQLite。
GUI 可以直接选择第一步生成的 data/cfs_2022_pums.sqlite，无需预先生成固定城市对 JSON。

论文解法核心文件：

```text
intercity_delivery/algorithms/paper_rolling_horizon.py      # 论文专用滚动时域接口
intercity_delivery/algorithms/state_dependent_mip.py        # Algorithm 1
intercity_delivery/algorithms/bhh_priority_heuristic.py     # Algorithm 2
intercity_delivery/data/cfs_processor.py                    # 官方 CFS PUMS 转换
```

## 主要功能

- 精确 MIP 与 Rolling Horizon 两种求解方式。
- 模型参数、算法参数、订单参数三类动态单因素灵敏度分析。
- GUI 与 CLI 使用同一个实验规格生成、求解和结果输出链路。
- CSV 汇总结果与可复现实验输入、订单和详细解的 JSON 结果。

## 环境

- Python 3.11+
- Gurobi Optimizer 与有效许可证
- `numpy`、`pandas`
- `customtkinter`

推荐使用项目环境：

```bash
conda activate pavane
```

## 项目结构

```text
.
├── main.py                         # GUI 与 CLI 统一入口（根目录唯一 Python 文件）
├── intercity_delivery/
│   ├── configuration.py            # 模型、算法、订单配置
│   ├── data/
│   │   ├── loader.py               # 时间弧、容量系数与订单结构
│   │   ├── cfs_processor.py        # 官方 CFS 数据处理
│   │   ├── sqlite_store.py         # 大型 CFS CSV 的索引缓存
│   │   └── cfs_catalog.py          # SQLite 列名、城市对和官方区域名称
│   ├── models/
│   │   ├── base_optimizer.py       # 基础 Gurobi 模型
│   │   ├── flexible_direct_optimizer.py
│   │   └── gurobi_results.py       # 状态相关属性安全读取
│   ├── algorithms/
│   │   ├── rolling_horizon.py
│   │   ├── paper_rolling_horizon.py
│   │   ├── state_dependent_mip.py
│   │   └── bhh_priority_heuristic.py
│   └── experiments/
│       ├── solvers.py               # 统一求解器注册表
│       ├── core.py                  # 实验规格与结果输出
│       └── gui.py                   # 可视化界面
├── tests/                            # 回归测试
├── docs/                             # 使用与实现说明
└── results/                          # CSV、JSON 输出
```

## 三类动态参数

参数全部定义在 `intercity_delivery/configuration.py`：

| 类别 | 配置类 | 参数键前缀 | 适用范围 |
|---|---|---|---|
| 模型参数 | `DeliveryConfig` | `model.*` | 精确 MIP、Rolling Horizon |
| 算法参数 | `RollingHorizonConfig` | `algorithm.*` | Rolling Horizon |
| 订单参数 | `OrderGenerationConfig` | `order.*` | 订单生成及全部求解器 |

实验核心通过 dataclass 字段反射自动发现参数。以后向这三个配置类新增字段，无需同步
修改 GUI 参数列表、CLI 参数列表或 CSV 配置列。

默认灵敏度水平按基准值自动生成。GUI 和 CLI 都可以用 JSON 数组覆盖水平，例如：

```text
[6,8,10]
[[0,2],[0,5],[0,8]]
[{"1":10,"2":10},{"1":30,"2":30}]
```

算法参数灵敏度规格只交给支持算法参数的 Rolling Horizon。选择全部求解器时，精确
MIP 会自动跳过算法参数规格，避免记录参数变化但算法实际未使用的重复结果。

## 可视化界面

直接运行：

```bash
python main.py
```

界面结构：

- 顶部：实验场景、求解器、种子数、时间限制与运行按钮。
- 中部：模型参数、算法参数、订单参数三个宽幅标签页。
- 底部：实验计划预览和运行日志并排显示。

参数不再集中在左侧窄区域。每个标签页自动读取对应配置类字段，并使用左右两组布局
充分利用窗口宽度。

## 命令行

CLI 与 GUI 调用相同的 `ExperimentPlan`、`build_specs()` 和
`run_experiment_suite()`。

查看全部动态参数：

```bash
python main.py --list-parameters
```

快速比较两个求解器：

```bash
python main.py --cli --scenario quick --solver all --time-limit 60
```

运行灵敏度分析：

```bash
python main.py --cli --scenario sensitivity --solver all --seeds 3 --time-limit 300
```

覆盖一个或多个参数水平：

```bash
python main.py --cli --scenario sensitivity --solver all --seeds 1 \
  --level "algorithm.prediction_horizon=[6,8,10]" \
  --level "order.num_orders=[20,50,100]"
```

PowerShell 中也可以给 `KEY=JSON` 整体加单引号。

只检查实验矩阵：

```bash
python main.py --cli --scenario sensitivity --solver all --seeds 1 --dry-run
```

主要参数：

- `--scenario`：`quick`、`sensitivity`、`all`
- `--solver`：`exact_mip`、`rolling_horizon`、`flexible_direct_mip`、
  `flexible_direct_rolling`、`paper_candidate_mip`、
  `paper_priority_heuristic`、`all`
- `--seeds`：每个参数水平的随机种子数
- `--time-limit`：每次求解的总时间限制
- `--data-source`：`generated` 或 `real`
- --real-data-path：真实数据模式下的 CFS SQLite 或兼容 JSON
- --city-a、--city-b：CLI 使用 SQLite 时选择的两个 CFS Area
- `--level KEY=JSON`：覆盖动态参数水平，可重复使用
- `--list-parameters`：列出三类参数及默认水平
- `--dry-run`：打印规格但不求解

## 结果

输出目录为 `results/`：

- full_experiment_summary_<城市对>__<算法>__<时间戳>.csv：一行对应一次实际求解。
- full_experiment_results_<城市对>__<算法>__<时间戳>.json：完整批次结果。
- detail_<算例>_<城市对>__<算法>__<时间戳>.json：单算例详细结果。

CSV 参数列按类别动态展开：

```text
Model_T
Algorithm_prediction_horizon
Order_num_orders
```

JSON 分别保存：

```text
model_parameters
algorithm_parameters
order_parameters
generation_parameters
```

结果格式版本当前为 4。CSV 新增城市对代码和名称列；完整 JSON 新增 city_pair、city_names、real_data_metadata 和 result_context。Rolling Horizon 的 `Best_Bound` 和全局 `MIP_Gap` 留空，
各窗口状态和窗口 Gap 位于 `detail.windows`。

## 相关文档

- [Rolling Horizon 实现与修改说明](docs/rolling_horizon.md)
- [直送—换装协同模型实现说明](docs/flexible_direct_model.md)
- [可视化界面操作说明](docs/gui_usage.md)
- [实验结果字段说明](docs/result_fields.md)

## SQLite 城市对 GUI 与结果命名（2026-08-18）

- GUI 真实数据入口改为直接加载 SQLite，并异步显示列名。
- 城市 1/城市 2 使用双向关系联动选择，显示两个方向的原始记录数。
- 结果文件名包含城市区域名称、CFS Area 代码和所选 Solution Approach。
- CSV/JSON 同步保存城市对名称、代码、抽样统计和模型建议。

## 修改记录

### 2026-07-01

#### Rolling Horizon

- 实现“外部一次调用、内部多窗口求解”的 Rolling Horizon 控制器。
- 增加预测区间 `prediction_horizon` 和滚动步长 `rolling_step`。
- 固定已经提交的历史弧，关闭预测区间外的决策，并保留跨窗口车辆与货物流状态。
- 将 Rolling Horizon 正式接入统一求解器注册表、GUI、CLI、CSV 和 JSON。
- 修正自动驾驶车辆城市库存平衡约束符号。

#### 三类动态参数

- 将参数明确拆分为 `DeliveryConfig`、`RollingHorizonConfig` 和
  `OrderGenerationConfig`。
- 删除订单参数硬编码字典，三类灵敏度参数统一通过 dataclass 字段动态发现。
- 实验规格同时携带模型、算法、订单三类配置。
- CSV 和 JSON 分类别动态记录全部参数。
- 精确 MIP 自动跳过不适用的算法参数灵敏度规格。

#### GUI 与 CLI

- GUI 改为顶部控制栏、中部三类参数宽幅标签页、底部预览/日志双栏布局。
- CLI 新增 `--list-parameters` 和可重复使用的 `--level KEY=JSON`。
- GUI 与 CLI 统一使用同一实验核心，并显示真实的适用求解次数。
- 已通过 CLI 同时运行精确 MIP 和 Rolling Horizon 的快速回归测试。

#### 直送—换装协同模型

- 新增独立的 `FlexibleDirectOptimizer`，允许自动化换装运输与人工跨城直送共存。
- 直送车辆与城市人工车共享库存，并在跨城后进入目的城市车队。
- 新增直送容量、成本、换装时间/成本和直送比例边界参数。
- 新增完整 MIP 与 Rolling Horizon 两个统一求解器接口。
- 新增求解级单元测试，覆盖自由直送选择和固定 50% 直送比例。

### 2026-06-27 动态灵敏度实验与完整结果输出

- 将实验类别精简为快速测试和灵敏度分析；订单规模水平同时承担基准与规模扩展实验。
- 灵敏度分析改为动态发现 `DeliveryConfig` 的全部 dataclass 字段，新增配置字段后无需修改 GUI。
- 将订单数、时间窗缓冲、大订单概率及大小订单货量范围全部纳入单因素灵敏度分析。
- GUI 的参数水平统一采用 JSON 数组，支持数值、布尔值、区间、列表和城市字典。
- CSV 自动输出全部 `Config_*` 参数列，并新增明确的灵敏度参数、水平及完整订单生成字段。
- 新增完整批次 JSON，同一算例的订单只保存一次，并集中记录多个求解器的结果。

### 2026-06-27 模型公式一致性修正

- 将人工车辆和自动驾驶车辆的承运货量变量由整数变量调整为非负连续变量，与模型中的 \(\mathbb{R}^+\) 定义一致。
- 将自动驾驶车辆容量约束（7）由“同一弧全部订单货量求和”调整为模型原式中的逐订单约束。
- 为约束（2）—（11）补充业务含义、累计时序关系和容量计算方式的详细代码注释。

### 2026-06-26 模型一致性修正

- 修正随机订单生成中的未服务惩罚成本：`OrderBatch.penalty_lost` 现在保存单位未服务惩罚成本，避免在目标函数中重复乘以订单需求量。
- 修正人工车辆时间弧生成逻辑：时间弧现在包含 `j <= i + f^k(M)` 的边界情况，并限制 `j` 不超过规划期末。
- 修正自动驾驶车辆跨城平衡约束：约束 (4)(5) 现在按模型区分车辆出发时间 `i <= t` 和到达时间 `j <= t`。
- 修正转运节点流守恒约束：约束 (9)(10) 现在按订单 `l` 分别建立，避免不同订单之间的货量相互抵消。
- 暂未调整货量变量类型与自动驾驶车辆容量约束，保持当前整数货量变量和总容量约束实现不变。

### 2026-06-26 论文仿真实验框架

- 将主程序由交互式选择改为命令行实验套件，支持快速测试、基准测试、规模扩展测试、灵敏度分析和全量实验。
- 随机订单生成器新增时间窗缓冲区间和大订单比例参数，便于研究时间窗紧迫程度与需求结构变化。
- 实验结果新增随机种子、总需求量、时间窗参数、大订单比例、MIP Gap 和 Best Bound 等论文分析字段。
- 支持 `--dry-run` 查看实验计划，便于在正式长时间求解前确认仿真矩阵。

### 2026-06-26 可视化与求解器接口重构

- 将 `main.py` 拆分为入口、实验核心、求解器接口和可视化界面四层结构。
- 新增 `customtkinter` 可视化实验界面，可在窗口中选择实验场景、求解方式和参数范围。
- 新增统一求解器接口，当前支持 `exact_mip`，并预留 `rolling_horizon` 扩展入口。
- 保证同一个算例的订单数据只生成一次，再传给不同求解器，便于后续算法公平对比。
- 新增可视化界面操作说明和运行结果字段说明文档。
