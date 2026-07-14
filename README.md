# Intercity Delivery System Design

基于混合整数规划（MIP）的城际配送系统优化与仿真实验平台。系统协同调度城市内
人工驾驶车辆和城际自动驾驶车辆，在车辆、容量、换装时序和订单时间窗约束下最小化
车辆成本与未服务惩罚。

## 论文 Solution Approach（2026-07-14）

`Intercity_Operation.pdf` 的两个解法已经实现，并作为两个独立求解器接入实验框架：

- `paper_candidate_mip`：状态相关候选弧生成 + 削减后的滚动 MIP（Algorithm 1）。
- `paper_priority_heuristic`：动态 BHH-aware 优先级构造启发式（Algorithm 2）。

两者使用新的 `paper_rolling_horizon.py`。该控制器支持控制窗口、预测起始窗口、
扩展完成窗口、订单逐期揭示和已提交车辆/货流状态；原有 `rolling_horizon.py` 未被修改。
GUI 中两个论文解法是独立复选框，可全选或只选其中一个，旧求解器保留为可选基准。

对应说明：

- [解法 1：状态相关候选弧削减 MIP](docs/state_dependent_candidate_mip.md)
- [解法 2：动态 BHH-aware 优先级启发式](docs/dynamic_bhh_priority_heuristic.md)

### 生成数据与真实数据

GUI 的“测试数据”区域必须明确选择数据来源：

- **生成数据**：使用 `OrderGenerationConfig` 和随机种子构造订单。
- **真实数据**：选择 `cfs_data_processor.py` 输出的 `cfs_model_orders.json`。

真实模式按实验种子从文件中确定性抽样 `order.num_orders` 条记录，重新编号并使用当前
模型的 `penalty_lost`。若订单时间窗超出当前 `T`，程序会报错并提示使用相同规划期
重新处理 CFS 数据，避免静默裁剪。

官方 2022 CFS PUMS 的处理方法见：

- [CFS 2022 数据内容与处理](docs/cfs_2022_data_processing.md)

示例处理命令（在 `pavane` 环境运行）：

```powershell
conda activate pavane
python cfs_data_processor.py --input path\to\cfs_2022_pumf.zip `
  --output-dir data\cfs_processed --num-orders 100 --seed 42
```

新增核心文件：

```text
paper_rolling_horizon.py       # 论文专用滚动时域控制与统一窗口接口
state_dependent_mip.py         # Algorithm 1
bhh_priority_heuristic.py      # Algorithm 2
cfs_data_processor.py          # 官方 CFS PUMS 转换为模型订单
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
├── config.py           # 模型、算法、订单三类参数 dataclass
├── data_loader.py      # 时间弧、集合、容量系数与订单数据
├── optimizer.py        # Gurobi 数学模型
├── flexible_direct_optimizer.py # 直送与换装共存模型
├── rolling_horizon.py  # Rolling Horizon 窗口控制和决策提交
├── solvers.py          # 统一求解器接口
├── experiment_core.py  # 动态参数、实验规格、结果输出
├── experiment_gui.py   # 可视化实验界面
├── main.py             # GUI 与 CLI 统一入口
├── docs/               # 使用与实现说明
└── results/            # CSV、JSON 输出
```

## 三类动态参数

参数全部定义在 `config.py`：

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
  `flexible_direct_rolling`、`all`
- `--seeds`：每个参数水平的随机种子数
- `--time-limit`：每次求解的总时间限制
- `--level KEY=JSON`：覆盖动态参数水平，可重复使用
- `--list-parameters`：列出三类参数及默认水平
- `--dry-run`：打印规格但不求解

## 结果

输出目录为 `results/`：

- `full_experiment_summary_*.csv`：一行对应一次实际求解。
- `full_experiment_results_*.json`：参数、订单、多个求解器结果和详细解。
- `detail_*.json`：需要保存明细的单个算例结果。

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

结果格式版本当前为 `3`。Rolling Horizon 的 `Best_Bound` 和全局 `MIP_Gap` 留空，
各窗口状态和窗口 Gap 位于 `detail.windows`。

## 相关文档

- [Rolling Horizon 实现与修改说明](docs/rolling_horizon.md)
- [直送—换装协同模型实现说明](docs/flexible_direct_model.md)
- [可视化界面操作说明](docs/gui_usage.md)
- [实验结果字段说明](docs/result_fields.md)

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
