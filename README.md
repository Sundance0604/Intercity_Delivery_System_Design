# Intercity Delivery System Design

这是一个基于混合整数规划（MIP）的城际物流配送系统优化模型。该系统旨在通过协同调度**人工驾驶车辆**和**自动驾驶车辆**，在满足订单时间窗和运力限制的前提下，最小化总运营成本。

## ✨ 主要功能

* **多车型协同**：支持人工车辆（短途/集散）与自动驾驶车辆（长途/干线）的混合调度。
* **精确建模**：考虑了车辆流守恒、载重限制、服务时间窗及 BHH 服务效率函数。
* **灵敏度分析**：支持对自动驾驶车队规模、成本系数等关键参数进行批量压力测试。
* **详细日志**：自动保存实验结果汇总（CSV）及详细调度方案（JSON）。
## 🛠️ 环境依赖

* **Python 3.11+**
* **Gurobi Optimizer** (需要有效的 License)
* **customtkinter** (用于可视化实验界面)

## 📂 项目结构
.
├── config.py           # 参数配置中心 (车辆参数、成本、时间窗设置)

├── data_loader.py      # 数据预处理 (生成时间弧、计算载重系数、加载订单)

├── optimizer.py        # 核心优化引擎 (Gurobi 变量定义、约束构建、目标函数)

├── main.py             # 程序入口 (启动可视化界面或命令行批处理)

├── experiment_core.py  # 仿真实验核心逻辑 (实验计划、订单生成、批量运行、结果保存)

├── solvers.py          # 求解器接口层 (精确MIP、Rolling Horizon等算法扩展入口)

├── experiment_gui.py   # 可视化实验界面

└── results/            # 输出目录 (自动存放 CSV 和 JSON 结果文件)

### 第三部分：运行指南与实验配置


## 🚀 快速开始

1.  **配置参数**：
    在 `config.py` 中调整基础参数（如时间段 `T`、车辆数 `N`、单位成本 `cost` 等）。

2.  **运行模型**：
    直接运行主程序，默认打开可视化实验界面。

    ```bash
    python main.py
    ```

    也可以通过命令行参数运行论文仿真场景：

    ```bash
    python main.py --cli --scenario baseline --solver exact_mip --seeds 5 --time-limit 300
    python main.py --cli --scenario scale --solver exact_mip --seeds 5 --time-limit 500
    python main.py --cli --scenario sensitivity --solver exact_mip --seeds 5 --time-limit 500
    python main.py --cli --scenario all --solver exact_mip --seeds 5 --time-limit 500
    ```

    如需先查看实验计划而不求解：

    ```bash
    python main.py --cli --scenario sensitivity --solver exact_mip --seeds 3 --dry-run
    ```

3.  **查看结果**：
    运行结束后，前往 `results/` 目录查看：
    * `experiment_summary_*.csv`: 所有实验组的成本、服务率等 KPI 汇总。
    * `detail_exp_*.json`: 特定实验的车辆路径和订单分配详情。

## ⚙️ 实验配置

详细操作说明请见：

* [可视化仿真实验界面操作说明](docs/gui_usage.md)
* [仿真实验运行结果说明](docs/result_fields.md)

`main.py` 目前支持四类实验套件：

* **quick**：默认快速测试，订单规模为 20，用于检查模型能否正常运行。
* **baseline**：小规模基准测试，订单规模为 20、50，用于和精确 MIP 最优解或后续算法结果对比。
* **scale**：规模扩展测试，订单规模为 100、200、500、1000，用于分析求解时间、服务率和成本随规模的变化。
* **sensitivity**：灵敏度分析，分别考察自动驾驶车辆数量、自动驾驶车辆单位成本、人工车辆数量、时间窗紧迫程度和大订单比例。
* **all**：依次运行 baseline、scale 和 sensitivity。

主要命令行参数：

* `--cli`：使用命令行模式；不加该参数时默认打开可视化界面。
* `--scenario`：选择实验套件，可选 `quick`、`baseline`、`scale`、`sensitivity`、`all`。
* `--solver`：选择求解器，可选 `exact_mip`、`rolling_horizon`、`all`。
* `--seeds`：每个参数水平运行的随机种子数量，用于得到均值和波动范围。
* `--time-limit`：每个算例的 Gurobi 求解时间上限，单位为秒。
* `--dry-run`：仅打印实验计划，不执行求解。

实验结果会记录 `Scenario`、`Seed`、`Num_Orders`、`Total_Demand`、`Buffer_Min`、`Buffer_Max`、`Large_Order_Prob`、`MIP_Gap`、`Best_Bound`、`Total_Cost`、`Unserved_Rate`、`Auto_Usage` 和 `Manual_Usage` 等字段，便于后续论文制表和算法对比。

## 修改记录

### 2026-06-26 模型一致性修正

* 修正随机订单生成中的未服务惩罚成本：`OrderBatch.penalty_lost` 现在保存单位未服务惩罚成本，避免在目标函数中重复乘以订单需求量。
* 修正人工车辆时间弧生成逻辑：时间弧现在包含 `j <= i + f^k(M)` 的边界情况，并限制 `j` 不超过规划期末。
* 修正自动驾驶车辆跨城平衡约束：约束 (4)(5) 现在按模型区分车辆出发时间 `i <= t` 和到达时间 `j <= t`。
* 修正转运节点流守恒约束：约束 (9)(10) 现在按订单 `l` 分别建立，避免不同订单之间的货量相互抵消。
* 暂未调整货量变量类型与自动驾驶车辆容量约束，保持当前整数货量变量和总容量约束实现不变。

### 2026-06-26 论文仿真实验框架

* 将主程序由交互式选择改为命令行实验套件，支持快速测试、基准测试、规模扩展测试、灵敏度分析和全量实验。
* 随机订单生成器新增时间窗缓冲区间和大订单比例参数，便于研究时间窗紧迫程度与需求结构变化。
* 实验结果新增随机种子、总需求量、时间窗参数、大订单比例、MIP Gap 和 Best Bound 等论文分析字段。
* 支持 `--dry-run` 查看实验计划，便于在正式长时间求解前确认仿真矩阵。

### 2026-06-26 可视化与求解器接口重构

* 将 `main.py` 拆分为入口、实验核心、求解器接口和可视化界面四层结构。
* 新增 `customtkinter` 可视化实验界面，可在窗口中选择实验场景、求解方式和参数范围。
* 新增统一求解器接口，当前支持 `exact_mip`，并预留 `rolling_horizon` 扩展入口。
* 保证同一个算例的订单数据只生成一次，再传给不同求解器，便于后续算法公平对比。
* 新增可视化界面操作说明和运行结果字段说明文档。
