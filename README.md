# BHH / Intercity：模型同步与调度实现

当前论文终稿入口：[final_manuscript/main.tex](research/paper/final_manuscript/main.tex)。新稿已补引言、文献综述、优化模型与数学推理，原TeX保留；数值实验章节及实验计划暂未写，未编译。结构和审查入口见[终稿说明](research/paper/final_manuscript/README.md)。

论文写作与QVI研究最新进度见 [PAPER_WRITING_PLAN](coordination/PAPER_WRITING_PLAN.md)：新版steady作主稿、linear作简化基准；限定共同子模型下Wave与QVI阈值一致的证明和toy核对已完成，完整随机QVI及原TeX修订尚未完成。

分支：`research/qvi-multiagent`。更新：2026-10-18。

本分支把新论文的有限时域 BHH wave 模型落实到可运行的 Python 优化程序，并保留旧模型和历史实验作比较。**当前已经实现 V1-M、V1-F、V1-F rolling horizon 和货流剪枝；V1-F 三种运行方式已在小样本 CFS 上跑通。**

V1-M 是强制换装，货物走“取货—自动驾驶干线—配送”；V1-F 是 flexible，允许上述换装链与人工车直送并存。这里的 F 表示运输方式可选择。V2 的随机到达、内生 wave 和 QVI 理论另行研究。

## 当前可运行的入口

| 求解器名称 | 行为 |
|---|---|
| `v1_flexible_mip` | 所有订单已知的完整 V1-F MILP |
| `v1_flexible_rolling` | 按释放时刻揭示订单，跨窗口保留已执行决策 |
| `v1_flexible_pruned_rolling` | 同一滚动控制，加实际稀疏货流变量筛选 |

三个名称均已接入 `main.py` 和共享求解器注册表，GUI 可选。原 V0 名称和默认选择保持兼容。普通实验入口的适配通过纯测试；下述真实 CFS 求解证据来自专用小样本执行器。

在项目根目录运行：

```powershell
# 只查看数据来源、参数与预算，不调用求解器
python -B experiments/specs/run_v1f_cfs_smoke.py

# 三种方式各最多15秒，固定同一组8条CFS订单
python -B experiments/specs/run_v1f_cfs_smoke.py --run

# 查看普通实验入口的计划
python -B main.py --cli --solver v1_flexible_pruned_rolling --data-source real --real-data-path data/cfs_processed_sqlite/cfs_model_orders.json --seeds 1 --time-limit 15 --dry-run
```

真实求解需要可用的 `gurobipy` 与许可证。本机已验证环境为 `C:/Users/32599/.conda/envs/pavane/python.exe`、Gurobi 10.0.3。脚本默认将完整非零方案和来源证据输出到终端；可选 `--output 新文件.json`，会在任何求解前检查能否创建文件，且不覆盖现有文件。

## 本次 CFS 验证结果

城市对为 Los Angeles `06-348` ↔ San Francisco `06-488`；seed 4101，8条订单，T=24，每期60分钟，城际时间9期，滚动预测8期/步长2期/延伸6期。

| 方法 | 完成状态 | 服务量 / 需求 | 总成本 |
|---|---|---|---|
| 完整 V1-F | OPTIMAL，独立核验通过 | 8 / 8 | 1100 |
| 滚动 | COMPLETED，12窗口，独立核验通过 | 7 / 8 | 2025 |
| 剪枝滚动 | COMPLETED，12窗口，独立核验通过 | 7 / 8 | 2025 |

剪枝把12个窗口的货流变量总数从8265减为397，实际删除7868个，约95.20%。保留车辆母弧和独立空车调拨。单个样本相同成本不证明一般等价；离线完整模型与在线滚动的信息集不同，不直接用成本差计算算法最优性差距，也不据此宣称加速。

CFS 当前使用“一条 shipment = 一单位 stop”的显式运行代理，保留原 SHIPMT_ID、重量、权重和旧合成时间窗。这只验证能运行，尚未校准真实客户停靠点。旧 V0 的重量数量不随之改动；混跑按求解器隔离转换。普通 V1 JSON 入口采用来源的城际时长建议，显式敏感性覆盖另行披露。

详细记录：[实施与验证报告](experiments/reports/v1_f_runtime_smoke.md)、[原始可重放结果](experiments/reports/v1_f_cfs_smoke_results.json)、[独立评审](coordination/reviews/v1_f_runtime_review.md)。

## 文件结构与关键路径

本机根目录：`D:/mycodelife/workshop/AI/multiagent-test`。下表均为相对根目录的路径。

| 路径 | 作用 |
|---|---|
| `main.py` | GUI / CLI 入口 |
| `AGENTS.md`、`.codex/agents/` | 主协调者及 Literature、Experiment、Theorist、Referee 的职责 |
| `research/paper/main_v2_BHH.tex` | V1 有限时域模型的首要数学来源 |
| `research/paper/main_BHH.tex` | 旧论文，对照审计使用 |
| `research/paper/steady_state_model*.tex` | 两个稳态变体，与有限时域 V1 分开 |
| `research/theory/` | 数学比较、容量合同、QVI 研究笔记 |
| `research/literature/` | 文献定位和核验记录 |
| `intercity_delivery/models/`、`algorithms/` | 保留的 V0 优化器和原滚动、剪枝、启发式算法 |
| `intercity_delivery/repositioning/` | 先前实现的 V0-F 独立空车扩展 |
| `intercity_delivery/v1/configuration.py` | 作者确认的基础配置、订单与 profile |
| `intercity_delivery/v1/capacity.py`、`network.py` | 原式 f/h/K、local 与 AV 母弧 |
| `intercity_delivery/v1/mandatory_optimizer.py` | 已实现的完整 V1-M |
| `intercity_delivery/v1/flexible_configuration.py` | V1-F 参数、空车与终点政策 |
| `intercity_delivery/v1/flexible_capacity.py` | 方向相关 direct 容量与独立 empty 母弧 |
| `intercity_delivery/v1/flexible_optimizer.py` | V1-F MILP：one-hot wave、换装/直送/空车与库存 |
| `intercity_delivery/v1/flexible_rolling.py` | 订单揭示、历史提交、窗口和总预算控制 |
| `intercity_delivery/v1/pruning.py` | 释放/期限、正容量与三腿时间支持筛选 |
| `intercity_delivery/v1/flexible_validation.py` | 用原始公式独立核容量、库存、货流和成本 |
| `intercity_delivery/v1/flexible_runner.py` | 完整/滚动统一结果、状态、来源散列 |
| `intercity_delivery/v1/adapter.py` | 旧接口和 CFS 的显式适配 |
| `intercity_delivery/experiments/solvers.py`、`core.py` | 求解器注册、独立数据口径、结果输出 |
| `tests/test_v1_flexible*.py`、`test_v1_pruning.py`、`test_v1f_integration.py` | 模型、跨窗、剪枝、独立核验、入口适配测试 |
| `experiments/specs/run_v1f_cfs_smoke.py` | 有界 CFS 执行检查，默认不求解 |
| `experiments/reports/` | 审计、实施报告和本轮结果 |
| `data/cfs_processed_sqlite/cfs_model_orders.json` | 已处理 CFS 样本，默认小检查输入 |
| `data/cfs_2022_pums.sqlite` | 本机原始 SQLite，可按索引只读抽样 |
| `simulation/runs/` | 先前数值实验配置和历史结果，保留 |
| `coordination/` | 模型合同、任务、主张、迁移计划和评审 |

本地 TeX、数据及部分历史文件尚未进入 Git；其他机器需取得对应输入。数据来源与代码 SHA-256 随结果记录，Git HEAD 单独不能标识整个本地工作区。

## 模型政策与尚未实现的范围

- wave 沿用已确认 D1–D6-A：按起止期、城市和方向分组，连续 stop 量，分钟与车小时费率，tour结束交付，允许同刻交接。
- local 用联合 BHH 容量；direct 用新论文方向相关两端 tour 公式。local selector 上界改为全车队数量，使迁入车辆可在新城市服务，动态库存仍严格限制车辆。
- 空调拨有独立变量、弧和费用，无订单也能移动。源文保留的 τ 长度零货 direct 弧仍可空行，metadata/报告披露；载货由独立货流变量表达。
- 每个计算窗口不要求车辆回原城市。真实期末默认无目标；完整模式可显式指定目标，滚动模式暂拒绝该配置，等待尾部可达性保障。
- 已启动任务按绝对时刻保留，货物可跨窗口在途或留在 hub。部分运行不给完整总成本；滚动不给全日 bound/gap。
- 本轮完成 V1-F、滚动和保守货流剪枝。V1 启发式初解、warm start、local branching、正式大规模回归、稳态验证和随机 QVI 尚未实施。

来源和决定见 [MODEL_CONTRACT](coordination/MODEL_CONTRACT.md)、[运行合同](coordination/V1_F_RUNTIME_CONTRACT.md)；进度见 [PROGRESS](coordination/PROGRESS.md)，后续依赖见 [MIGRATION_PLAN](coordination/MIGRATION_PLAN.md)。本轮没有修改论文、旧优化器、历史实验数据，也没有提交或推送 Git。
