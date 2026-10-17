# BHH / Intercity 多 Agent 科研分支

当前分支：`research/qvi-multiagent`。更新日期：2026-10-17。

本分支用于梳理 BHH / Intercity 城际配送研究中的数学模型、Python 实现与理论扩展，建立可追溯的科研协作和模型迁移流程。核心工作是核对最新论文实际定义的模型，找出现有程序的差距，经过独立审查后形成实施计划。

目前已完成一轮模型同步审计。**实际代码迁移尚未开始，稳态数值验证和完整随机 QVI 理论也尚未完成。** 当前 Python 程序保留原有模型与算法，审计报告的提交不表示新模型已经实现。

## 1. 分支用途与模型边界

系统包含两个城市、城市内人工驾驶车辆、城际自动驾驶车辆及两端换装枢纽，研究取货、干线运输、配送、直送与车辆调度之间的关系。本分支区分：

| 对象 | 含义 | 首要依据 |
|---|---|---|
| V0 | 现有 Python 实际实现，包括强制换装、灵活直送及对应滚动算法 | `intercity_delivery/` 下的源码 |
| V1 | 新论文的有限时域 wave 容量与调度模型 | 本地 `research/paper/main_v2_BHH.tex` |
| 稳态变体 | 共同 timetable 下的批量、载量、成本与期限分析；两稿单列比较 | 本地 `steady_state_model.tex`、`steady_state_model_v1_linear.tex` |
| V2 | 随机订单抵达、内生 wave 与动态发车决策的 QVI 研究目标 | 用户研究定义与本地理论笔记 |

V0 以实际代码为准，旧论文不能替代代码审计。V1、稳态变体和 V2 不默认等价。本地 QVI 笔记目前主要处理确定性常率或时变需求率，尚不能代表完整随机到达模型。

本分支的产出是数学比较、代码映射、文献核验、独立审查、模型合同及迁移计划。实际实现和实验须按依赖与批准范围推进。

## 2. 建议阅读顺序

| 顺序 | 路径 | 作用 |
|---|---|---|
| 1 | [coordination/PROGRESS.md](coordination/PROGRESS.md) | 当前完成事项、限制及下一步 |
| 2 | [coordination/MODEL_CONTRACT.md](coordination/MODEL_CONTRACT.md) | 模型命名、来源、假设边界及输入散列 |
| 3 | [coordination/MODEL_DIFF.md](coordination/MODEL_DIFF.md) | V0、V1、稳态与 QVI 的差异索引 |
| 4 | [coordination/reviews/model_alignment_review.md](coordination/reviews/model_alignment_review.md) | 独立审查、严重问题及结论 |
| 5 | [coordination/MIGRATION_PLAN.md](coordination/MIGRATION_PLAN.md) | 待确认问题、实施依赖、验证与完成标准 |

具体结论的证据位于三份专业报告：

| 路径 | 负责 Agent | 内容 |
|---|---|---|
| [research/theory/model_comparison.md](research/theory/model_comparison.md) | Theorist | 逐公式比较、代码结构映射、假设与证明缺口、QVI 边界 |
| [experiments/reports/model_code_gap.md](experiments/reports/model_code_gap.md) | Experiment | 文件/函数/行号对应、可复用部分、迁移与测试设计 |
| [research/literature/model_positioning.md](research/literature/model_positioning.md) | Literature | 研究定位、公开引文核验及其实际支持范围 |

## 3. 文件结构

以下目录树描述当前本地工作区。标注“本地输入”的文件未纳入本轮审计提交，远端克隆不会自动获得；工作目录也不表示已有新实验实现。

~~~text
.
├── README.md                            # 本分支用途、结构与阅读入口
├── AGENTS.md                            # 科研协作、证据等级与修改规则
├── .codex/
│   ├── config.toml                      # 项目级 Agent 配置
│   └── agents/
│       ├── literature.toml              # 文献与写作职责
│       ├── experiment.toml              # 优化程序与实验职责
│       ├── theorist.toml                # 数学模型与理论职责
│       └── referee.toml                 # 独立审查职责
├── coordination/
│   ├── MODEL_CONTRACT.md                # 模型来源、定义与对齐合同
│   ├── MODEL_DIFF.md                    # 模型差异索引
│   ├── MIGRATION_PLAN.md                # 迁移阶段与作者确认清单
│   ├── CLAIMS.md                        # 主张、假设与证据状态
│   ├── TASKS.md                         # 有界任务、依赖与验收
│   ├── PROGRESS.md                      # 已核实进度和开放问题
│   ├── FIRST_RUN_PROMPT.md              # 初始四 Agent 演练模板
│   └── reviews/
│       └── model_alignment_review.md    # 本轮独立审查报告
├── research/
│   ├── paper/                           # 论文数学来源；本地输入
│   │   ├── main_BHH.tex                 # 旧论文
│   │   ├── main_v2_BHH.tex              # 新有限时域模型
│   │   ├── steady_state_model.tex       # deadline / sorting 稳态稿
│   │   └── steady_state_model_v1_linear.tex # 较早线性时间成本稳态稿
│   ├── theory/
│   │   ├── model_comparison.md          # 已提交的数学比较报告
│   │   └── Common Wave 的内生化与非均匀需求扩展.md # 本地 QVI 输入
│   ├── literature/
│   │   └── model_positioning.md         # 已提交的文献定位报告
│   └── toy_models/                      # 独立小型理论测试工作目录
├── experiments/
│   ├── reports/
│   │   └── model_code_gap.md            # 已提交的模型—代码审计
│   └── specs/                           # 实验方案工作目录
├── intercity_delivery/                  # 现有 V0 优化程序
│   ├── configuration.py                # 模型、算法及订单配置
│   ├── models/                         # 强制换装和灵活直送优化器
│   ├── algorithms/                     # 滚动时域、候选网络、启发式
│   ├── data/                           # 订单、CFS 处理和 SQLite
│   └── experiments/                    # 求解器注册、实验核心、GUI
├── main.py                             # 现有 GUI / CLI 入口
├── tests/                              # 现有回归测试
├── simulation/                         # 原有实验协议、运行与对比记录
├── docs/                               # 原有算法、数据与操作说明
├── data/                               # 本地原始/缓存数据，Git 忽略
└── results/                            # 本地程序输出，Git 忽略
~~~

本轮未在 `research/toy_models/` 或 `experiments/specs/` 新增实验程序或方案。计划中的 `intercity_delivery/v1/` 尚未创建。

## 4. 关键路径与职责

### 科研协调

- [AGENTS.md](AGENTS.md)：主 Agent 协调职责、专业角色分工和修改边界。
- [.codex/agents/](.codex/agents/)：四个专业角色的实际配置。各自写独立报告，Referee 在交付后审查，主 Agent 负责综合。
- [coordination/CLAIMS.md](coordination/CLAIMS.md)：记录重要主张的假设、来源、证明/实验证据及审查状态；静态核实不等于定理成立。
- [coordination/TASKS.md](coordination/TASKS.md)：记录交付、待批准实施阶段和独立研究任务。

### 现有程序

| 路径 | 作用 |
|---|---|
| [intercity_delivery/models/base_optimizer.py](intercity_delivery/models/base_optimizer.py) | V0 强制换装模型 |
| [intercity_delivery/models/flexible_direct_optimizer.py](intercity_delivery/models/flexible_direct_optimizer.py) | V0 直送/换装共存及人工车辆迁移 |
| [intercity_delivery/data/loader.py](intercity_delivery/data/loader.py) | BHH 函数、时间弧、容量系数与订单结构 |
| [intercity_delivery/algorithms/paper_rolling_horizon.py](intercity_delivery/algorithms/paper_rolling_horizon.py) | 滚动窗口控制和历史决策 |
| [intercity_delivery/algorithms/state_dependent_mip.py](intercity_delivery/algorithms/state_dependent_mip.py) | 状态相关候选网络与窗口模型 |
| [intercity_delivery/algorithms/bhh_priority_heuristic.py](intercity_delivery/algorithms/bhh_priority_heuristic.py) | BHH 优先级构造初解 |
| [intercity_delivery/experiments/solvers.py](intercity_delivery/experiments/solvers.py) | 求解器注册与结果适配 |
| [main.py](main.py) | GUI / CLI 入口 |

原有操作说明见 [GUI](docs/gui_usage.md)、[滚动时域](docs/rolling_horizon.md)、[直送模型](docs/flexible_direct_model.md)及[实验协议](docs/numerical_experiment_protocol.md)。它们描述现有实现，不能作为 V1 已迁移的证据。

### 本地数学来源

当前审计工作区根目录：

~~~text
D:/mycodelife/workshop/AI/multiagent-test
~~~

四份 TeX 位于该目录下的 `research/paper/`，QVI 笔记位于 `research/theory/Common Wave 的内生化与非均匀需求扩展.md`。V1 的完整路径为：

~~~text
D:/mycodelife/workshop/AI/multiagent-test/research/paper/main_v2_BHH.tex
~~~

审计提交 `93b2dcb` 仅包含 10 份审计与协调文档，未包含上述本地 TeX 和 QVI 输入。散列已登记在模型合同中；其他机器复核时须取得相同版本。两份 main 稿所需的 `导出的条目.bib` 及部分稿称校准脚本尚未在可读工作树找到，具体限制见专业报告。

## 5. 当前状态与下一步

本轮结论：**三份专业报告可作为静态审计依据接受，模型与算法同步仍需重大修订。**

主要差距包括 wave 联合容量与 stem 未落入代码、论文求解章节保留旧容量逻辑、空返和终端恢复缺失、候选弧及窗口定义不一致。CFS 重量单位不能直接当作 V1 的停靠点数。

自动复核确定的下一步优先级：

1. **G0-M：先整理 V1 强制换装最小决策规格。** 建立公式—代码—验收矩阵；将 wave 分组、单位、信息及窗口等选择留给作者确认。
2. **G1：批准后实现独立容量表与弧模块。** 验证公式代回、零容量、物理上限、取整和单位。
3. **G2＋对应 G4：实现最小 V1 强制换装模型与独立核算。** 公式可追溯、成本/可行性独立检查；微型求解有预算，V0 保持原样。

`coordination/V1_M_DECISION_SPEC.md` 是下一项拟议交付，当前尚未创建。直送、滚动时域、剪枝、启发式及稳态/QVI 研究按迁移计划的依赖分别推进。

本轮审计没有运行测试或求解，没有修改论文、生产代码及历史实验数据。提交和推送用于保存审计成果，不表示模型歧义已解决或实际迁移已获批准。
