# BHH 结论、来源与标签迁移

2026-10-18。当前主文件：[main.tex](main.tex)。本表用于追溯已有成果，未产生新模型等价或创新性认证。章节号按源顺序描述；以作者实际编译结果为准。

## 贡献与证据

| 表述 | 现位置与依据 | 必须保留的范围 |
|---|---|---|
| wave容量进入有限车队调度 | §3 service_capacities；§4 optimization_model，mod:local-capacity、mod:local-links、mod:human-inventory、mod:demand | 给定路由近似；固定整数车队与订单窗 |
| 固定local arc的每车容量非递减、超可加 | 原Prop.4.1→§3 Prop.3.1，prop:analysis-capacity；附录A完整证明 | 同城市、方向、起止期；不含Direct容量或全局凸性 |
| 条件最优collection/distribution载荷 | 原Prop.4.2→§5 Prop.5.1，prop:analysis-loads；附录A证明 | common-wave；正成本条件；0<n_i≤min(M,q) |
| 最优重复wave与静态成本最小值一致 | 原Thm.4.1→§5 Thm.5.1，thm:analysis-consistency；附录B日历证明 | 恒定到达、全量reset、逐批可分成本及声明的可容许策略 |
| 相对值和QVI验证 | 原Thm.4.2→附录B Thm.B.1，thm:analysis-qvi；定理与证明一起移动 | 尾包络、零点等待边界、a.e.方程、增长与横截条件；非MILP验证 |
| Poisson与fluid整数阈值比较 | 原Prop.4.3→附录B Prop.B.1，prop:analysis-poisson；完整证明保留 | 相同正整数full-reset域和C；成本率相差omega/2 |
| rolling horizon实施 | §4末 solution_method；附录C完整filter/replay | 固定历史及完整任务；partial可见；窗口界不作全日界 |
| BHH几何解释 | 新eq:wave-geometry；main_v2_BHH.tex:71、steady_state_model.tex:62–69；Banerjee印刷1331页 | b=barbeta sqrt(S)是解释性参数化；没有实际区域设计或校准数据 |

两项主要贡献为“容量与有限车队调度建模”和“受限解析解释”。rolling horizon、过滤与replay是实施支撑。BHH渐近定理、renewal比值和QVI方法已有文献前史，未仅凭这些工具宣称新颖性。

## 章节分工

| 当前位置 | 内容 | 原来源 |
|---|---|---|
| §3 service_capacities | 服务系统、BHH解释、容量与pooling命题、Direct容量 | 原§3.1及§4.1；新增解释式、符号表、示意图 |
| §4 optimization_model + solution_method | 完整MILP、模式与终点、核心滚动过程 | 原§3.2–3.6、§5.1 |
| §5 mathematical_analysis | 基准假设、载荷、重复wave结论、简短验证概述 | 原§4.2–4.3 |
| 附录A proofs | 容量反解与载荷证明 | 原§4.1推导及原附录A.1 |
| 附录B dispatch_verification | 日历界、完整QVI和Poisson结论及证明 | 原§4.3余项、§4.4–4.5及原附录A.2–A.4 |
| 附录C implementation_details | cargo过滤与独立replay | 原§5.2–5.3 |

删去的是重复的模型差异说明、重复总结和非必要扩展讨论。关键局部条件、MILP约束与全部已有命题及证明保留。两图是机制示意，无实验含义。

## 旧标签到新文件

所有68个旧标签保留，另新增10个结构、图表和解释标签。下表中路径均相对于本目录；移动后沿用原label名称。

| Label | 原文件 | 新文件 |
|---|---|---|
| alg:auto-support | solution_method.tex | implementation_details.tex |
| alg:delivery-support | solution_method.tex | implementation_details.tex |
| alg:downstream-cut | solution_method.tex | solution_method.tex |
| alg:local-replay | solution_method.tex | implementation_details.tex |
| alg:origin-cut | solution_method.tex | solution_method.tex |
| alg:time-bounds | solution_method.tex | implementation_details.tex |
| alg:window | solution_method.tex | solution_method.tex |
| app:proofs | proofs.tex | proofs.tex |
| ass:analysis-accounting | mathematical_analysis.tex | mathematical_analysis.tex |
| ass:analysis-benchmark | mathematical_analysis.tex | mathematical_analysis.tex |
| eq:analysis-average-control | mathematical_analysis.tex | mathematical_analysis.tex |
| eq:analysis-calendar-bound | mathematical_analysis.tex | dispatch_verification.tex |
| eq:analysis-capacity | mathematical_analysis.tex | proofs.tex |
| eq:analysis-consistency | mathematical_analysis.tex | mathematical_analysis.tex |
| eq:analysis-dispatch-cost | mathematical_analysis.tex | mathematical_analysis.tex |
| eq:analysis-first-dispatch | mathematical_analysis.tex | dispatch_verification.tex |
| eq:analysis-generalized-cost | mathematical_analysis.tex | mathematical_analysis.tex |
| eq:analysis-inverse | mathematical_analysis.tex | proofs.tex |
| eq:analysis-load-domain | mathematical_analysis.tex | mathematical_analysis.tex |
| eq:analysis-local-bridge | mathematical_analysis.tex | proofs.tex |
| eq:analysis-n1 | mathematical_analysis.tex | mathematical_analysis.tex |
| eq:analysis-n2 | mathematical_analysis.tex | mathematical_analysis.tex |
| eq:analysis-poisson | mathematical_analysis.tex | dispatch_verification.tex |
| eq:analysis-qvi | mathematical_analysis.tex | dispatch_verification.tex |
| eq:analysis-superadditive | mathematical_analysis.tex | service_capacities.tex |
| eq:analysis-tail-envelope | mathematical_analysis.tex | dispatch_verification.tex |
| eq:analysis-value | mathematical_analysis.tex | dispatch_verification.tex |
| eq:analysis-verification-bound | mathematical_analysis.tex | dispatch_verification.tex |
| eq:proof-growth-lower | proofs.tex | dispatch_verification.tex |
| eq:proof-growth-upper | proofs.tex | dispatch_verification.tex |
| eq:proof-zero-limit | proofs.tex | dispatch_verification.tex |
| mod:auto-before-delivery | optimization_model.tex | optimization_model.tex |
| mod:auto-busy | optimization_model.tex | optimization_model.tex |
| mod:auto-capacity | optimization_model.tex | optimization_model.tex |
| mod:auto-inventory | optimization_model.tex | optimization_model.tex |
| mod:demand | optimization_model.tex | optimization_model.tex |
| mod:direct-arcs | optimization_model.tex | service_capacities.tex |
| mod:direct-capacity | optimization_model.tex | service_capacities.tex |
| mod:direct-coefficients | optimization_model.tex | service_capacities.tex |
| mod:direct-duration | optimization_model.tex | service_capacities.tex |
| mod:direct-links | optimization_model.tex | optimization_model.tex |
| mod:direct-windows | optimization_model.tex | optimization_model.tex |
| mod:human-inventory | optimization_model.tex | optimization_model.tex |
| mod:local-arcs | optimization_model.tex | service_capacities.tex |
| mod:local-capacity | optimization_model.tex | service_capacities.tex |
| mod:local-function | optimization_model.tex | service_capacities.tex |
| mod:local-links | optimization_model.tex | optimization_model.tex |
| mod:local-windows | optimization_model.tex | optimization_model.tex |
| mod:movement-arcs | optimization_model.tex | service_capacities.tex |
| mod:objective | optimization_model.tex | optimization_model.tex |
| mod:optional-terminal | optimization_model.tex | optimization_model.tex |
| mod:pickup-before-auto | optimization_model.tex | optimization_model.tex |
| prop:analysis-capacity | mathematical_analysis.tex | service_capacities.tex |
| prop:analysis-loads | mathematical_analysis.tex | mathematical_analysis.tex |
| prop:analysis-poisson | mathematical_analysis.tex | dispatch_verification.tex |
| sec:analysis | mathematical_analysis.tex | mathematical_analysis.tex |
| sec:discussion | discussion.tex | discussion.tex |
| sec:introduction | introduction.tex | introduction.tex |
| sec:method | solution_method.tex | solution_method.tex |
| sec:model | optimization_model.tex | optimization_model.tex |
| sec:related | related_work.tex | related_work.tex |
| subsec:analysis-limits | mathematical_analysis.tex | dispatch_verification.tex |
| subsec:benchmark | mathematical_analysis.tex | mathematical_analysis.tex |
| subsec:capacity-analysis | mathematical_analysis.tex | service_capacities.tex |
| subsec:dynamic-wave | mathematical_analysis.tex | mathematical_analysis.tex |
| subsec:qvi | mathematical_analysis.tex | dispatch_verification.tex |
| thm:analysis-consistency | mathematical_analysis.tex | mathematical_analysis.tex |
| thm:analysis-qvi | mathematical_analysis.tex | dispatch_verification.tex |

## 新增标签

| Label | 文件 |
|---|---|
| app:dispatch | dispatch_verification.tex |
| app:implementation | implementation_details.tex |
| app:filter | implementation_details.tex |
| app:replay | implementation_details.tex |
| tab:model-scope | mathematical_analysis.tex |
| sec:capacity | service_capacities.tex |
| tab:notation | service_capacities.tex |
| fig:service-architecture | service_capacities.tex |
| eq:wave-geometry | service_capacities.tex |
| fig:rolling-timeline | solution_method.tex |

