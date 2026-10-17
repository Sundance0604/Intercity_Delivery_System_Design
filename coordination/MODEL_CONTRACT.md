# MODEL CONTRACT — 模型来源与对齐合同

2026-10-17（Asia/Shanghai）。负责人：Principal Investigator。

状态：**SOURCE_AUDITED / ALIGNMENT_NOT_ESTABLISHED / MIGRATION_NOT_STARTED**。本轮核实文件定义和核心代码结构，未运行测试、模型或实验，未完成所有命题证明。Referee接受三份静态审计；源模型—算法同步为MAJOR REVISION。未决定义不可作为实现默认值。

## 1. 来源与命名

| 对象 | 首要本地来源 | 范围 |
|---|---|---|
| V0-M | `intercity_delivery/models/base_optimizer.py`、`data/loader.py` | 实际强制换装实现；AV容量逐订单、z整数 |
| V0-F | `intercity_delivery/models/flexible_direct_optimizer.py` | 实际直送/换装共存实现；共享AV容量、连续z、累计人工迁移 |
| V0-RH | `intercity_delivery/algorithms/`、`experiments/solvers.py` | 上述模型加窗口、信息集、历史固定及候选弧 |
| 旧论文 | `research/paper/main_BHH.tex` | 论文比较依据；不替代实际V0代码 |
| V1-M/F | `research/paper/main_v2_BHH.tex` | 首要有限时域新模型；正文与求解章节尚未完全同步 |
| S-current | `research/paper/steady_state_model.tex` | deadline、finite sorting、慢车因素的稳态变体 |
| S-linear | `research/paper/steady_state_model_v1_linear.tex` | 较早线性时间成本稳态变体；版本用途待确认 |
| V2-local | `research/theory/Common Wave 的内生化与非均匀需求扩展.md` | 确定性常率/时变率的renewal代数与形式QVI |
| V2-target | 用户提出的随机抵达、内生wave、动态发车研究 | 尚未完整形式化；不可用V2-local自动替代 |

证据：[数学比较](../research/theory/model_comparison.md)、[代码差距](../experiments/reports/model_code_gap.md)、[文献定位](../research/literature/model_positioning.md)、[独立审查](reviews/model_alignment_review.md)。差异和实施门槛见 [MODEL_DIFF](MODEL_DIFF.md)、[MIGRATION_PLAN](MIGRATION_PLAN.md)。

## 2. 逐维模型合同

| 维度 | V0实际实现 | V1正文 | S-current / S-linear | V2实际与缺口 |
|---|---|---|---|---|
| 系统/目标 | 固定车队、车辆时间成本+拒单罚损；F可加处理费 | 同类目标，F加direct成本；无ω lead-time | 每shipment广义成本含ω平均时间；current另有Tmax≤L | waiting running cost+dispatch cost；随机期望目标未定义 |
| 变量 | 整数车辆、连续货流；M的z整数、F连续；无u | 整数x/y/w、binary u/ũ、连续g/ĝ/g̃/z | 连续W/n1/n2；按负荷配车 | q或(t,q)、reset；fleet/pipeline/age未闭合 |
| 需求/信息 | 离线生成/载入；paper RH按s_l≤t认定已知 | 活动订单确定性MILP，未给随机分布 | 恒定、平衡、对称、空间均匀 | local仅确定性Λ/λ(t)；概率空间/marks/filtration缺失 |
| 人工BHH | x×f^-1(Δ)，无stem和显式M截断 | K(v)=min{vM,f^-1(v[Δ−2ρ]^+)}，one-hot选v | 每车2ρ+(a+b/√W)n；current含慢车δ | 约化service cost需周期可分条件 |
| direct/库存 | F已有累计人工出入，旧两端单车合并容量 | origin zoned/destination unzoned，方向相关K̃；空返计费；终端人工复位 | 平衡返程、连续AV装载近似 | 无车辆状态不能代替全网络决策 |
| AV容量 | M逐订单限额；F同弧总货量 | 同弧总货量≤Mhat y | chatτ/min(W,Mhat)非一般整数装载精确式 | 未定义 |
| 车队/约束 | 固定N、初始位置、RH历史固定 | 固定初始N，F允许迁入；AV无终端复位要求 | 按负荷配车、忽略车辆排队；平均busy非峰值保证 | 跨周期fleet/sorting耦合可能破坏一维reset |
| 时域/终点 | 离散有限期，full/reduced弧域不同 | τ为正整数期数；全域弧在T完成；RH端点/义务待确认 | τ为物理时间、长期连续近似，省略首末期 | average-cost/有限期形式；terminal/growth未给 |
| 期限/服务 | 单订单窗，可拒绝 | pickup≥s、delivery≤e；订单类连续拆分 | current有硬L；linear只有线性时间成本、无硬deadline | 随机服务标准和年龄状态未定义 |
| hub/同步 | F固定θ/fee可配置 | 累计货流时序；无sorting μ、慢车δ、共同timetable | current μ=κΛ、等待慢车；有限sorting下common wave仍为政策假设 | stage synchronization开放 |
| 数据/单位 | CFS quantity来自重量换算 | d/M/Mhat需按stops明确 | 一shipment一pickup+delivery，连续近似 | arrival marks与stops/volume关系待定义 |
| 指标/证据 | 历史结果和已有tests本轮未审定运行 | 静态映射；未实现/数值验证 | 稿内证明、illustrative数值，未完整复核/复现 | 限定周期代数可核；形式QVI非verification theorem |

Δ=(j−i)t0，f_k(q)=a_kq+b_k√q，V1 a_k>0、b_k≥0。其逆为 `((sqrt(b_k²+4a_kθ)−b_k)/(2a_k))²`。local联结为 `x=Σv u_v`、`Σu_v≤1`、`Σg≤ΣK(v)u_v`；零车用全部u=0表示，正文Z+零值约定需确认。

direct：`f̃_d(q,v)=2ρ_origin+f_origin(q)/v+τt0+f_dest(q/v)`，在[0,vM]中二分得到满足f̃≤Δ的最大q。τ期零容量空返为正文另给特例，不可因正stem删除，业务解释待确认。依据main_v2:71–82、106–110、141、184–188。

代码定位：base_optimizer:68–88、164–193；flexible_direct_optimizer:175–226、228–375；loader:85–89、166–180；paper_rolling_horizon:162–166、233–275；cfs_processor:528–534。数学定位：main_v2:69–216；S-current:57–145、355–364。完整逐式证据见专员报告。

## 3. 不可跨越的推理边界

1. 旧论文≠全部V0，V0-M≠V0-F，窗口MILP≠全域在线控制。最后窗口bound/gap不提供全日证书；partial objective不与complete成本直接排名。
2. W=q、n=q/v使local tour形式匹配，不推出V1≡S。S-current:360明确program缺ω/sorting/timetable。目标归一化、车队峰值、同步、期限与首末期均须对齐。
3. K(v)/v非减及定义域内超可加是Referee接受的模型代数；K被线性cap截断后不必凸。零tour budget、b=0、饱和不支持严格超线性。stem和M使V1/V0容量没有统一大小关系；不是有限样本routing精度定理。
4. S的common-wave比较限于nested、无sorting约束、同δ等条件；finite sorting同步最优未证。S-current:89的n≤W未完整进入P/loads，相关全域结论待重审。deadline active、正乘子、strict slack不可混用。
5. Poisson平均等待不提供样本路径有限最长等待界；BHH渐近或拟合δ不能自动保证真实逐单硬期限。
6. V2 complete-cycle恒等式J(q)=ΛG°(q)仅限确定性常率、common-wave、cost-bound、renewal-separable。任意日历平均需余周期控制；随机到达、stage synchronization和全网络最优未证。
7. 固定步长RH/随机输入不等于QVI。单位Poisson生成元含λ[V(q+1)−V(q)]，不能以确定性λV_q替代。形式QVI未给verification/existence/threshold theorem。

## 4. 未决项与依赖

MIGRATION_PLAN的Q1–Q12为作者确认清单。wave分组、迁入车上界、空返/终端、单位、start/ext是精确V1实施门槛；S/V2另立研究任务，不在迁移中猜定。

四份TeX无input/include，宏和定理环境在本文件。main两份依赖的`导出的条目.bib`在可读树未找到，公开BHH来源已核实不等于本地条目核实。`steady_state_check.py`、`steady_state_symbolic.py`、`bhh_tour_check.py`未定位；稿内数值/拟合是作者报告，未复现。`.pytest_cache`不可访问限制全库扫描的绝对断言。未编译TeX。

## 5. 输入版本

Git HEAD：`3b3c34b37df4f50d34ae6160bcd1303a696afd3b`。已有大量untracked论文/实验，HEAD不能唯一标识本轮来源。SHA-256如下：

| 来源 | SHA-256 |
|---|---|
| main_BHH.tex | B62D543037ED581E1F53E1FAEEB1E102B8106FA2119AAB4C722D7F47EDA3A267 |
| main_v2_BHH.tex | 500BE1ADA854317BE79C358E0FF1269786DD4C7EA8E1102B501354EFF04A3457 |
| steady_state_model.tex | 980CFAEF58AC3DD25712205A673599A74E52BE53BFE8D4DCCAC1B78065AC7083 |
| steady_state_model_v1_linear.tex | 05F09EEBD5D3ADE1BB8B7022E460878A37C2A3C4566138852C62CC94F737DF80 |
| Common Wave 的内生化与非均匀需求扩展.md | A508B86D981F89CC57EAEDD58FE9071D9D884025C0370A36D99504EE39226CCA |

未来实验另记model_id、source hashes、数据语义、信息集、完成状态与bound_scope；本合同不授予实施或求解预算。
