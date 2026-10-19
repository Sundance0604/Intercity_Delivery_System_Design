# V1 数值实验模型冻结稿

2026-10-18（Asia/Shanghai）。负责人：Principal Investigator；文档整理：Literature。范围：P0模型冻结；后续小规模真实求解另按 [V1_P1_BOUNDED_PROTOCOL.md](../experiments/specs/V1_P1_BOUNDED_PROTOCOL.md) 的阶段和预算执行。

**状态：FROZEN_AUTHOR_CONFIRMED_SYNTHETIC_PROFILE。** 作者已确认需求单位、direct聚合终点和T2最低平均载量三项选择。冻结对象是该合成数值profile，不是实测路线校准。更新后的源字节须由PI重新生成manifest并只读通过预检，之后才进入有预算的小规模真实求解；本稿不预报运行结果。

## 1. 来源、版本与优先级

数值实验以 `research/paper/final_manuscript/main.tex`、其 `sections/*.tex` 和 `references.bib` 为当前稿源。附件为 `D:/download/Intercity_Operation_BHH (3).pdf`，21页；运行手册为 `D:/download/BHH_CFS_V1_Numerical_Experiment_Runbook.md`。手册提及的 `(3)(1).pdf` 未作为本轮实得来源。不能仅凭文件名，把该缺失版本与实得 PDF 判为同一版本。

源文件清单、SHA-256、Git状态及附件身份由 [v1_source_manifest.json](../experiments/specs/v1_source_manifest.json) 固定。已归档PDF为 [Intercity_Operation_BHH_3.pdf](../experiments/specs/source_snapshots/Intercity_Operation_BHH_3.pdf)，SHA-256 `a7f49a07adc6ac499a91143f218cdc665c0e81dd68f6fd618c26e7cca97e55a0`。公式语义对照见 [equation_to_code_matrix.md](../experiments/specs/equation_to_code_matrix.md)。PDF和TeX的核心公式对齐检查，不等于逐字版本相等、相同编译输出或PDF由该工作树生成的证明。本轮不改论文，不重新编译或补写文献。

既有已确认定义继续有效：`V1M-D1-D6-A-2026-10-18`；见 [V1_M_IMPLEMENTATION_PROFILE.md](V1_M_IMPLEMENTATION_PROFILE.md)、[V1_F_RUNTIME_CONTRACT.md](V1_F_RUNTIME_CONTRACT.md) 和 [MODEL_CONTRACT.md](MODEL_CONTRACT.md)。历史合同按日期记录进展，本稿不把其中早期“未实现”的叙述当作当前代码状态。

## 2. 作者决定

| 决定 | 作者确认方案 | 对实验的影响 |
|---|---|---|
| P0-D1 需求单位：AUTHOR_CONFIRMED | 作者确认“一票一合成任务等效量”；decision_id=`V1P0-JOB-UNIT-2026-10-18`。每条抽样 CFS shipment 的 `D_l=1`、unit=`synthetic_job_equivalent`；优化器内仍为可分连续量 | 不把源 pounds 当作 stops，不把该量声称为真实停靠数；所有容量、a/b、罚金按同一合成单位解释 |
| P0-D2 Direct终点：AUTHOR_CONFIRMED | decision_id=`V1P0-DIRECT-ENDPOINT-2026-10-18`。保留原聚合服务公式，车辆在任务完成节点j计入目的城市抽象可用节点，并可从j衔接后续任务 | 不增加隐含目的端回hub路段，不声称真实hub/服务区连接或实际接驳行程已校准；城市可用性采用冻结的抽象语义 |
| P0-D3 T2机制：AUTHOR_CONFIRMED | decision_id=`V1P0-T2-MINLOAD-2026-10-18`。`e=0` 且每条direct wave满足 `sum_l h_ijl >= m w_ij`；主值m=0.1，敏感性0.05/0.1/0.2 | 新增最低平均装载政策，单位为synthetic_job_equivalent/车；改变可行域，不能当作PDF原式已含的约束 |

P0-D3按每车平均货量联结，防止零货direct wave；它不提供逐车装载分配或每辆车严格载货的证明。上述值是作者采用的合成实验政策，非文献或实证校准。以后改变选择须新登记decision和源快照，不能只改结果标签。

## 3. 冻结可行域与变量语义

两城市、双方向、离散绝对节点 `0,...,T`。`T,tau` 为正整数，所有时间窗满足 `0<=s_l<=e_l<=T`；`Delta t,M,M_A,a_k` 正且finite，`b_k,rho_k` 非负且finite，初始车数非负整数。需求和罚金非负且finite；订单ID唯一。零需求、零车队允许，核函数的零车/零时域扩展不自动扩大完整实例输入域。

local wave按 `(i,j,city,direction)` 分组。相同开始但不同结束的任务属于不同wave，不跨wave额外pooling。local与direct selector均遍历 `v=1,...,N_1+N_2`，全不选表示0车；不能用城市初始车数截断迁入后的可用规模。

| 项目 | 冻结定义 |
|---|---|
| Local母弧 | `1<=j-i<=ceil((2rho_k+f_k(M))/Delta t)`，且 `0<=i<j<=T` |
| AV母弧 | `j-i=tau` |
| Empty母弧 | `j-i=tau_E`；默认 `tau_E=tau` |
| Direct母弧 | `tau<=j-i<=tau+ceil((2maxrho+f_1(M)+f_2(M))/Delta t)` |
| Local容量 | `K(v)=min(vM,F_k(v[(j-i)Delta t-2rho_k]^+))` |
| Direct容量 | `Psi_d(q,v)=2rho_o+f_o(q)/v+tau Delta t+f_r(q/v)`；以该不等式求最大q并截断vM；`j-i=tau` 的正货容量为0 |
| AV容量 | 同弧跨订单货量和 `<=M_A*y`，不能逐单各用一次容量 |
| 车辆变量 | `x,y,w,e` 非负整数；`u,u_direct` binary；货量 `g,g_auto,h` 和拒单 `z` 非负连续量，`z<=D_l` |
| 货流 | release/deadline、pickup完成先于AV发车、AV到达先于distribution开始、两端需求平衡，完整结果三阶段总货量相等 |
| 库存 | 人工车同时计local半开占用与direct/empty累计迁移；AV按出发/到达累计库存，各城各节点库存非负 |

所有任务都在真实T前完成；母弧包含零载货车辆任务。自动车可无货跨城。完整T3原式也允许零货direct，独立empty不是唯一人工迁移渠道。可行性不靠 `direct_ratio=0` 关闭车辆移动。

## 4. 单位、成本与数据解释

| 量 | 单位/转换 |
|---|---|
| `D,q,M,M_A,g,h,z` | `synthetic_job_equivalent`；每条抽样CFS记录提供1单位合成任务，允许连续分量 |
| `Delta t,rho,f(q)` | 分钟 |
| `tau,tau_E,s,e,T,H,h,E` | 期数，乘Delta t转分钟 |
| `a` | 分钟/服务等效量 |
| `b` | 分钟/服务等效量的平方根 |
| `c_*_hour` | 货币/车小时；成本只除60一次 |
| `p_l` | 货币/未服务等效量 |

目标为local、AV、direct、独立empty完整任务车时费加拒单罚金；任务内闲余时间也按所赋完整时长收费。默认empty小时费率沿用direct，但允许显式覆写并记录。有限期目标不含等待时间、平均lead time、sorting queue或稳态目标中的omega项。

本轮核心默认 `transfer_time_periods=0`、`transfer_cost_per_unit=0`、`direct_ratio_min=0`、`direct_ratio_max=1`。改变上述值必须标扩展情景。CFS原SHIPMT_ID、OD、重量、抽样权重等来源字段保留；release/deadline为沿用生成程序的合成窗，不能标成CFS观测。路线里程到tau的转换是模型构造，不能标成观测运营行程时间。输入语义和observed/synthetic分类由 [experiment_data_contract.json](../experiments/specs/experiment_data_contract.json) 记录。源稿的stop-equivalent词汇在本次数值构造中采用合成job服务等效量解释；论文原文未因本稿自动修改。冻结后，单位映射的变化必须重标实例版本并重新校准全部关联参数；不能直接继承V0的重量容量和罚金。

## 5. 信息、历史、交接与终点

O（offline）事前知道全日订单；RH只使用 `s_l<=t` 已揭示订单，并保留已过期订单以核算历史货物。所有时间为绝对时间。开始于t前的车辆、selector和货流决策完整固定，历史未启动键固定为0；不得截断在途任务或清空已完成迁移。

交接在同节点可即时进行；local占用 `i<=t<j`，完成节点可接下一任务。交付在destination tour或direct任务末尾记账，不以平均途中交付时间替代期限口径。按P0-D2，direct车辆在j到达目的城市抽象可用节点，可在j衔接local或跨城任务；该节点语义不认证真实hub细节。

RH使用 `b=min(T,t+H)`、`c=min(T,b+E)`、`a=min(T,t+h)`。新origin pickup/direct/empty开始 `<b`，新AV/destination distribution开始 `<c`，均在c内结束；历史先固定，不能被新窗截断。提交 `[t,a)` 启动的整段任务；z不是物理提交事件。真正终点默认城市分配自由，但仍要求任务在T内完成。计算窗不强制回城；full可额外指定守恒的真实T人工库存，RH对该非空目标当前拒绝，因为尾部可达性尚未实现。

## 6. 四机制对照

| 机制 | 约束/允许 | 科学解释 |
|---|---|---|
| T0 | `w=e=h=u_direct=0` | 换装货运、人工车城市专属；AV仍可空行 |
| T1 | `w=h=u_direct=0`，允许e | 换装货运、允许人工独立空调拨 |
| T2 | `e=0`，`sum h>=m*w`、主值m=0.1 | 加最低平均载量的direct政策；与原T3具有额外政策差别 |
| T3 | 原式，e允许、w允许零货；主基准m=0 | 完整柔性聚合模型 |

同实例跨机制固定需求、成本、车队、时间、终点及其他参数；T2敏感性独立标记。T0/T1不能靠direct比例上界实现。T3不保证一定选用direct。m=0的T3与m>0的T2比较不能把全部差异解释为独立empty的单一效应；额外匹配 `T3,m=0.1` 与 `T2,m=0.1`，再比较 `T3,m=0.1` 与 `T3,m=0`，分别检查empty和minimum政策的效应。

## 7. BHH参数与P0验收边界

参数准备须显式覆盖：(a) `0<K(v)<vM` 的routing时间约束生效区间；(b) `K(v)=vM` 的载量饱和区间；(c) 车队资源紧张的可解释配置。必须保留a/b/rho/M/Delta t的完整单位和来源，区分合成参数与真实校准。容量表中的pooling差异只说明本模型的机制；没有运营求解或客户坐标验证时，不报告成本改善、真实routing准确率或现实车队的绑定频率。

参数设计产物为 [v1_p0_capacity_design.json](../experiments/specs/v1_p0_capacity_design.json) 与 [v1_p0_parameter_design.md](../research/theory/v1_p0_parameter_design.md)，对应纯测试 `tests/test_v1_p0_capacity_design.py`；参数已经协调者登记为synthetic设计，具体测试证据以设计报告和PI验收为准。本稿不宣称已完成45次配对先导。作者现授权冻结预检后做小规模真实求解：仅10/20/40三算法配对、10单机制和载量敏感性，按独立协议预算执行；100单及以上不在授权批次。

## 8. 核验、比较与证书

独立replay用原f及direct时长式核容量，不用优化器K表自证；另核变量域、selector、库存、共享AV容量、窗、precedence、最终平衡和全成本。完整结果才能给全期总成本；partial只核已启动承诺和账单，延期最终平衡/比例/未来可达性。

O若证明最优，是给定有限期模型和全知输入的最优成本；限时incumbent不能写成最优值。O与RH的信息不同。RH末窗bound/gap不构成全日证书；不把partial成本与完整成本排序。dense/pruned对照固定同一车辆及selector母域、物理模型、信息、历史、窗口和预算，只改变新货流候选；现有time-support过滤无通用无损或加速定理。

## 9. 非目标与冻结完成条件

本稿不改变旧V0、原论文或历史结果，不开展长求解、依赖更新、道路级BHH校准、随机QVI、稳态实验、warm-start或论文数值写作。既有保存的CFS smoke仅为流程证据。

第2节三个明确选择及decision_id现已齐备；逐式矩阵、单位合同、合成参数和机制回归/独立审查沿P0产物保存。PI须对本次决定更新后的源文件重新生成manifest并只读复核，运行冻结预检后再执行下一阶段。作者profile冻结不自动等于预检已经运行通过，也不保证下一批求解完成；这些结果分别保存实际证据。
