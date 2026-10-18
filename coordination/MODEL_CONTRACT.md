# MODEL CONTRACT — 模型来源与对齐合同

2026-10-17（Asia/Shanghai）。负责人：Principal Investigator。

最新状态（2026-10-18）：**STRUCTURE_REVIEWED / AUTHOR_COMPILATION_PENDING**；新稿已按容量→MILP/RH→独立稳态重排，见第12节。Direct物理路线待作者确认；数值章节和计划延期；既有T0与V1结论范围不变，不宣称全文已完成投稿验收。

## 1. 来源与命名

| 对象 | 首要本地来源 | 范围 |
|---|---|---|
| V0-M | `intercity_delivery/models/base_optimizer.py`、`data/loader.py` | 实际强制换装实现；AV容量逐订单、z整数 |
| V0-F | `intercity_delivery/models/flexible_direct_optimizer.py` | 实际直送/换装共存实现；共享AV容量、连续z、累计人工迁移 |
| V0-RH | `intercity_delivery/algorithms/`、`experiments/solvers.py` | 上述模型加窗口、信息集、历史固定及候选弧 |
| 旧论文 | `research/paper/main_BHH.tex` | 论文比较依据；不替代实际V0代码 |
| V1-M/F | `research/paper/main_v2_BHH.tex` | 首要有限时域新模型；正文与求解章节尚未完全同步 |
| V1-M已采用基准 | `intercity_delivery/v1/mandatory_optimizer.py`、`V1_M_IMPLEMENTATION_PROFILE.md` | 作者确认D1–D6-A的offline/arc-wave强制换装；不包含F/RH/S/QVI |
| V1-F已实现扩展 | `intercity_delivery/v1/flexible_optimizer.py`、`V1_F_RUNTIME_CONTRACT.md` | 沿用已采用wave定义，含方向相关direct、独立empty和迁入selector修正；RH为release揭示，非S/QVI |
| V0-F-EMPTY | `intercity_delivery/repositioning/`及`coordination/VEHICLE_REPOSITIONING_SPEC.md` | V0-F货流/容量加独立空车、绝对库存和新滚动控制；不是完整V1-F |
| S-current | `research/paper/steady_state_model.tex` | deadline、finite sorting、慢车因素的稳态变体 |
| S-linear | `research/paper/steady_state_model_v1_linear.tex` | 较早线性时间成本稳态变体；用作T0共同基准的简化对照 |
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

后续G0-M已完成[最小决策规格包](V1_M_DECISION_SPEC.md)及[独立审查](reviews/v1_m_decision_spec_review.md)：仅针对拟议的全时域offline强制换装基准，将必要决定收敛为D1–D6。规格准备ACCEPT，作者选择与实施许可仍PENDING；本合同没有因此采用推荐方案。V1-F的迁入/空返、RH窗口/信息、S/V2问题继续保留，可在该限定M基准之外延期，不一概阻断M规格准备。

最新作者评审已确认两项，现不再笼统记为OPEN：**VR-DEC-01计算rolling窗口不强制期末城市复位；VR-DEC-02空调拨与载货服务独立、无订单也可生成。** 见[修正规格](VEHICLE_REPOSITIONING_SPEC.md)。原论文完整horizon的Σw+=Σw−不能复制到每窗；若真实T_op要求目标，须计w+e和真实完成状态，并维护前序决策后的终点可达性。真实关停模式/目标、空行时间/费用及跨cut新任务表示尚未决定；不能推定全日一定复位或永不复位。当前代码本无该terminal等式，缺口主要是独立empty网络/变量及全链路状态接入。

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

## 6. 2026-10-18批准后的有限实施合同

用户“好，请执行”与“继续，有额度了”授权车辆扩展。实现隔离在`intercity_delivery/repositioning/`；model_id=`v0_f_explicit_empty_v1`的v1表示扩展版本号，不表示论文V1已实现。入口为`flexible_direct_empty_mip`和`flexible_direct_empty_rolling`；保留V0默认入口、原容量及原模型/算法。

默认empty时长沿用direct、小时费率沿用cost_direct，terminal_target=None；metadata披露回退来源，API可覆写。完整模式支持显式真实T目标，rolling暂拒绝目标，等待全程可达性保障。计算窗无城市复位约束；新任务须在completion_end完成，已有绝对事件不截断或重置。原文V1语义、容量、单位和V2问题不因本扩展解决。

初轮34项有限检查通过，见[实施报告](../experiments/reports/vehicle_repositioning_implementation.md)。车辆账本只验证库存、标量与核算；后续货物检查和真实换装用例已由第7节单独补齐，不将车辆账本本身冒称全模型核验器。

## 7. 连续两步交付：货物流核验与V1纯内核

第一步已增加`repositioning/cargo_validation.py`并接入runner：逐单hub库存、处理延迟、货流先后、共享V0-F容量、释放/期限、需求预约与完整链平衡；full额外核对r/q/z，partial明确延期最终平衡/比例，不能推断未来可达。NaN/溢出假阳性在独立审查中发现并修复。新增31项有限测试，含真实三控制窗pickup→AV→destination微例，手算成本7.2。见[报告](../experiments/reports/repositioning_cargo_validation.md)及[审查](reviews/cargo_validation_review.md)。

第二步已实现`intercity_delivery/v1/`的f/h/K、local/AV母弧和半开占用，依据main_v2:71/74/82/106/110；22项纯测试通过。显式统一物理时间输入；v=0/T=0为工具扩展，正M等为工具输入域，不冻结完整模型D1–D6。严格computed-float ceil，有限中间结果溢出报错，下溢遵循float。见[数学合同](../research/theory/v1_kernel_contract.md)、[实现报告](../experiments/reports/v1_kernel_implementation.md)及[审查](reviews/v1_kernel_review.md)。

目前进入V1基础工具迁移，完整V1-M/F优化器未实现；无one-hot决策接入、源模型与全算法等价或QVI声明。两步测试与旧扩展合计87个不同用例分批通过，不是一次全套大实验。新V1工具未替换旧容量；论文、历史数据及原优化器保持不变。

## 8. 作者确认后的完整V1-M基准

作者对异步定义问题明确回复“采用这组定义”，确认D1–D6-A，决定编号`V1M-D1-D6-A-2026-10-18`。采用offline已知订单、逐(i,j,city,方向)wave、连续stop量和零车合法、分钟/车小时费率、tour-end/同刻交接、0≤s≤e≤T整数窗。输入正性/finite等按[基准合同](V1_M_IMPLEMENTATION_PROFILE.md)冻结；源71与逐arc公式的文字张力仍在TeX，不假称已改稿。

完整M01–M15已落入独立configuration/mandatory_optimizer/validation/runner。模型用K表和one-hot；validator不用K表，直接核原f不等式、两城占用/AV库存、逐单货流/窗口/需求及三项成本。结果model_id=`v1_mandatory_arc_wave_v1`，semantic_status=`author_confirmed_profile`，附decision_id/source/input SHA、完整六组计划和全域证书范围；无incumbent不给成本或通过标记。

初版8微型run及配置/validator/kernel共74项PASS；后续最终70纯测试PASS，含4个runner异常边界；确认后两次定向真实复验成本12/5且完整payload保存。共78个不同用例分批通过，其中8个不同微型求解用例累计实际跑10次，20秒上限内。初版日志写入失败/进程exit1与未捕获raw诚实保留，详见[实施记录](../experiments/reports/v1_mandatory_implementation.md)、[最终raw结果](../experiments/reports/v1_mandatory_confirmed_micro_results.json)和[独立审查](reviews/v1_mandatory_review.md)。这不是V1-F/RH等价证明、真实数据实验或QVI成果。

## 9. V1-F、RH、剪枝与CFS小样本已交付

本轮用户要求V1-F、rolling horizon及剪枝一次实现，仅验证CFS可运行。model_id=`v1_flexible_arc_wave_empty_v1`，具体源式、API和默认政策见[运行合同](V1_F_RUNTIME_CONTRACT.md)。继承D1–D6-A，full离线、RH按release揭示。direct用main_v2:141的方向相关闭式逆；local selector显式修正为全车队上界以允许迁入服务，动态库存限制城市可用量。独立empty无订单可生成；源τ长度零货direct特例保留，不能将w都解释为逐车严格载货。默认真实终点无目标，full可指定、RH暂拒绝hard目标；计算窗无回归。

三入口v1_flexible_mip、v1_flexible_rolling、v1_flexible_pruned_rolling已注册，旧默认兼容。剪枝实际稀疏创建cargo，保留vehicle/selector母弧和历史pickup/AV后继；独立核验用原f而非K表自证。partial无完整总成本，RH无全日bound/gap。core按solver隔离CFS数量和来源τ。

[保存结果](../experiments/reports/v1_f_cfs_smoke_results.json)：同8条shipment-stop代理，seed4101，06-348↔06-488、T24、τ9、H8/step2/ext6。full OPTIMAL成本1100服务8，两RH均12窗COMPLETED成本2025服务7，全部独立核验通过。货流变量跨窗累计8265→397，删除7868。只认证小样本运行，不证明一般无损剪枝、加速、源文逐字等价或QVI；CFS代理尚无真实stop校准。

最终73纯PASS；7不同微例累计8run分批通过（含一次同价退化断言修正），3次CFS总约0.953秒。[实施记录](../experiments/reports/v1_f_runtime_smoke.md)与[Referee限定ACCEPT](reviews/v1_f_runtime_review.md)含完整边界。原代码/四TeX共21来源散列一致，无大实验、历史改写或Git提交推送。V1 warm start/local branching、真实终点RH尾计划、S/QVI另列。

## 10. QVI–Wave共同子模型的一致性研究

用户要求比较两个steady并规划QVI如何融入论文，不编译TeX。版本内部记录确认current为较新机制主稿，linear为简化对照；[总写作计划](PAPER_WRITING_PLAN.md)及[逐章审计](../research/literature/paper_integration_plan.md)给出精确行号和后续改稿顺序。

共同基准T0明确为恒定确定性到达、对称common-wave、κ=∞/δ=0/无绑定deadline、连续负荷、按负荷配车及逐批成本可分；保留0<n_i≤min(M,q)，要求ω>0。定义R=G°−ωq/(2Λ)、C=qR后，J=ΛG°，静态argmin与最优重复发车阈值集合相同。[自包含证明](../research/theory/qvi_wave_equivalence.md)补齐任意日历liminf余项、tail-inf全域QVI和横截条件。[独立审查](reviews/qvi_wave_paper_review.md)限定ACCEPT，不覆盖全部current/有限车队/V1-F/RH；真实pipeline没有因backlog reset而消失。

单位Poisson在同一整数全量reset阈值域、同一C及线性holding下目标率比fluid少ω/2，故同argmin；连续与整数域不同，一般adapted随机控制、marks/年龄/期限/车队等仍需独立verification。原QVI-001/BHH-001宽泛研究目标不升级。

[纯toy](../research/toy_models/qvi_wave_consistency_check.py)以原式稳态数值优化对独立Bellman，三类边界q约1.973515/8/61.304753，对应.05网格1.95/8/61.3；[保存raw](../research/toy_models/qvi_wave_consistency_results.json)。PI最终复跑exit0约2.132秒，无Gurobi/CFS/大实验。原TeX/生产代码未改，未编译；current现稿可投稿状态仍MAJOR REVISION。QVI作为短桥梁主文+附录，不能把既有CFS smoke当其验证。

## 11. 新建论文终稿源项目（不含数值实验）

用户明确要求创建新TeX作为终稿、完成引言/综述、在原优化模型上加入数学推理，并交子Agent审查。新入口为[final_manuscript/main.tex](../research/paper/final_manuscript/main.tex)，同目录sections和references.bib组成独立项目，旧四TeX不覆盖。结构含引言、相关文献、完整有限期模型、受限数学分析、实际RH/cargo pruning、讨论结论及证明附录。

有限期部分采用当前V1-F的迁入selector、独立empty和默认free terminal；核心为零transfer delay/fee、无direct-share限制，扩展单独说明。理论完整保留T0的n≤min(M,q)、J=ΛG°、日历余项、tail-inf QVI与受限Poisson结论。途中平均交付/固定车队/不同目标等不等价边界明示，不将完整MILP称为QVI离散化，也不照搬旧优先初解或无损剪枝声明。

引言与综述重新成稿，独立Bib含12条实际引用且有DOI；Referee发现的直接shipment-consolidation/renewal前史已补入Çetinkaya–Bookbinder2003，未以renewal比值本身声称创新。详见[终稿项目说明](../research/paper/final_manuscript/README.md)与[终稿独立审查](reviews/final_manuscript_review.md)。

本轮不写数值实验章节或基于新稿的实验计划；没有运行求解、编译或生成PDF。终稿指当前指定的论文源入口，不宣称全部数值研究/排版/投稿验收已完成。原71份生产/旧tests/TeX/Bib来源SHA保持一致。

## 12. Banerjee参照下的结构精简（2026-10-18）

用户确认直接重排并精简当前新稿。当前入口仍为[final_manuscript/main.tex](../research/paper/final_manuscript/main.tex)，标题改为Wave-Based Intercity Freight Scheduling with Fleet Circulation；正文为容量→MILP/RH→独立稳态，完整QVI/Poisson与实现细节保留附录。已有数学假设和结论未升级，生产实现未改。18条文献含Banerjee2022。

新增BHH几何式仅解释现有均衡分区；b是输入系数，不声称完成校准。w=e=h=0明确为city-dedicated human-driven fleets的transfer-only变体。Direct路径/目的接驳/再派位置仍REQUIRES_AUTHOR_CONFIRMATION，旧式未补项。

[结构报告](../research/paper/final_manuscript/BHH_structure_revision_report.md)与[来源及标签映射](../research/paper/final_manuscript/BHH_source_and_claim_map.md)记录当前证据。Referee限定ACCEPT；原53显示块、5结论、2假设与5proof均保留，最终11TeX/78labels/68refs/18Bib静态通过。作者明确自行编译，PDF排版待作者验收；本轮无实验或新理论。
