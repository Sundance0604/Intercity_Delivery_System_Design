# CLAIMS — 证据与科学状态账本

更新：2026-10-18。最新PAPER-FINAL条目记录新终稿正文、12条实际文献与独立审查；数值章节和计划暂未写，未编译。QW的T0限定证明及既有V1实现证据保留；不升级完整随机理论、性能或创新性。旧阶段状态只指当时，当前主张见文末。

保留科学状态词：CONJECTURE / PROVED / REFUTED / NUMERIC_TOY / NUMERIC_FULL / LITERATURE / NEEDS_REVIEW。另用 **AUDITED_SOURCE** 表示文件/代码定义已核，不表示科学定理；**CONDITIONAL_ALGEBRA** 表示明确假设内的代数已独立核对，不是全网络最优定理；**AUTHOR_DECISION**仅记录作者确定的设计政策。Referee的ACCEPT针对所列范围；报告ACCEPT不等于源模型已正确或迁移获批。

共同来源：[数学报告](../research/theory/model_comparison.md)（T）、[代码报告](../experiments/reports/model_code_gap.md)（E）、[文献报告](../research/literature/model_positioning.md)（L）、[独立审查](reviews/model_alignment_review.md)（R）。完整命名/散列见[合同](MODEL_CONTRACT.md)。

| ID | 精确主张/范围 | 假设 | 状态 | Owner | 来源/证明证据 | 实验证据 | Referee |
|---|---|---|---|---|---|---|---|
| QVI-001 | 适当内生batching问题可建立QVI；不声称随机full-network问题已验证 | state/information/admissibility/boundary待完整定义 | CONJECTURE，保留原状态 | theorist | 本地QVI笔记§6/§9有形式方程；T§6、R14；无verification theorem | 无 | 形式方程可接受；完整理论声明REJECT |
| BHH-001 | BHH/common-wave结构能否扩至非均匀/随机到达 | 政策、随机过程、sorting/fleet条件未定 | CONJECTURE，未因笔记升级 | theorist | 笔记确定性λ(t)且threshold以“如果”限定；T§6、R14 | 无 | 扩展未证 |
| ALIGN-001 | 实际V0至少区分M/F/RH，旧论文不能替代代码 | 当前本地文件 | AUDITED_SOURCE | experiment | base:68–69、184–192；flex:228–269、352–357；旧main:103；E§3、R9 | 未运行 | ACCEPT静态差异 |
| ALIGN-002 | V1正文已换K(v)/K̃与selectors，现代码仍旧线性容量 | main_v2定义的模型域 | AUDITED_SOURCE | experiment/theorist | main_v2:82、106–110、141、184–188；base:164–178、flex:320–374；T§2/3、R2 | 无 | ACCEPT差异；“已实现”REJECT |
| BHH-002 | K(v)/v非减、枚举域内超可加；K不必凸、非无条件严格超线性 | h凸且h(0)=0，D固定，K=min(vM,h(vD))；v1+v2在域内 | CONDITIONAL_ALGEBRA | theorist | T§3.1、R§5.1独立代数；不是有限样本routing定理 | 无 | ACCEPT限定代数 |
| ALG-001 | 旧启发式不能普遍保证新V1容量可行 | ρ可正、仍用旧C=h(Δ) | REFUTED：被否定的是“旧算法始终V1可行” | theorist | f(q)=q、ρ=1、Δ=2、M≥2，旧C=2而K(v)=0；T§4.6、R2手工反例 | 未运行 | REJECT原可行性claim |
| ALG-002 | history-only availability=0并非一般安全删弧条件 | 新direct能在本窗口调入 | REFUTED：无条件安全claim | referee | R3初始一车被占用+另城新调入例，避免N=0 selector缺陷掩盖；代码state_dependent:47–65、164–167 | 未运行 | REJECT无条件claim；特殊固定城子模型可另证 |
| ALIGN-003 | V1-F终端正反direct车次平衡恢复两城HV总数 | 所有direct弧在T完成、累计库存式成立 | CONDITIONAL_ALGEBRA | theorist | main_v2:154/212/221；T§2.3、R§5.3；不要求货量平衡/逐车原路配对 | 无 | ACCEPT限定终端代数；RH政策待确认 |
| ALIGN-004 | CFS quantity是重量归一化，不能直接当V1 stops实证 | 当前数据转换 | AUDITED_SOURCE | experiment | cfs_processor:528–534；main_v2:69；E§6、R5 | 无 | ACCEPT差异；现有stop校准claim REJECT |
| ALIGN-005 | full/reduced弧域、论文端点/完成上界不一致 | 当前窗口规格 | AUDITED_SOURCE | experiment/referee | full_window:45–61、state_dependent:153–181；main_v2:234–263/429–443；R8 | 无 | MAJOR REVISION |
| S-001 | common-wave比较可在nested、无sorting约束、同δ下成立 | current κ=∞；one-by-one比较另需δ=0 | NEEDS_REVIEW：稿内条件命题，比较代数已核 | theorist | S-current:103–111；T§5、R10/14；有限sorting同步未证 | 无 | 条件边界接受，不认证完整有限sorting定理 |
| S-002 | S-current负载闭式与后续全域结论需处理n≤W | setting89 vs P132/loads138–139 | NEEDS_REVIEW | theorist | W=1/16、a=b=c=ρ=ω=1，n1=√(2/5)>W；T§5.1、R6 | 无 | MAJOR REVISION |
| S-003 | deadline active不等价于λ>0，阈值/重根/不可行边界待限定 | 内点ω>0，L可等于cost-bound Tmax | NEEDS_REVIEW | theorist/referee | S-current:145/205/234；R7临界等号反例 | 无 | MAJOR REVISION |
| ALIGN-006 | V1与两份S没有自动等价/现成验证关系 | 目标、fleet、μ/δ、timetable、整数/终点不同 | AUDITED_SOURCE | PI | S-current:360；T§5、E§5、R10；仅local形式可匹配 | 无stationary验证 | ACCEPT差异；现成等价claim REJECT |
| QVI-002 | complete-cycle J(q)=ΛG°(q)，重复最优batch可达周期平均最小值 | 确定性常率、common-wave、cost-bound、renewal-separable、存在minimizer | CONDITIONAL_ALGEBRA | theorist | 本地笔记§4/5，T§6、R14；任意日历平均另需余周期/时间趋无穷控制 | 无 | ACCEPT限定代数；不认证随机/全网络最优 |
| EXP-001 | partial objective与last-window gap不能直接当全日成本/界 | 预算可中止、窗口求解 | AUDITED_SOURCE | experiment | paper_rolling:148–151/233–275，solvers:395–406；R12；完成目标也需独立账本核算 | 历史结果影响未审定 | MAJOR REVISION结果解释 |
| LIT-001 | BHH及相关公开文献存在、主题可核；精确有限波等式/novelty不随之成立 | 出版社/原始机构元数据或摘要的有界核验 | LITERATURE | literature | L§4含作者年/URL/支持范围/2026-10-17；R13另抽查3项 | 稿中数值表/拟合未复现 | ACCEPT有界文献审计 |
| SOURCE-001 | Bib及稿称校准/符号脚本在可读树未找到 | 扫描受.pytest_cache权限限制 | AUDITED_SOURCE | literature | main_v2:7；S-current:19–25/358；L§1、R13 | 未编译/运行 | ACCEPT有界缺失报告 |
| G0M-SOURCE-001 | 新V1不能原样复用只导出y/z且过滤≤0.1的旧结果层来独立核算连续解 | V1连续g/ĝ/z；当前V0-M的z仍为integer | AUDITED_SOURCE | experiment | `intercity_delivery/experiments/solvers.py:114–118`；[规格M15](V1_M_DECISION_SPEC.md)、[新审查R6](reviews/v1_m_decision_spec_review.md)；是复用风险，不是旧V0已运行的小量故障 | 未运行 | ACCEPT静态设计要求 |
| G0M-ORACLE-001 | F1–F5的手工容量/共享AV/分数z/成本/小量导出预期成立 | 规格所列有效配置；单位/域按D1–D6候选解释 | CONDITIONAL_ALGEBRA | theorist后referee | [规格§6](V1_M_DECISION_SPEC.md)、[新审查R5](reviews/v1_m_decision_spec_review.md)；仅手算验收oracle，F2分配可非唯一 | 无测试/求解记录 | ACCEPT验收设计；不认证数值最优运行 |
| VR-DEC-01 | rolling计算窗口不得强制人工车恢复初始城市分布 | 指计算边界，非已定义的真实关停目标 | AUTHOR_DECISION | 作者/PI | 最新用户评审第1点；[车辆修正规格§1/5](VEHICLE_REPOSITIONING_SPEC.md) | 非实验主张 | 作者已确认；真实T_op/R另定 |
| VR-DEC-02 | empty repositioning独立于loaded服务，不要求真实订单 | 同一人工fleet，e无货流不创造货量 | AUTHOR_DECISION | 作者/PI | 最新用户评审第2点；[修正规格§3](VEHICLE_REPOSITIONING_SPEC.md) | 非实验主张 | 作者已确认；严格逐车载货语义另定 |
| VR-SOURCE-001 | 当前Python无Σw+=Σw−终端式，缺独立e；完工截断不同于车辆复位 | 当前V0-F/RH实现 | AUDITED_SOURCE | experiment | flex_optimizer:217–269，full/reduced候选；[修正规格§2/6](VEHICLE_REPOSITIONING_SPEC.md) | 未运行生产 | 不声称已删除不存在约束 |
| VR-ALG-001 | 绝对事件账中A=N0+到达(w+e)−出发(w+e)−local忙车；城市idle+local忙车+跨城在途守恒 | 历史/新决策互斥、无车创建销毁、正确事件时点；有限给定任务 | CONDITIONAL_ALGEBRA | theorist/PI | [修正规格§4](VEHICLE_REPOSITIONING_SPEC.md)，不证明完整routing/最优性 | 另有下行有限toy检查 | [新审查VR-R3](reviews/vehicle_repositioning_review.md)：ACCEPT限定账本 |
| VR-TOY-001 | 10个正反例通过车辆库存/调拨/跨窗/真实目标/双算检查 | 脚本固定有效输入、tau=2，无schema/完整cargo/cost验证 | NUMERIC_TOY（确定性账本检查，非优化） | theorist | [独立脚本](../research/toy_models/vehicle_repositioning_ledger_check.py)及修正规格§7；最终exit0、约0.09秒 | 实际运行10例PASS；无Gurobi | [新审查VR-R7](reviews/vehicle_repositioning_review.md)：ACCEPT有限原型，未重跑；不能升级NUMERIC_FULL |

| VR-IMPL-001 | 隔离V0-F扩展支持订单无关empty与计算窗不复位 | 默认政策/有限完成窗见合同§6，非完整V1 | IMPLEMENTED / MICRO_TESTED | experiment/PI | `intercity_delivery/repositioning/`；[实施报告](../experiments/reports/vehicle_repositioning_implementation.md) | 34项检查，6完整微型求解测试+1真实RH测试 | [实施审查](reviews/vehicle_repositioning_implementation_review.md)，不推广一般正确性 |
| VR-CARGO-001 | 给定V0-F已提交事件可独立重放货流、容量、期限和完整余额；真实三窗换装微例通过 | 声明输入域/数值容差；车辆可用性另由ledger验证；partial不保未来可达 | MICRO_TESTED / AUDITED_SOURCE | experiment/PI | [cargo实现报告](../experiments/reports/repositioning_cargo_validation.md) | 新31项PASS，真实链0.05/费用7.2；NaN反例修复后PASS | [cargo审查](reviews/cargo_validation_review.md)ACCEPT限定范围 |
| V1-KERNEL-001 | 原式f/h/K、local/AV母弧、半开占用已有隔离纯工具实现 | 统一显式物理单位、声明float域；无完整wave政策/订单/目标 | SOURCE_KERNEL / MICRO_TESTED | theorist/experiment/PI | [数学合同](../research/theory/v1_kernel_contract.md)、[内核报告](../experiments/reports/v1_kernel_implementation.md)，main_v2:71/74/82/106/110 | 22纯例实施者、PI、Referee分别PASS；无solver | [内核审查](reviews/v1_kernel_review.md)ACCEPT；完整V1、D1–D6及QVI不升级 |
| V1M-DEC-001 | D1–D6-A作为首版offline强制换装基准定义 | 不覆盖F/RH/S/QVI，原稿文字未修改 | AUTHOR_DECISION | 作者/PI | 作者对完整六项问题答复“采用这组定义”；[决定编号/合同](V1_M_IMPLEMENTATION_PROFILE.md) | 非实验主张 | 来源为人类决定，不由审查代选 |
| V1M-IMPL-001 | 已采用profile下M01–M15完整MILP、完整结果与独立原f核验已实现 | 显式D1–D6-A及声明数值域/容差；有限微例 | IMPLEMENTED / MICRO_TESTED | experiment/PI | [实施记录](../experiments/reports/v1_mandatory_implementation.md)，main_v2:67–137 | 初轮74PASS含8真实；最终70纯PASS，2次确认后复验12/5保存raw；不重复累计测试数 | [审查](reviews/v1_mandatory_review.md)ACCEPT限定基准，非一般等价或性能证明 |

不把稿内“proved/checked”注释当作独立证据。PROVED只用于完整假设和证明获相应独立认可的确切命题；本轮没有新增全域研究定理。有限测试不代表完整V1或一般MILP/RH正确性已验证。不得改变原QVI-001/BHH-001范围来制造已完成状态。

## V1-F / RH / 剪枝最新主张

旧表“未实现”保留当时V0差异语境；以下为新独立V1模块的当前证据。

| ID | 主张/假设 | 状态 | Owner | 来源与证明证据 | 实验证据/独立审查 |
|---|---|---|---|---|---|
| V1F-IMPL-001 | 方向direct、联合wave、独立empty与迁入库存已实现；global selector和终点政策是显式修正 | IMPLEMENTED / MICRO_TESTED | experiment/PI | main_v2:139–216、[运行合同](V1_F_RUNTIME_CONTRACT.md)、[实施记录](../experiments/reports/v1_f_runtime_smoke.md)；非源文逐字等价 | 4模型微例PASS、CFS full服务8/成本1100；[限定ACCEPT](reviews/v1_f_runtime_review.md) |
| V1F-RH-001 | release揭示、绝对历史和跨窗货流；无计算窗复位；无hard真实终点目标 | IMPLEMENTED / MICRO_TESTED | experiment/PI | flexible_rolling/configure_window及原f独立核验；partial不保证未来可达 | 3RH微例分批PASS、CFS12窗完成服务7/2025；ACCEPT有限执行，非全局gap |
| V1F-PRUNE-001 | 实际稀疏cargo，保留vehicle母弧和历史后继；无一般等价证明 | IMPLEMENTED / CFS_SMOKE | experiment/PI | pruning.py、global_safety_proof=False；必要时间支持筛选 | 跨12窗8265→397，删7868；本例成本服务一致；ACCEPT实现，不认证一般无损/加速 |
| V1F-CFS-001 | 三方式同CFS来源小样本完成并独立核验；一shipment一stop代理、旧合成窗、参数未校准 | CFS_SMOKE，非正式全规模NUMERIC_FULL | experiment/PI | [完整输入/非零方案/散列](../experiments/reports/v1_f_cfs_smoke_results.json) | full8/1100、两RH7/2025；Referee三方案纯重放PASS |
| V1F-ENTRY-001 | 三新入口已注册；CFS数量/τ和汇总按solver隔离，旧默认不变 | IMPLEMENTED / PURE_TESTED | experiment/PI | adapter/core/solvers/main；普通CLI持久化写盘本轮未实跑 | 最终73纯用例含11数据/入口及4旧入口回归；ACCEPT此范围 |

V2/QVI与稳态主张不升级；full离线与RH在线信息不同，不能从成本差推算法最优性差距。

## QVI–Wave与论文规划：本轮限定结果

原QVI-001/BHH-001的完整随机/非均匀/全网络研究保持CONJECTURE，以下为严格更小对象。

| ID | 精确主张 | 假设/边界 | 状态 | Owner | 证明/来源 | 数值证据与审查 |
|---|---|---|---|---|---|---|
| QW-DET-001 | J(q)=ΛG°(q)，最优静态batch集合=最优重复阈值集合；任意successive wave不降低全日历平均成本 | T0：恒定确定性、common-wave、完整n≤min(M,q)、连续运力、ω>0、全量reset、逐批可分会计；不含有限fleet/pipeline耦合 | PROVED（限明确T0模型；物理应用CONDITIONAL） | theorist | [证明§2–5](../research/theory/qvi_wave_equivalence.md)：余周期下界、tail包络、QVI不等式与横截 | [Referee限定ACCEPT](reviews/qvi_wave_paper_review.md)，不认证一般PDE唯一性 |
| QW-POISSON-001 | 单位Poisson整数阈值JP=JD−ω/2，所以相同整数域argmin相同 | 同C、线性holding、全量reset阈值类；一般adapted随机verification未完成 | PROVED（限阈值类）/其他OPEN | theorist | 证明§6；continuous3/2与integer2反例是域差，不推翻同域结果 | 整数toy均20，rate差.5；限定ACCEPT |
| QW-NUM-001 | 原式稳态优化与独立Bellman在三toy一致至网格精度 | T0、.2/.1/.05网格、上界160/320；非真实同日达校准 | NUMERIC_TOY | theorist checks/PI | [方法记录](../research/theory/qvi_wave_numerical_check.md)、[脚本](../research/toy_models/qvi_wave_consistency_check.py) | [raw](../research/toy_models/qvi_wave_consistency_results.json)：1.973515→1.95、8→8、61.304753→61.3；审查ACCEPT，非连续收敛定理 |
| PAPER-PLAN-001 | current概念演进较新，linear适合T0基线；建议一篇整合，QVI短节+附录 | 稿内27–41版本说明，非mtime；原稿仍有域/边界/算法错位 | AUDITED_SOURCE / WRITING_PLAN | literature/PI | [专员计划](../research/literature/paper_integration_plan.md)、[主计划](PAPER_WRITING_PLAN.md) | 5条primary文献限定核验；规划ACCEPT，稿件本身MAJOR REVISION、创新性OPEN |

不得将这些结果表述为完整steady、V1-F、RH、CFS和QVI已等价；现代码目标、交付、固定fleet、有限期边界均未与T0统一。

## 新终稿源项目

| ID | 精确主张 | 范围/状态 | Owner | 证据与审查 |
|---|---|---|---|---|
| PAPER-FINAL-TEXT-001 | 新英文终稿含完成的引言、综述、优化模型、数学分析、实际方法、讨论/结论及完整证明 | WRITTEN / STATIC_REVIEWED；数值章节和计划有意延期，不是完整投稿完成声明 | literature/theorist/experiment/PI | [main.tex](../research/paper/final_manuscript/main.tex)、[项目说明](../research/paper/final_manuscript/README.md)、[独立审查](reviews/final_manuscript_review.md) |
| PAPER-FINAL-LIT-001 | 文献库含12条实际引文/DOI，直接集并发车前史已纳入 | BIBLIOGRAPHY_CHECKED；元数据/公开摘要有界核验，非穷尽novelty认证 | literature/referee | references.bib；新增Çetinkaya–Bookbinder2003回应独立审查，不把renewal ratio当新意 |
| PAPER-FINAL-MATH-001 | 已接受T0证明被写入新稿，与有限模型边界分开 | SOURCE_TRANSFER_REVIEWED；继承QW-DET/QW-POISSON限定范围，未新增全网络随机理论 | theorist/referee/PI | mathematical_analysis.tex、proofs.tex；finite源与实际V1-F区别披露；不升级原QVI-001/BHH-001 |

未编译，因此没有PDF排版或编译成功声明。旧数据和CFS smoke没有被写成新论文的效果实验；实验计划本轮未撰写。

## 当前结构修订的证据（2026-10-18）

| ID | 精确主张与边界 | 状态/Owner | 来源、证明与实验证据 |
|---|---|---|---|
| PAPER-STRUCTURE-001 | 容量→MILP/RH主线与独立稳态分工完成；原结论及证明全部保留，QVI/Poisson移附录 | WRITTEN / STATIC_REVIEWED，PI/Referee | [结构报告](../research/paper/final_manuscript/BHH_structure_revision_report.md)及[映射](../research/paper/final_manuscript/BHH_source_and_claim_map.md)；独立限定ACCEPT；无新实验 |
| PAPER-GEOMETRY-001 | 在均匀均衡分区近似下，sqrt((S/v)(q/v))=sqrt(Sq)/v解释现有b项 | EXPLANATORY_IDENTITY，Theorist/PI | main_v2:71、steady:62–69及新稿eq:wave-geometry；非有限样本定理或实证标定 |
| PAPER-DIRECT-ROUTE-001 | Direct物理连接与库存可再派位置尚未确认 | REQUIRES_AUTHOR_CONFIRMATION，作者 | 结构报告路径账本；代码复现原式不构成路线证明 |
| PAPER-LIT-018 | 当前18条实际引文包含已读原PDF的Banerjee2022 | SOURCE_CHECKED，Literature | 原PDF首页、1331/1333/1336/1339页；未查其在线附录，不认证novelty |

旧12条/17条记录保留为相应阶段历史。原QW及V1范围不变；作者自行编译，静态审查不等于排版或正式投稿验收。
