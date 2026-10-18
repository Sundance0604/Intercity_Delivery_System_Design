# PROGRESS — 已核实的模型同步进展

更新：2026-10-18（Asia/Shanghai）。**新英文终稿模型/理论正文、引言与综述已完成，子Agent全文审查限定ACCEPT。** 数值章节和实验计划按用户要求暂不写，未编译。此前T0证明与V1-F/RH/剪枝保持已有验收；一般随机QVI、warm start/local branching和大型回归未完成。最新证据见文末。

## 已完成的有界交付

1. 主协调者读取AGENTS、初始共享合同/账本/队列/进度、Git状态、四份TeX及相关本地源码；定位实际QVI笔记。
2. 并行Theorist/Experiment/Literature分别交付[数学比较](../research/theory/model_comparison.md)、[代码差距](../experiments/reports/model_code_gap.md)、[文献定位](../research/literature/model_positioning.md)。
3. 等待三稿后启动Referee，直接核对原始来源并交付[独立审查](reviews/model_alignment_review.md)。三稿ACCEPT限静态审计用途；源模型—算法对齐MAJOR REVISION。
4. PI更新[MODEL_CONTRACT](MODEL_CONTRACT.md)、[CLAIMS](CLAIMS.md)、[TASKS](TASKS.md)及本文件，新建[MODEL_DIFF](MODEL_DIFF.md)、[MIGRATION_PLAN](MIGRATION_PLAN.md)。合同登记源文件SHA-256，计划包含依赖/方法/验证/验收及Q1–Q12。
5. 按用户要求提交推送上述10份文档（93b2dcb），并完整重写分支README后推送（f26b4c9）。推送未包含本地TeX、QVI笔记、生产修改或历史数据。
6. 用户随后授权按评审推进下一步：Theorist和Experiment并行只读核对，PI交付[V1_M_DECISION_SPEC](V1_M_DECISION_SPEC.md)，Referee在具体交付后完成[独立规格审查](reviews/v1_m_decision_spec_review.md)。规格含M01–M15逐式/代码/验收矩阵、D1–D6作者选项、F1–F5手工oracle与有限后续预算建议。输入域/案例小修已解决，规格准备ACCEPT；未登记整个G0已完成。
7. G0-M收尾核对：35份受保护Python/测试/TeX来源的前后SHA-256一致；8份相关文档的本地链接无缺失，文档差异格式检查通过。实际变更为2份新规格/审查文档，以及README、合同、计划、主张账本、任务、进度6份更新；没有新代码或数值验证。
8. 最新作者评审后，Theorist提供车辆账本/真实终点数学，Experiment只读列出e全链路接入，PI交付[车辆修正规格](VEHICLE_REPOSITIONING_SPEC.md)。新增[独立toy账本脚本](../research/toy_models/vehicle_repositioning_ledger_check.py)，最终10个正反例PASS、exit0、约0.09秒；只验证有限车辆事件，不验证BHH/MILP/完整RH货物或最优性。
9. Referee在具体规格/脚本交付后完成[车辆独立审查](reviews/vehicle_repositioning_review.md)：规格及有限原型ACCEPT，生产对齐NOT IMPLEMENTED；静态检查代码与10例逻辑，执行结果引用Theorist记录，未重复运行。两项作者政策已确认，剩余参数/cut/真实终点模式明确保留。
10. 本车辆阶段新增修正规格、独立审查和toy脚本，更新README及合同/计划/主张/任务/进度；保留上一阶段未提交成果。35份受保护源码/旧tests/TeX散列一致，10份相关Markdown链接无缺失，文档格式差异检查通过。未改生产或历史结果，未提交推送。

## 核实的知识与负面结果

- V0实际有M/F/RH不同模型/算法分支；旧论文与base的AV容量已不一致。F已有部分V1结构，不能一概当未实现。
- V1新增wave联合容量、stem、one-hot与direct终端/空返机制；正文、旧算法与代码未同步。已给旧启发式容量和history-only零车剪枝的限定手工反例，没有运行实验。
- V1与两份稳态稿尚不等价；sorting/δ/ω/timetable、车队/目标/连续近似及首末期需桥接。S的n≤W、deadline等号/可行边界需重审。
- CFS quantity是重量映射，不能当stop-count校准；partial horizon成本和window bound/gap不得解释为全日证书。
- 本地QVI笔记已找到，实际主要为确定性常率/时变率。限定周期代数可核；随机到达、完整verification、threshold与finite sorting同步最优仍未解决。
- Literature记录多项公开文献的存在性/主题核验与支持边界（清单含13项）；Referee另作有限抽查。未作穷尽novelty认证、未逐篇付费全文审定。Bib与稿称校准脚本缺失/未定位。
- Referee曾怀疑b=0破坏稳态2/3定律，复读as:bhh的b>0及cost-bound的ω>0后撤回；未把域外反例登记为稿内漏洞。V1与S参数域仍须分别保留。
- G0-M补核：V0-M导出仅y/z且过滤≤0.1，若V1连续解复用将缺少完整验算证据。新V1要求完整x/y/u/g/ĝ/z与独立重算；未声称旧V0整数z运行已出现0.05丢失故障。F1–F5只完成手算与独立复核，不是测试/求解结果。

## 未完成与明确限制

初轮审计和G0-M阶段未运行Python/tests/solver。最新车辆任务仅以bundled Python执行标准库toy账本：初次9例，静态补齐真实closing后的未来任务反例后最终10例；没有运行生产tests、Gurobi、真实数据、TeX编译或依赖安装。四份TeX、生产Python、tests及历史数据不变。已有生产tests/solver许可证状态仍UNTESTED。稿内表格与校准不是本轮复现证据。

初次审计后曾按用户明确要求提交推送文档，见上列提交；此后的G0-M与车辆修正准备阶段未提交推送。新toy不放入生产模块，不覆盖旧结果，不持续后台运行。

已有大量untracked研究/实验文件，工作树不干净；`.pytest_cache`拒绝访问导致Git/文件扫描警告，不能断言全库所有角落均被遍历。没有修改权限或清理缓存。HEAD及输入散列见MODEL_CONTRACT，不能用HEAD替代untracked模型版本。

收尾检查：`intercity_delivery/`、`tests/`、`research/paper/`下的Python/TeX/Bib文件与`main.py`前后SHA-256快照一致；`git diff --check`通过，仅提示现有LF/CRLF转换策略。四份原协调文档更新、六份获准报告/计划新建；其余既有untracked状态不解释为本轮产物。Referee对六份协调文档作只读一致性复核，无阻断错误；文献清单数量已按复核结果更正。这不增加任何科学证明或数值验证状态。

## 待作者决定与下一步

最新两项政策已按作者意见确认，不再重复询问窗口是否应回归或空调拨是否独立。车辆修正还需明确真实关停模式/目标、空行时长费用与跨cut新任务表示；真实hard目标需前序窗口保持终点可达，不只在最后窗补救。当前代码无terminal等式，不能虚称已删除。

此前D1–D6推荐仍未一并采用，V1-M规格独立保留。wave、单位、连续域等只在对应批准范围内冻结；S/V2仍独立，不由本次车辆意见自动解决。

后续按车辆修正规格的事件账/empty母域→e模型与费用库存→候选/history→完整核算→启发式/邻域顺序准备，实际生产实施需相应规格和范围获批。微型Gurobi预算仍只是建议，正式/大型实验另申请。当前只完成有界规格与独立toy，不自行推进生产迁移或研究循环。

## 进展口径

审计发现、反例和明确的未决条件属于进展；文件数、页数、代码量、Agent数量不代表新定理或数值可靠性。CLAIMS保留QVI-001/BHH-001为CONJECTURE；新增VR-TOY-001仅为10例确定性账本检查，没有NUMERIC_FULL或全网络PROVED成果。

## 2026-10-18首次车辆实施（历史记录）

两个Experiment分别拥有模型/控制/runner和政策/账本；PI新增注册、配置与CLI入口。34项检查通过：ledger18、integration4、optimizer6、rolling6。6个微型完整求解测试各≤2秒单线程，真实RH整次≤2秒，总预算60秒内完成。CLI dry-run通过，未运行真实数据或大型实验。Gurobi10.0.3许可证可用。partial状态修复后又定向重跑1项stub通过；独立审查静态接受修复。证据见[实施报告](../experiments/reports/vehicle_repositioning_implementation.md)。

结果限于V0-F加独立empty。计算窗不复位，历史迁移持久；完整模式支持真实终点目标，rolling目标暂拒绝。partial不冒充完整成本，窗口界不冒充全局界。独立Referee限定ACCEPT；跨窗换装链与全套货流独立验证仍OPEN。原模型/算法/数据/四TeX保留，新增3处入口和隔离模块；未提交推送。下一优先级是换装链跨窗微例及独立验算，然后按需要实施终点可达性；不直接启动完整V1迁移或大型实验。

## 连续两步完成：从车辆修正进入V1基础迁移

用户要求第一步完成后直接执行第二步，本轮按依赖完成：

1. Experiment交付cargo_validation、专用测试与[报告](../experiments/reports/repositioning_cargo_validation.md)，PI接runner。先运行原34+当时新22共56项PASS；补4例后cargo26项PASS；Referee复现并指出NaN容量假阳性，修复后最终纯30项PASS（0.003秒），真实换装1项定向PASS（0.051秒）。当前新增31个不同用例与原34合计65，不声称最终版本一次性运行65项。真实链三个控制窗分别pickup(0,1)、AV(1,2)、delivery(2,3)，服务0.05、手算成本7.2。[审查ACCEPT](reviews/cargo_validation_review.md)。
2. 在第一步修复关项、纯负例通过后启动第二步。Theorist先只读交付[原式合同](../research/theory/v1_kernel_contract.md)，Experiment实现`intercity_delivery/v1/`纯容量/弧工具。22纯测试实施者PASS（0.073秒）、PI独立PASS（0.043秒）、Referee独立PASS（0.035秒）；[实施报告](../experiments/reports/v1_kernel_implementation.md)、[审查ACCEPT](reviews/v1_kernel_review.md)。没有调用solver或替换旧容量，未冻结D1–D6。

累计87个不同用例分批通过，重复运行不增加数量。新增换装真实RH共4整次、每次5个微型窗口，20个窗口MILP共享各整次2秒预算；旧6完整测试及1个原direct-RH兼容回归另14秒上限，总理论上限22秒。实际各进程均小于1秒；没有声称精确累计solver CPU时间，也没有大型或真实数据实验。第二步纯测试不消耗solver预算。

V1当前仅基础工具就绪：f/h/K、stem/cap、ceil母弧与半开占用；完整V1-M/F优化器、one-hot决策、V1-RH、稳态和QVI仍未完成。旧审查中的具体跨窗换装覆盖缺口已关闭，但partial未来可达性、复杂双向多单滚动、真实终点尾计划及任意浮点范围精度不在已验证范围。

收尾检查：21份受保护原models/algorithms/data和四TeX的前后SHA-256一致；21份相关Markdown的本地链接无缺失；7份本轮相关Python源码/测试AST有效；`git diff --check`通过，仅原LF/CRLF提醒。Git状态前后已读取，原有大量untracked输入及`.pytest_cache`权限警告保留。本轮原有生产变更仅为新扩展runner接入cargo checker；旧3处入口变更来自上一轮并保留。

本轮新增4份源码（cargo checker及V1三文件）、2份测试、1份数学合同、2份实施报告、2份独立审查和1份预算方案；更新runner、README、MODEL_CONTRACT/CLAIMS/TASKS/PROGRESS/MIGRATION_PLAN及原V1-M规格的阶段注记。没有改原优化器、旧tests、论文或历史数据，未安装依赖、提交或推送Git。两个任务均已完成后停止，没有自行进入第三步完整MIP或后台研究。

## 作者确认及完整V1-M基准完成

用户随后要求继续，两个Experiment分别实现configuration/mandatory_optimizer/runner和独立validation，Theorist只读核对M01–M15与手算oracle。初次8个完整微型run及配置/validator/kernel共74项PASS（suite0.500秒）。脚本在测试结束后保存JSON遇Python workspace PermissionError，外层exit1，未保存那8份raw；PI如实保存[断言记录](../experiments/reports/v1_mandatory_micro_assertions.json)，将执行器改stdout及显式输出预检，未为该日志故障重跑整套求解。其后子Agent因workspace额度不足中断，恢复后补齐review；额度错误不是自动审批拒绝。

作者对异步六项问题明确回复“采用这组定义”，D1–D6-A冻结为`V1M-D1-D6-A-2026-10-18`；不再把该基准登记为待作者确认。更新model_id=`v1_mandatory_arc_wave_v1`、semantic_status=`author_confirmed_profile`、decision_id，数学约束未变。确认后2次真实定向复验OPTIMAL/独立核验通过，成本12/5，完整raw已保存于[最终结果](../experiments/reports/v1_mandatory_confirmed_micro_results.json)，进程exit0、0.395秒。先前8run和最终2run累计10次，单次2秒/Threads1/Seed0，总理论上限20秒，实际未跑大型或真实数据。

最终新增4个runner纯mock，连同配置8/validator36/kernel22共70项PASS（0.043秒、exit0）；新MIP八用例已在初轮执行，最终有78个不同用例分批通过，不声称一次全跑78。Referee此前亲跑44纯例，最后亲跑4个mock并纯重放两份raw都通过，[最终审查ACCEPT](reviews/v1_mandatory_review.md)。本次新增56个不同用例，22工具回归不重复记新增；连前阶段共143个不同用例有执行证据，不推广全库/一般数学正确性。

M范围G0/G1/G2和独立结果核算完成；原V0/RH与TeX/历史数据保留，未接旧CLI/GUI注册表，未提交或推送。当前下一步是G4-M独立实验入口与schema适配，随后按V1-F/RH各自合同推进。详见[本轮实施报告](../experiments/reports/v1_mandatory_implementation.md)。

最终收尾：本阶段前后21份原models/algorithms/data和四TeX散列一致；21份相关Markdown本地链接无缺失；12份V1相关Python AST有效；正式确认后两raw的source SHA与当前文件全部一致；`git diff --check`通过，Git状态前后已核。保留原有untracked输入、LF/CRLF及缓存权限提醒。新增4个V1源码、3测试、profile/方案/执行器、实施及review和2份证据，协调/README已同步；文件责任与历史产物区分见实施报告§6。

## 最新完成：V1-F、rolling horizon、剪枝及CFS

按用户要求复用已有入口/数据/实验方法，完成独立V1-F、RH和保守货流剪枝。Theorist只读核原式，四Experiment分别拥有模型、控制/剪枝、核验/runner、CFS/入口，Referee最后审查。[运行合同](V1_F_RUNTIME_CONTRACT.md)、[实施报告](../experiments/reports/v1_f_runtime_smoke.md)、[独立审查ACCEPT](reviews/v1_f_runtime_review.md)。

[保存JSON](../experiments/reports/v1_f_cfs_smoke_results.json)：同8条CFS、seed4101、06-348↔06-488、τ9；full OPTIMAL服务8/成本1100，两RH均12窗到T24、服务7/成本2025，全部独立核验PASS。实际cargo累计8265→397，删除7868（95.20%）；本样本成本服务一致，不认证一般等价或速度优势。full与RH信息不同，不从成本差推在线gap；CFS为shipment-stop代理，未作实证校准。

最终73纯tests一次PASS（含4旧入口回归）；7不同微例累计8run分批通过。初轮1项empty非零断言因同价τ长度零货direct退化失败，只调测试empty费率后成本2.5定向通过，未改生产可行域。CFS只3run、总约0.953秒；批准上限65秒，实际各run按上限合计61秒，不冒称精确solver CPU时间。Referee独立三方案重放、source散列核对均PASS；status类型注解LOW修复关闭。

新增8个V1运行源码、5测试、CFS执行器、运行合同、实施报告/JSON/审查；必要更新main/configuration/core/solvers及README和5协调文档。21份原models/algorithms/data及四TeX来源散列一致，25相关Python AST有效，git diff --check通过。原untracked、LF/CRLF和缓存权限提醒保留；无论文/历史数据修改、依赖安装或Git提交推送。普通CLI注册/转换以纯测试核对，实际CFS证据来自专用runner，未另行实跑普通CLI写盘套件。

用户本轮范围完成后停止。V1 warm start/local branching、真实终点RH尾计划、稳态、随机抵达/内生wave/QVI及正式大型实验尚未实施。

## 论文规划与QVI–Wave一致性研究完成

用户要求研究能否同参同解并规划如何融入现有论文，无需编译。根据current:27–41的10月1日/2日内部记录和earlier标记，选择steady_state_model.tex为主要机制稿、linear为简化基准。Theorist、Checks与Literature独立拥有[理论](../research/theory/qvi_wave_equivalence.md)、[toy数值](../research/theory/qvi_wave_numerical_check.md)、[逐章写作方案](../research/literature/paper_integration_plan.md)，交付后Referee给[限定ACCEPT](reviews/qvi_wave_paper_review.md)。PI整合[总计划](PAPER_WRITING_PLAN.md)。

已证明确T0共同子模型的J=ΛG°和最优重复阈值集合一致，补齐完整日历余项、全域tail-inf QVI及横截验证；保留n≤min(M,q)，没有继承现稿缺域闭式。单位Poisson只在同整数阈值类同argmin、cost rate差ω/2；一般随机控制仍未验证。Referee两处措辞LOW（0附近regularity、Poisson B=1例外）均由原owner修复。全部steady可投稿状态仍MAJOR REVISION；完整V1/RH/CFS/QVI等价声明REJECT。

纯一维脚本用原式稳态优化对独立Bellman：三例1.973515/8/61.304753对应.05网格1.95/8/61.3；容量、网格、160→320边界、同整数Poisson及遗漏holding负例通过。[PI最终raw](../research/toy_models/qvi_wave_consistency_results.json)exit0约2.132秒；专员3次小版本与PI1次均纯toy，重复不算新案例。5条primary文献只核支持范围，不认证novelty。没有Gurobi/CFS/大实验或TeX编译。

本轮新增理论报告、toy脚本/数值记录/raw、文献写作计划、独立审查和总规划；更新模型合同/主张/任务/进度及README入口。原TeX和生产源码/旧tests未改，历史数据未动，未提交推送。下一步是具体稿件修订，不是继续扩充高维QVI或重复已有大实验。

## 新论文终稿已形成（当前不含数值实验）

按用户明确要求，新建[final_manuscript/main.tex](../research/paper/final_manuscript/main.tex)独立源项目，保留原main/steady稿。Literature写完整英文引言/综述和Bib，Theorist写数学分析/证明，Experiment按当前源码写完整M/F模型与实际RH/剪枝，PI装配摘要、章节及讨论/结论。用户要求的子Agent独立全文审查见[final_manuscript_review](reviews/final_manuscript_review.md)。

首轮全文审查未发现模型或数学阻断错误，指出直接shipment-consolidation/renewal前史遗漏；补Çetinkaya–Bookbinder2003及明确方法前史后定向关项。现12条实际引文均有DOI。新文没有继承未验证的全域convexity、deadline iff multiplier、finite-sorting同步全局最优或V1=QVI声明。

最终源码静态检查为8TeX、12Bib、68labels、57ref uses，无missing input/ref/cite、重复label、括号/环境问题；未编译，项目无PDF/aux/bbl/log。71份原生产/旧tests/TeX/Bib与开始散列一致。主入口与文件结构见[项目说明](../research/paper/final_manuscript/README.md)。

本轮新增终稿项目与独立审查，更新论文/仓库README和模型合同/claims/tasks/progress。没有写数值实验章节、没有写基于终稿的实验计划、没有运行solver/大实验或改旧数据；未提交推送。终稿为当前指定源入口，不代表包含实验的完整论文已可投稿。

## 参照Banerjee完成结构精简（2026-10-18）

用户明确选择“直接重排并精简新稿”。Literature核读Banerjee原PDF并完成两项贡献及三主题综述；Theorist只读确认均衡分区解释可用，Direct路线仍未确认。PI将容量命题前置、MILP与RH相邻，完整QVI/Poisson及filter/replay移入附录，增加两幅原创机制图和两张表。正文粗略英语词数6700→4479，含附录7784→6542；全部已有结论、假设与证明保留。

Referee发现时间线原j>c_t示意与固定窗口参数不符，已修为i<t<b_t<j≤c_t≤T并复核，最终限定ACCEPT。11TeX、10inputs、78labels、68refs、18Bib静态检查通过；本轮无数值实验、新研究或生产代码变更。

内置编译准备阶段仍失败；随后作者明确本机不编译、由其自行编译，故停止重试，不声明有新PDF或无排版溢出。[修订报告](../research/paper/final_manuscript/BHH_structure_revision_report.md)、[来源和迁移表](../research/paper/final_manuscript/BHH_source_and_claim_map.md)为当前交付；旧报告保留历史。未提交推送。
