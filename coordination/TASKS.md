# TASKS — 有界任务与依赖

更新：2026-10-19。论文正文任务、P0冻结、P1小规模阶段及固定N40机制后续实验均已完成，任务见文末。100–500单论文级实验仍DEFERRED；本轮不改生产模型、论文或旧结果。DONE仅对列明范围，不表示全部投稿、完整随机理论或数值研究完成。

## 本轮模型同步审计

| ID | Owner / 唯一写入范围 | 依赖 | 完成标准 | 状态 |
|---|---|---|---|---|
| MS-T | theorist：`research/theory/model_comparison.md` | 四TeX、本地代码/笔记、初始合同 | 逐公式新旧比较、代码映射、假设/证明/QVI缺口、行号 | DONE，Referee ACCEPT静态报告 |
| MS-E | experiment：`experiments/reports/model_code_gap.md` | 同上 | 文件/函数/行号映射，迁移依赖及测试验收；不运行求解 | DONE，Referee ACCEPT |
| MS-L | literature：`research/literature/model_positioning.md` | 同上及公开权威文献 | 问题/结构/贡献主张比较，引用核验及限制 | DONE，Referee ACCEPT有界核验 |
| MS-R | referee：`coordination/reviews/model_alignment_review.md` | MS-T/E/L全部交付后 | 独立原始来源复核，按严重度/证据/影响/补救/状态给verdict | DONE；源模型同步MAJOR REVISION |
| MS-PI | primary：MODEL_CONTRACT/CLAIMS/TASKS/PROGRESS、MODEL_DIFF/MIGRATION_PLAN | MS-R | 来源/主张/依赖/验收/作者问题分明；保护源文件 | DONE，文档交付；等待用户决定 |

## 后续G0-M最小决策规格任务

| ID | Owner / 写入范围 | 依赖 | 完成标准 | 状态 |
|---|---|---|---|---|
| G0-M-MATH | theorist，仅回复PI | MS报告、实际V1-M | 全部公式/域/单位和手工oracle只读核对 | DONE |
| G0-M-CODE | experiment，仅回复PI | MS报告、实际代码 | 根相对路径矩阵、可延期项、导出/验证要求、预算建议 | DONE |
| G0-M-PREP | PI：`coordination/V1_M_DECISION_SPEC.md`及导航/账本更新 | 两个专员回复 | 单一规格含源式、D1–D6选项/影响、M01–M15追溯、F1–F5、实施验收 | DONE，规格准备ACCEPT |
| G0-M-REVIEW | referee：`coordination/reviews/v1_m_decision_spec_review.md` | 具体规格已交付 | 直接原式/代码复核；确认手算；不替作者选择 | DONE，确认/实施仍PENDING |
| G0-M-CONFIRM | 作者+PI | 规格及独立审查 | D1–D6采用方案及实施范围书面决定，更新冻结版本 | DONE，作者采用A；decision_id见合同§8 |

## 最新作者评审：车辆调拨与滚动边界

| ID | Owner / 唯一写入范围 | 依赖 | 完成标准 | 状态 |
|---|---|---|---|---|
| VR-DEC-01/02 | 作者 | 最新评审意见 | 窗口不强制复位；空调拨独立、不依赖订单 | AUTHOR_CONFIRMED |
| VR-MATH | theorist：`research/toy_models/vehicle_repositioning_ledger_check.py`，数学回复PI | 合同/原式/作者意见 | w/e库存、真实终点政策、边界继承；有限正反例，不运行MILP | DONE；最终10例PASS、exit0、0.09秒 |
| VR-CODE | experiment，仅回复PI | 当前源码 | 逐函数接入矩阵；确认代码无现存复位等式，区分完工截断 | DONE，只读 |
| VR-SPEC | PI：`coordination/VEHICLE_REPOSITIONING_SPEC.md`及协调更新 | 两个专员交付 | 明确新政策、变量/库存/费用/跨窗/目标、实施与验收 | DONE，规格ACCEPT，生产未实施 |
| VR-REVIEW | referee：`coordination/reviews/vehicle_repositioning_review.md` | 具体规格和脚本已交付 | 独立核验，不把toy升级完整MILP验证 | DONE，规格/有限原型ACCEPT |
| VR-PARAM | 作者+PI | 车辆修正规格 | 真实终点模式/目标、空行时长费用、cut策略及所需业务语义 | OPEN，已确认政策不重复询问 |
| VR-IMPLEMENT | experiment后referee | 用户“好，请执行”及有限默认政策 | 独立V0-F变体全链路接通e；有界微型验证与审查 | IMPLEMENTED；证据见文末和实施报告 |

## 待批准的实施队列

下表是拟议有限任务，不是已委派执行。完整Q1–Q12、阶段方法/验证/验收见[MIGRATION_PLAN](MIGRATION_PLAN.md)。

| ID | 拟Owner | 依赖 | 交付与完成标准 | 状态 |
|---|---|---|---|---|
| G0 | PI+作者 | 本轮报告 | 按批准模型范围确认关键规格；M由D1–D6收敛，F/RH另定 | M-A已确认；F/RH未冻结 |
| G1 | experiment | G0-M-CONFIRM | local配置/容量/母弧域；direct表另行推进 | M范围DONE，F direct表未实施 |
| G2 | experiment | G1 | V1-M及公式追溯、独立schedule验证、批准后micro测试 | M范围DONE；M01–M15及微型验证 |
| G3 | experiment | G2及direct政策确认 | V1-F、空返、迁移、全日人工复位验证 | NOT_STARTED |
| G4 | experiment | G2/G3对应范围 | 新registry/schema、独立成本及scope、V0回归保护 | M独立schema/核算DONE，CLI/GUI注册接入仍待 |
| G5 | experiment | G4 | V1-RH历史/库存/reveal/罚损/terminal账本 | NOT_STARTED |
| G6 | experiment+theorist后referee | G5 | 同母弧域、保守剪枝/安全性证明；full/reduced检查 | NOT_STARTED |
| G7 | experiment | G6 | wave增量容量启发式、返程、selectors/warm start与可行性 | NOT_STARTED |
| G8 | experiment | G7 | local branching邻域规格、恢复/预算/证书验证 | NOT_STARTED |
| G9 | experiment后referee | G8+单独预算批准 | 隔离公平实验计划、回归、日志；先smoke再正式 | NOT_STARTED，正式预算未申请 |

## 独立研究队列

| ID | 拟Owner | 依赖 | 完成标准 | 状态 |
|---|---|---|---|---|
| S1 | theorist后referee | Q10确认 | n≤W/active set/期限临界及可行域完整重审 | OPEN，未授权改稿 |
| S2 | theorist+experiment后referee | S1、Q1/Q9、V1基本验证 | S↔V1共同子模型/归一化/峰值车队/整数/首末期规格 | OPEN |
| L1 | literature | Q11材料及范围 | Bib、全文支持定位、校准/脚本来源；不认证未经检索的创新 | OPEN |
| QV1 | theorist | Q12确认 | 随机或确定性V2的状态/信息/控制/目标/边界合同 | OPEN |
| QV2 | theorist后referee | QV1 | 明确verification/threshold等命题与独立证明 | OPEN |

## 旧首次运行队列保留记录

R-001 landscape.md、T-001 qvi_assessment.md、E-001 inventory.md、V-001 initial.md、PI-001依赖V-001的初始综合是安装时预置队列。本轮用户改为上述模型同步审计，这些原指定文件未交付，不登记DONE；状态为DEFERRED/SCOPE_REPLACED_BY_MS，若未来恢复须重新明确范围。不得把本轮不同名称报告当作原任务已完成。

## 2026-10-18车辆实施任务

| ID | Owner/范围 | 交付/验收 | 状态 |
|---|---|---|---|
| VR-I-LEDGER | experiment：config.py、ledger.py及ledger测试 | 显式政策、绝对事件库存与成本，18项纯测试 | DONE |
| VR-I-MODEL | experiment：optimizer/rolling/runner及对应测试 | 独立e、费用库存、历史继承、完整/滚动结果；12项检查 | DONE |
| VR-I-ENTRY | PI：main/configuration/solvers及integration测试 | 两个可选入口、旧默认保留，4项检查及CLI dry-run | DONE |
| VR-I-REVIEW | referee：implementation_review | 交付后独立审查；partial status修复静态复核 | DONE，限定ACCEPT；报告中开放项保留 |
| VR-NEXT-CARGO | experiment后referee | pickup→AV→destination跨窗微例及独立货流/容量/期限验算 | DONE，31项新增有限测试；审查ACCEPT |
| VR-NEXT-TAIL | experiment/theorist后referee | 如需rolling真实目标，补全程终点可达性保障 | OPEN；当前明确拒绝目标 |

VR-PARAM采用可覆写且披露来源的工程默认，见合同§6，不代表所有V1语义已决定。完整V1的G1–G9仍未完成；连续两步交付见下表，未启动大型实验。

## 连续两步交付与验收

| ID | Owner/独占写入 | 验收 | 状态 |
|---|---|---|---|
| STEP1-CARGO | experiment：cargo_validation.py、cargo测试/报告；PI：runner接入 | 30纯负/正例+1真实三窗换装；NaN修复；费用7.2；Referee审查 | DONE，31项PASS，ACCEPT |
| STEP2-MATH | theorist：research/theory/v1_kernel_contract.md | 源式、单位、域、稳定性、手算oracle，明确不代D1–D6 | DONE，源式合同ACCEPT |
| STEP2-KERNEL | experiment：v1三源码、test_v1_kernel.py、实施报告 | f/h/K、母弧、占用及22项独立oracle纯测试；不solver | DONE，实施者/PI/Referee各自PASS，ACCEPT |
| STEP2-REVIEW | referee：cargo_validation_review、v1_kernel_review | 各步交付后独立审查，唯一阻断NaN关项 | DONE |
| STEP2-SYNTHESIS | PI：README及共享协调文档 | 区分V0-F核验、V1工具、完整MIP及QVI；保护来源 | DONE，具体证据见PROGRESS |

后续完整V1-M优化器依赖D1–D6，纯工具就绪不代表作者确认；真实目标尾计划任务仍OPEN，本轮没有将其当作第二步实施。

## 后续已完成：作者确认后的完整V1-M

| ID | Owner/范围 | 交付/验收 | 状态 |
|---|---|---|---|
| V1M-CONFIRM-A | 作者/PI：六项规格及profile | 人类明确采用D1–D6-A，记录decision_id，不改TeX | DONE |
| V1M-BUILD | experiment：configuration/mandatory_optimizer/runner、mandatory测试 | M01–M15、one-hot/K、共享AV/库存/时序/需求/费用、完整结果 | DONE，8微型用例通过 |
| V1M-VALIDATE | experiment：validation及专用测试 | 原f独立核容量/全套约束/成本及36手工正反例 | DONE |
| V1M-STATUS | PI接管中断收尾：runner状态测试、metadata | 4纯mock、无incumbent不伪造有效、确认编号贯通 | DONE，最终70纯PASS |
| V1M-FINAL-EVIDENCE | PI：两份raw和实施记录 | 正式确认后12/5复验，历史74/exit1/raw缺口不改写 | DONE，raw保存/exit0 |
| V1M-REVIEW | referee：v1_mandatory_review.md | 输出交付后审查、44纯+4mock亲跑、最终raw纯重放 | DONE，限定ACCEPT |
| G4-M-ENTRY | experiment后referee | 独立实验注册/配置转换/schema，保留V0默认与单位边界 | NEXT，未在本轮启动 |

本表取代前一阶段“完整V1-M待确认/未实现”的当前状态；旧表保留阶段历史。V1-F/RH、真实终点尾计划和正式大型实验仍未完成。

## 最新任务：V1-F、RH、剪枝和CFS可运行性

| ID | Owner/独占范围 | 验收 | 状态 |
|---|---|---|---|
| V1F-MODEL | experiment：F配置/容量/optimizer及模型测试 | 源direction direct、迁入local、独立empty，4手算微例 | DONE |
| V1F-RH-PRUNE | experiment：flexible_rolling/pruning及两测试 | 绝对历史、release、预算、实际cargo删减、3RH微例 | DONE |
| V1F-CHECK | experiment：flexible_validation/runner及测试 | 原f独立核验、成本/状态边界，33纯用例 | DONE |
| V1F-CFS-ENTRY | experiment：adapter、core/solvers/configuration/main、执行器及集成测试 | 8条来源可追溯、单位/τ/混跑隔离，3方式各≤15秒 | DONE，均完成并独立核验 |
| V1F-REVIEW | referee：v1_f_runtime_review.md | 专业产物交付后审查及三保存方案重放 | DONE，限定ACCEPT；status注解LOW已关 |
| V1F-SYNTHESIS | PI：README、共享协调文档、实施记录与结果 | 73纯PASS；微例累计8run；CFS3run；保护来源 | DONE |

本轮用户范围已完成。[报告](../experiments/reports/v1_f_runtime_smoke.md)与[结果](../experiments/reports/v1_f_cfs_smoke_results.json)。后续独立任务为V1可行初解/warm start→local branching、真实终点RH尾计划、单位实证校准/正式大规模回归及S/QVI。G4-M专门注册既有V1-M接口仍延期，不阻断已交付V1-F入口；不自动运行后台研究或重复大实验。

## 论文规划与QVI–Wave一致性（2026-10-18）

| ID | Owner/唯一范围 | 产物/验收 | 状态 |
|---|---|---|---|
| QW-THEORY | theorist：research/theory/qvi_wave_equivalence.md | 同域原式、J=ΛG、日历余项、QVI tail包络、Poisson阈值界限 | DONE，Referee限定ACCEPT；两处LOW已修 |
| QW-CHECKS | theorist：toy脚本、numerical_check.md | 独立原式优化/Bellman，3边界+Poisson+遗漏holding负例，无solver | DONE；PI纯复跑与raw保存，非连续收敛证明 |
| QW-WRITING | literature：paper_integration_plan.md | 两steady选择、逐章缺口、主文/附录、文献和6步验收 | DONE，限定ACCEPT |
| QW-REVIEW | referee：qvi_wave_paper_review.md | 专员交付后独立逐式/代码/raw/规划检查 | DONE，无未解决阻断T0错误；原稿MAJOR REVISION |
| QW-SYNTHESIS | PI：PAPER_WRITING_PLAN、共享账本、raw JSON | 证据与版本保护，区分T0/Poisson阈值/current/V1/V2 | DONE；不改/不编译TeX |

[下一步具体写作顺序](PAPER_WRITING_PLAN.md)：先修current的n≤W、乘子边界及common-timetable限定→写T0基准与桥梁节/附录→统一main_v2实际算法与模型描述→另定同口径比较→补引言/摘要/文献。这里只规划，未执行原稿编辑或正式新实验。一般随机verification、finite-sorting阶段同步、真实deadline/车队高维QVI继续OPEN。

## 新建终稿源项目（用户本轮明确授权）

| ID | Owner/唯一写入范围 | 验收 | 状态 |
|---|---|---|---|
| FINAL-LIT | literature：introduction/related_work/references.bib | 英文成稿、12实际引文及支持边界、补直接renewal集并前史 | DONE |
| FINAL-MATH | theorist：mathematical_analysis/proofs | 有限容量代数、完整T0域、全日历/QVI/Poisson限定证明，无未验regime挪用 | DONE |
| FINAL-MODEL | experiment：optimization_model/solution_method | 完整采用V1-F模型、真实RH/pruning文字对应源码，无实验运行 | DONE |
| FINAL-REVIEW | referee：reviews/final_manuscript_review.md | 具体成稿交付后独立审查；实质文献缺口补齐、source SHA记录 | DONE，当前正文范围限定ACCEPT，无未修阻断；9源SHA冻结 |
| FINAL-ASSEMBLY | PI：main/discussion/项目README及共享账本 | 单一入口、交叉引用/文献/环境/brace静态核对、旧来源保护 | DONE |

当前数值实验章节与基于终稿的新实验计划均DEFERRED，用户明确“先不写”。不因终稿命名自动运行实验、编译、提交或推送；未来实验必须以这份源版本重新定义口径。

## 结构精简交付（2026-10-18）

| ID | Owner/范围 | 状态 |
|---|---|---|
| STRUCTURE-LIT | Literature：标题/摘要/引言/综述/Bib，原PDF核读 | DONE，18条引文 |
| STRUCTURE-MATH | Theorist：只读核几何解释、Direct账本、transfer-only及证明依赖 | DONE，未添加模型假设 |
| STRUCTURE-ASSEMBLY | PI：容量前置、MILP/RH相邻、理论和方法附录、两图两表及报告 | DONE，正文约减少33% |
| STRUCTURE-REVIEW | Referee：产出后独立审查 | DONE，时间线不当j>c_t示意已修，限定ACCEPT |
| STRUCTURE-COMPILE | 作者：以现有main.tex自行编译及排版核对 | AUTHOR_HANDOFF；不继续本机重试 |
| DIRECT-ROUTE-DECISION | 作者：明确Direct连接段与再次可用位置 | REQUIRES_AUTHOR_CONFIRMATION |

## P0 数值实验准备（2026-10-18）

用户要求“完成P0”；范围是模型/数据单位冻结、四机制配置、纯容量参数和验证，
不含正式批量求解、历史结果改写或论文改稿。

| ID | Owner/唯一范围 | 验收与当前状态 |
|---|---|---|
| P0-MECHANISMS | experiment：F configuration/optimizer/validation/runner、新mechanism测试 | IMPLEMENTED/PURE_REVIEWED；机制与独立replay相符，含零货容差修复 |
| P0-PARAMETERS | theorist：capacity_design JSON、parameter_design说明、新容量测试 | 合成基线预注册冻结，原式独立二分与时间链PURE_CHECKED，不是运营结果 |
| P0-FREEZE-TEXT | literature：V1_MODEL_FREEZE、equation_to_code_matrix | DONE；式1–26/Appendix C及三项作者决定均冻结，63来源/预检通过 |
| P0-INPUT-SOURCES | PI：experiment_data_contract、experiment_inputs、来源快照/manifest/preflight及测试 | DONE；最终63来源哈希一致，SQLite两次同seed映射一致；三项作者决定已确认，gate通过且拒绝pending替身 |
| P0-REVIEW | referee：reviews/v1_p0_preparation_review.md | DONE；准备产物限定ACCEPT，75纯测及来源核验通过；正式冻结待作者选择 |
| P0-AUTHOR | 作者：direct终点和T2最低装载政策 | DONE；用户逐项明确确认，见MODEL_CONTRACT§14及机器决定记录 |

## P1有界真实验证与小规模实验

| ID | Owner/写入范围 | 验收/状态 |
|---|---|---|
| P1-ENDPOINT | theorist：v1_direct_endpoint_contract与direct_continuity纯测 | DONE；抽象节点语义与j同刻复用，提前使用拒绝；不证明真实hub路径 |
| P1-INSTRUMENT | experiment：F runner/rolling/validation及instrumentation纯测 | DONE/PURE_REVIEWED；phase分区、模型规模、原约束残差、母域SHA与optimize次数，合并195纯测通过 |
| P1-HARNESS | experiment：run_v1_p1、experiment_metrics、harness纯测 | DONE；输入冻结、单次claim、累计预算、micro门、失败保留和配对统计，真实75例校验通过 |
| P1-PROTOCOL | literature：冻结稿/矩阵及V1_P1_BOUNDED_PROTOCOL | DONE；三作者决定、T2平均语义、80次/600秒上限、N<=40 |
| P1-MICRO | PI：run_v1_p1_micro及真实raw | DONE；5顶层/8optimize、3预期不可行、成本5/4，两可行例原f核验通过 |
| P1-RUN-REPORT | PI：独立新batch、汇总/图、共享账本与报告 | DONE_SMALL_SCALE；15固定实例、75批次结果、CSV/JSON/2图，80总顶层/413opt/136.108秒 |
| P1-REVIEW | referee：reviews/v1_p1_validation_review | DONE/LIMITED_ACCEPT；全量75独立重放/身份/统计/校准/计时通过，明确非无损/非机制收益证据 |

P1的100–500单、广城市和更紧车队正式效果实验DEFERRED，未在当前80次任务范围内实施；不自动追加求解。

详见[结构报告](../research/paper/final_manuscript/BHH_structure_revision_report.md)。不自动启动数值实验、模型扩展或代码迁移。
## 2026-10-19 固定N40样本的机制后续实验

用户要求继续下一步数值实验。保留原63来源冻结、五组N40订单和全部基础参数；本轮不扩大到100–500单。

| ID | Owner/唯一写入范围 | 验收/预算 | 状态 |
|---|---|---|---|
| FOLLOWUP-DESIGN | theorist：research/theory/v1_followup_design_20261019.md | 解释原N40 Direct已启用；同历史剪枝恢复的证据边界；不运行solver | DONE |
| FOLLOWUP-HARNESS | experiment：run_v1_followup_20261019.py、test_followup_20261019.py、新batch目录 | 30新full，5秒软预算/15秒硬限，累计300秒；固定输入/计划/源绑定、独立replay、无覆盖重试 | DONE；13纯测、30新solve/35replay通过；26OPTIMAL/4TIME_LIMIT |
| FOLLOWUP-DOCS | literature：V1_FOLLOWUP_20261019_PROTOCOL.md、v1_followup_20261019_results.md | 预先固定7配对、全部状态/证书、探索性结论与可追溯链接 | DONE；报告LIMITED_ACCEPT |
| FOLLOWUP-REVIEW | referee：coordination/reviews/v1_followup_20261019_review.md | 专员产物完成后独立全量审查；不运行新solver | DONE；35原式replay/35配对/七组统计/证书和报告一致 |
| FOLLOWUP-FIGURE | PI：plot_v1_followup_20261019.py、batch figures_final | 从固定analysis绘成本证书区间/服务量，保存SHA；无新solve | DONE；已视觉核验，初版图另保留 |

新策略为T0/0、T1/0、T2/.1、T3/.1、T2/.05、T2/.2各5seed，另复用旧T3/0五行作为比较参考。旧限时incumbent不升级为精确最优值，OPTIMAL亦保留solver容差/gap。

## 固定N四方向自动实验脚本

| ID | Owner/唯一范围 | 验收 | 状态 |
|---|---|---|---|
| FIXED-N-CORE | experiment：experiments/fixed_n中pooling.py以外模块、run_fixed_n.py、test_fixed_n_round.py | 固定N、共享基准、四阶段、预算/恢复/身份门/独立重放/汇总 | DONE；15核心纯测PASS，真实14任务+零新增resume通过 |
| FIXED-N-POOL | 首派experiment因服务容量两次失败；PI补pooling.py、test_fixed_n_pooling.py | 仅local容量替换、独立原式额外核验；不改旧63源 | DONE；8纯测、真实separate与Referee独立原式核验通过 |
| FIXED-N-THEORY | theorist：research/theory/fixed_n_round_contract.md | 14任务、15配对、同模型/信息/单因素边界 | DONE，无新solver |
| FIXED-N-GUIDE | literature：experiments/specs/FIXED_N_ROUND_GUIDE.md | 实际CLI使用、恢复/预算/不同N、真实验证证据 | DONE；实际help/plan核对，PI补奇数/锁/真实验证，12链接通过 |
| FIXED-N-REVIEW | referee：coordination/reviews/fixed_n_round_review.md | 专员产物后审查，并核有界四方向真实验证 | DONE；23独立纯测/14独立replay/15配对/48窗/恢复SHA，LIMITED_ACCEPT |

## N40完整四方向运行（作者明确要求）

| ID | Owner/唯一范围 | 验收 | 状态 |
|---|---|---|---|
| N40-EXEC | experiment：simulation/fixed_n_runs/FIXED_N40_20261019_01/ | 5seed/70任务、600秒累计、5秒逻辑/15秒硬限、四方向自动汇总与真实状态 | DONE；70完整可行/290optimize，181.313秒，无补跑 |
| N40-REPORT | 原literature，额度中断后由PI完成fixed_n40_20261019_results.md及统计audit | 四方向配对/证书/服务/时间与局限，全部分母保留 | DONE；75配对/15组统计850数值比较通过，协调者记录非独立终审 |
| N40-REVIEW | referee，PI整理已完成消息至fixed_n40_20261019_review.md | 专员产物后全量独立复验及统计/来源/预算审查，无新增solve | PARTIAL/OPEN；70独立原式与来源/预算已过；最终统计与prose因工作区额度不足未完成 |
| N40-GITHUB | PI：当前research/qvi-multiagent分支 | 用户明确授权完成后推送；保存精确源/产物字节，核commit与remote一致 | AUTHORIZED；提交包含此前未同步的P0/P1及四方向实现与实验 |
