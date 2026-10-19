# V1 P1 有预算的小规模真实求解协议

2026-10-18（Asia/Shanghai）。负责人：Principal Investigator。状态：**PROTOCOL_READY / RESULTS_NOT_REPORTED_HERE**。本文件规定准备、执行顺序、记录和停止规则；实际结果由PI在运行后独立报告，不在本文件预报成功。

## 1. 授权与冻结前提

作者已确认需求单位、direct原式及目的城市抽象可用节点、T2主值m=0.1与敏感性0.05/0.1/0.2，要求完成模型冻结和预检后做小规模真实求解，不开展大规模实验。三决定分别为 `V1P0-JOB-UNIT-2026-10-18`、`V1P0-DIRECT-ENDPOINT-2026-10-18`、`V1P0-T2-MINLOAD-2026-10-18`。

权威定义见 [V1_MODEL_FREEZE.md](../../coordination/V1_MODEL_FREEZE.md)，机器决定为 [v1_p0_decisions.json](v1_p0_decisions.json)，来源为 [v1_source_manifest.json](v1_source_manifest.json)，预算为 [v1_p1_budget.json](v1_p1_budget.json)。运行前必须重新生成最终源字节manifest并只读核对，`v1_p0_preflight.py`须通过。任一来源漂移、决定不一致或独立纯测错误，先修复并留痕，再重新审查/快照，不以刷新哈希掩盖差异。

单位固定 `synthetic_job_equivalent`，每条抽样CFS shipment提供1个连续可分等效量；CFS源重量保留，不用作需求或额外重量约束。Direct车辆在完成节点j进入目的城市抽象可用节点，并可在j开始后续任务；不认证真实hub接驳细节。T2的 `sum_l h_ijl>=m*w_ij` 约束是wave平均载量政策，不证明逐辆车非空。论文正文及附件PDF不改，缺失 `(3)(1).pdf` 不被假称存在。

## 2. 运行上限与预算

| 项目 | 硬范围 |
|---|---|
| 总新增真实运行数 | 至多80；失败、超时和重试也占次数，禁止自动重试 |
| 总求解编排墙钟 | 至多600秒，累计输入prepare、5micro和各执行session（含worker启动、建模、solver及run内replay）；启动新worker前核剩余预算。人类等待与随后只读重放/统计绘图另报耗时，不作为新增求解预算 |
| Solver设置 | 单线程，solver seed=0；实际版本、许可状态、容差、gap设置原样记录 |
| Micro | 5次，每次逻辑预算至多3秒 |
| 三算法pilot | 45次，每次逻辑预算至多5秒；RH为整次run共享预算，不是每窗5秒 |
| 机制及敏感性新增 | 30次，每次逻辑预算至多3秒 |
| Worker硬超时 | 至多15秒，并受剩余总墙钟限制；单例逻辑预算与进程硬保护分别记录 |
| 需求规模 | 只运行10/20/40；机制与minimum敏感性仅10单；本批不运行100单及以上 |

逻辑预算应覆盖可测的构建、求解、replay阶段；solver `TimeLimit` 只使用扣除前序开销后的余额。进程硬超时覆盖导入/环境启动或不可控调用，不得以15秒代替逻辑预算。剩余总时间不足安全启动下一个worker时，未启动行标 `NOT_RUN_BUDGET`，保留完整计划分母。总预算耗尽不能解释为全部计划通过。

## 3. 阶段与运行计数

| 阶段 | 固定设计 | 新增运行数 | 进入/停止门 |
|---|---|---:|---|
| A 手算真实micro | T2强制每车.05货量违反m=.1不可行；direct后同刻空货local任务成本5；提前一期local不可行；T0强制direct不可行；跨窗direct执行成本4 | 5 | 两个可行例通过独立replay及手算，三个不可行例返回已证不可行，才进入B；任一失败立即停止后续真实求解 |
| B 三算法配对pilot | N=10/20/40，seeds=4101/4102/4103/4104/4105；同实例O-full、RH-dense、RH-pruned，T3/m=0 | 45 | A通过、全部不可变输入先生成；每run单独保存结果，失败不删行 |
| C 四机制 | N=10、同5 seeds、offline full的T0/T1/T2(m=.1)/T3(m=0) | 15 | 复用B中相同N10/seed的5个T3-full，不新增重复求解；总比较行20 |
| D 最低载量敏感性 | N=10、同5 seeds，offline T2/m=.05及.2 | 10 | 中间主值T2/m=.1复用C；只改minimum |
| E 匹配minimum的empty对照 | N=10、同5 seeds，offline T3/m=.1 | 5 | 与C的T2/m=.1比较empty效应，再与B的T3/m=0比较minimum效应 |
| 合计 | 5+45+15+10+5 | 80 | 不因未完成而增加预算或规模 |

五个micro的具体输入、手算期望、容差和选择理由须在求解前登记，不能在看到结果后换掉失败例。
80次指顶层独立实验attempt（包括失败、超时），不是Gurobi optimize调用数。
一次RH的多个窗口共用该run的3/5秒逻辑预算；逐窗optimize调用次数另行统计。
重复顶层attempt也消耗上限且本协议禁止自动重试，本设计按五次micro顶层运行核算。
许可证首次不可用、环境初始化失败或模型规模超许可时，保存原错误，停止真实求解；仍可完成输入、纯测和报告基础设施，不伪造incumbent。

上述local是车辆连续性任务，货量为零；不将其写成真实载货换装验证。
本轮5个micro没有单独证明AV共享容量、正empty最优选择或自然最优解中的minimum绑定频率；
这些约束另有纯核验测试，本批机制运行仍须独立检查，不能由direct微例替代。

B之后的常规时间限制/无incumbent/partial保留为真实结果，不自动重试；若发现独立replay失败、母域不一致、订单/单位混用、成本口径错误或历史丢失等正确性问题，停止后续阶段并交PI修复。修复后的版本/重跑须重新登记，不覆盖原raw，且不能突破当前预算。无新批准时不追跑未运行行。

## 4. 固定输入与配对

主城市对 `06-348↔06-488`。按 [experiment_data_contract.json](experiment_data_contract.json) 复用既有SQLite抽样与合成窗生成。每个 `(N,seed)` 在比较前只生成一次不可变订单JSON，连同订单ID、方向、D=1、release/deadline、penalty、原CFS记录、observed/synthetic标签和SHA-256保存；所有方法读取同一文件，不在solver内重新抽样。同seed不同N不是同一个实例，除非另行明确嵌套设计，不将其视为逐订单完全相同。

本批15个需求实例及全部预定运行行在首个比较run前写入新batch目录，完整run键至少包含：源manifest内容SHA、instance SHA、N、seed、solver、信息集、机制、minimum、H/h/E、参数与cost scope。相同实例机制/算法对照固定车辆、单位、成本、终点、时间窗和其他政策。实例JSON和源快照在预算不足时仍保留；未运行行明确原因。

基线读取 [v1_p0_capacity_design.json](v1_p0_capacity_design.json)：near_transition的T24、Delta t=60分钟、tau9、a=5、b=20、rho=5、M=4、M_A=8、两城人工车各6/AV各2，人工/AV/direct/empty小时费率20/15/25/25，penalty100。RH采用已冻结pilot设计H12/h2/E12，不改变既有全局运行默认。transfer delay/fee为0，direct share边界[0,1]，真实T城市分配自由。以上是合成设计，非CFS实证标定；运行前核机器参数与本描述一致，不悄悄调整到得到期望收益。

## 5. 信息公平性与证书

O-full使用全日已知订单；RH仅按release揭示。O与RH差异同时包含信息与滚动决策限制，不能称为相同信息的算法优劣。只有O已证最优时才使用“optimal”；有incumbent的限时full报告incumbent、有效bound及gap。完整可行RH成本低于已证最优O且超容差，立即核模型/输入/费用口径并停止相应比较。

RH-dense与RH-pruned使用相同物理模型、vehicle/selector母弧、窗口、预算和揭示规则；剪枝只影响货流候选。完整trajectory可能因先前决策而分叉，故其端到端比较属于控制策略比较；若要定位单窗剪枝的增量影响，需在相同历史承诺下保存母域签名与候选计数，不能把不同历史窗口伪作严格同子问题。候选减少不提供一般无损或速度改善证明。

RH窗口bound/gap仅属于各窗子问题，不提供全日证书。完整结果以独立replay成本排序；partial/no-incumbent/error/worker-timeout/NOT_RUN不进入完整成本排名，也不赋零成本；它们保留在计划总数和失败/完成率分母。

## 6. 运行记录与指标

使用新 `simulation/v1_runs/<batch_id>/`，batch ID唯一，不覆盖旧实验；保存manifest、不可变orders、计划行、每run raw/log、validation、每窗统计、summary及全部错误。任何重用的T3结果按相同run ID/SHA引用，不复制成多次独立观测。参数、来源和运行环境身份在顶层及每run metadata中可追溯。

计时必须区分源读取/抽样与JSON导出、候选构建、模型构建、solver、replay、其他开销及总墙钟。RH逐窗保存上述可测阶段、t/b/c/a、已知订单数、历史承诺摘要、变量/约束数、dense/retained cargo数、vehicle/selector母域签名、status、incumbent/bound/gap与预算余额。已有字段不能测量的阶段标 `null` 及原因，不用solver时间冒充端到端时间。输入预处理共享成本单列；比较算法总时间时明确是否摊入，所有方法保持同口径。

成本报告replayed total及local/AV/direct/empty/penalty/transfer分项；数量报告服务/拒单/货运direct/货运换装份额；车辆报告独立empty车次/成本与零货direct单独数量，不能把e计数称为全部人工空行。连续量下fractional service按货量统计，不把其直接解释为整票履约数。

利用率用忙车时间：人工总忙时为 `sum (j-i)Delta t*(x+w+e)`，分母 `N*T*Delta t`；AV同理用y和A。报告local/direct/empty分项及车辆在途时间，不用任务数比率替代利用率。两城指标分别列local忙车时、跨城到发和库存曲线；若另报城市利用率，先明示采用的车辆可用时间分母及迁入迁出归属，不能直接以初始N_k作为动态城市全部车时。N或A为0时比率为null并说明，不能除零。交付延迟按任务末尾交付、相对release计算，并标明货量加权还是order平均。

状态至少区分 `OPTIMAL`、`TIME_LIMIT_WITH_INCUMBENT`、`NO_INCUMBENT`、`PARTIAL_RH`、`INFEASIBLE`、`ERROR`、`WORKER_TIMEOUT`、`NOT_RUN_BUDGET`；具体原solver status并列保存。完成且replay通过不等于证明最优；reject-all的完整可行结果也须如实报告，不把“有正服务”当作可以删除样本的条件。

## 7. 小样本统计与运营解释

主要比较以同N同seed的完整配对差值进行，报告每个seed原值、差值、平均、标准差、中位数和有效配对数n。成本差符号预先定义为“比较对象成本减参考成本”，负值表示比较对象成本低；相对差的分母为 `max(epsilon,abs(reference cost))`，epsilon和近零参考数须披露。机制主比较相对T0；empty效应用T2/.1与T3/.1；minimum效应用T3/.1与T3/0。T2/.1与T3/0同时改变两政策，不用其单一差值归因empty。

95%区间主方法采用配对差的Student-t区间 `mean(d) ± t_(0.975,n-1)*sd(d)/sqrt(n)`，n>=2才计算，n=1仅列单值、n=0不作成本推断。计划仅5 seeds，近似独立/常态差值的覆盖假设不由数据自动保证；区间作小样本描述，不能证明全国或所有城市普适收益。若失败导致仅部分配对完整，明示complete-pair筛选和分母，不把其当作全部计划的无偏表现。重用T3不增加n；三规模、四机制及敏感性属于多重探索，不作未经校正的显著性筛选。

剪枝同时给货流删减率、完整成本配对差与端到端speedup，亚秒运行时间标记噪声限制，不因短时有差就称数量级改善。没有改善、没有direct采用或成本相等均是可报告结果，不额外调参挑选有利点。

## 8. 阶段交付与停止

本批完成应报告：80次上限内的实际运行数/复用数/未运行数；所有预定行的状态；micro手算与replay证据；三算法配对表；N10机制及minimum匹配对照；计时/利用率/输入身份；剩余不确定性。只运行部分时准确写完成的阶段与失败原因。

P0作者模型冻结、来源预检、真实求解micro门、配对pilot和机制解释各自验收，不互相替代。后续N>=100、更长预算、其他城市、全因子、稳态/QVI或论文效果写作均须新的明确任务与预算。本协议不授权Git推送、网络发布、论文主张修改或无人值守研究循环。
