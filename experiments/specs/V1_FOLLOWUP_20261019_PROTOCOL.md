# V1 N40固定样本机制后续实验协议

2026-10-19（Asia/Shanghai）。Batch：`P1_FOLLOWUP_20261019_01`。负责人：Principal Investigator；协议整理：Literature。状态：**PROTOCOL_READY / RESULTS_NOT_REPORTED_HERE**。

## 1. 目的、授权与解释范围

用户在P0冻结及有界P1完成后要求“继续完成下一步数值实验”。本阶段在原有最大规模N40上补齐offline机制与minimum对照，不扩大需求规模、不调高惩罚或改动车队以诱导direct。旧N10机制样本没有自然采用direct/empty，旧N40基线则已经可见；本阶段是知悉既有结果后的固定样本后续探索，**不是独立复现，也不是未看数据的确认性试验**。

复用 [P1_SMALL_20261018_01](../../simulation/v1_runs/P1_SMALL_20261018_01/) 内N40、seeds=4101/4102/4103/4104/4105的原始orders和基础config。Direct已在原基线启用，不能把本轮描述为修复了原先禁用direct的实验。旧 [结果报告](../reports/v1_p1_results.md) 与 [独立审查](../../coordination/reviews/v1_p1_validation_review.md) 的限定结论保留。

原63文件冻结来源manifest内容SHA-256为 `5c8b233fbc50cda4d23383f404915027e2b8d63de5faa03a30f6703063b1c6c4`，本轮不得修改该集合或刷新哈希掩盖漂移。补充协议、脚本与测试另行列明SHA并绑定到新batch。原订单、原结果、论文源及生产模型不覆盖。

## 2. 固定模型与输入

采用 [模型冻结合同](../../coordination/V1_MODEL_FREEZE.md) 与 [MODEL_CONTRACT](../../coordination/MODEL_CONTRACT.md) 已确认的profile：每条CFS shipment映射1个连续可分 `synthetic_job_equivalent`，重量仅保留来源；Direct采用当前聚合时长式，在完成时刻进入目的城市抽象可用库存并可同刻再派；不认证真实hub接驳路径。

全部比较固定原N40订单窗、方向、penalty、时间单位、容量、a/b/rho、费用、车辆、真实终点和扩展设置。T=24、每期60分钟、tau=9；每城人工6/AV2；a=5、b=20、rho=5，M=4、M_A=8；local/AV/direct/empty小时费率20/15/25/25，penalty=100。它们是原有合成设计，不是本轮重新标定的运营参数。机器实例及基础config的字节/内容身份先核验，再只允许机制与minimum两个预定字段变化。

| 政策 | Direct | 独立empty | wave最低平均载量 |
|---|---|---|---|
| T0/m0 | 关闭 | 关闭 | 不适用 |
| T1/m0 | 关闭 | 开启 | 不适用 |
| T2/m.1 | 开启 | 关闭 | .1 |
| T3/m0 | 开启 | 开启 | 0，原零货direct特例保留 |
| T3/m.1 | 开启 | 开启 | .1 |
| T2/m.05 | 开启 | 关闭 | .05 |
| T2/m.2 | 开启 | 关闭 | .2 |

最低平均载量为每条direct弧 `q >= m*w`。除零货wave外，正货但平均低于m的wave也会被排除；不能声称只改变q=0一点，也不能推断每辆车严格非空载。T1仍无direct，T3/m0中的零货direct与显式e单独报告，不以e计数代替全部人工空行。

## 3. 运行分母、预算和停止规则

新batch仅新增六个政策各五个seed，共 **30个full顶层attempt**。T3/m0的五个full直接引用旧batch原run ID与raw SHA，合并比较35个观测，复用不增加独立样本或求解次数。旧 `n40_s4103_T3_m0_full` 为TIME_LIMIT且有incumbent/bound，保持原状态，不重跑、不升级为精确oracle。

| 项目 | 本轮上限或规则 |
|---|---|
| 新增顶层attempt | 30；不运行RH、micro或其他规模 |
| 每run逻辑预算 | 5秒，记录实际runner与parent墙钟；solver只用扣除前序开销后的余额 |
| 每worker硬保护 | 15秒，同时不超过累计预算余额 |
| 累计求解编排墙钟 | 300秒，计入新batch prepare与全部执行session，包括worker启动、建模、solver和run内replay |
| 后续只读工作 | 全量replay、统计和图表单列实际耗时，不新增solver调用 |
| Solver | Threads=1，solver Seed=0，实际软件/许可/容差/gap原样保存 |
| 不可变产物 | 输入与计划在首个新solve前冻结，run ID独占写入；不覆盖、不自动重跑 |

每个新run保留request、worker claim、raw、状态、incumbent/bound/gap、独立replay和执行日志；optimize调用数与顶层attempt分别计数。SOURCE/决定/config身份不符、许可或依赖错误、worker错误、独立replay失败等正确性问题出现后停止后续solve，保留原错误与完整计划分母。常规TIME_LIMIT保留可行incumbent或无incumbent状态，不以删行、补跑或更换seed补齐。

启动前核剩余总预算，导入/request开销后再核安全worker时限；余额不足则标 `NOT_RUN_BUDGET`，不启动新worker。逻辑预算超出与硬保护分别披露；有累计余量不掩盖单run超出。后续修复或重新求解需要新的版本/预算登记，不在本批覆盖失败。

## 4. 配对、证书与报告指标

全部为全日已知订单的offline full，使用同实例、同求解器设置与5秒预算。完整结果必须独立按原f/direct式核容量、货流、库存、窗口、机制和成本；全部35个保存方案（含五个旧raw）在本轮事后只读全量replay。报告原solver status、可行incumbent、有效下界、gap和证书范围。OPTIMAL只称为求解器设定容差内最优，仍保留bound/gap，不声称数学精确最优；旧seed4105的gap约4.6687e-5同样不抹零。

成本差预定义为“比较对象减参考对象”；负值表示比较对象可行解成本低。七项配对固定如下，五seed逐项列原值及差值，不按结果另选比较。

| 比较对象 | 参考对象 | 能回答的问题/边界 |
|---|---|---|
| T1/m0 | T0/m0 | 关闭direct时添加独立empty |
| T2/m.1 | T0/m0 | 无独立empty下添加带minimum的direct政策 |
| T3/m0 | T0/m0 | 同时开启direct与empty；不能拆成单机制因果效应 |
| T3/m.1 | T2/m.1 | 同minimum下添加独立empty |
| T3/m.1 | T3/m0 | T3中增加minimum |
| T2/m.05 | T2/m.1 | T2中降低minimum |
| T2/m.2 | T2/m.1 | T2中提高minimum |

五seed配对差给出n、均值、标准差、中位数、min/max与95% Student-t描述区间，完整五例时df=4。不是重复计时试验；若执行采用固定政策顺序，时间对比亦含顺序/系统噪声。七组为多重探索，不进行未经校正的显著性筛选，不宣称一般收益。

若一对任一结果限时，则incumbent差不能归因纯机制或当作最优成本差。对最小化且都有有效证书的A、B，另列最优差 `C*_B-C*_A` 的证书区间 `[LB_B-UB_A, UB_B-LB_A]`，UB取独立replay通过的完整incumbent，LB取有效full下界；任一证书缺失则不填造区间。该区间是优化证书界，与跨seed的95%描述区间分别标注。可行域嵌套只提供理论最优值方向，不保证5秒incumbent同向。

完整replayed成本及local/AV/direct/empty/penalty/transfer分项、服务货量/比例、direct货量/份额、显式empty次数/成本、零货direct另列。连续服务量不转述为完整实票履约数。忙车时间与release至任务末尾的货量加权lead-time沿用原指标；不从单一任务数推导利用率。失败、无incumbent、超时与NOT_RUN保留在全计划分母，不赋零成本，不进入完整成本均值。

## 5. 验收产物与结论边界

新batch保存supplement manifest、原输入/旧run身份、30行新增计划、逐run结果及预算session，派生逐run表、七项配对表和证书区间。实际结果写入 [v1_followup_20261019_results.md](../reports/v1_followup_20261019_results.md)，专业产物完成后交Referee独立审查；审查不新增solve。

成功验收指来源与运行可追溯、全部计划状态真实、完整方案通过原式重放、统计和证书解释正确，**不要求发现direct/empty收益**。无采用、零差、反向incumbent差、限时或预算未完成均须报告。结论限这五个既有N40合成实例及预算，不外推城市/规模，不证明一般最优策略、RH剪枝无损、真实物理接驳或QVI等价。本协议不授权N>=100、调参追逐效果、论文主张修改、Git推送或网络发布。
