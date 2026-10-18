# 车辆调拨与滚动边界规格独立审查

日期：2026-10-17（Asia/Shanghai）。Owner：Referee。

审查对象：`coordination/VEHICLE_REPOSITIONING_SPEC.md`和`research/toy_models/vehicle_repositioning_ledger_check.py`。两项专员有界交付及具体规格已由PI指明后开展本审查。

**规格：ACCEPT（修正设计与边界披露）。独立原型：ACCEPT（有限确定性车辆事件账本）。生产对齐：NOT IMPLEMENTED，不能宣称通过。**

VR-DEC-01（计算窗不强制末端复位）和VR-DEC-02（独立、无订单依赖的空调拨）已由作者确认；不重新要求确认这两项。空调拨数值行驶时间/费率、一般cross-cut表示、真实T_op/R模式及严格逐车配载政策仍待决定，不能由本审查代为采用。旧V1-M的D1–D6也未被两项政策一并确认。

## 审查证据与执行范围

直接复核源`research/paper/main_v2_BHH.tex:141,145–154,208–252`；当前`intercity_delivery/models/flexible_direct_optimizer.py:50–76,175–598`的目标、约束构造、全部约束方法与滚动接口；`algorithms/full_window_mip.py:31–128`、`algorithms/state_dependent_mip.py:136–212`、`algorithms/paper_rolling_horizon.py:19–40,114–135`。完整静态阅读toy脚本，手工检查状态转移和10个断言的逻辑。没有重跑脚本、测试或求解器，没有联网。

执行记录来自Theorist经PI提供的最新记录：**10 cases PASS、exit 0、0.09 sec**。本Referee确认当前代码确实包含第10例及`terminal_target`对所有任务`end<=time`的检查；该执行记录不是本Referee另一次运行证据。脚本只导入标准库dataclasses，不导入生产模型、不调用Gurobi、不写结果文件。

## Findings（severity / evidence / impact / remedy / status）

### VR-R1 [Major / 集成缺口已披露] 源全期复位与现有代码不可混称

- **Evidence**：源212为Σw+=Σw−，221明确完整horizon末城市恢复；源230–252只规定滚动控制/起始/延展窗。当前`flexible_direct_optimizer.py:217–226`调用的全部约束方法无terminal回归等式；228–269已经实现累计w迁移。当前direct从tau+1生成，且候选需订单。
- **Impact**：本次工作不能声称“删除已存在的窗口复位代码”。完工截断与城市归位是不同限制。未来若照搬源212到每个计算窗，仍会违反已确认VR-DEC-01。
- **Remedy**：独立新变体实施e网络并保留真实库存；只在明确真实终点模式下启用全局目标。旧源文保持不改，本修正另有作者决策来源。
- **Status**：规格准确披露差别；生产缺口未修复。**Verdict：ACCEPT（审计事实与修正）；NOT IMPLEMENTED（生产）。**

### VR-R2 [Verified] 独立空网络、货量隔离及计费原则正确

- **Evidence**：规格第3、6节；源141将空驶混在w内，当前生产没有e。拟议A_E按车辆时间/方向生成且不依赖订单，e为非负整数，费用独立按物理时长。
- **Impact**：e必须不进入Ktilde(w+e)、served、direct share、q、处理费或需求守恒；同一行动不同时作为w和e收费。空弧存在不代表当地必有车，库存仍控制其可行性。
- **Remedy**：未来网络/模型/历史/结果所有接口均传e，full/reduced共用明确母域；验收无订单调拨及e不增货量/处理费，独立重算费用和单位。
- **Status**：设计清楚，暂无实现。**Verdict：ACCEPT（规格）。**

### VR-R3 [Verified / 条件化代数] 绝对事件账本守恒、占用与快照互斥正确

- **Evidence**：规格第4节与toy `state`、`feasible`。A=N0−累计跨城出发+累计跨城到达−local忙车；P只统计跨城在途，L统计local忙车。
- **Impact**：每个跨城任务出发扣源城、到达加目的城，完成后迁移永久保留；local只在[start,end)扣当地。故ΣA+ΣL+P=ΣN0且没有再扣一遍P。到达时可调度与源文一致。快照已吸收的迁移若再叠原始历史会双算。
- **Remedy**：保留唯一任务身份与history/new互斥；若未来rebasing，分别保存切点idle、local释放、跨城到达和未来固定任务/货物，不能只保留忙车后重置N0。
- **Status**：手工检查恒等式成立；仅限任务合法且计一次的账本。**Verdict：ACCEPT（条件化账本）。**

### VR-R4 [Major / 未决实现范围] 不复位不等于没有有限前瞻偏差

- **Evidence**：规格第5、7–8节；当前`full_window_mip.py:45–78`与`state_dependent_mip.py:158–208`的completion截止；`flexible_direct_optimizer.py:574–579`的新任务越窗关闭。
- **Impact**：j<=b_h的有限完工域仍会删掉可有利的跨cut新任务；即使无城市回归约束也存在前瞻偏差。允许新任务越cut则必须保留真实到达/local释放/货物状态，不能只放开一个上界。既承诺任务不能被后窗截短或删除。
- **Remedy**：先明确有限完工域或cross-cut+边界状态政策、物理时长及新起点域，再统一full/reduced/commit。保持e整数固定，不能进入flow容差松弛。
- **Status**：规格已正确披露两种选择及偏差；一般cut策略尚未确认/实现，toy不覆盖候选网络。**Verdict：ACCEPT（范围披露）；PENDING（策略冻结与实现）。**

### VR-R5 [Major / 运营政策条件] 真实关停目标须含所有移动且及早检查可达性

- **Evidence**：规格第5节；toy `terminal_target`要求所有task.end<=time、feasible、local=transit=0及idle=target；第10例是未来6→8任务对time=5目标的拒绝。
- **Impact**：即使关停当刻idle等于目标，仍有未来固定任务也不能判定关停完成。若确有hard T_op/R，应计w+e，且前期提交后仍需保证将来可达；最后窗才补检查可能太晚。平衡w车次不能替代全部移动、在途清空和目标城市分布。
- **Remedy**：真实T_op/R明确时才启用，前序控制决策另做同一真实终点的可达性/尾部调拨验证。该问题不要求每窗即时回归。数值target不由本审查设置。
- **Status**：源全域等式适用条件及新helper边界准确；尚未实现early viability或一般目标求解。**Verdict：ACCEPT（规范/helper有限逻辑）；PENDING（运营模式/完整保障）。**

### VR-R6 [Major / 语义风险已披露] w不保证真载货，e不等于全部实际空驶

- **Evidence**：规格第3节及toy第9例允许loaded cargo=0。当前`flexible_direct_optimizer.py:363–375`也只给货流容量上界。
- **Impact**：Σcargo<=K(w)无法推出w>0有正货，更不能推出每辆车均有货。把所有w统计为实际载货车次会夸大载货，反之e也不能代表所有真实空驶。严格g>0不是普通闭合MILP可直接表达的约束。
- **Remedy**：保留w“可载货服务调度”、e“专门空调拨”的诚实语义，分别导出车辆数和货流。不得偷偷加epsilon；严格分类要另批准最小装载量/不可分需求/逐车配载。零货w不能不检查时长/成本/库存就自动转成e。
- **Status**：规格已完整警示；未把上界当充分真载条件。**Verdict：ACCEPT（披露）；REJECT（若宣称现有式保证逐车载货）。**

### VR-R7 [Moderate / 证据等级] 10个小例是有限账本原型，不是生产验收

- **Evidence**：toy全部代码与10个check，规格第7节已说明不完整schema、不验证BHH/货物网络。例子覆盖无平衡窗口、无订单empty、在途不可用、到达即用/完成迁移、真目标含w+e、快照双算、empty cargo=0、固定正时长、零货w及关停后任务。
- **Impact**：`valid_task`不全面检查整数/有限值/初始fleet/唯一ID；`state`不验证任务；脚本无cost/BHH/candidate/commit/Start/邻域/完整结果字段，也没有早期目标可达性。10例PASS不能推广为一般输入正确性或MILP/RH已修复。
- **Remedy**：继续保持现有有限原型标签。生产前新增完整schema/唯一ID及独立schedule验证，成本、货流隔离、整数承诺、cut策略和warm/neighborhood恢复单独验收；微型求解另获明确预算。
- **Status**：已静态复核，执行结果仅引用PI转交记录，未重跑。上述边界已披露，不构成本原型拒收理由。**Verdict：ACCEPT（有界原型）；NOT VERIFIED（通用验证器/生产策略）。**

## 最终状态与停止

本次取得的是两项确认政策、可接受的修正规格及有限车辆账本证据；不是生产修改、求解验证或新最优性定理。接下来仍需冻结数值时长/费用、cut表示及真实终点模式；源模型其他wave/单位/需求域问题按实际实施范围继续保留。

唯一写入`coordination/reviews/vehicle_repositioning_review.md`。未改TeX、生产代码、旧结果或他人报告，未运行solver/长实验，未commit。Git状态存在既有untracked文件，不称工作树干净；完成后仅核对授权相关文件状态。审查到此停止，交付PI。
