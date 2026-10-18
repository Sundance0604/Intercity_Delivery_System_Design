# 车辆调拨与滚动边界修正规格

2026-10-17（Asia/Shanghai）。Owner：Principal Investigator。Theorist负责独立账本小例，Experiment只读核对生产实现；Referee在本具体规格交付后独立审查。

状态（2026-10-18）：**两项作者决定已确认；隔离V0-F独立空车扩展已实现并通过有限微型验证。** 见[实施报告](../experiments/reports/vehicle_repositioning_implementation.md)与[实施审查](reviews/vehicle_repositioning_implementation_review.md)。原规格及[规格审查](reviews/vehicle_repositioning_review.md)保留；本文不代表源TeX或完整V1已修改。工程默认与有限完成窗见MODEL_CONTRACT§6；rolling真实目标暂拒绝。此前[V1-M规格](V1_M_DECISION_SPEC.md)的D1–D6仍未一并确认。

## 1. 已确认的两项政策

| ID | 最新作者意见 | 处理状态 |
|---|---|---|
| VR-DEC-01 | 计算rolling窗口不是运输业务终点，不得在每个子问题尾部强制车辆恢复初始城市分布 | AUTHOR_CONFIRMED；窗口尾允许异地/在途，保留真实库存与承诺 |
| VR-DEC-02 | 载货运输与空车调拨分开；空车行动不依赖真实订单 | AUTHOR_CONFIRMED；新增独立空调拨弧/变量/费用/历史/结果字段 |

**尚未确认：**真实运营是否存在关停时点T_op、届时目标车队分布R、空调拨实际行驶时间/费率、一般跨完成截断任务表示，以及严格“每车有货”的配载政策。不能把窗口不复位推成全日永不复位，也不能把“全局可以要求”推成默认必须复位。

## 2. 源文、代码与新修正的区别

- `research/paper/main_v2_BHH.tex:141`将τ期零容量空调拨并在w内；`:212/221`在完整规划期令Σw+=Σw−；`:154`用累计出入表达人工库存。`:230–252`是滚动结构，没有理由把每个计算窗当真实结束。
- **当前Python没有上述terminal return等式。** `intercity_delivery/models/flexible_direct_optimizer.py:217–226`及各约束方法没有Σw+=Σw−，`:228–269`已有累计人工车出入。因此本轮不是“删除代码中已存在的窗口复位式”。
- 当前direct弧从τ+1开始；full/reduced候选需有订单，缺独立empty决策。相关起点/终点限制是候选完工截断，不是城市复位，二者不能混称。
- 本修正保留真实车辆守恒，拆开empty；若未来设置真实终点目标，目标须计所有车辆移动w+e。原TeX和历史审计报告保持原貌，修正来源以本次作者意见登记。

## 3. 变量与独立空调拨网络

local车数`x_ij^{kd}`、可载货direct服务车数`w_ij^d`继续存在；新增`e_ij^d∈Z_≥0`为专门空调拨车数（代码建议`empty_vehicles`，避免与订单deadline e_l混淆）。x/w/e属于同一人工车队，不是三份独立运力。

空弧集合建议：

`A_E^d={(i,j):i为允许的新调度起点，j=i+τ_E^d，τ_E^d为已定义的正整数期数}`。

两方向都按车辆时空母域生成，不要求有同方向订单，不用货物时间窗筛选，不从loaded容量大于0的弧中抽取。既有已承诺空弧必须无条件继承。候选存在不代表城市当前就有车，最终由库存约束控制。

基准若按原文hub-to-hub空驶可取τ_E=τ；若实际需城市到高速入口的接驳或turn-around，须明确计入τ_E或独立事件。本次不默认为loaded的stem/tour时长，也不将道路通行时间设为0。

e没有订单ID、没有货流变量，**不进入**direct服务量q、需求守恒、BHH容量、换装处理费、served或direct份额。尤其不能将e加进K̃(w+e)为货物创造容量。只用`m=w+e`统计车辆出入。

费用新增：`C_E=Σ_{d,(i,j)∈A_E^d} c_E^d (j−i)t0 e_ij^d`，c_E按所用物理时间计。若费率按车小时、t0分钟，则乘t0/60。载货w仍按自身整段服务时长计费；e不再作为w收费，同一行动只计一次。

### 连续货流下的“loaded”精度

`Σ_l g̃_ij^l≤K̃_ij(w_ij)`只给容量上界，不能保证w>0时有正货，更不能证明每辆车均有货。严格`g>0`不是可直接使用的普通闭合MILP约束；不得偷偷加epsilon最小载量。

当前可安全采用的变量语义是w=“可载货服务调度”、e=“专门空调拨”。结果分别保存w、e、direct货量和正货量弧；不把所有w直接标成实际载货车次，也不把e标成全部真实空驶。若作者要求精确逐车loaded/empty分类，须批准最小配载量或不可分订单/逐车配载扩展。零货量w是否可改写成e还涉及时间/费用和后续库存，不能无检查自动替换原始解。

## 4. 推荐的绝对时间车辆事件账本

先用完整事件账本保证正确性，再另审历史压缩。初始人工车数N_k^0；历史承诺H与当前新增D互斥，每个任务有唯一ID且只计一次。所有求和对H∪D进行。

`L_k(t)=Σ_{d,i≤t<j} x_ij^{kd}`（local忙车）。

城市可再调度车数：

`A_k(t)=N_k^0−Σ_{origin(d)=k,i≤t}(w_ij^d+e_ij^d)+Σ_{dest(d)=k,j≤t}(w_ij^d+e_ij^d)−L_k(t) ≥0`。

跨城在途：`P(t)=Σ_{d,i≤t<j}(w_ij^d+e_ij^d)`。

独立核验：`Σ_k A_k(t)+Σ_k L_k(t)+P(t)=Σ_kN_k^0`。跨城出发已经扣掉车，不能再从A_k减一遍P；local也只能扣一次。同刻到达在j可用、占用为[i,j)，若实际有turn-around须明确新增时间。

**已完成跨城迁移不能随滚窗消失。** 车辆到达后持续留在目的城，直到下一次离开；不能只保留当前忙车再把库存重置为最初N。

若改用重基快照，须明确定义“已计切点到达、尚未计切点新发车”的idle库存，另存local剩余释放、跨城在途目的/到达、未来固定任务、hub/在途货物及订单剩余量。快照吸收的完成迁移不能再叠完整历史；两种记账方式不可混用。当前生产没有历史清理，本条防止未来压缩时引入错误，不冒称已修复现有清理bug。

## 5. 计算窗、控制窗与真实终点

记窗口起点a_h、控制结束c_h、新调度最晚起点域S_h、有限扩展完成截断b_h。仅开始于[a_h,c_h)的新行动实际提交；未来计划可重算。e与x/y/w一样提交整数车数、方向和真实起止时刻，**不得进入货流容差松弛**。

在b_h不施加A_k=N_k^0，也不施加Σ(w+e)^+=Σ(w+e)^−。车辆可在另一城或在途；保存状态不等于要求车辆返回。

需要单独区分两种弧表示策略：

1. **有限完工域：**新弧必须j≤b_h。此时不能宣称允许任意新任务越b_h；应把扩展域设得足以覆盖批准的起点+物理时长，并承认短域截断仍可能产生前瞻偏差。
2. **越cut任务+边界状态：**允许明确上限内的新弧j>b_h，并在尾部记录其终点、数量、目的城及未完成状态；j之前目的地不可用，后续窗必须继承到达事件。local释放和货物状态也要继承，不能只放开变量上界。

无论选哪种，已出发/已承诺的跨cut任务都不可被后窗截掉或改方向。新任务范围有限、时长有物理定义；允许越计算窗不意味着允许越真实T_op。e起点可先建议与其他新调度共用S_h、与订单无关；若允许在延展区计划，full/reduced须共享同一明确母域，仅控制区内开始者提交。

### 真实关停目标只按明确运营政策启用

如果业务明确要求真实T_op全部停妥且目标R_k，需同时：所有已执行/固定任务在T_op前完成，L_k(T_op)=P(T_op)=0，并有：

`N_k^0−Σ_{origin=k,i≤T_op}(w+e)+Σ_{dest=k,j≤T_op}(w+e)=R_k`。

无初始在途、全部任务完成且R=N^0时，两城才可简化为`Σ(w^++e^+)=Σ(w^-+e^-)`；只平衡载货w不正确，有在途时只比较出发车次也不足以证明恢复。若为持续运营、没有真实关停目标，不启用此条件。

如果确有hard T_op/R，不能等最后窗口才检查：前面已提交行动可能让剩余时间无法到达目标。需在控制决策提交前检查“现状态/未来固定任务在剩余时域能否到达R”，可用独立终端可达性/尾部调拨子问题或有效安全约束。该检查指向同一个真实T_op，**不是每窗就地复位**；其正确性/预算另验。软未来价值若采用，须标为近似政策，不用无限惩罚伪装窗口硬回归。

## 6. 生产接入改动矩阵（未实施）

以下路径相对`intercity_delivery/`，完整位置已由Experiment核对；建议独立新变体，不改V0默认。

| 位置 | 当前事实 | 新变体必须完成 |
|---|---|---|
| `models/flexible_direct_optimizer.py:50–76,90–173` | direct从τ+1开始，无e | 独立empty_arcs/e整数变量，无订单键/货流 |
| 同文件`:175–269` | 无empty费用；已有w累计迁移；没有terminal回归式 | 目标加C_E；库存计w+e；不宣称删除不存在的等式 |
| 同文件`:320–521` | 货量、服务窗、处理费、需求拆分 | e只影响车辆与费用，绝不创造货量或direct share |
| 同文件`:546–598` | 六变量组，滚动上界/按起点提交 | 新vehicle组e；保持真实起止和整数承诺；明确cut策略 |
| `algorithms/full_window_mip.py:31–128` | direct需订单且j≤completion_end | 与reduced共用订单无关empty母域；并回固定历史 |
| `algorithms/state_dependent_mip.py:20–25,47–65,136–212` | 无empty域；history-only可用车筛选 | empty_arcs/迁移账；不能因历史当地0车删除可由新e调入支撑的服务 |
| 同文件`:330–378,388+` | origin起点限制、history fix、flow小容差松弛 | 所有构造器传e；e只固定，不作为flow松弛 |
| `algorithms/paper_rolling_horizon.py:19–40,114–178` | 六组历史，vehicle_groups仅x/y/w | 初始组/整数归一化/commit含e；跨窗到达与完成迁移不丢失 |
| 同文件`:233–269` | 最后窗口目标及x+w计数 | 独立已执行账本成本，empty车次/成本/在途/终端库存；partial/bound_scope分离 |
| `algorithms/bhh_priority_heuristic.py:27–38,124+,503–586,644–682` | 按订单构造，无empty行动 | 独立调拨动作或明确初版不覆盖；e不触发货物服务，费用一次 |
| `algorithms/warm_started_mip.py:34–80,82+` | 通用group，构造器只传direct | 网络/Start含e；不由e重算q/r，不覆盖固定历史 |
| `algorithms/local_branching_mip.py:34,72–158,170+` | 邻域只有x/y/w | 明确e邻域/capture/restore/恢复完整域，不遗漏新组 |
| `experiments/solvers.py:194–305,316+,373+` | 导出/统计无e | 完整e和全部验算字段；车次不是物理车辆数；q仍只货流 |

## 7. 验收与本轮可运行原型

唯一新增程序：[独立车辆事件账本检查](../research/toy_models/vehicle_repositioning_ledger_check.py)。标准库、固定微型确定性用例、不导入生产模型，不运行Gurobi。它不是完整输入schema、BHH或货物网络验证器。

必要检查包括：不平衡窗口合法、无订单仍可调拨、在途不得提前用车、同刻到达可用、完成迁移跨窗持久、原始历史/快照不能双算、e无货物、正物理时长、真实目标含w+e、关停后任务不应被关停目标helper忽略。不把脚本通过登记成新MILP或RH策略已实现。

### 实际检查记录

执行者：Theorist。标准库原型最初9例通过；PI静态检查发现真实closing helper还应拒绝“全部开始在T_op之后”的任务，加入`all(task.end<=T_op)`及该反例后，最终**10例PASS、退出码0、约0.09秒**。未生成结果文件，Python用`-B`禁止本地字节码缓存。Referee静态审查执行证据和源码，没有重复运行。

本机实际命令：

```powershell
& 'C:\Users\32599\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B research/toy_models/vehicle_repositioning_ledger_check.py
```

最后一行stdout：`10 cases passed; deterministic ledger only; no MILP claim.` 系统`python`别名首次启动退出1、未产出测试结果；切换bundled解释器后才得到上述有效执行。该原型没有完整schema验证、任务ID去重、费用/货流/候选cut构造器，不作为生产validator复用。

最终脚本SHA-256：`1E03C9DB6479E50CCE6C6AEA69DB22B5D5267E8837D80ED9E5F402FDD159E75F`。生产源码、旧tests与四份TeX等35份受保护文件的前后散列一致；文档差异检查通过，10份相关Markdown的本地链接无缺失。本阶段未提交推送。

后续完整接入还须验证：loaded和empty只记一次费；e不增加served/direct share/handling；越cut新弧按选定策略明确拒绝或保留在途，不截短到达；固定e不松整数；Start/邻域完整恢复；真实hard终点在前序控制决策后仍可达。Gurobi micro验收仍需明确预算，不自动运行旧大套件。

## 8. 接下来实施的依赖

`已确认两项政策 → 冻结空行时间/费用、cut域及真实终点模式 → 独立事件/empty网络模块 → 新变体e变量/费用/库存 → full/reduced及history接通 → 完整导出与核算 → 启发式/warm/local → 限额微型求解`。

本轮完成的是可审查修正规格与独立库存原型。实际新optimizer、原论文修改、真实数据校准和正式实验均未完成。此前wave/单位/连续stop等未决事项仍按批准模型范围处理，不能由这两项作者政策推定全部D项已同意。
