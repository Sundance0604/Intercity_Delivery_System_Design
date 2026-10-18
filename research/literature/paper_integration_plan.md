# Wave、QVI 与现有论文的整合写作计划

日期：2026-10-18（Asia/Shanghai）。责任：Literature / Writing Agent。本轮唯一写入本文件；未修改或编译 TeX、未运行求解、未提交或推送 Git。

本计划依据本地实际稿件与当前实施合同。数学一致性证明由 Theorist 独立负责，轻量核验由 Checks 负责，结论的最终状态以 PI 整合与 Referee 审查为准。以下区分“已读到的稿件内容”“建议采用的论文结构”和“等待复核后才可写入的数学结论”。

## 1. 两份 steady：哪份更新，哪份适合本次比较

**建议以 `steady_state_model.tex` 为后续稳态主稿，以 `steady_state_model_v1_linear.tex` 为简化对照基线；不要把两稿直接拼接。**

判断不依靠文件修改时间：两稿封面都写 Working draft, 2026-09-30。`steady_state_model.tex:27–41` 的内部决策记录进一步包含 2026-10-01 的不同步回归及 2026-10-02 的慢车系数决定，并在 40–41 明确称 linear 为 earlier draft。因此可判断 current 在概念演进上较新；封面日期尚未同步。

| 项目 | linear 稿：适合一致性基准 | current 稿：适合论文主线 |
|---|---|---|
| 定位 | 运营中的三段 batching 与共同 wave；direct 仅 benchmark（25–43） | 同日达、最优 headway/loads、deadline 与 cost 两区（27–51） |
| 城市与需求 | 恒定、对称、双向平衡、空间均匀（49–50） | 同样的确定性流体背景（57–58） |
| 城市 tours | 等负荷分区且同时返回（53–61） | 引入最慢车因子 δ，同时返回假设已撤回（61–69） |
| 时间 | 线性平均 lead-time 成本 ω，无硬 deadline（69–70） | 线性平均成本 ω 与硬最长期限 L 并存（85–86） |
| hub | 固定 θ、h，无有限 sorting | μ=κΛ、κ>2，origin 等最慢 collection 车（81–82） |
| 决策 | W1,m,W2,n1,n2，推出共同 wave（75、101–114） | W,n1,n2；finite sorting 下 common timetable 为政策假设（61–62、103–111） |
| 候选主结论 | common wave、负荷比、城市/干线尺度、direct 分解 | deadline/cost 两区、负荷比、期限比较静态 |
| 本次用途 | 在无 sorting、δ=0、deadline 不约束下提供清洁共同子模型 | 保留运营解释与后续扩展，但不能把所有机制塞入一维 backlog QVI |

“current 更好”指它的运营问题定义更完整、清楚记录后续机制，并不表示所有命题已经复核通过。它需要先修正下文的可行域和命题限定。相反，linear 更简单，适合检验 QVI 是否恢复同一个问题的最优解，却不应因此替代 current 的整个研究问题。

路径与名称也需在未来编辑时统一：current:41 指向不存在的 `archive/steady_state_model_v1_linear.tex`，实际文件与它同目录；两稿引用 `main_v2.tex`，实际文件名是 `main_v2_BHH.tex`。本轮只记录，不改原稿。

## 2. 当前论文实际写到了哪里

### 2.1 主 MILP 稿还不是完整论文

`main_v2_BHH.tex` 当前只有 Model Formulation 与 Solution Approach 两个实质章节：

| 位置 | 当前内容 | 写作完成度与下一步 |
|---|---|---|
| 45–60 | Title / Author / Address / Abstract 占位，Introduction 空白 | 尚不能写成“论文已基本完成”；最后根据已验收结果重写摘要与贡献 |
| 62–79 | 系统、订单、stop 单位、BHH 与 wave | 是可复用的模型骨架；补充 wave 按起止期分组的已采用定义，区分近似与原始 BHH 定理 |
| 81–137 | Mandatory MILP | 保留为清晰基线；说明 offline 模型与 release-revealed 控制之间的信息差别 |
| 139–223 | Flexible MILP 与 direct、迁移、期末复位 | 按实现修正说明 independent empty、selector 全车队上界、可选真实终点政策；不要将窗口末端写成运营终点 |
| 224–252 | RH、start/control/extension 窗口 | 226 的“over entire planning horizon optimal”缺全局证书；改为每窗优化并承诺控制窗决策的近似在线策略 |
| 253–445 | state-dependent arc pruning 与两个 proposition | 与当前实际货流剪枝不同；需逐项重审其充分性和适用域 |
| 448–690 | priority ranking / construction heuristic | 仍沿用旧单车逆容量，尚不能称 V1-aware 实现已完成 |
| 692–694 | bibliography 与结束 | 当前无独立计算实验节、结果讨论节或结论节 |

### 2.2 稳态稿有结果框架，但证据链未闭合

current 的 Setting（53 起）、Lead times（91 起）、Batching problem（124 起）、Two regimes（154 起）、Direct reference（247 起）、Illustration（261 起）、Open issues（355 起）已经形成研究叙事。它没有正式摘要、Introduction/相关工作论证和正式参考文献；候选文献只在 70–75 的注释中。linear 也只有候选引用，尚不构成已写好的文献综述。

current:19–25 报告符号与数值检查，358 报告 BHH/δ 仿真；本地未定位到 `steady_state_check.py`、`steady_state_symbolic.py`、`bhh_tour_check.py`，其中第三个被稿件标为项目外脚本。故 263–350 的表格和 358 的误差百分比应保持“作者稿内示例、待归档复现”，不能当作本轮新验证结果。文件扫描遇到既有 `.pytest_cache` 访问限制，不作不可读区域内绝对不存在的断言。

`main_v2_BHH.tex:7` 依赖 `导出的条目.bib`，当前可读树中未找到。公开文献元数据已核验不等于本地 cite key 已修复；需要未来补齐精确条目。四份指定 TeX 的数学内容自包含，未发现 input/include 链。

## 3. 必须在正式写作前解决的缺口

| 缺口与来源 | 对论文的影响 | 建议处理方式 |
|---|---|---|
| current:89 明确 n≤min(M,W)，但 P:132 与 closed form:138–139 只写 n≤M；linear:75 有 n≤W，107 才附条件 | 小 wave 会出现“单车装载超过整批”的假解；某些凸性/比较静态不能直接全域照抄 | 共同子模型保留 n≤min(M,W)，重新最小化；已有闭式只在 n≤W 不绑定时引用。边界分支单独报告 |
| current:145 把“deadline slack”与 λ=0 的上端负荷比写成 iff | 边界可 active 且 λ=0，ω=0 还存在退化；strict slack、非绑定乘子不可混用 | 陈述为在明确内点和正性条件下由乘子决定比值，单列临界边界；Theorist/Referee确认后再改 |
| current:103–111、359 仅在 κ=∞/nested/同 δ 等范围支持 common-wave dominance | finite sorting 下主模型使用 common timetable 不等于证明同步全局最优 | 主文明确“within the common-timetable class”；跨阶段同步最优保留 open question |
| current:101 将平均等待结论推广到 Poisson | Poisson 的期望结果不能给样本路径硬最长等待保证；连续批量与整数阈值也不同 | 明确确定性基准与特殊 Poisson 计算的域、目标、值差；一般随机 deadline 需年龄状态或概率保证 |
| current:51 与 360；main_v2:223 | 一处称 finite-horizon counterpart，另一处明确 MILP 缺 ω/sorting/timetable | 改为“共享 BHH primitive、待桥接的两类模型”；在目标、服务定义和政策类一致前不称相互数值验证 |
| current:119–122 将 delivery 视为 tour 内平均/最后 stop；V1 实施合同采用 tour-end delivery | 同参数仍有不同 lead-time 目标、可行窗和最优解 | 桥接实验必须选择同一个 delivery convention，记录其选择；不能用一个 ω 换算消除全部差别 |
| current:78、244 是负荷配车/平均 busy；V1 固定初始车辆并逐期约束 | 平均 fleet 不是可行的峰值车队；稳态政策未必能嵌入有限 fleet | 对比先使 fleet 不绑定，或明确另加峰值/周期库存可行性检查 |
| main_v2:71 按同起点分 wave 的文字 vs 按(i,j,城市,方向)容量；已有实施档明示 arc-wave | 不同完成期是否共享密度会改变容量和成本 | 稿件未来采用已批准的 arc-wave 口径，并说明与稳态 common wave 的映射条件 |
| main_v2:106/110 的 N^k 与 F 的迁入；141/212/221 的空驶与终点 | 论文未体现已实施的 independent e 与迁入 selector 修正；默认真实T目标也不同 | 按 `V1_F_RUNTIME_CONTRACT.md` 写成明确模型版本，区分源稿、作者决定与工程修正 |
| main_v2:419–426 基于历史 available=0 删除车辆弧 | 对允许未来空车迁入的 F 不能无条件使用；当前实现有意保留车辆/selector 母弧 | 正式算法节按实际 cargo pruning 重写，若保留旧命题，先限定无新迁入条件并独立证明 |
| main_v2:525/527/688 仍用单车 f逆，未用 joint K(v) 增量 | priority construction 不能直接产生与 V1 容量一致的可行初解；代码未完成该迁移 | 算法节只声称已实施 MIP/RH/cargo pruning；warm start/local branching 放后续或待验收再写 |
| current:49 对 AV 影响“不显著”的概括 | 特定 regime 的解析独立性与少量示例敏感度被混合 | 只在 AV 满载/明确条件分支写独立性，其余写受条件限制的比较静态或示例观察 |
| CFS smoke 与旧大实验不属于同一 model_id | 将旧 V0 结果换名为 V1 或将一条shipment代理stop称校准，均不可接受 | V1-F 八单结果只作可运行性附录；正式效率/泛化结论另需统一模型、数据和信息集 |

上述表不是要求先重做所有研究。为了尽快进入可读论文，先修共同子模型、统一声称范围和模型表，再让高维随机扩展独立排队。

## 4. 本轮“QVI 与 trivial Wave 同解”应如何进入论文

共同子模型由 Theorist 审定，不从“同名参数”直接推等价。可比较域应至少写清：恒定确定性 Λ、对称平衡、连续 wave/loads、δ=0、κ=∞、deadline 不绑定、common-wave、无车队/排序/pipeline 跨周期耦合、全量 reset、同一服务成本与可行域；尤其保留 n≤min(M,W)。

共享的局部几何表达可直接说明：取 q=W、v=W/n，`2ρ+f(W)/v = 2ρ+(a+b/√W)n = 2ρ+s(W)n`。这里 W/q 是每批shipment量，n 是单车load，v 是车辆数，不能把三者都叫 wave size。此代数映射不消除 V1 的 integer v、one-hot、离散起止期、arc-wave分组与首末期边界，也不消除前表的 tour-end 交付差异。

Theorist 本轮完整结果见 [qvi_wave_equivalence.md](../theory/qvi_wave_equivalence.md)。该报告第4节给出：定义优化后服务成本 R(q) 为同一每shipment目标剔除 dispatch 前等待项，C(q)=qR(q)，则周期平均成本率满足 J(q)=ΛG°(q)，从而 argmin 集合相同。它解释 stationary optimal wave 如何在允许不同 successive batches 的政策类中恢复。报告已经补上非Zeno admissibility、日历时域未完成周期余项控制，并在第5节给出 tail-inf 包络的全域 QVI verification。这些属于报告内明确假设下的证明，独立审查完成前不升级为论文已接受定理；不能仅将原笔记 complete-cycle 加权平均证明标为任意无限时域控制定理。

特殊 Poisson 扩展应精准陈述：Theorist 报告第6节证明单位 Poisson 到达、整数全量 reset 阈值、同一 C(B) 与线性等待成本时，J_P(B)=J_D(B)−ω/2，因此同一个整数阈值域上 argmin 相同、最优值差一常数。它不是连续 wave 与整数随机阈值字面相等的保证，也不覆盖 marked orders、部分发运、车队不可用、随机服务或硬期限。对所有 adapted 随机干预策略的最优性还需要可积性及期望横截条件，不由阈值类比较自动成立。这个有限结论若获审查，可做边界说明/附录；无需把全部随机 QVI 抬为本篇主模型。

原笔记 `Common Wave 的内生化与非均匀需求扩展.md`：246–459 是周期比较及 J=ΛG 的原始讨论；463–545 是形式平均成本 QVI；549–769 是确定性时变到达与有限期形式；807–901 自己说明 stage synchronization、年龄、sorting workload/pipeline 未完成。未来引用 Theorist 修订证明时，保留该笔记为研究历史，不通过改写笔记掩盖原有缺口。

“同解”建议分三项报告：最优值的单位换算、argmin 集合、实际周期/阈值政策。存在多个最优 wave 时不得挑两种算法的不同 minimizer 宣称矛盾；离散域不同或有限期终点不同也不得宣称失败。数值一致仅为 theorem 的实施核验，不能替代证明；不同模型的数值不同则应解释其结构来源。

## 5. 建议一篇整合，并为第二篇设启动条件

**当前先形成一篇以三段货运运营为中心的整合稿。QVI 作为 stationary benchmark 的解释与校验桥梁，不单独占据标题或贡献首位。** 这是写作建议，未自动重组任何 TeX。

主问题可以表述为：城市密度经济、负荷、hub 处理与期限如何共同决定三段运输的批量，以及这些解析规律如何支持有限车队的调度？

拟议贡献只能在相关验收完成后写入：

1. 清楚区分波次规模与单车负荷的 BHH 近似，并在已说明的政策类中给出稳态结构。
2. 对不含有限排序/慢车/绑定期限的共同基准，给出稳态 wave 与更新式 dispatch 问题的一致性解释。它是模型内部一致性结果，文献创新性尚未认证。
3. 提供可复现的 V1 MILP/RH/cargo pruning，并说明解析基准与有限车队实施之间的适用条件和差距。当前仅第三项的实现与 CFS 可运行性已建立，性能与大规模结论尚未建立。

| 拟章节 | 主文保留 | 附录/补充材料 | 依据与依赖 |
|---|---|---|---|
| 1 Introduction + related work | 研究问题、三层模型图、精确贡献与边界 | 全量文献核验表 | 先完成下面各章再写摘要，不从空白Introduction推断创新 |
| 2 System and common primitives | stop语义、BHH近似、wave与load、共同参数和模型差异表 | 参数单位和符号对照 | main_v2:65–79；current:53–89 |
| 3 Stationary operations | 先无sorting/δ/绑定期限的基准，再逐项增加current机制；保留经过复核的命题 | 全证明、n=W/AV容量拐点、退化边界 | current:91–245；先修域和命题限定 |
| 4 Dispatch interpretation | 简短更新周期一致性命题、含义、不能推出什么 | 全域QVI verification、余周期条件；特殊Poisson整数阈值结果若审查通过 | 新理论审查通过后纳入；无需将形式PDE占据全章 |
| 5 Finite-horizon implementation | V1-M/F、independent empty、动态库存、RH承诺与cargo pruning | 完整MILP、伪代码、接口/状态/单位、独立核验器 | 论文需跟实际新实现一致；不保留未实现启发式的完成式表述 |
| 6 Evidence | 同参同域的小型一致性、机制分解、以后统一口径的有限期实施实验 | 输入、raw结果、完整复现说明、CFS smoke | 当前八单CFS不能填充完整效果/效率章节；旧大实验逐项按model_id审计再决定复用 |
| 7 Discussion | common timetable有限政策类、校准与工程取舍 | 附加敏感性 | 随机/有限sorting同步最优仍开放 |
| 8 Conclusion | 回答已完成研究问题，明确可执行结论 | — | 摘要与结论的每句结果均能回到claim/evidence |

direct reference 可在第3节末保留简短成本分解，长的 whole-city/EOQ/critical-distance推导放附录，避免同时追求“同日达机制”“direct市场切换”“随机控制”三个主问题。current:27–31 已明确不以市场结构比较作为主线；本计划遵守该已记录定位。

**第二篇的触发条件**：当随机模型具有清楚的概率空间/信息集、完整状态、reset/action/cost、可验证 QVI 与存在性/政策结构结果，并有区分均值替换/固定wave/RH的可复现证据时，再考虑以 stochastic adaptive dispatch 为独立主问题。仅将 deterministic λV_q 换成 Poisson 生成元、或重复恢复已有最优wave，不足以自动支持单独论文。是否新颖需要进一步的目标期刊范围检索与完整对比，当前五条背景文献并非穷尽性检索。

## 6. 写作顺序与验收门槛

| 顺序 | 有限任务 | 输入依赖 | 可验收产物与完成标准 |
|---|---|---|---|
| W1 | 冻结共同子模型及命名 | 两steady、实施合同、Theorist结果 | 一页参数/单位/可行域/目标/政策对照；写明current主稿、linear简化基线；不靠参数数值相同证明同问题 |
| W2 | 完成一致性证据 | W1 | 同一G/R/C来源、argmin/值/政策映射、日历时域与QVI验证的独立审查；轻量算例复核；边界不被省略 |
| W3 | 先修结构性不一致 | W1及数学审查 | n≤W完整进入问题、乘子边界、sorting下共同wave限定、真实终点政策、算法容量与变量同步；只编辑获批准范围 |
| W4 | 写第2–4节 | W2–W3 | 主文短命题+解释，完整证明可追溯；QVI不夸大为全网络最优控制 |
| W5 | 写第5–6节 | 当前V1实现报告及W3；另需正式比较规格 | 每项算法描述对应实际代码；CFS smoke标记为smoke；旧数据不得变更模型标签；尚无证据的表保留待完成而不填造数值 |
| W6 | 文献、引言、摘要、结论收口 | W4–W5 | 补齐Bib；每条贡献有source/assumptions/proof/experiment/owner；无placeholders和“已完成”的空泛宣称 |

本轮授权是研究和规划，不需要编译 TeX，也不需要为判断两稿定位重跑大实验。W3之后的原文编辑、正式数值研究按PI汇总确认的范围推进，不因本计划自动触发。若W2证明成立但主稿复杂扩展仍开放，可立即写清楚的基准节及附录，不必等待完整随机QVI才开始论文。

## 7. 可复用写作段落（草稿，仅在结论审查后使用）

**模型关系段落：**

> We distinguish a stationary common-wave benchmark from a finite-horizon fleet scheduling model. They share the same approximation for local routing, but differ in their service-time convention, objective, fleet constraints, and admissible dispatch policies. Their numerical outputs are therefore comparable only after these features have been aligned. In particular, a rolling computational horizon does not impose a physical end-of-day return requirement.

**一致性结果段落，等待理论审查后采用：**

> In a deterministic constant-arrival benchmark with separable complete-wave costs, the stationary wave can be recovered from a dispatch problem that permits successive batch sizes to differ. After removing pre-dispatch waiting from the per-shipment objective and charging it as a running cost, the cycle-average objective is a positive constant multiple of the stationary objective. The two formulations consequently share the same minimizing batches on their common feasible domain. This interpretation does not establish optimal synchronization across stages with finite sorting capacity, nor equivalence to the finite-fleet scheduling problem.

**随机扩展限制段落：**

> Stochastic arrivals require a separate state and admissibility specification. In a restricted unit-Poisson model with integer full-reset thresholds, an additive correction to average waiting cost may preserve the minimizing integer thresholds. This special case does not justify replacing a general stochastic arrival process by its mean rate, and it does not provide a sample-path delivery-time guarantee.

**算法证据段落：**

> The implemented rolling-horizon procedure solves successive finite subproblems and commits dispatches within each control window. Its pruning implementation removes cargo candidates using time and stage compatibility while preserving vehicle movement opportunities and inherited cargo. The current CFS exercise establishes feasibility and executable integration on a small shipment-to-stop proxy instance; broader performance and calibration claims require separate evidence.

这些段落未给未核创新性、未虚构定理编号或数值，也没有写入任何TeX。正式写作应以最终Referee接受范围替换其中的条件式状态。

## 8. 本轮精简文献核验（仅5条，均为原始研究/作者机构/出版社）

访问日期均为 **2026-10-18**。核验范围为出版社/作者机构的元数据与可见摘要；未宣称阅读全文或核验其所有定理。检索只用公开题名与一般概念，没有向外网上传未发表稿件、公式或项目数据。

1. **Jillian Beardwood, J. H. Halton, J. M. Hammersley (1959)**, “The shortest path through many points,” *Mathematical Proceedings of the Cambridge Philosophical Society* **55**(4), 299–327. DOI **10.1017/S0305004100034095**。[Cambridge原始页面](https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/shortest-path-through-many-points/F1C28B5730B94887F4659FCBF8A1F2BB)。支持大样本随机点最短路径的平方根标度；不直接证明有限少stop分区的精确时长或δ估计。正式年份为1959，数字化上线2008不是发表年。

2. **G. F. Newell (1971)**, “Dispatching Policies for a Transportation Route,” *Transportation Science* **5**(1), 91–105. DOI **10.1287/trsc.5.1.91**。[INFORMS原始页面](https://pubsonline.informs.org/doi/10.1287/trsc.5.1.91)。研究非恒定平滑到达率下的发车时刻与乘客总等待成本，以及容量影响。说明内生发车/时变需求的宽泛概念有明确前史；单线固定车辆班次数不等于本项目三段BHH dispatch模型。

3. **Dimitris J. Bertsimas, Garrett van Ryzin (1991)**, “A Stochastic and Dynamic Vehicle Routing Problem in the Euclidean Plane,” *Operations Research* **39**(4), 601–615. DOI **10.1287/opre.39.4.601**。[INFORMS原始页面](https://pubsonline.informs.org/doi/10.1287/opre.39.4.601)，[MIT作者论文](https://www.mit.edu/~dbertsim/papers/Vehicle%20Routing/A%20stochastic%20and%20dynamic%20vehicle%20routing%20problem%20in%20the%20Euclidean%20plane.pdf)。Poisson到达、空间均匀需求和动态路由的背景；其中的轻重负荷政策不是本项目多阶段QVI证明。

4. **P. L. Lions, B. Perthame (1986)**, “Quasi-Variational Inequalities and Ergodic Impulse Control,” *SIAM Journal on Control and Optimization* **24**(4), 604–615. DOI **10.1137/0324036**。[SIAM原始页面](https://epubs.siam.org/doi/10.1137/0324036)。研究反射扩散的遍历脉冲控制和对应QVI。用于交代方法谱系；不能不检查过程、边界与可容许控制便直接套到确定性backlog reset。

5. **Christoph Belak, Sören Christensen, Frank Thomas Seifried (2017)**, “A General Verification Result for Stochastic Impulse Control Problems,” *SIAM Journal on Control and Optimization* **55**(2), 627–649. DOI **10.1137/16M1082822**。[SIAM原始页面](https://epubs.siam.org/doi/10.1137/16M1082822)，[作者机构成果页](https://www.fis.uni-hamburg.de/en/publikationen/detail.html?id=7380d581-3307-4d68-a998-55f351cf4ecb)。元数据与机构摘要确认其verification、粘性解/比较原理和迭代最优停止框架；支持“形式QVI还需verification”的写作要求，不自动认证本项目特定最优阈值。

以上文献界定相关已知研究，不构成项目创新性认证或完整文献地图。此前较广的物流/分区背景仍见 `research/literature/model_positioning.md`，但正式稿每个实际引用仍应以其确实支持的命题范围使用。
