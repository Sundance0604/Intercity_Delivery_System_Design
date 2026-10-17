# 模型同步独立审查

日期：2026-10-17（Asia/Shanghai）。审查人：Referee。主协调者通知三份专员交付全部完成后启动本审查。

## 1. 审查结论与证据范围

**审计交付：ACCEPT，作为静态差距审计与迁移设计的依据。模型—算法同步状态：MAJOR REVISION。实际迁移：尚不具备无歧义规格，须先取得作者决策与用户批准。**

ACCEPT 不代表证明、代码、历史结果、运行环境或创新性通过认证。本轮未运行 Python、测试、solver 或实验，没有改论文、专员报告、生产代码、历史数据或 Git 提交。唯一写入为本报告。

已读 AGENTS.md、初始 MODEL_CONTRACT（INITIAL / UNVERIFIED）、三份报告，核对四份 TeX 的模型与相关证明/算法段、本地 QVI 笔记及实际代码。对报告核心公式和高风险代码位置作直接逐行复核；并非对每一个命题给出新的完整证明，也不是逐篇文献全文复审。Git 状态存在大量既有 untracked 文件，且 .pytest_cache 权限警告；不能称工作树干净或用 HEAD 唯一标识研究输入。源码行号指本轮当前文件。

| 交付 | Verdict | 独立核验结果与接受范围 |
|---|---|---|
| research/theory/model_comparison.md | ACCEPT | 新旧容量、两份稳态变体、QVI 形式与未决条件区分可靠；模型内代数与全模型定理没有混用。第 5 节的完整稳态命题仍需专门证明审查。 |
| experiments/reports/model_code_gap.md | ACCEPT | 核心文件、函数、变量域、容量/库存/rolling差异经直接源码核验成立；迁移与测试是计划，未执行通过。 |
| research/literature/model_positioning.md | ACCEPT | 文献存在性与适用范围、缺失 Bib、未复现数值及不能宣称 novelty 的界限合理；公开来源抽查范围见 R13。 |

以上报告可接收为有边界的审计成果；其中识别的源模型问题不因报告被接收而消失。以下每项分别列 severity、evidence、impact、remedy、status 与科学/实施 claim verdict。

## 2. 严重发现

### R1 [Critical] V1 波定义及 local 车数上界尚未闭合 — MAJOR REVISION

- **Evidence**：research/paper/main_v2_BHH.tex:71 将 wave 定为同城、同方向、同开始期；:106–110、:172–176 却逐完整 arc (i,j) 选 v。:141、:154、:221 允许 direct 完成后迁入当地并执行 local，local 枚举仍 v≤初始 N^k。
- **Impact**：相同 i、不同 j 能否拆成多个独立空间分区波，会改变容量及可行域。N1=1、N2=2 时先迁入两车，可用库存达到三车；当前 local selector 仍把单波封顶一车。N1=0 时迁入后 local 也被 selector 域禁止。这是数学政策选择，不能在实现时自行改成总车队上界。
- **Remedy**：作者决定 start-wave 或 arc-wave；若 start-wave，需给不同时长车辆共享区域的规则。决定迁入车是否可组成超过初始 N^k 的 local 波，再确定表域/库存口径。
- **Status**：静态冲突已确认；经营语义与修正规格待作者确认，阻断精确 V1-F 实施。

### R2 [Critical] 核心波容量已经更新，求解章节仍是旧模型 — REJECT（“现有算法已产生 V1 可行初解”）

- **Evidence**：V1 main_v2_BHH.tex:82、:106–110 是 K(v)=min{vM,h(v[(j−i)t0−2ρ]^+)}。:525–565、:623–645 仍使用单车逆容量与 ceil(q/C)，:686–690 宣称可行；:340、:369 弧界还是旧 f(M)、未定义新参数的 tilde-f(M)。代码 intercity_delivery/models/base_optimizer.py:164–178、models/flexible_direct_optimizer.py:320–374 与 algorithms/bhh_priority_heuristic.py:65–119 也是车辆数乘固定单车容量。
- **Impact**：只替换一个系数不能迁移 MILP/启发式。手工反例：f(q)=q、ρ=1、Δ=2、M≥2，旧 C=2，而新 K(v)=0；旧算法在有车辆时分配正货量便违反新容量。这是代数反例，未运行数值实验。
- **Remedy**：容量表、binary selector、增量容量 K(v+m)−load、历史固定、warm start 与初解残差验证必须一起更新；直送同理。先统一正文—算法规格，不能照抄旧伪代码当新模型 oracle。
- **Status**：不一致已确认；V1 算法可行性声明目前不能接受；实施/测试未执行。

### R3 [Critical] history-only 可用车为零不足以安全剪枝 — REJECT（无条件安全性 claim）

- **Evidence**：main_v2_BHH.tex:419–428 使用 inherited availability；intercity_delivery/algorithms/state_dependent_mip.py:_manual_availability:47–65 只计 committed schedules，generate:164–167 以此筛 local，:203–206 筛 direct。V1 允许本窗口新增直送调入。
- **Impact**：在“继承库存0，但本窗口可由另一城新调入”的情形，未来 local 并非必定不可行。为避免把 R1 的 N=0 表域缺陷误当反例，可取城1初始一车已在 local 占用至 t=5，继承未来 t=3 库存0；城2另一车能在 t=3 前直送到城1，随后在 [3,4) 做一辆 local，满足单波 v≤N1=1，且可安排以后回流。history-only 筛选会删掉该 local。
- **Remedy**：证明所用 profile 是对所有本窗口可行调入都成立的真正上界；或对可被调入改变的城市/时段保守保留。固定城市无迁移子模型可另证明安全性。
- **Status**：静态逻辑反例已确认；不意味着所有剪枝规则都错。未来 tiny full/reduced 一致只能作反例筛查，不能替代一般证明。

### R4 [Critical] 空返、终端恢复与滚动承诺需要统一 — MAJOR REVISION

- **Evidence**：main_v2_BHH.tex:141 明定 j−i=τ 的零货量空 reposition，:212、:221 要终端正反车次数相等。flexible_direct_optimizer.py:68–76 排除 τ 弧，set_constraints:217–226 无终端恢复式；full_window_mip.py:67–78 与 state_dependent_mip.py:195–208 按有订单过滤直送。V1:238–252 只承诺 control-window 起始任务。
- **Impact**：无反向订单时必须允许空返并收费；反向总货量相等不等于车次恢复。若将全日恢复强加每短窗口，将人为缩小可行域；若从不追踪恢复义务，则末日虚增直送收益。AV 没有同样的终端归位要求，不可凭对称直觉添加。
- **Remedy**：确认空返绕过 stem 的业务特例、恢复期限与 rolling terminal obligation；用完整累计离开/到达账本保存车辆位置，保持已承诺货流或显式 hub/在途库存源项。
- **Status**：V0 缺口已确认；rolling 终端政策待作者确认，阻断 V1-F。

### R5 [Critical] CFS 重量单位不能直接变成 stops — REJECT（“现有 CFS 已校准 V1 stops”）

- **Evidence**：V1:69 每单位需求是一处 pickup 与一处 drop-off；intercity_delivery/data/cfs_processor.py:528–534 使用 SHIPMT_WGHT/pounds_per_model_unit 后取整截断。
- **Impact**：BHH 依赖 stops，而容量可能依赖 volume/weight；重量归一化不是 stop-count 数据证据。沿用旧数据并改字段名称会改变研究解释且不可复现。
- **Remedy**：本阶段合同明确禁止此重标。后续先合成 stops 验证；真实数据需明确 stop 映射来源或经批准增加 stops/volume 双维模型。
- **Status**：数据语义差异已确认；真实 stop 校准未提供。

## 3. 主要数学与集成发现

### R6 [Major] 稳态 n≤W 未进入优化问题/负载式 — MAJOR REVISION

- **Evidence**：steady_state_model.tex:89 要 n_i≤min(M,W)，:132 只写 n_i≤M，:138–139 仅截 M。Theorist 的手工例 a=b=c=ρ=ω=1、δ=0、W=1/16、M≥2，s=5，n1=√(2/5)>W；可取足够大 L 使 deadline 不约束。S-linear:107 至少标有“provided these do not exceed W”。
- **Impact**：负载唯一性及 KKT 闭式只对不缺 W-cap 的规定子问题有效；稀疏域可得少于一辆的“分区车”。由该消元得到的全域凸性、regimes 与阈值不得自动继承。
- **Remedy**：决定限制大波适用域，或纳入 W-cap（固定 W 负载应同时截 M 与 W）后重新审查 W-dependent active set、feasibility 与所有后续结论；不能只改一句文字便视为全域证明完成。
- **Status**：遗漏与手工反例已复核；完整重推尚未执行，不能登记为全域已证。

### R7 [Major] deadline active、正乘子和阈值存在须分开 — MAJOR REVISION

- **Evidence**：steady_state_model.tex:145 将内点比率达到上界与 deadline slack 作 iff，:205 写 deadline binds iff Λ<Λ°。:234 使用较大根 g'_M(W+)>0 的隐函数。
- **Impact**：ω>0 且未触 cap 时，取 L 恰为 cost-bound 解的 Tmax，则 deadline active、λ=0，比率仍为上界；所以 active 不等于正 multiplier。Λ 临界等号的文字也应区分 inactive/active/λ>0。很小 L 可能根本无可行解；full-load 双根合并时 g'=0，不能套该隐函数结论。
- **Remedy**：用“λ=0”陈述代数等价，明确非退化/严格互补的附加条件；给可行性条件与临界等号约定，限定 simple-root comparative statics。不得将这些边界当成已验定理。
- **Status**：等号反例为静态推理；完整 threshold/feasibility theorem 待证明复审。

### R8 [Major] start/ext 弧域、取整与 completion 上界不统一 — MAJOR REVISION

- **Evidence**：main_v2_BHH.tex:234–248 用 H−1、H+L−1，Algorithm 输入:262–263 却含 H、H+L。:429–443 把已定义为期数的 τ 与物理 f 相加后统一除 t0，并缺 stem、分腿取整。full_window_mip.py:45–61 限各腿起点<start_end；state_dependent_mip.py:153–181 允许下游起点至 completion_end，再在:330–345 仅限制 origin pickup。
- **Impact**：现有 full/reduced 并非同一母弧域上的纯剪枝，reduced 可能含 full 未含的下游弧。强行比较“删弧比例/最优差距”会混入模型差异。只限定各腿 driving time 也不能界定任意 hub 等待后的完工时间；“存在可完工路径”与“任意 option 均能完工”不同。
- **Remedy**：明确半开区间、只限制首次 pickup 还是限制所有腿；用同一母弧域/相同 history ledger 比较。completion 命题须按新物理时间、逐腿取整和等待/可用车假设重新陈述证明。空容量 arc 不可作为载货最快路径。
- **Status**：源文及代码差异确认；合法统一规格待作者决定，bound 未证明。

### R9 [Major] V0 不等于旧 TeX，也没有唯一 V0 baseline — ACCEPT（差异 claim）；MAJOR REVISION（统一比较方案）

- **Evidence**：旧 main_BHH.tex:103 和新 V1:114 都是 AV 货量求和；base_optimizer.py:184–192 实为逐订单容量。flexible_direct_optimizer.py:352–357 已是总量容量。base:68–69 z 是 INTEGER，而 V1:134 连续；flex z/r/q 连续。flex:228–269 已有累计 HV 迁移，不能说这是 V1 才首次在代码出现。
- **Impact**：某些 V0 变体本来就有不同可行域；不分支命名会把修正共享容量、stem、wave、回空和变量域等多种效果混称 BHH 规模收益。
- **Remedy**：保留 V0-M/V0-F/V0-RH 标签与独立 fixtures；论文/合同逐项列差异。精确 V1 模式需中性化 θ/transfer fee/direct ratio/独立 direct capacity 等未入正文的扩展。
- **Status**：关键源码位置复核通过；V0 历史数值未审定。

### R10 [Major] S-current 与 V1 的验证桥梁不存在自动等价 — REJECT（现成等价/验证 claim）

- **Evidence**：V1:84、:145–146 为车辆成本+拒单惩罚，固定车队、整数车辆、离散有限时域；steady_state_model.tex:57–89、:119–132 为对称常率需求、可配车队、连续批量、ω lead-time、sorting、δ 和 common timetable。:360 自己声明 program 未包含这些机制。
- **Impact**：仅共同 local tour 形式不能证明相同优化问题。steady 平均 busy fleet:244 不是峰值可行车队；chatτ/min(W,Mhat) 在 W>Mhat 是连续车数近似，不是 ceil(W/Mhat) 的整数成本。
- **Remedy**：先建立受控共同子模型，统一 served/offered 归一化、需求拒绝、deadline 的交付时点、处理/排序/同步、峰值fleet、时间与整数近似、warmup/terminal，再允许报告 stationary 对照；S-linear 和 S-current 单列。
- **Status**：结构差异确认；桥梁尚待设计与批准。本轮没有验证稳态数值命题。

### R11 [Major] Poisson 的期望不能变成硬最长等待界 — REJECT（随机硬界 claim）

- **Evidence**：steady_state_model.tex:94–101 固定常率/nested lemma 后仅把 averages 扩到 Poisson expectation；:120 则用确定性 W/Λ 构造 Tmax。
- **Impact**：Poisson 填满有限批次的等待有无界尾部，任何有限确定上界均不能由均值替代获得。BHH 有限小样本拟合和常数 δ 也不能独立给每单路径保证。
- **Remedy**：保留确定性常率定理范围；随机模型另定义 almost-sure、chance constraint 或平均/分位数服务标准，并给对应证明或数值误差证据。
- **Status**：期望/路径界区别确认；不指控当前 lemma 已将 Poisson 的最大值断言为定理，但禁止迁移时作此延伸。

### R12 [Major] 滚动输出的 partial cost 与 window gap 不能作全域证书 — MAJOR REVISION

- **Evidence**：paper_rolling_horizon.py:148–151 总预算耗尽可提前退出，:238–248 记录 completed，:265 返回 last_solution.objective；experiments/solvers.py:395–406 将最后窗口 best_bound/mip_gap 放进统一 result。
- **Impact**：未完成时订单可能未揭示，低 partial cost 不代表算法优越；last-window gap 证明的是该窗口 MILP，不能证明在线策略/全日全域最优或提供全日 lower bound。完成时的最后目标可能包含历史，但仍需独立决策账本核算才能确认口径，不能一律宣称必错或必对。
- **Remedy**：独立重算已执行成本及到期罚损；显式标 completed_horizon、cost_scope、bound_scope、incumbent feasibility、信息集；全域 reference 只来自同一数学/数据/信息条件下的有效下界。
- **Status**：字段路径确认；历史结果是否受影响须后续隔离审计，不能覆盖重写历史结果。

## 4. 文献、QVI 与证据等级

### R13 [Major] 文献存在性可接受，novelty/数值复现仍未通过 — ACCEPT（有界文献审计）

- **Evidence**：四 TeX 未发现 input/include，主要 theorem/proof 定义在文件内；main 两份:7 的 导出的条目.bib 在可读文件清单未找到。steady:358 的 calibration 与符号/校验脚本未定位。Referee 另抽查公开 BHH、Newell 1971、Belak/Christensen/Seifried 2017 权威元数据/摘要；没有逐篇完整重做 Literature 的全部检索。
- **Impact**：不能认证本地 cite key/Bib 条目准确、稿中表格已复现或研究首创。BHH 渐近结果不自动证明 stem、有限 stops、均衡分区、同时回库与 deadline 保证。
- **Remedy**：补正式 Bib 及脚本、参数、seed、routing certificates；取得具体篇页/公式后再支持强声明。Roundy 1986 的作者意图、教材 −1/3 具体出处保留 unresolved。
- **Status**：关键存在性/主题抽查通过；全文支持、穷尽 novelty 与数值复现不在本轮完成范围。

公开来源（访问日 2026-10-17）：[BHH 原始出版社页面](https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/shortest-path-through-many-points/F1C28B5730B94887F4659FCBF8A1F2BB)；[Newell 1971 出版社页面](https://pubsonline.informs.org/doi/10.1287/trsc.5.1.91)支持其确定性非恒定到达率下 dispatch-time/等待问题已存在；[2017 verification 出版社检索元数据](https://epubs.siam.org/doi/abs/10.1137/16M1082822)与[作者原始成果目录](https://sites.google.com/site/soerenchristensen01/publications/papers-mathematics)支持该方法文献存在。SIAM 页面直接打开错误，未以该失败声称读到全文。

### R14 [Major] V2 只有限定 renewal 代数与形式 QVI — ACCEPT（限定结果）；REJECT（完整随机/全网络最优 claim）

- **Evidence**：research/theory/Common Wave 的内生化与非均匀需求扩展.md:246–361 给 complete-cycle 加权平均；:463–515 的 min{ωq−g+ΛV′, C(q)+V(0)−V(q)}=0；:652–709 使用确定性 λ(t)Vq，:713 明以“如果 threshold structure”限定。未定义随机到达的概率空间、marks、filtration、verification/终端条件。
- **Impact**：对 common-wave、renewal-separable、常率确定性流体的 complete-cycle 平均恒等式可接受。推广到任意日历平均需时间趋无穷与余周期控制；fleet/pipeline/sorting、期限age结构破坏一维 reset 的闭合性。形式 QVI 符号与成本最小化相容，但不是存在、阈值或最优验证定理。
- **Remedy**：另立 V2 状态/信息/控制/成本/边界规范；规定 q=0/no-Zeno、partial dispatch、剩余任务、terminal/transversality。若单位 Poisson，生成元应为 λ(t)[V(t,q+1)−V(t,q)]，与确定性导数不同；是否采用 Poisson 须作者决定。
- **Status**：限定周期代数与形式区别复核通过；随机 QVI、三段同步与 full-network optimality 未证明，不纳入本次代码迁移成果。

### R15 [Moderate] 可复现性与批准门槛 — ACCEPT（审计保护方案）

- **Evidence**：三份报告只写其授权报告，Experiment 未声称运行测试；当前 git status 的研究/实验目录有大量 untracked。计划明确隔离 V0/V1、保存源文件散列/seed/solver 状态并另申请微型求解预算。
- **Impact**：HEAD 不能指代 untracked TeX/笔记/报告；“有更多文字”不能登记为数值验证或理论突破。
- **Remedy**：主协调者更新合同及 CLAIMS 时将 static-confirmed、conditional algebra、proof pending、experiment not run、author decision 分开；源文件以散列/版本快照登记。后续实施保持 V0 的入口与历史结果，批准后再分层执行测试。
- **Status**：审计报告可接收；未批准代码迁移、求解预算或网络发布。

## 5. 接受的独立代数核验与更正记录

1. 对 h=f^{-1}，h 凸且 h(0)=0，h(vD)/v 非减，故 K(v)/v=min{M,h(vD)/v} 非减，并在枚举域允许 v1+v2 时给 K(v1+v2)≥K(v1)+K(v2)。线性 cap 后 K 本身不必凸；b=0/D=0/饱和都不支持无条件严格超线性。与无 stem 的旧容量没有统一大小关系。ACCEPT（模型内代数）。
2. 将两城 AV 累计库存式相加得到总在途不超过初始总数；闭合终端与两城已服务量相等时，累计货流式夹逼得到 linehaul 总量相等。该加强式若放入带初始 hub/在途存货的窗口，须重新写源项。ACCEPT（限定终端代数）。
3. direct 正反总车次相等配合所有弧在 T 完成，恢复各城 HV 总数；不要求货量平衡或逐车身份原路返程。ACCEPT（限定全域终端代数）。
4. **更正已撤回的潜在误报**：审查过程中一度怀疑稳态 cost-bound 允许 b=0 导致 2/3 失效；复读 steady_state_model.tex:66 明定 b>0、ρ>0，:158 明定 ω>0，故这不是该稿既定域下的反例，未列为缺陷。V1-main 则 a>0、b≥0；两个模型的参数域不可混用。该更正已向主协调者反馈。

## 6. 迁移前的阻断项与交付判定

阻断实际实施的最高优先事项是 R1 的 wave/枚举域，R4 的空返与全日终端恢复，R5 的 stops/重量语义，R8 的 start/ext 与时间单位规格，以及 R2/R3 的新容量和安全候选网络。缺少这些作者决策时，只能形成显式条件化方案，不能声称“精确实现最新模型”。

稳态证明修订 R6/R7、S↔V1 桥梁 R10 和 V2 的 R14 可作为独立后续研究范围；它们不必被悄悄塞进 V1 迁移，但会阻断相应稳态验证/QVI 成果声明。已有专员建议顺序与隔离 V0 的方案可用，最终协调文档须保留这些门槛。

**最终建议：三份静态审计 ACCEPT；源模型与算法对齐 MAJOR REVISION；现阶段“已完成迁移/验证/随机理论”之类结论 REJECT。等待用户确认规格并批准实际代码迁移。**
