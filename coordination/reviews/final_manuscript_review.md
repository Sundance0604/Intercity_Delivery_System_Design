# 新终稿独立审查

日期：2026-10-18（Asia/Shanghai）。审查人：独立 Referee。

审查在 PI 明确告知 Literature、Theorist、Experiment 的稿件已交付并完成装配后开展。审查对象为 `research/paper/final_manuscript/main.tex`、七个 section 源文件及 `references.bib`。本 Referee 只写本报告，未改论文、证明、代码或结果；未编译 TeX、未运行求解器、未开展数值实验，也未撰写实验计划。

## 1. 结论与范围

**本次用户要求的引言、文献综述、有限时域优化模型与数学推理正文：限定 ACCEPT。** 文献前史的一项实质遗漏已由原 owner 修订并读回关闭；没有尚未修复的阻断性数学或模型对应错误。

这不是完整投稿流程通过的结论。**全篇作为含实证验证、创新性定位和出版排版的可投稿论文，仍为 MAJOR REVISION / 尚待完成**：数值实验及其计划依用户要求未写，本文未认证算法性能、CFS 泛化、全面文献覆盖或理论首创性，亦未认证编译或排版成功。后续补入这些材料须按最终目标、信息集与数量单位重新审查，不能直接移植历史 V0 结果。

| 交付或主张 | 决定 | 被接受的精确范围 |
|---|---|---|
| `sections/introduction.tex`、摘要及 discussion | ACCEPT | 三个数学对象及其差异明确；没有声称完整 MILP、RH、稳态与 QVI 相等，没有虚构实验或自动化成本优势 |
| `sections/related_work.tex`、`references.bib` | ACCEPT，限所述背景和边界 | 有真实可定位来源；补入直接 shipment-consolidation/renewal 前史；不接受由此推断穷尽检索或理论首创 |
| `sections/optimization_model.tex` | ACCEPT，限明确的零 transfer-delay/fee、无 direct-share 约束核心 | 容量表、selector、需求、先后、车队及独立空车变量构成一致 MILP；目标和终点政策与采用的 runtime 核心相符 |
| `sections/solution_method.tex` | ACCEPT，限实现说明 | history、cutoff、货流稀疏规则及 certificate scope 对应当前源码；没有一般递归可行性、无损剪枝或加速定理 |
| local capacity、conditional loads 命题与附录 | ACCEPT，限采用的聚合 routing 近似 | 保留 wave-load 上界；超可加不被错误提升为凸性或严格性；非凸反例正确 |
| static/dynamic common-wave theorem | ACCEPT，限 T0 假设 | 同一成本和可行域内，任意 admissible successive waves 不能改善限定长期平均成本；不覆盖完整运输网络 |
| global QVI theorem 及附录 | ACCEPT，限确定性一维 reset、a.e. QVI 与所述横截条件 | tail envelope、0 态边界、calendar remainder 和 verification 自包含；没有宣称一般 viscosity comparison 或唯一性 |
| unit-Poisson 命题 | ACCEPT，限同一正整数全量阈值域 | 成本率相差常数 `omega/2`、argmin 集合相同；不覆盖一般 adapted stochastic control |
| 一般随机、多阶段、有限车队 QVI 等价或剪枝无损 | REJECT，若被作为已证主张提出 | 当前稿已明确不作这些主张 |
| 全文可立即投稿、已完成性能验证或 TeX 已编译通过 | 不接受；MAJOR REVISION / 尚待相应证据 | 实验、创新性全面定位及出版排版未在本轮完成 |

## 2. 按严重度列示的发现

| 严重度 | 证据 / 文件位置 | 影响 | 补救 | 状态 |
|---|---|---|---|---|
| MEDIUM | `sections/related_work.tex`，Dynamic dispatch subsection 的初版 | 直接的 freight consolidation / renewal quantity-dispatch 文献缺位，容易让一般 renewal 比值被误读为本文的新 dispatch 原理 | 原 owner 已增加 Çetinkaya–Bookbinder (2003) 的段落与 Bib 条目，说明 quantity/time policies、renewal ratio 前史，限定本文到 BHH-dependent service cost 与所证关系 | **RESOLVED**；Referee 已读回新段及新 Bib |
| MEDIUM，范围限制 | `main.tex`、全稿目前无 numerical experiments section | 尚无最终模型下的性能、消融、实证或统计证据；历史 smoke 与旧大实验不能替代 | 按用户指示本轮不写实验或计划；后续须依此稿最终目标/单位/信息集制定并执行验证 | **DEFERRED BY USER**；不阻断本轮正文交付，不计为已经完成 |
| MEDIUM，范围限制 | `sections/related_work.tex` 与十二条背景来源 | 有限来源存在及相关，不等于 exhaustive novelty audit；不能由本次文本认定首创性或没有竞争方案 | 保留当前克制表述；全面创新性定位、近期竞争模型和投稿目标匹配留待专门文献审查 | **OPEN**；稿件没有使用“首个”等排他性主张 |
| LOW，整合事项 | introduction roadmap；model 的 order-set、物理时间符号及 method 路标 | 未定义集合或章节顺序/记号漂移会影响可读性，旧 roadmap 还可能误述核心包含独立 transfer fee | 原 owner 已明确 `L=union_d L^d`、统一 `tau_phys`、改为显式 method 引用并按最终章节顺序写 roadmap | **RESOLVED**；已定向读回，未改数学内容 |
| LOW，验证边界 | `main.tex`、全部输入与 Bib | 静态可解析不意味着编译、引用排序或版式已通过 | 本轮依用户“不编译”要求只做静态检查；后续实际编译与视觉检查再认证 | **NOT PERFORMED**；没有冒称编译成功 |

没有发现需以新证明或大规模求解来弥补的隐藏 HIGH / CRITICAL 问题；这句话仅适用于下述已读版本与明确限定范围。

## 3. 模型与实现的独立交叉核查

除四份原论文来源与 MODEL_CONTRACT、上一轮 QVI review 外，逐式对照了：
`intercity_delivery/v1/flexible_optimizer.py`（包括 `configure_window`）、
`flexible_rolling.py`、`pruning.py`，并核对已采用的 runtime 合同。没有执行这些求解流程。

- local 表对应 `min(v M, f_inverse(v [duration-2 rho]^+))`；direct 源式的两个系数为 `(a_origin+a_dest)/v` 与 `b_origin/v+b_dest/sqrt(v)`。论文保留方向差异及 tau 长度零货 convention，没有将其错误改成对称 local 公式。
- selector 上界用全人工车队 N，而城市动态库存限制实际可用车辆；这避免把初始城市车辆数误当所有未来 local wave 的上界。人工库存计入 direct/empty 的累计出发与到达以及 local 半开占用；AV 城市库存与 implied busy bound 一致。
- 独立 `e` 不依赖订单；`w` 不被误称为强制载货，零货移动仍可行。默认真实 T 无固定城市目标，所有弧仍在真实 T 前完成；可选 shutdown stock 不被复制到每个 rolling cut。
- 起点 release、终点 deadline、逐单累计 pickup-before-AV、AV-before-destination 与两个 local 需求等式在 T 共同强制完整链数量一致。它们不允许把已 pickup 的货物仅通过增加拒单变量而消失。direct 与 transshipment 是需求的不同部分。
- 车小时费率只换算一次；数学中的分钟成本与 `tau_phys=tau Delta t` 相符。有限模型明确采用 tour-end delivery，稳态明确采用 mean within-tour delivery；lead-time 目标也没有偷换成 finite-horizon vehicle/rejection 目标。
- 新任务起止规则、历史 start<current 的值及隐式零固定、全任务跨窗保留、z 不执行、release 揭示、总预算与末窗 bound 的局部范围对应源码。文中诚实保留潜在后窗不可行，不宣称 recursive feasibility。
- cargo filter 保留正历史 pickup/AV 对 downstream 的支持，且保留 vehicle/selector 母域。它使用乐观时间/容量支持，没有用“当前城市空闲为零”误删未来迁入服务。文中不把筛选通过当成联合可行，也不宣称一般保最优或性能结果。
- 当前模型核心有意取零 transfer-delay/fee、无额外 direct-share bounds；runtime 中这些可配置扩展已被文字明确隔开，没有把未展示的参数悄悄算入数学核心。

## 4. 理论核查

1. `n_i<=min(M,q)` 保留在可行域和条件最优负荷的 clipping 中。两个负荷公式从独立的一阶条件取得，不依赖旧 steady 稿遗漏 q 上界的简式。
2. 容量比 `K(v)/v` 单调推出超可加；附录给 a=b=B=1、M=1/2 的整数二阶差分反例，正确阻止无根据的 convexity / strictness 升级。
3. C 除去的仅是 dispatch 前 holding，服务中的 lead time 留在 C；费用预付与逐批可分作为假设，backlog reset 不等于车辆归位或真实 pipeline 消失。
4. C 的正下界及至多线性上界、J 在零与无穷处发散，使最小值存在且 minimizer set 非空紧。calendar remainder 有统一有限下界，故不是只在完整周期末比较。
5. tail-inf F 构造使 V 在全正状态的 a.e. QVI 互补成立。F 只在正开区间局部 Lipschitz；m 在首个 minimizer 前为零，0 态右导数与禁空干预边界明确。value matching 不被替代为所有高状态立即发车；kink 用一侧条件，不强加经典 smooth fit。
6. 有限平均成本子序列上，末 backlog=O(sqrt(T)) 与 V 至多线性共同验证横截余项。此处没有假设一般随机 Dynkin verification 已经成立。
7. Poisson 计算使用状态 0 至 B-1 的平均停留，各周期期望时长 B/Lambda。相同整数域的常数差与不同连续域的反例数值正确；B=1 无凑批等待、B>1 无确定最大凑批时间的区分保留。

这些是本稿限定证明的检查结论，不是 BHH 有限样本误差保证、finite-sorting 同步定理或 full V1-F/QVI 等价证明。

## 5. 文献与静态源检查

本 Referee 独立复核若干关键 publisher/author-institution 来源及其支持范围，包括 [Daganzo 1984](https://pubsonline.informs.org/doi/10.1287/trsc.18.4.331)、[Karp 1977](https://pubsonline.informs.org/doi/10.1287/moor.2.3.209)、[Belak et al. 2017](https://epubs.siam.org/doi/10.1137/16M1082822)、[Çetinkaya–Bookbinder 2003](https://www.sciencedirect.com/science/article/pii/S0191261502000607)，以及作者机构的 Bertsimas routing 来源。部分 DOI 直接打开失败，已用可得的 publisher search / author-institution metadata 交叉核对；没有声称读取了十二篇全文。其余条目与交付中的来源定位及基本元数据对照，未发现不存在的引用。文献论述没有把所引一般定理自动套给当前模型。该检查不提供穷尽新颖性认证。

对冻结源码做了纯文本静态检查：**8 个 TeX、12 个 Bib 条目、68 个 label、57 个 ref/eqref、12 个实际被引用的不同 Bib key；缺失 input、未定义 ref/cite、重复 label/Bib key、括号或环境栈问题均为 0。** 没有调用 LaTeX/BibTeX 编译工具。检查支持源文件装配完整，不支持 PDF 排版结论。

## 6. 冻结审查版本

以下路径均相对 `research/paper/final_manuscript/`。任何实质改动须重新检查相应结论；此 SHA 表不包含本报告本身或之后 PI 更新的项目账本。

| 文件 | SHA-256 |
|---|---|
| `main.tex` | `D29987839A0DD44AC2A6A6D3F0817E758648D0DD396A760AB5B60946F920B1B8` |
| `references.bib` | `2457CFE8944D86142B2196154B2553BDB7CB588230BCEC3825D04C7CE6790830` |
| `sections/discussion.tex` | `E37C78BB356406C9C345F78F9563868C101BF38C716F4F2D007067C49EAB4002` |
| `sections/introduction.tex` | `064719D15F65C13CE9936A500CAC115B2F97F732F450B964587D0EAFBCD50609` |
| `sections/mathematical_analysis.tex` | `022F6EA4A152745F146673535A140D2C2915714B6EF4731EF3A079F41C61D023` |
| `sections/optimization_model.tex` | `F64D1DD5EBB41A610BFB30057C18556E7272B95F0EE704B4E8C87F5185A2DDA5` |
| `sections/proofs.tex` | `2222C1E2BBAA5654BB0BE00D91429EAACDA22AB6B1E00792E79E0813AC4253D7` |
| `sections/related_work.tex` | `E1642396B31E3DB6B27F1450FB3789CE956F29C17C90A20E03E0CE2A368C507A` |
| `sections/solution_method.tex` | `D49E449D7D810F5844F2A8CDF6F866804F5A8BA19E05A48B7A5C56E2DE2EDDB9` |

