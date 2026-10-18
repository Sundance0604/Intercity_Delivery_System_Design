# QVI–Wave 一致性与论文规划独立审查

2026-10-18。Referee。审查在三个专员交付后进行；只写本审查文件，未改理论、论文、生产代码或结果，未编译 TeX，未运行求解器。本报告不把 PI 与专员对同一脚本的重复执行算成新案例。

## 1. 决定与证据范围

| 对象 | 决定 | 允许使用的结论 |
|---|---|---|
| `research/theory/qvi_wave_equivalence.md` §§2–5 | **ACCEPT，限明确的 T0 约化模型** | 确定性常率、全量 reset、逐批可分成本下，静态最优 wave 与最优重复阈值集合相同；任意 admissible successive waves 不能降低全日历长期平均成本；tail-inf 包络给出所陈述的几乎处处 QVI 与 verification |
| 同报告 §6 | **ACCEPT，限整数阈值策略类；一般随机控制结论不接受为已证** | 单位 Poisson、同一整数动作域及同一 C 下，阈值目标只差常数 ω/2，故 argmin 集合相同；不能推出所有 adapted 策略全局最优、连续 fluid 动作相同或最优值相同 |
| `research/toy_models/qvi_wave_consistency_check.py`、`research/theory/qvi_wave_numerical_check.md`、保存 raw JSON | **ACCEPT，限 toy 数值核验** | 三个同参、同域的独立稳态优化/Bellman 对照及边界检查通过；不是连续 QVI 收敛证明或运输全模型实验 |
| `research/literature/paper_integration_plan.md` | **ACCEPT，限写作规划及明确限定的背景定位** | current 为主稿、linear 为简化基线，建议主文放短桥梁命题、附录放 verification；不认证理论创新性或完成了穷尽文献检索 |
| 两份现存 steady 稿作为可直接投稿的理论正文 | **MAJOR REVISION** | 须修可行域、边界、同步与算法对应关系；本轮没有实际修改 TeX，不能称稿件已修好 |
| “QVI、完整 steady、V1-F/RH、CFS 全部等价” | **REJECT** | 目标、状态、政策类、交付时刻和可行域不一致，无这样的证明或实验 |

证据由本地两份 steady 源码、原始 Common Wave 笔记、MODEL_CONTRACT 与上述新交付组成。独立逐式审查和脚本代码检查由本 Referee 完成；实际运行日志由专员/PI 提供，本 Referee 本轮没有重新执行。文献部分只审查交付的引用范围、论断强度与来源记录，没有冒称自己重新联网检查全部条目或阅读全文。

## 2. 数学核心核查

1. **可行域及成本没有被偷偷替换。** T0 保留 `0<n_i≤min(M,q)`，从原始目标重求负荷；不是照抄稿中遗漏 `n≤W` 的简式。取 `R(q)=min_n G(q,n)−ωq/(2Λ)`、`C(q)=qR(q)`，dispatch 前等待仅在 running cost 记一次，dispatch 后 lead time 仍保留在 C。C 的预付会计、无有限资源争用及可分性均写成假设，未把 backlog reset 当成真实车辆归位。
2. **周期恒等式及存在性成立。** `J(q)=ΛC(q)/q+ωq/2=ΛG°(q)`。正的 c、ρ、ω、Λ 及 `n_i≤q` 给出 `C(q)≥4cρ`，故 J 在 0 与 ∞ 发散，达到最小值 g；无须先继承未复核的全域凸性或唯一性命题。两城总成本率应另乘 2，argmin 不变。
3. **日历余项补齐了原笔记的缺口。** 完成周期 excess 非负，余周期项为 `ωΛr²/2−gr≥−g²/(2ωΛ)`，因此任意 admissible 策略的完整日历 liminf ≥g。重复任一 minimizer 达到 g。该界不依赖预设残余周期有界，非 Zeno 条件也已显式规定。平均最优策略可有有限 transient 或混合 minimizer，不能说每次 dispatch 唯一相同。
4. **QVI 符号、全域包络与互补关系成立。** 对最小化问题，两项为 `ωq−g+ΛV'≥0` 和 `C−V≥0`，min 为零。定义 `F=C+ωq²/(2Λ)−gq/Λ`、`m(q)=inf_{z≥q}F(z)`，则 `V=(gq−ωq²/2)/Λ+m`；m 非降，严格非接触区域局部常数，故几乎处处满足 QVI。报告没有错误地把 `V=C` 延拓到所有高于首次阈值的状态。
5. **横截说明充分，且未默许任意状态光滑。** C 非负、增长至多线性；V 有常数下界和线性上界；有限 liminf 的子序列上，已付 waiting 使 `Q(T)=O(√T)`，所以 `V(Q(T))/T→0`。这足以支持报告的确定性 verification。AV kink 不能强制经典 smooth fit；其他远处 valley、多个最优 wave 和 q=0 无空干预均有区分。
6. **Poisson 结论范围正确。** 对整数全量 reset 阈值 B，期望周期长 B/Λ，等待成本率 `ω(B−1)/2`，故 `J_P(B)=J_D(B)−ω/2`。同整数域 argmin 相同是精确特殊结果，不应写一般随机性必然改变阈值。C=9/8、Λ=ω=1 的连续域反例 q*=3/2、整数 B*=2 及对应值正确。离散 QVI 不替代 stopped Dynkin 的可积性/横截证明，报告将 all-adapted 最优性列为 conditional，处理恰当。

## 3. 发现、影响与补救

| 严重度 | 证据/位置 | 影响 | 补救 | 状态 |
|---|---|---|---|---|
| HIGH | `steady_state_model.tex:89,132,138`；linear 的 `n≤W` 与后续简式 | 显示优化问题和负荷公式可能允许单车负荷超过整批；后续全域命题不能直接照搬 | 正式稿完整写入 `n≤min(M,W)` 并重审边界分支；新 T0 已保留 | 新理论已规避；TeX 仍待修，**MAJOR REVISION** |
| HIGH | current:103–111、359；`main_v2_BHH.tex` 目标与 V1 实施合同 | finite sorting 下同步最优、完整 V1 与稳态/QVI 相互验证都缺条件 | 明示 common-timetable 政策类；分离平均途中交付与 tour-end、固定车队与按负荷配车、waiting 与拒单罚损；不把 RH 当 QVI 离散化 | 交付规划已准确列明；原稿待修 |
| MEDIUM | 新理论 §2、§6 | 预付会计不等于真实在途费用；Poisson 阈值计算不等于所有随机控制的 verification | 进入论文时保留可分性及会计边界；随机 all-adapted 结论须另给概率空间、可积性、横截与 admissibility | **OPEN/CONDITIONAL**，不得升级 |
| LOW | 新理论 §5 原“F 连续、局部 Lipschitz” | F 含 √q 项，不应含混宣称在 0 邻域 Lipschitz；0 态正则性需要单独说明 | 明确 F 在 `(0,∞)` 局部 Lipschitz，以有限右极限延拓；由于首个最优 b>0，m 在 `[0,b]` 恒为 0，V 在 0 的右导数为 g/Λ，且 0 禁止 dispatch | **RESOLVED**；已定向读回第119行确认原 owner 修订 |
| LOW | 新理论 §6 末原“Poisson 最老订单等待无确定性上界” | B=1 立即发运，没有凑批等待，是该笼统句的反例 | 限定非平凡凑批阈值 B>1；逐单交付期限还须另核 service time/其他随机机制 | **RESOLVED**；已定向读回第201行确认 B>1 与 B=1 区分 |
| MEDIUM | current 现存数值表及未定位的原检查脚本；CFS 八单 smoke | 稿内旧数值与新 toy 证据、生产实现证据若混用会夸大进展 | 原示例待归档复现；toy 仅核桥梁；CFS 仅核可运行；旧 V0 大实验不能直接换 V1 标签 | 规划处理正确，正式实验章节尚未完成 |
| MEDIUM | 新 literature 规划 §8，五条背景来源 | 条目存在/相关不等于公式可直接套用，更不认证创新 | 保留出版社/作者机构来源、支持范围与访问日期；正式写作补齐本地 Bib、精确前史与竞争工作检索 | **ACCEPT 背景记录范围**；完整创新性审查 **OPEN** |

## 4. 数值证据核查

脚本有两条实际不同路径：原始 G 用 scipy 内层优化负荷，外层分段搜索保留容量 kink；动态路径用解析截断负荷形成服务费后，做反向 Bellman 加 average-rate 二分归一化。动态求解没有调用静态 argmin，也没把静态最优阈值传进去。两者共用物理参数与成本定义是应有的对齐，不能误称两次同一 argmin 的自证。

已读保存的 `research/toy_models/qvi_wave_consistency_results.json`，其来源 SHA、参数、三网格和 160/320 边界记录与报告吻合。三例稳态 q 为约 1.97351547、8、61.30475281；细网格 dispatch 为 1.95、8、61.3。阈值误差小于一个 .05 网格，最大成本率差约 .0001384。八探针的成本路径误差约 5.69e−14。离散 residual 为零是有限递推的内部核验，不是连续 QVI 的全域求解证书。

同整数域 fluid/Poisson 皆选 B=20，成本率差 .5=ω/2；这与解析恒等式一致。遗漏 holding 的负例返回 80、160，明确是随截断边界移动的解，报告未将其冒充无界有限最优点。上边界扩大只证这几个案例的边界稳定，不证所有参数无截断误差。没有随机抽样，因而不需要种子置信区间，也没有蒙特卡洛显著性可报告。

## 5. 论文整合判断

current 的注释明确记录 2026-10-01/02 扩展及 earlier linear 稿，因此 current 应作为主要机制稿，linear 更适合作为基准/解析参照。“较新”不代表其所有命题已正确；上表缺口仍须修。

接受把确定性一致性结果作为论文内部桥梁：主文讲清同一 G、R、C、政策类及 `J=ΛG°`，说明允许逐次改变 wave 并不改善 T0 的长期平均成本；附录给全日历余项、tail 包络、边界与 verification。特殊 Poisson 整数阈值常数差可作有限边界说明。该结果不要求把整篇转成随机控制论文。

该规划已有可验证知识：自包含限定证明、独立数值对照、原稿问题清单与明确写作位置。尚未完成的是原 TeX 修订、完整随机 QVI、finite sorting 同步最优、统一目标下的有限车队桥接实验及正式性能章节。不得用新增文本页数或同一脚本重复运行代替这些交付。

**最终范围：T0 理论核心、toy 数值核验、论文规划均为限定 ACCEPT；原稿及任何完整模型等价/创新性声明仍按上述 MAJOR REVISION / REJECT / OPEN 处理。两处 LOW 措辞已由原 owner 修订，本 Referee 已读回确认。**
