# 论文写作与 QVI–Wave 一致性：主协调者规划

后续执行状态（2026-10-18）：用户已授权新建终稿，入口为[final_manuscript/main.tex](../research/paper/final_manuscript/main.tex)。本计划保留研究与写作依赖记录；新稿已完成当前模型/理论正文，数值实验章节及实验计划仍按用户要求延期，原TeX未覆盖。

日期：2026-10-18。对象：BHH / Intercity 现有论文。范围：研究、轻量独立验证与写作规划；本轮不修改或编译 TeX，不改生产算法，不运行 Gurobi 或大规模实验，不提交推送 Git。

## 1. 当前判断

**以 `steady_state_model.tex` 为主稳态稿，以 `steady_state_model_v1_linear.tex` 为简化对照。QVI 在明确的共同子模型下可以恢复同一个最优 Wave；这一结果适合成为论文的短桥梁命题与附录证明。它没有建立完整稳态稿、V1-F 或 rolling horizon 与随机 QVI 的等价。**

current 不是仅因名字或mtime较新：两稿封面同为2026-09-30，但current:27–41包含10月1日/2日的后续决定，并称linear为earlier draft。current新增同日达deadline、sorting和慢车因素，研究问题更完整；linear适合一致性基准。current中的公式域、乘子边界和同步结论仍有需修问题，不能把“较新”当作“全稿已数学验收”。

本轮实际产物：

| 产物 | 位置 | 责任与证据 |
|---|---|---|
| 自包含一致性证明 | [qvi_wave_equivalence.md](../research/theory/qvi_wave_equivalence.md) | Theorist，原式、日历平均、全域QVI验证及Poisson边界 |
| 独立toy脚本 | [qvi_wave_consistency_check.py](../research/toy_models/qvi_wave_consistency_check.py) | Checks，用原式稳态优化对比独立Bellman递推 |
| 数值方法与完整stdout | [qvi_wave_numerical_check.md](../research/theory/qvi_wave_numerical_check.md) | 真实执行、误差、网格与边界检查，非生产实验 |
| PI复跑结构结果 | [qvi_wave_consistency_results.json](../research/toy_models/qvi_wave_consistency_results.json) | exit0；Python/NumPy/SciPy及来源SHA；参数与全部结果 |
| 逐章写作审计及整合方案 | [paper_integration_plan.md](../research/literature/paper_integration_plan.md) | Literature，版本/行号、缺口、章节、英文段落与5条primary文献 |
| 独立审查 | [qvi_wave_paper_review.md](reviews/qvi_wave_paper_review.md) | Referee，在三专员交付后审查；具体结论以报告为准 |

## 2. “同等参数、相同解”实际需要对齐什么

比较基准称为 **T0**：恒定确定性到达率Λ、对称平衡城市、共同Wave、连续批量/负荷，δ=0、κ=∞、不施加绑定deadline；按负荷配车，忽略有限资源跨周期争用。保留θ/h及车辆容量，特别是

\[
0<n_i\le\min(M,q),\qquad s(q)=a+b/\sqrt q.
\]

T0 要求ω>0及理论报告声明的正性条件。current允许ω=0，但同时去掉deadline后，有限最优Wave可能不再存在，不能照搬本轮结论。

两种求解必须同时对齐：参数和单位、可行域、连续/整数决策域、到达过程、服务及交付口径、成本目标、长期/有限期边界、资源和政策限制。只填相同Λ、a、b、ρ、M、τ不够。Wave的q是每批shipment量，n是单车负荷，v=q/n是连续近似车数，不能把q与V1 one-hot中的v混为一谈。

从两份steady共同原式重新最小化n，得到

\[
n_1(q)=\min\{M,q,\sqrt{2c\rho/(\omega s(q))}\},\quad
n_2(q)=\min\{M,q,\sqrt{4c\rho/(\omega s(q))}\}.
\]

这补齐了现稿闭式未完整包含n≤W的边界；本轮没有据此自动重证现稿全部凸性、单调性或regime命题。

## 3. 已得到的一致性结果

令G°(q)为T0优化负荷后的每shipment广义成本，剔除dispatch前的凑批等待：

\[
R(q)=G^\circ(q)-\frac{\omega q}{2\Lambda},\qquad C(q)=qR(q).
\]

动态模型等待时 dQ/dt=Λ，运行等待费用率ωQ；每次把未发backlog全量发出、重置为0，并记整批未来服务成本C。**这是会计约化，真实车辆和在途货物不会瞬间清零。** 成立所需的逐批可分性已显式列明。

周期时长q/Λ，周期等待成本ωq²/(2Λ)，因此

\[
\boxed{J(q)=\frac{C(q)+\omega q^2/(2\Lambda)}{q/\Lambda}
=\Lambda G^\circ(q).}
\]

于是最优批量集合完全相同：重复任何静态最优Wave即可实现动态长期平均最优，headway为H*=q*/Λ；相同q下的最优负荷也相同。成本值应按单位换算为g*=ΛminG°，双方向总率为2g*，不能要求不同单位的数值直接相等。

这次不止重述原笔记的完成周期平均：理论报告补齐任意日历时域的剩余周期项，下界为−g²/(2ωΛ)，并构造tail-inf relative value验证

\[
\min\{\omega q-g+\Lambda V'(q),\ C(q)+V(0)-V(q)\}=0.
\]

全域包络覆盖超过最小最优阈值后仍可能等待的状态。主结论是最优**重复阈值集合**对应静态argmin，不是“所有平均最优策略每一批都一样”，也没有自动证明全状态唯一阈值、PDE唯一性或finite-sorting下阶段同步最优。

**单位Poisson的特殊补充：** 在同一整数阈值B域、同一C(B)、线性holding、每次全量发运的阈值政策类中，

\[
J_P(B)=\Lambda C(B)/B+\omega(B-1)/2=J_D(B)-\omega/2.
\]

最优整数阈值集合相同，成本率相差常数。连续q与整数B是不同决策域；一般带标记/批量抵达、年龄/期限、有限车队、随机服务和所有adapted控制的verification均不由此获得。Poisson非平凡阈值B>1的凑批等待没有确定性上界，不能转成逐单硬期限保证；B=1即时发出无凑批等待。

## 4. 数值核对已经实际执行

稳态路径直接数值最小化原式的n1/n2及q；动态路径另用解析服务费构造C，在网格上求Bellman平均率，不调用稳态argmin。包含负荷受Wave限制、AV容量拐点、人工车容量上界三种情形。

| 情形 | 稳态连续q* | QVI/Bellman网格Δq=.05 | 判断 |
|---|---:|---:|---|
| 小Wave，n≤q绑定 | 1.973515 | 1.95 | 误差小于一格 |
| AV容量拐点 | 8 | 8 | 一致 |
| 人工车容量绑定 | 61.304753 | 61.30 | 误差小于一格 |

网格.2/.1/.05、状态上界160→320均检查；离散残差通过不等于连续方程收敛定理。单位Poisson与fluid在相同整数网格均选20，成本率分别80.2035558774和80.7035558774，恰差ω/2=.5。

另作目标不匹配负例：遗漏holding后，截断最优点从80随上界扩大到160，正确holding问题的网格阈值约20.4。80/160是人为截断边界，不是无界问题有限最优解；该例说明换目标会变解。

PI最终复跑exit0、约2.132秒；专员最终版约2.055秒，另有两次小版本检查，均为纯一维计算。重复运行不增加不同case计数，没有Gurobi、CFS、蒙特卡洛或大型实验。该toy没有同日达L约束，低需求下较长headway不应被解读为现实same-day可行方案。

## 5. 论文目前的完成度与必须先修的点

| 当前材料 | 已有内容 | 主要欠缺 |
|---|---|---|
| main_v2_BHH.tex | M/F模型、BHH容量、RH/候选/旧优先启发式框架 | 标题摘要占位、Introduction空；无正式实验/结论；算法仍有旧容量与不安全删弧；没有ω/sorting/common timetable |
| steady_state_model.tex | 同日达setting、lead times、负荷、cost/deadline regimes、示例 | n≤W未贯穿所有公式、active期限与正乘子不等价、有限sorting共同Wave仍是假设；示例/δ原仿真未归档复现 |
| steady_state_model_v1_linear.tex | 清楚的简化运营框架 | 较早机制和未覆盖边界；适合对照，不作为第二份平行“最终主稿” |
| QVI研究笔记 | 周期解释、形式QVI、时变需求扩展设想 | 本轮已另报告补齐T0证明；原时变有限期仍缺终端规格，高维随机/阶段同步仍开放 |

正式桥接V1代码尤其需要统一以下项，不能只改一项费率：

1. **目标**：V1为车辆成本+拒单罚损；steady另计ω平均lead time，且通常全部服务。
2. **交付**：现V1已采用tour-end交付，steady含distribution途中平均n2/2；会改变时间成本与负荷最优解。
3. **资源/政策**：固定初始整数车队、离散arc-wave，与连续按负荷配车、common timetable不同。
4. **边界/机制**：有限期首末、独立空车/城市迁移、sorting/delta及硬期限需各自对齐。
5. **证据**：旧V0大规模结果不能重标V1；现V1 CFS八单只证运行，不验证steady或QVI。

这些问题不阻断本轮T0理论一致性，但阻断“现有MIP已经直接验证全部稳态理论”的写作。

## 6. 建议如何融入现有论文

当前建议先形成**一篇以三段货运运营为中心的整合稿**。以current稳态稿组织理论主线，吸收main_v2的有限期实现；不把所有随机扩展强行塞入主贡献。

| 拟章节 | 进入主文的内容 | 附录/后续 |
|---|---|---|
| Introduction / related work | 运营问题、Wave与单车负荷的区别、已验证贡献 | 文献核验清单；最后写摘要 |
| Common primitives | stop、BHH、统一量纲、模型层次表 | 参数与source对照 |
| Stationary operations | 先T0，再逐项引入current的deadline/sorting/delta，限定政策类 | 已复核完整证明与边界分支 |
| Dispatch interpretation | 一条短命题J=ΛG及长期重复Wave最优，解释适用范围 | 日历余项、tail-inf QVI、特殊Poisson整数结果 |
| Finite-horizon implementation | V1-M/F、独立empty、库存、RH、实际cargo pruning | 完整MILP和伪代码；未完成的warm start不写成已实现 |
| Evidence | 本轮同参同域一致性检查；以后经对齐的实施比较 | 原始结果、来源、CFS smoke；旧大实验逐项核model_id |
| Discussion / conclusion | 回答已证问题，区分规模经济、同步假设与工程实施 | 一般随机、高维QVI、有限sorting同步、年龄与尾计划 |

QVI的作用是解释**为什么静态最优Wave在允许逐批变化的动态政策类中仍被恢复**；不是宣称原Wave外生、现在才内生，也不把与现有最优解一致本身认证为文献创新。相关方法前史和五条核验primary文献见[写作专员计划§8](../research/literature/paper_integration_plan.md)。

建议主文约1–2页讲桥梁，技术verification放附录。是否单独写随机QVI第二篇，应等完整状态/信息/干预/边界、独立理论增量和对应证据齐备后再判断。

## 7. 接下来写什么，完成标准是什么

| 顺序 | 具体工作 | 当前状态/验收标准 |
|---|---|---|
| P1 | 冻结T0参数、域、单位和成本拆分 | 本轮已完成；证明和两个独立数值路径可追溯 |
| P2 | 独立检查理论、数值与写作边界 | 独立审查限定ACCEPT；两处LOW已修复；不升级全部随机控制 |
| P3 | 修current的n≤W、乘子边界、common timetable限定与版本/路径 | 本轮列出精确改动清单，尚未修改TeX；每个命题须在完整域重新核验 |
| P4 | 写T0基准节和dispatch桥梁节，附完整证明 | 可直接基于本轮产物写入；保持“共同子模型”与物理会计假设 |
| P5 | 对齐main_v2的模型/算法描述，设计与steady真正同口径的比较 | 先决定lead-time/交付定义与fleet/policy，未对齐前不重跑大实验冒充验证 |
| P6 | 补文献与正式证据章节，最后写Introduction/Abstract/Conclusion | Bib可解析、每条主张有依据，不保留placeholder/未复现的完成式表述 |

P3–P6是后续具体写作任务；本轮用户要求的是研究和规划，故只交付清晰方案，不自动改论文或启动新实验。完整current全部定理的修订、所有adapted随机策略的验证、一般随机/高维QVI和正式数值研究保持开放。

## 8. 本轮证据状态

本轮可主张：T0限定数学一致性、自包含证明、独立离散数值观察、稿件现状审计及可执行整合方案。不能主张：完整V1/S/V2等价、一般随机阈值定理、连续QVI数值收敛、完整稿件已可投稿或创新性已认证。所有原TeX保持原样；没有编译。

收尾核验：71份生产/旧tests/TeX/Bib来源前后SHA-256一致；保存toy的4项来源SHA与当前文件一致；10份相关Markdown链接无缺失或控制字符，toy AST有效，git diff --check通过。原有工作树未提交变更、untracked及缓存权限/换行提醒保留。

