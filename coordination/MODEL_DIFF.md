# MODEL_DIFF — 新旧模型差异索引

2026-10-17。静态审计经Referee接受；差异未实施。命名见[MODEL_CONTRACT](MODEL_CONTRACT.md)；完整公式/代码位置见[数学报告](../research/theory/model_comparison.md)、[代码报告](../experiments/reports/model_code_gap.md)。

## 1. V0 → V1有限时域模型

| ID | 旧稿/实际代码 | V1公式/章节依据 | 迁移含义；Referee编号 |
|---|---|---|---|
| D01 | CFS quantity为重量换算 | main_v2:69每单位一pickup/drop-off stop | 禁止重标；先合成stops，真实映射另定；R5 |
| D02 | loader:85–89用int(f(M))当期数 | main_v2:82 ceil((2ρ+f(M))/t0) | 新独立弧生成器，单位/取整统一；R8 |
| D03 | x f^-1(Δ)，无stem/manual M截断 | main_v2:82/106 K(v)=min(vM,f^-1(v[Δ−2ρ]^+)) | 新city/arc/v表；不能线性插值代替整数选择；R2 |
| D04 | 无wave binary | main_v2:110/176 x=Σvu、Σu≤1 | 新selector及history/start/导出；R1/R2 |
| D05 | base z整数，flex连续 | main_v2:134/216 z连续 | 新分数需求fixture；R9 |
| D06 | 旧稿auto vol已求和，base却逐单；flex已求和 | main_v2:114/180同弧总量≤Mhat y | base有缺口，不能视作V1首次提出；R9 |
| D07 | flex合并两端单车inverse×w | main_v2:141 origin zoned、dest unzoned，方向/v/stem | 新K̃、ũ，两城非对称不能共用旧表；R2 |
| D08 | flex已有累计HV迁移 | main_v2:154动态城市库存 | 可复用结构，确认迁入后的local wave上界；R1/R9 |
| D09 | direct仅τ+1起、有订单才保留 | main_v2:141 τ期零货量空返 | 无货也保留/计费，不许空弧载货；R4 |
| D10 | flex无terminal return | main_v2:212 Σw+=Σw− | 全日人工复位，不自动加AV或短窗复位；R4 |
| D11 | F另有θ/fee/ratio/direct参数 | main_v2无相应扩展 | 精确V1中性化，其他变体独立标签；R9 |
| D12 | candidate/heuristic单车inverse | main_v2正文已更新，算法:340/369、525–645/688仍旧 | 先解源规格冲突，不能照抄旧伪代码；R2/R8 |
| D13 | inherited库存0即删弧 | V1-F当窗可新调入 | profile未必为供给上界，安全性待证；R3 |
| D14 | full/reduced不同起点域 | main_v2正文H−1，算法H | 统一母域、端点、下游起点；R8 |
| D15 | 最后窗口objective/bound/gap返回顶层 | rolling无全域最优保证 | 独立成本核算，partial/window/global分域；R12 |

V1目标仍为运营费+拒单罚金，不自动增加ω、sorting、δ。stem/物理cap/wave收益同时变化，目标优劣方向不能预设。旧函数/数据/结果保持原样。

## 2. V1与稳态变体

W=q、n=q/v时，`2ρ+f(q)/v=2ρ+(a+b/√W)n`；共同local形式不证明整体等价。

| 项目 | V1-main | S-linear | S-current |
|---|---|---|---|
| 需求/时域 | 活动订单、离散有限期、可拒绝 | 恒定平衡、连续长期 | 同左，另有同日达L |
| 政策 | 每弧wave，索引语义未决 | nested三段波下推common wave | common timetable；finite sorting同步未证 |
| 车队 | 固定N、库存守恒 | 按负荷配车、无队列 | 同左；平均fleet非峰值 |
| 目标 | 车辆费+拒单罚金 | 每shipment广义成本+ω时间+Θ | 车辆费+ωTmean+2h，Tmax≤L |
| hub | 累计流，无μ/δ | 固定transfer | μ=κΛ、慢车δ、整波等待 |
| 域/单位 | a>0,b≥0；τ为期数 | 按各命题限定 | a≥0,b>0,ρ>0；cost-bound另定ω>0；τ物理时间 |
| 数学限制 | 正文/算法未同步 | 负载注明n≤W适用域 | P/loads漏W-cap，deadline临界等号需改述 |
| AV | 整数车 | 连续装载近似 | 同左 |

依据main_v2:69–216；S-linear:49–132；S-current:57–145、158–240、360；审查R6/R7/R10/R11。稿中表格和外部校准脚本未复现；不能称代码已经验证稳态命题。

## 3. V2边界

本地笔记是确定性常率/时变率。complete-cycle恒等式J(q)=ΛG°(q)限于common-wave、cost-bound、renewal-separable、无跨周期残留耦合；形式QVI不是verification/threshold theorem。

随机V2另需process/marks/filtration、stopping times、状态充分性、干预算子、q=0/no-Zeno、terminal/growth。Poisson jump generator不同于λV_q；finite sorting的stage synchronization未由successive-batch stationarity解决。依据笔记第4–14节、审查R14。

## 4. 论文与来源

四TeX无input/include，宏在文件内；main两稿Bib缺失。BHH1959支持渐近标度，不能直接认证有限小区精确等式或最慢车硬界。公开文献来源/支持范围见[文献报告](../research/literature/model_positioning.md)，创新性未认证。所有待作者决定事项见[迁移计划](MIGRATION_PLAN.md)，完整verdict见[独立审查](reviews/model_alignment_review.md)。
