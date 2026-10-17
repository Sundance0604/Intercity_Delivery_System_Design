# CLAIMS — 证据与科学状态账本

更新：2026-10-17。仅登记本轮有来源的结论；不存在本轮运行的NUMERIC_TOY/NUMERIC_FULL结果。

保留科学状态词：CONJECTURE / PROVED / REFUTED / NUMERIC_TOY / NUMERIC_FULL / LITERATURE / NEEDS_REVIEW。另用 **AUDITED_SOURCE** 表示文件/代码定义已核，不表示科学定理；**CONDITIONAL_ALGEBRA** 表示明确假设内的代数已独立核对，不是全网络最优定理。Referee的ACCEPT针对所列范围；报告ACCEPT不等于源模型已正确或迁移获批。

共同来源：[数学报告](../research/theory/model_comparison.md)（T）、[代码报告](../experiments/reports/model_code_gap.md)（E）、[文献报告](../research/literature/model_positioning.md)（L）、[独立审查](reviews/model_alignment_review.md)（R）。完整命名/散列见[合同](MODEL_CONTRACT.md)。

| ID | 精确主张/范围 | 假设 | 状态 | Owner | 来源/证明证据 | 实验证据 | Referee |
|---|---|---|---|---|---|---|---|
| QVI-001 | 适当内生batching问题可建立QVI；不声称随机full-network问题已验证 | state/information/admissibility/boundary待完整定义 | CONJECTURE，保留原状态 | theorist | 本地QVI笔记§6/§9有形式方程；T§6、R14；无verification theorem | 无 | 形式方程可接受；完整理论声明REJECT |
| BHH-001 | BHH/common-wave结构能否扩至非均匀/随机到达 | 政策、随机过程、sorting/fleet条件未定 | CONJECTURE，未因笔记升级 | theorist | 笔记确定性λ(t)且threshold以“如果”限定；T§6、R14 | 无 | 扩展未证 |
| ALIGN-001 | 实际V0至少区分M/F/RH，旧论文不能替代代码 | 当前本地文件 | AUDITED_SOURCE | experiment | base:68–69、184–192；flex:228–269、352–357；旧main:103；E§3、R9 | 未运行 | ACCEPT静态差异 |
| ALIGN-002 | V1正文已换K(v)/K̃与selectors，现代码仍旧线性容量 | main_v2定义的模型域 | AUDITED_SOURCE | experiment/theorist | main_v2:82、106–110、141、184–188；base:164–178、flex:320–374；T§2/3、R2 | 无 | ACCEPT差异；“已实现”REJECT |
| BHH-002 | K(v)/v非减、枚举域内超可加；K不必凸、非无条件严格超线性 | h凸且h(0)=0，D固定，K=min(vM,h(vD))；v1+v2在域内 | CONDITIONAL_ALGEBRA | theorist | T§3.1、R§5.1独立代数；不是有限样本routing定理 | 无 | ACCEPT限定代数 |
| ALG-001 | 旧启发式不能普遍保证新V1容量可行 | ρ可正、仍用旧C=h(Δ) | REFUTED：被否定的是“旧算法始终V1可行” | theorist | f(q)=q、ρ=1、Δ=2、M≥2，旧C=2而K(v)=0；T§4.6、R2手工反例 | 未运行 | REJECT原可行性claim |
| ALG-002 | history-only availability=0并非一般安全删弧条件 | 新direct能在本窗口调入 | REFUTED：无条件安全claim | referee | R3初始一车被占用+另城新调入例，避免N=0 selector缺陷掩盖；代码state_dependent:47–65、164–167 | 未运行 | REJECT无条件claim；特殊固定城子模型可另证 |
| ALIGN-003 | V1-F终端正反direct车次平衡恢复两城HV总数 | 所有direct弧在T完成、累计库存式成立 | CONDITIONAL_ALGEBRA | theorist | main_v2:154/212/221；T§2.3、R§5.3；不要求货量平衡/逐车原路配对 | 无 | ACCEPT限定终端代数；RH政策待确认 |
| ALIGN-004 | CFS quantity是重量归一化，不能直接当V1 stops实证 | 当前数据转换 | AUDITED_SOURCE | experiment | cfs_processor:528–534；main_v2:69；E§6、R5 | 无 | ACCEPT差异；现有stop校准claim REJECT |
| ALIGN-005 | full/reduced弧域、论文端点/完成上界不一致 | 当前窗口规格 | AUDITED_SOURCE | experiment/referee | full_window:45–61、state_dependent:153–181；main_v2:234–263/429–443；R8 | 无 | MAJOR REVISION |
| S-001 | common-wave比较可在nested、无sorting约束、同δ下成立 | current κ=∞；one-by-one比较另需δ=0 | NEEDS_REVIEW：稿内条件命题，比较代数已核 | theorist | S-current:103–111；T§5、R10/14；有限sorting同步未证 | 无 | 条件边界接受，不认证完整有限sorting定理 |
| S-002 | S-current负载闭式与后续全域结论需处理n≤W | setting89 vs P132/loads138–139 | NEEDS_REVIEW | theorist | W=1/16、a=b=c=ρ=ω=1，n1=√(2/5)>W；T§5.1、R6 | 无 | MAJOR REVISION |
| S-003 | deadline active不等价于λ>0，阈值/重根/不可行边界待限定 | 内点ω>0，L可等于cost-bound Tmax | NEEDS_REVIEW | theorist/referee | S-current:145/205/234；R7临界等号反例 | 无 | MAJOR REVISION |
| ALIGN-006 | V1与两份S没有自动等价/现成验证关系 | 目标、fleet、μ/δ、timetable、整数/终点不同 | AUDITED_SOURCE | PI | S-current:360；T§5、E§5、R10；仅local形式可匹配 | 无stationary验证 | ACCEPT差异；现成等价claim REJECT |
| QVI-002 | complete-cycle J(q)=ΛG°(q)，重复最优batch可达周期平均最小值 | 确定性常率、common-wave、cost-bound、renewal-separable、存在minimizer | CONDITIONAL_ALGEBRA | theorist | 本地笔记§4/5，T§6、R14；任意日历平均另需余周期/时间趋无穷控制 | 无 | ACCEPT限定代数；不认证随机/全网络最优 |
| EXP-001 | partial objective与last-window gap不能直接当全日成本/界 | 预算可中止、窗口求解 | AUDITED_SOURCE | experiment | paper_rolling:148–151/233–275，solvers:395–406；R12；完成目标也需独立账本核算 | 历史结果影响未审定 | MAJOR REVISION结果解释 |
| LIT-001 | BHH及相关公开文献存在、主题可核；精确有限波等式/novelty不随之成立 | 出版社/原始机构元数据或摘要的有界核验 | LITERATURE | literature | L§4含作者年/URL/支持范围/2026-10-17；R13另抽查3项 | 稿中数值表/拟合未复现 | ACCEPT有界文献审计 |
| SOURCE-001 | Bib及稿称校准/符号脚本在可读树未找到 | 扫描受.pytest_cache权限限制 | AUDITED_SOURCE | literature | main_v2:7；S-current:19–25/358；L§1、R13 | 未编译/运行 | ACCEPT有界缺失报告 |

不把稿内“proved/checked”注释当作独立证据。PROVED只用于完整假设和证明获相应独立认可的确切命题；本轮没有新增此类全域研究定理。不得改变原QVI-001/BHH-001范围来制造已完成状态。
