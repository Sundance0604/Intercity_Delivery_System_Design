# BHH / Intercity 新旧模型定位与文献审计

日期：2026-10-17（Asia/Shanghai）。作者：Literature Agent。状态：模型定位审计，等待 Theorist 与 Referee 的独立数学复核；不是创新性认证、证明认证或数值复现报告。

## 1. 范围、材料与依赖

已读取 `AGENTS.md`、初始 `coordination/MODEL_CONTRACT.md`、Literature 配置、四份指定 TeX、本地 QVI 笔记 `research/theory/Common Wave 的内生化与非均匀需求扩展.md`。通过逐段阅读与新旧文本差异检查核对公式、正文及算法。这里只对代码入口作有限只读抽查，V0 的完整代码判定以 Experiment 报告为准，不能用旧论文替代实际代码。

四份 TeX 未发现 `\\input` 或 `\\include`；定理环境及 proof 宏在各自主文件内定义，没有发现须另读的外部数学宏文件。`main_BHH.tex:7` 与 `main_v2_BHH.tex:7` 均依赖 `导出的条目.bib`；`research/paper` 的含隐藏文件目录检查确认该文件不存在，仓库范围 `rg --files -uu -g '*.bib'` 也未发现 Bib 文件。扫描遇到 `.pytest_cache` 拒绝访问，因此不作该目录内绝对不存在的断言。当前可读工作树不能恢复该 cite key 的原始 Bib 元数据，也未编译论文。两份 steady-state 稿没有正式 `cite`/bibliography 命令，只有候选文献注释。

`steady_state_model.tex:19-24,358` 提及 `steady_state_check.py`、`steady_state_symbolic.py`、`bhh_tour_check.py`；本次仓库文件名扫描未找到这些脚本。稿内说明 tour 检查脚本在项目外。不能把作者注释中的“已检查”当成本轮独立复现证据。`steady_state_model.tex:38` 指向 `archive/steady_state_model_v1_linear.tex`，实际所给文件在 `research/paper/steady_state_model_v1_linear.tex`；两份稳态稿还用 `main_v2.tex` 指代配套程序稿，实际文件名为 `main_v2_BHH.tex`。这些交叉指向需未来整理，本轮没有修改。

开始时 Git 状态含大量既有未跟踪研究与实验文件，不能将这些视为本轮产出。本 Agent 唯一写入文件为本报告；未改 TeX、优化代码、历史数据，没有运行求解或提交推送。外部检索仅使用已公开作者、题名及一般概念，未上传未发表稿件或搜索本地未发表公式。

## 2. 四份稿件与 V0 / V1 / V2 的关系

| 对象 | 实际研究问题与政策类 | 证据 | 应避免的等价性推断 |
|---|---|---|---|
| 旧论文 `main_BHH.tex` | 有限时域、双向人工/自动车辆调度；拒单惩罚、三段中转与人工直送，单车逆 BHH 容量线性乘车数；滚动时域与优先启发式 | 67-78、99、138-193、198 起 | 旧论文不是 V0 所有实现与修复的穷尽说明 |
| V0：现有 Python | 本轮只抽查到 base 与 flexible 优化器；后者有单车直送容量预处理与 `_add_capacity_constraints` | `intercity_delivery/models/base_optimizer.py:72,90`；`flexible_direct_optimizer.py:50,78,320` | 不能凭稿件宣称代码已有 wave 联合容量、排序或 QVI |
| V1：`main_v2_BHH.tex` | 仍是有限时域 MILP，新增 zoned wave 联合容量、stem、城市索引、直送物理迁移与期末回流；决策更新网格预先给定 | 67-83、106-134、141-223、228-252 | 内生 wave 规模并不表示已内生优化连续时间干预时刻；也未强制全系统 common timetable |
| V1-S：`steady_state_model.tex` | 对称城市与恒定平衡需求的连续稳态近似；共同时间表下优化 W、n1、n2；硬服务期限与平均 lead-time 成本并存；排序容量及慢车等待 | 57-89、119-132 | 是单独的解析变体，不是 main_v2 自动成立的等价缩约 |
| V1-S-linear：`steady_state_model_v1_linear.tex` | 更早的稳态线性时间成本稿；无硬 deadline、有限排序和慢车系数；common wave 推导与整城单车政策对照 | 49-78、95-132、185-294 | 文件名中的 v1 不等于本轮三层命名中的 V1；whole-city 的平方根法则不能移植到 zoned wave |
| V2：QVI 扩展 | 本地笔记研究确定性恒定或时变需求、common-wave、renewal-separable dispatch；随机到达、期限、排序与多阶段状态尚待建立 | QVI 笔记 13-17、246-459、463-545、549-769、809-901 | 确定性时变需求不是随机订单模型；形式 QVI、条件性周期成本恒等式不是 full-network 最优性定理 |

主定位变化是从“调度优化与双路径成本节省”转向“城市分区密度如何影响三段批量耦合，以及期限/排序如何塑造 wave”。这是一项候选研究主线，尚不构成文献创新性结论。

## 3. 研究目标和论证结构的具体变化

### 3.1 旧论文与 main_v2

1. 旧稿 `main_BHH.tex:193` 将“无一般性损失的大规模可计算性”“客户行为敏感性”列为贡献，但正文没有需求响应模型或复杂度证据。新稿 `main_v2_BHH.tex:223` 改为预计算容量保留 MILP、批量耦合与稳态验证，更符合公式范围。仍不能将“保持 MILP”写成“已证大规模高效”。
2. 旧容量 `main_BHH.tex:99,164` 为 `sum g <= x f^{-1}(D)`，规模经济来自单车装载；新稿 `main_v2_BHH.tex:81,106-110` 为 `K(v)=min{vM,f^{-1}(v[D-2rho]^+)}` 的 one-hot 表。后者依赖 wave 总量的空间密度，需要新的模型解释、有限样本检验及求解规模评估。
3. 旧直送 `main_BHH.tex:138` 用 `f1(lambda)+f2(lambda)+tau`，新稿 `main_v2_BHH.tex:141` 用 `2rho_origin+f_origin(lambda)/v+tau*t0+f_dest(lambda/v)`：出发端分区、目的端整城。文献不能代替“为何目的端不能重排分区”的运营假设；需明确订单 OD 独立及车辆不在目的端互换货物。
4. 新稿 `main_v2_BHH.tex:154,212,221` 以累计离开/抵达和期末双向次数相等处理人工车回流，并允许空驶；这比仅将直送行程视为原城市车辆占用更具体。期末复位与稳态平衡回程是不同约束，不能称完全一致。
5. 新旧 Introduction、abstract 与 title 仍是占位（`main_v2_BHH.tex:45-60`；旧稿相同）。因此所谓贡献定位主要来自模型节结尾及两份独立稳态稿，而不是已完成的论文论证链。

### 3.2 两份稳态稿

线性稿 `steady_state_model_v1_linear.tex:35-43` 的主线是最大批次只支付一次等待、common wave 与负载、直送价值分解；后半保留整城单车政策、独立 EOQ/平方根法则和临界距离。

更新稿 `steady_state_model.tex:43-51` 将主线调整为同日达、同步收益、负载比与 deadline/cost 两种 regime。新增 `mu=kappa*Lambda`（81-83）、慢车系数 delta（61-69）、硬期限 L（85-87），并只将直送作为 reference（247-259）。两份稿不可混引用：

- 线性稿未截断负载比为 sqrt(2)（103-109）；新稿有 deadline 乘子与慢车效应，其内点负载比介于 1 与 sqrt(2(1+delta))（135-151）。容量截断下上端等号等说法还需逐项检查。
- 线性稿将 common wave 作为命题结论（101-115）；新稿命题只在排序不约束、nested waves、同一 delta 下成立（103-113），有限排序时 common timetable 是保留的政策假设（111-113、359）。
- 新稿两种 regime 与高需求 2/3 标度均附特定条件（164-188、201-240）；不能用“AV 不影响 wave”概括所有区域。AV 未满载时有 tau、hat-c 的影响；满载/条件化分支才可独立。
- 线性稿 zoned 临界距离单调性明确无证明（181-183）；whole-city 命题有稿内证明（258-274）。新稿不再把这一 benchmark 作为 headline，且加入 deadline 后原“模仿可行解”需重新证明期限可行。

## 4. 文献核验与实际支持范围

所有下列链接访问日期为 **2026-10-17**。核验级别：出版社/作者机构页面元数据和摘要；除明确说明的公开原文外，没有宣称逐页审读全部付费全文。文献存在性、主题相关性与“支持本稿某个定理”是三项不同判断。

### 4.1 实际引用

**Jillian Beardwood, J. H. Halton, J. M. Hammersley (1959).** “The shortest path through many points.” *Mathematical Proceedings of the Cambridge Philosophical Society*, 55(4), 299-327. DOI 10.1017/S0305004100034095。[Cambridge 出版社原始页面](https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/shortest-path-through-many-points/F1C28B5730B94887F4659FCBF8A1F2BB)。元数据与摘要核实，发表年是 1959，页面数字化上线 2008 不应当作发表年。

对应 `main_BHH.tex:71`、`main_v2_BHH.tex:71` 的 `Beardwood1959Shortest`。支持二维大样本随机点最短闭合路径的平方根数量/面积标度；handling 项 `a lambda`、hub stem、均衡分区、每辆车恰好用 `f(lambda)/v`、少量 stops 下误差及慢车期限保证，均不是该引文直接证明的内容。因 Bib 缺失，只能确认最可能意图引用的论文存在，不能确认现有 cite key 对应条目的作者、年、题名书写完全正确。

### 4.2 稳态稿注释中的候选文献

| 文献（已核元数据） | 权威来源 | 对本项目的有限支持与不足 |
|---|---|---|
| Richard M. Karp (1977), “Probabilistic Analysis of Partitioning Algorithms for the Traveling-Salesman Problem in the Plane,” *Mathematics of Operations Research* 2(3), 209-224 | [INFORMS，10.1287/moor.2.3.209](https://pubsonline.informs.org/doi/10.1287/moor.2.3.209)；[作者机构原始报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1977/28889.html) | 分区、子巡回与拼接的渐近算法依据；不直接给出有限小区 tours 总长与全城单巡回的精确相等，也不保证同时返回 |
| Carlos F. Daganzo (1984), “The Distance Traveled to Visit N Points with a Maximum of C Stops per Vehicle: An Analytic Model and an Application,” *Transportation Science* 18(4), 331-350 | [INFORMS，10.1287/trsc.18.4.331](https://pubsonline.informs.org/doi/10.1287/trsc.18.4.331) | 车队配送距离连续近似的直接背景；需进一步阅读全文核对 stem 与分区公式适用条件，不能仅凭摘要认定等同本稿公式 |
| Lawrence D. Burns, Randolph W. Hall, Dennis E. Blumenfeld, Carlos F. Daganzo (1985), “Distribution Strategies that Minimize Transportation and Inventory Costs,” *Operations Research* 33(3), 469-490 | [INFORMS，10.1287/opre.33.3.469](https://pubsonline.informs.org/doi/10.1287/opre.33.3.469) | 运输、库存/时间与配送策略权衡已有基础；不能把成本-等待权衡本身说成新贡献。其 direct 与 peddling 定义也不能直接等同本项目三段中转 |
| Carlos F. Daganzo (1987), “The Break-Bulk Role of Terminals in Many-to-Many Logistic Networks,” *Operations Research* 35(4), 543-555 | [INFORMS，10.1287/opre.35.4.543](https://pubsonline.informs.org/doi/10.1287/opre.35.4.543) | 货运 terminal、集并与拆分的既有网络背景；不是本稿人工/自动混合、硬时限负载比或有限排序同步的证明 |
| Gordon F. Newell (1971), “Dispatching Policies for a Transportation Route,” *Transportation Science* 5(1), 91-105 | [INFORMS，10.1287/trsc.5.1.91](https://pubsonline.informs.org/doi/10.1287/trsc.5.1.91) | 已研究给定非恒定平滑到达率下选择 dispatch 时刻与总等待成本；所以“需求随时间变化、发车时刻内生”这一宽泛概念不能直接主张首创；其固定车辆班次数、乘客单线路问题不同于 BHH 三段货运 |
| Dimitris J. Bertsimas, Garrett van Ryzin (1991), “A Stochastic and Dynamic Vehicle Routing Problem in the Euclidean Plane,” *Operations Research* 39(4), 601-615 | [INFORMS，10.1287/opre.39.4.601](https://pubsonline.informs.org/doi/abs/10.1287/opre.39.4.601?journalCode=opre) | Poisson 随机抵达、随机位置/服务、平均系统时间与轻/重负荷政策是 V2 的基础背景；不证明本项目 QVI 或多阶段共同 wave |
| Dimitris J. Bertsimas, Garrett van Ryzin (1993), “Stochastic and Dynamic Vehicle Routing in the Euclidean Plane with Multiple Capacitated Vehicles,” *Operations Research* 41(1), 60-76 | [INFORMS，10.1287/opre.41.1.60](https://pubsonline.informs.org/doi/10.1287/opre.41.1.60)；[作者 MIT 原文](https://web.mit.edu/dbertsim/www/papers/Vehicle%20Routing/Stochastic%20and%20Dynamic%20Vehicle%20Routing%20in%20the%20Euclidean%20Plane%20with%20Multiple%20Capacitated%20Vehicles.pdf) | 多车与容量对动态随机路由的影响；不能跳过 fleet/pipeline 状态直接引用其结果认证 backlog-only QVI |
| Robin Roundy (1985), “98%-Effective Integer-Ratio Lot-Sizing for One-Warehouse Multi-Retailer Systems,” *Management Science* 31(11), 1416-1430 | [INFORMS，10.1287/mnsc.31.11.1416](https://pubsonline.informs.org/doi/10.1287/mnsc.31.11.1416) | 多阶段补货、整数比与政策取整的比较背景；近似保证依赖其库存结构，不能移给 BHH、时间窗或有限排序 |
| Robin Roundy (1986), “A 98%-Effective Lot-Sizing Rule for a Multi-Product, Multi-Stage Production/Inventory System,” *Mathematics of Operations Research* 11(4), 699-727 | [INFORMS，10.1287/moor.11.4.699](https://pubsonline.informs.org/doi/10.1287/moor.11.4.699) | 固定外部需求、多产品多阶段库存政策；只能作结构对比。注释仅给作者年份，无法确认作者原意唯一指向本条，应在正式引用前确认 |
| Carlos F. Daganzo (2005), *Logistics Systems Analysis*, 4th ed., Springer | [出版社，10.1007/3-540-27516-9](https://link.springer.com/book/10.1007/3-540-27516-9) | 存在性、版本、目录已核；物流连续近似与 transshipment 的综合背景。注释所称 headway 的 -1/3 法则尚未定位到具体页码/公式，不列作已核实该指数的原始出处 |

以上均是相关背景；没有任何一篇已被本轮证实含本稿的“3-leg + hard deadline + finite sorting + delta”的完整定理。也没有进行穷尽现代文献检索，因此不能据未找到相同模型宣称新颖性。

### 4.3 V2 的补充方法文献（并非 TeX 已有引文）

- P. L. Lions, B. Perthame (1986), “Quasi-Variational Inequalities and Ergodic Impulse Control,” *SIAM Journal on Control and Optimization* 24(4), 604-615。[SIAM，10.1137/0324036](https://epubs.siam.org/doi/abs/10.1137/0324036)。出版社元数据/摘要核实；讨论反射扩散的遍历控制与 QVI，支持方法类别存在，但不自动满足本项目确定性 reset 模型的假设。
- C. Belak, S. Christensen, F. Seifried (2017), “A General Verification Result for Stochastic Impulse Control Problems,” *SIAM Journal on Control and Optimization* 55(2), 627-649。[作者机构原始成果页](https://www.fis.uni-hamburg.de/en/publikationen/detail.html?id=7380d581-3307-4d68-a998-55f351cf4ecb)，[DOI 10.1137/16M1082822](https://doi.org/10.1137/16M1082822)。元数据与摘要核实（检索索引保留的机构摘要可读，直接打开机构页未完整返回成果正文；作者名保留所核元数据的缩写）；存在性、粘性解与迭代最优停止的框架提示 V2 必须有 verification 层，不能只展示 QVI 形式。

这里不以 American option 作为已完成理论依据。本地笔记 545 附近已将其限定为直觉，并以 repeated/regenerative impulse control 描述重复发车。

## 5. 论证完整性的主要缺口

### 5.1 模型桥梁未闭合（迁移/论文验证前置问题）

`main_v2_BHH.tex:83,146` 目标是运营成本与未服务惩罚，无 omega lead-time cost；城市/订单可异质，车辆数固定，采用有限离散时域。`steady_state_model.tex:57-89,127-132` 则恒定平衡需求、对称城市、可按负载扩充车队、无车辆排队，且有硬 L、平均时间成本、有限 sorting 与共同 timetable。正文 `steady_state_model.tex:360` 自己也承认 program 无时间成本/排序/timetable。

因此 `main_v2_BHH.tex:223` 和稳态稿 51 的“finite-horizon counterpart / validation”目前是研究目标，不是已成立映射。需要明确共同子模型、目标归一化、总需求或已服务需求分母、拒单的处理、车辆非绑定或显式 sizing、排序及慢车因子如何对齐、整数/离散误差与首末日边界。只给 MILP 扩 wave 容量，不能声称已验证稳态 regime。

### 5.2 BHH 的证据层级

`main_v2_BHH.tex:71` 以“partitioning property”直接说各小区 tours 加总为单个 tour，再除以 v；BHH 与 Karp 支持渐近背景，有限 v/少量 stops、均衡 workload 与长短车差异仍是建模近似。新稳态稿 `358` 报告误差 2%、0.85 拟合常数、15%-50% 最慢车偏差等，但脚本/样本/种子/求解证书不在本工作树，故一律为**作者报告的尚未复现的模拟观察**，不是本轮验证结果，也不是普遍定理。硬 deadline 不能从平均 stem 与拟合常数自动获得逐单几乎必然保证。

### 5.3 命题条件和表述应先审查

- `steady_state_model.tex:89` 写 n_i<=min{M,W}，但问题 (P) 131-132 只显示 n_i<=M，闭式负载 137-145 也没显式截断 W。稀疏市场若 n_i>W，空间分区与车辆数 W/n_i 失去解释。需要 Theorist 检查 domain、可行性与所有封闭式结论。
- `steady_state_model.tex:145` 内点比率后“if and only if deadline slack”等等号，须区分 omega=0、乘子为零但约束恰等号、容量截断等边界。稿内 proof 不等于全部文字推论已经独立审核。
- 新稿引言中 AV “不显著影响”是参数依赖的解释，不能替代 164-188/219-240 的条件结论；尤其 loads 未满载 deadline regime 的比较静态明确无证明。
- consolidation Lemma（新稳态 93-101；旧稳态 82-90）要求 nested integer batches、固定阶段时延、顺序及无队列等条件。Poisson 版本仅说明某些平均值可取期望，不能把 (q-1)/Lambda 写成 Poisson 样本路径最大等待上界；Poisson 间隔无有界最长等待。随机期限应另立定义。
- AV 成本 `hat-c*tau/min{W,hat-M}` 是连续车数利用近似；若 W/hat-M 不是整数，实际同时发车成本为离散装载问题。论文需区分近似、取整测试与精确 MILP。

### 5.4 求解章节未随核心数学更新

新旧稿 diff 显示 Solution Approach 基本未修改。`main_v2_BHH.tex:340,369` 仍使用旧 f(M)/tilde-f(M) 弧界，未带 stem/periods/新的 v 参数；429-444 的完成窗口命题也仍用旧式单位与 duration。`453-457` 将 lambda 称为 congestion，并且三段中转的第三段误指 direct arc。启发式 `534` 附近继续用 `f^{-1}(D)` 及“增加车辆 × 单车容量”，遗漏 K(v)、新 u 和 stem；`689-691` 却仍宣称产生可行初解。应由数学/实现专员逐点核查并在修复验证前把这些“可行/最优”视为未证。

`main_v2_BHH.tex:226` 宣称 rolling horizon 实现全时域 optimal dispatch，缺少全局最优性证明。解决窗口 MILP 到最优也不蕴含在线/全日最优；`254` 的显著缩减 search space 需要实例与统计结果。这里不能用 RL 类比（254）替代信息结构定义，也不属于独立 DRL_CO 项目。

### 5.5 V2 结果范围

本地笔记 `246-459` 的 q*=W° 来自周期平均成本加权恒等式，明确限于恒定确定性需求、common wave、cost-bound 与 renewal-separable。不得推广为 fixed fleet、finite sorting pipeline、多阶段独立发车或随机订单模型。`463-545` 是 average-cost 形式 QVI；`652-769` 是确定性时变 lambda 的有限期形式，threshold 还使用“如果”作为条件。终端成本、q=0 的 admissibility、防止无限频率干预、价值函数正则性、verification 与阈值结构均需另证。笔记的“stationary special case”应以这些条件为限定，不作为创新性或 full-model 等价结论。

## 6. 建议的论文组织（建议，不改稿）

1. 先明确 same-day freight operations 的问题、已知空间/库存/dispatch 文献与可验证研究问题；将需求响应、定价或“首创 QVI”移出当前已完成贡献范围。
2. 用统一符号表给出 V1-S 的全部 domain、成本单位、时限定义与条件；分别标明 modelling assumption、稿内命题、数值观察。
3. 依赖顺序为 routing approximation → 嵌套批次等待 lemma → 在无排序约束下 common-wave 条件 → 给定 W 的负载 → cost-bound wave → deadline regime。有限排序下共同 timetable 留作政策类假设；未完成同步最优性单列讨论。
4. 将 main_v2 作为经过共同假设对齐后的 finite-horizon 数值模型，报告近似来源与修复后的算法，不把旧启发式正确性直接沿用。
5. 直送保留 benchmark。说明与“仅直送”“灵活双路径”三个实验场景的关系，不把 benchmark 名称误当成模型禁止混合直送。
6. 数值部分先小样本 routing 校准及可复现性，再 stationary 对照与整数误差，再非平稳/非对称压力测试。所有当前表格维持 illustrative/unverified replication 状态。
7. V2 作为独立理论扩展或独立后续文稿；先写明确定性周期等价的假设，再讨论随机抵达、时间依赖边界、多阶段同步的不同未决问题。

## 7. 需要作者确认的问题

1. 实际迁移目标是否只实现 main_v2 的 wave MILP，还是也要在另一个阶段增加稳态模型的 lead-time cost、sorting、delta 和共同 timetable？三份“最新稿”目前并非相同模型。
2. main_v2 的 wave 是相同起点+方向的所有车辆，还是相同 (i,j,k,direction) 的车辆组？正文 71 的定义与 81/110 的 arc 分组可能产生不同政策类，尤其同起点不同完成时刻。
3. 硬 deadline 是按近似 tour 的确定性上界，还是平均/分位数/逐单保证？稳态 delta 的模拟来源能否提供本地脚本与记录，以便后续批准后复现？
4. n_i<=W、最少一件/一辆、分区数取整、AV 最后一辆部分装载如何统一到论文命题与数值程序？
5. `导出的条目.bib` 的正式来源及 Roundy/Logistics Systems Analysis 的作者意图版本是什么？“headway -1/3”应补确切页码，不能仅列作者年份。
6. finite sorting 下 common wave 当前是否仍为政策假设？笔记明确未解决此同步问题，不应因动态 successive batches 的周期论证就把它升级为结论。

## 8. 交付界限

本轮完成公开引文存在性与主题支持审计、四稿定位比较、条件与论证缺口标记。未证明 novelty，未复现稿中表格/脚本，未审定 QVI verification，也未修改任何源模型。后续实际迁移应等待主协调者整合、Referee 审核以及用户批准。
