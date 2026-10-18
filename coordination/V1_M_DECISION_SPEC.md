# V1-M 最小决策规格包（G0-M）

日期：2026-10-17（Asia/Shanghai）。Owner：Principal Investigator。Theorist和Experiment已分别只读核对原始数学与代码，Referee已完成[独立审查](reviews/v1_m_decision_spec_review.md)。

**当前状态（2026-10-18）：作者已回复“采用这组定义”，D1–D6-A全部确认；对应offline V1-M基准已实现并有微型验证。** 决定ID为`V1M-D1-D6-A-2026-10-18`。后文保留最初规格与手算设计；实际实施及执行证据见[实施记录](../experiments/reports/v1_mandatory_implementation.md)，不把作者确认扩大到V1-F/RH或QVI。

阶段补记：先前只交付capacity/network纯工具；最新已交付configuration、mandatory_optimizer、validation和runner。下文“未创建/未运行/待批准”均是最初规格准备阶段的历史；当前范围及作者确认以本文顶部和[基准profile](V1_M_IMPLEMENTATION_PROFILE.md)为准。原TeX未改。

## 1. 本项交付解决什么

上一轮[MIGRATION_PLAN](MIGRATION_PLAN.md)的Q1–Q12同时覆盖V1-M、直送、滚动、稳态与V2。本规格将首项工作收敛为一个可单独批准的基准：**main_v2强制换装正文、全时域、全部给定订单在起始时已知的确定性MILP**。

这是建议的offline reference，源文第69行仍叙述实时抵达；不得把本基准称为已实现的实时策略或随机控制问题。范围和下面D1–D6仍待作者确认。

本规格包含：原式完整清单、参数量纲、公式—代码—验收矩阵、有限的作者选项、可手算案例、后续G1/G2的交付与预算建议。不把直送、滚动、sorting、慢车δ、ω时间成本或QVI悄悄加入V1-M。

## 2. 来源和版本

- 首要来源：`research/paper/main_v2_BHH.tex:67–137`，尤其Mandatory Transshipment Scheme。下文使用源label和源码行号，未编译确认排版后的公式数字。
- 当前代码基线HEAD：`f26b4c96907082c2ebb7b0104263831e7100aa26`；现有V0不改。
- 本次V1输入SHA-256：`500BE1ADA854317BE79C358E0FF1269786DD4C7EA8E1102B501354EFF04A3457`。其余数学输入及限制见[MODEL_CONTRACT](MODEL_CONTRACT.md)。
- 依据：[数学比较](../research/theory/model_comparison.md)、[代码差距](../experiments/reports/model_code_gap.md)、[前轮独立审查](reviews/model_alignment_review.md)。本轮补查包括结果导出的完整性。
- 文中代码路径均相对仓库根；实际求解器在`intercity_delivery/experiments/solvers.py`，实验核心在`intercity_delivery/experiments/core.py`，不是顶层`experiments/solvers.py`或`experiments/core.py`。

## 3. 作者决策表：建议不等于已经采用

六行方案A均已由作者确认，只覆盖本offline强制换装基准，没有自动扩大到G3–G9。

| ID / 原问题 | 推荐方案A（待确认） | 其他解释及影响 | 决策状态 |
|---|---|---|---|
| D1 / Q1 | 先实现全时域offline V1-M；所有给定订单已知，不做direct/RH/S/V2 | 若先做在线策略，还须定义reveal、窗口、历史货流和终端义务，不能直接沿用本规格 | AUTHOR_CONFIRMED_A |
| D2 / Q2 | wave按完整(i,j,k,方向)定义，逐式对应82/106/110；明确它与71的同start语言有张力 | start-wave若只允许一个j，是额外政策限制；允许多j则需新增共同分区/不同tour预算规则，现稿未定义 | AUTHOR_CONFIRMED_A |
| D3 / Q5/Q6 | d/g/z/M/Mhat为stop等价量；t0/ρ/f用分钟；a为分钟/stop、b为分钟/√stop；费用每车小时，目标只做一次t0/60换算；先合成数据 | 若费用为每期，就不能再乘物理t0；若真实客户不可分或容量为重量，则须新模型/数据映射，不能重标CFS | AUTHOR_CONFIRMED_A |
| D4 / Q5/Q6 | 保留连续g/ĝ/z，解释为订单类流体近似；x/y允许0，u binary；允许0车队/0需求 | 整数客户/整数z会改变134的域；严格正车数会强迫所有弧出车并破坏全拒单基准 | AUTHOR_CONFIRMED_A |
| D5 / Q9 | 以destination tour结束j≤e为服务口径；origin返回j后才可装AV；同刻交接/再出发可用；不加processing delay或AV终端复位 | 用最后顾客实际送达而非tour结束，需额外交付时点；真实逐单期限保证不由BHH平均式自动给出 | AUTHOR_CONFIRMED_A |
| D6 / 输入边界 | 采用下述显式参数域；首批合成基准s/e为整数节点且0≤s≤e≤T；未定义的跨域输入拒绝，不裁剪 | 来源未逐项声明这些输入域。实值窗阈值、e>T、负release或跨期存量须另给规则；不能将受限基准冒称全输入等价 | AUTHOR_CONFIRMED_A |

作者对上述组合明确回复“采用这组定义”；本基准的解释、单位和输入合同现按A冻结。后续若改用其它语义，应新建profile/版本及相应验收，不能静默更换。

D6-A的完整建议域：所有数值有限；T、τ为正整数；两城人工Nᵏ与AV Nhatᵏ均为非负整数（拒绝分数车队）；t0>0、a_k>0、M>0、Mhat>0；b_k≥0、ρ_k≥0、d_l≥0、车辆费用和罚金≥0。s_l/e_l采用整数节点且0≤s_l≤e_l≤T；零车队和零需求合法。实值时间窗作为阈值在原不等式中可能有解释，但不在本推荐首版输入合同内；不得悄悄取整。这里的正容量、整数窗和范围限制是待批基准合同，不是声称源文已逐一限定。

### D2的具体差别

源71以同城、同方向、同开始期描述wave；82/106/110的u和K却逐(i,j)索引。推荐arc-wave是为了先忠实实现写出的MILP，不表示源文矛盾消失。

不能把各j车数相加后，向每条弧重复赋同一个总K。若start-wave额外要求每个(i,k,方向)只用一个j，可加跨j选择限制，但这严格收缩arc-wave政策类，需要作者批准。若允许不同j，必须定义不同时长车辆如何共享区域和load；一般start-wave的可行域目前不足以和arc-wave作包含比较。

手工区别示意（非实验）：f(q)=q+√q、ρ=0、t0=1、N=2、M≥2；(0,1)和(0,2)各派一车，arc-wave分别可载h(1)=(3−√5)/2≈0.382和h(2)=1。若另加“合成一个等load/等tour-time start-wave”的规则，每车需f(1.382)/2≈1.279，短弧车便不能完成。这里只比较这个明确规则，不推断所有start-wave定义。

## 4. 可由源式确认的模型骨架

### 集合、参数与时间

城市k∈{1,2}，方向d∈{+,-}；η+=1、η−=2，目的城市为另一城。时间节点𝒯={0,…,T}，τ为正整数期数。订单l∈𝒟ᵈ具有s_l、e_l、需求d_l、罚金δ_l。固定人工车数Nᵏ、各hub初始AV数Nhatᵏ、容量M/Mhat。依据67/69/71。

令Δ_ij=(j−i)t0，f_k(q)=a_kq+b_k√q，a_k>0、b_k≥0，h_k=f_k^{-1}，每车stem=2ρ_k：

`h_k(θ)=((sqrt(b_k²+4a_kθ)−b_k)/(2a_k))²`（74）。

local弧：`A_k={(i,j)∈𝒯²:0<j−i≤ceil((2ρ_k+f_k(M))/t0)}`；AV弧：`Ahat={(i,j)∈𝒯²:j=i+τ}`。占用均为`i≤t<j`，t=0,…,T（82）。保留正文母弧域；弧存在不表示有正载货容量。

`K_kij(v)=min(vM,h_k(v[(j−i)t0−2ρ_k]^+))`，v=1,…,Nᵏ（82）。K为预计算常数；不是旧式v h_k(Δ)。零tour budget时K=0；Nᵏ=0时selector集合为空，若采用非负域则x=0、该城流量0。

### 变量与目标

x_ij^{kd}为local车数，y_ij^d为AV车数，u_ijv^{kd}二元；g_ij^{kl}、ĝ_ij^l、z_l为非负连续。g的方向由l确定；x/y的0值约定对应D4。依据79/110/134。

源目标（84，`eq:obj`）：

`Σ_l δ_l z_l + Σ_{k,d,(i,j)∈A_k} c (j−i)t0 x_ij^{kd} + Σ_{d,(i,j)∈Ahat} chat τt0 y_ij^d`。

完整决策向量包括u，即使源目标下标未列u。若采用D3分钟/小时方案，代码费用应为`c_hour (j−i)t0/60`和`chat_hour τt0/60`；这对应源式费率`c=c_hour/60`、`chat=chat_hour/60`按分钟计。不能把原数值费率原封不动换单位，也不能做两次换算。77的“unit-period cost”与84量纲冲突仍待D3确认。

### 全部约束及追溯矩阵

以下代码位置均在`intercity_delivery/`内；拟议V1模块均未创建。

| SPEC ID / TeX位置 | 完整要求 | 旧代码对应与差距 | 拟议模块 / 验收 |
|---|---|---|---|
| M01 / 74 | f/h及参数域 | `data/loader.py:42–83`有旧逆函数 | `v1/capacity.py`；独立代回f(h)=θ，零值和数值稳定性 |
| M02 / 82 | local/AV母弧域、半开占用、K(v)表 | `data/loader.py:85–104,166–180`：manual未按新单位/stem/cap | `v1/network.py,capacity.py`；ceil、j=T、τ>T、零容量检查 |
| M03 / 89 `eq:num man veh` | 每城每t，Σ_dΣ_{i≤t<j}x≤Nᵏ | `models/base_optimizer.py:95–114`主体可参考 | `v1/mandatory_optimizer.py`；重叠不可复用车、j时车已释放 |
| M04 / 93 `num auto veh` | 每t，Σ_dΣ_{i≤t<j}y≤Σ_kNhatᵏ | `models/base_optimizer.py:118–130` | 同上；保留冗余总量式并独立核验 |
| M05 / 97、101 `bal auto ±` | 两hub各自初始Nhat−累计本向发出+累计反向到达≥0 | `models/base_optimizer.py:132–163` | 同上；按0…T核库存、允许同刻到达再发，不要求终端复位 |
| M06 / 106 `BHH` | 每k/d/arc，Σ_{l∈Dᵈ}g_ij^{kl}≤Σ_vK_kij(v)u_ijv^{kd} | `models/base_optimizer.py:164–178`旧x×coeff | 同上；按共享wave表校验，不用线性插值 |
| M07 / 110 `wave link` | x=Σ_vv u_v，Σ_vu_v≤1 | `setup_variables:18–70`无u | 同上；空selector/零车、one-hot与x一致 |
| M08 / 114 `auto vol` | 每方向AV弧，Σ_{l∈Dᵈ}ĝ_ij^l≤Mhat y_ij^d | `models/base_optimizer.py:180–193`逐订单，非共享 | 同上；联合超载反例，空方向求和为0 |
| M09 / 118、137 `violate` | origin弧i<s_l则g=0；destination弧j>e_l则g=0 | `data/loader.py:128–163`，`models/base_optimizer.py:195–204` | `network.py,mandatory_optimizer.py`；不擅自对每腿加完整订单窗 |
| M10 / 122 `transfer step 1` | 每l/t，AV累计开始量≤origin累计已完成local量 | `models/base_optimizer.py:207–244` | `mandatory_optimizer.py,validation.py`；逐单逐t，不跨单借货 |
| M11 / 126 `transfer step 2` | 每l/t，destination累计开始量≤AV累计已到达量 | `models/base_optimizer.py:245–262` | 同上；AV到达前不得发末端车 |
| M12 / 130 `unserved` | 每城k每l，Σ_{A_k}g_ij^{kl}=d_l−z_l | `models/base_optimizer.py:267–278` | 同上；pickup/delivery总流相等，z连续 |
| M13 / 134 `domain` | u binary、x/y非负整数（待D4）、g/ĝ/z非负连续 | `models/base_optimizer.py:32–69`的z是integer | `mandatory_optimizer.py`；分数需求/z，未选车可为0 |
| M14 / 84 `eq:obj` | 只含人工/AV服务时间费用和未服务罚金 | `models/base_optimizer.py:72–88`；`configuration.py:161–189,245–250` | `configuration.py,validation.py`；三项费用独立重算，单位换算一次 |
| M15 / 独立验收要求 | 完整导出x/y/u/g/ĝ/z及索引/容差/域，以便重算 | `experiments/solvers.py:114–118`只导出y/z且过滤≤0.1 | 新V1结果层；0.05合法连续量不得丢失；V0导出不改 |

M15是实现验收要求，不是新增源模型约束。物理量非负/有限、整数fleet等输入验证是D6建议合同，不伪装为源文逐项已声明。

## 5. 由原式得到的核验性质及范围

- 两城AV库存式相加蕴含总在途车数上界；保留93有助追溯，不改变该闭合模型。
- 全部弧在[0,T]完工、起始无货物存量时，M10/M11在t=T与M12夹逼出每单AV总流=已服务量。该核验不能直接移入带历史货流的RH子问题。
- 按D4非负域，x/y/u/g/ĝ全0、z=d是全拒单可行解，条件为d/N非负、没有最低服务率/强制发车/非零历史承诺等额外限制。非负费用/有限罚金用于成本解释，不把可行性偷换成任意最优性结论。
- 无订单不需要出车；空订单方向仍可有AV空载迁移，不能按“无该向订单”删除所有该向车辆弧。无manual跨城迁移，因此V1-F的local迁入上界问题不阻断V1-M。
- K(v)/v非减等代数沿用先前审查的限定范围；不声称K凸、全域严格超线性、有限样本routing精确或真实逐单deadline保证。

## 6. 手工验收案例（仅设计、未执行）

案例以源式物理时间为单位；F4另给分钟/小时等价换算。所有未列车辆/selector/流均为0。

| ID | 输入与候选安排 | 手工预期 / 检查对象 |
|---|---|---|
| F1 零tour预算 | f(q)=q，ρ=1，t0=1，M=10，弧(0,2)，v≥1 | K(v)=h(v[2−2]^+)=0。任何正货违反M06；弧仍在定义域，可空跑，不等于强制删除弧 |
| F2 共享AV | T=3，τ=t0=1；N1=N2=1，Nhat1=1/Nhat2=0；两城a=.5,b=ρ=0,M=2,Mhat=1；同向两单d=.75、s=0/e=3；三腿(0,1)/(1,2)/(2,3) | local各容量2，AV合计仅1；两单各载.75违反M08。费用0、单位罚金1时最小未服务总量.5，手工可行解各服务.5、各z=.25 |
| F3 连续z | 同F2但只有一单d=1.5；三腿各流1，两个local的u1=1，AV y=1，z=.5 | 满足全部列式，费用0/罚金1时最优手工值.5。整数z会错误排除该解；同时验证未选弧车数0 |
| F4 成本/同刻交接/末端窗 | T=3，τ=1，t0=.5小时，a=.5小时/stop，b=ρ=0，M=Mhat=1；两城各1HV，origin 1AV；d=1、s=0/e=3，三腿(0,1)/(1,2)/(2,3)；c=10/chat=4每车小时，罚金100 | 车辆费2×.5×10+.5×4=12，最优服务全单。换成t0=30分钟、a=30分钟/stop仍为12；e改2则三腿无法完工，全拒单成本100。依赖D3/D5批准 |
| F5 小量完整导出 | 沿用F4等价分钟版本的有效配置，但所有车队置0；一单d=.05，单位罚金2、s=0/e=3；按D4全拒单 | z=.05，成本.1；完整结果和独立核算必须保留该量，不能沿用旧>0.1导出筛选 |

F2/F3的T=3使正服务必须经过三个连续正时长腿，唯一有效AV载货时段为(1,2)，因此shared Mhat=1给出相应上界；它们不是solver输出。数值相等先后以独立验算+手工/枚举oracle核验，不能用赋Start个数代替可行性。

## 7. G1/G2拟议实现及验收门槛

批准D1–D6及实施范围后，建议新建`intercity_delivery/v1/configuration.py`、`capacity.py`、`network.py`、`mandatory_optimizer.py`、`validation.py`及V1专用测试。路径尚未创建；不改V0类、loader、默认配置、registry旧名、tests期望或历史数据。第一版独立入口，现有CLI/GUI接入另作为相应G4工作。

| 阶段 | 必须交付 | 验证与完成标准 |
|---|---|---|
| G0-M-PREP 当前项 | 本规格、独立审查、决策表 | 每个源公式有出处/代码差距/验收；选项影响清楚；待作者选择未冒作已定 |
| G0-M-CONFIRM | 作者D1–D6决定及实施范围 | 采用方案和延期内容书面记录；仅“规格包准备完成”不能代替该状态 |
| G1 | 纯配置/逆函数/容量/母弧域模块 | 非法参数/NaN/Inf、θ=0/b=0/小θ稳定性、代回f(h)、K边界/cap、N=0、τ>T、ceil与j=T、半开占用测试通过 |
| G2＋对应G4 | 最小完整V1-M、完整导出、独立验证器 | M01–M15逐项可追溯；逐单逐t流/车库存/域/窗/terminal残差通过；三项费用重算一致；F1–F5手工oracle；V0保护检查 |

纯逆函数可用数值稳定等价式`h(θ)=[2θ/(sqrt(b²+4aθ)+b)]²`；θ=0单独处理，避免b=θ=0的0/0。测试应代回原f，不用同一个逆函数自证。

独立validator须从配置、订单、完整schedule重算：人工/AV占用与每hub库存、one-hot与车数、K与共享AV负载、每单每t两段前置关系、两端已服务及AV总流、时间窗、整数/非负域、三项成本。结果中记录model_id、source/data hashes、单位、offline信息标签、版本/容差/退出状态；未完成或无incumbent不得伪报通过。

未来micro预算建议（未执行、未批准）：最多12例、每例≤5秒、Threads=1、总solver预算≤60秒，T≤8、1–3笔合成单、每城0–2HV、AV总数≤2；另记总墙钟。先纯函数再微型MIP；license不可用记unavailable，限时有解记限时可行，无解不记测试通过，不自动调用旧全套Gurobi测试。

## 8. 对原Q1–Q12的范围收敛

| 原问题 | 本基准处理 |
|---|---|
| Q1/Q2 | D1/D2必须确认；不由PI替作者选择 |
| Q3/Q4 | 仅V1-F的迁入上界、manual空返/复位延期；V1-M不新增AV终端归位 |
| Q5/Q6 | D3/D4/D6；RH reveal单独延期，offline benchmark标签必须明确 |
| Q7 | RH窗口延期；本模型仅0…T全域及半开占用，不以旧candidate chapter作为oracle |
| Q8 | 不加载direct/ratio/transfer扩展，不进入V0-F配置；F政策留G3决定 |
| Q9 | D5确认tour-end；S/真实随机期限标准延期 |
| Q10/Q12 | S证明边界与V2形式化独立研究，不阻断本限定基准 |
| Q11 | Bib/校准缺失继续披露；不声称已编译、校准、文献创新或真实routing验证 |

正文与旧算法章节的冲突没有因本包而被修复。若批准D2-A，只能按记录的arc-wave公式解释推进；不能悄悄把解释再标成作者已修改原稿。
