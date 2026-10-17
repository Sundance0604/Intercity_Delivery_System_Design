# BHH 模型数学比较与迁移前置问题

审计日期：2026-10-17。责任人：Theorist。仅创建本报告；未运行实验或求解器，未修改 TeX、代码和历史结果。下列“已核实”指实际文件定义或本轮独立代数，不代表稿件命题已全部通过独立审查。

## 1. 来源与版本边界

已读 AGENTS.md、INITIAL/UNVERIFIED 的 MODEL_CONTRACT.md，核对 Git 状态。工作树原本已有大量未跟踪研究和实验文件，不作为本轮产物。读取全部四份 TeX、旧/新稿差异、local QVI 笔记及相关代码。

| 对象 | 本地首要依据 | 数学性质 |
|---|---|---|
| V0-base | intercity_delivery/models/base_optimizer.py 与 data/loader.py | 单车逆 BHH 容量、固定城市人工车队；AV 容量按每个订单分别限制 |
| V0-flex | intercity_delivery/models/flexible_direct_optimizer.py 及候选/滚动派生 | 旧容量上增加 direct、累计 HV 迁移、总 AV 容量、可选换装成本/时间、直送比例 |
| V1 | research/paper/main_v2_BHH.tex | 离散有限时域、整数车数、连续 stops、wave 容量表 MILP；求解章节尚沿用旧式 |
| S-current | research/paper/steady_state_model.tex | 恒定对称需求、common timetable、连续 wave/载量、deadline、sorting、慢车因子 |
| S-linear | research/paper/steady_state_model_v1_linear.tex | 无 deadline/finite sorting/慢车因子的早期稳态分析 |
| V2-local | research/theory/Common Wave 的内生化与非均匀需求扩展.md | 确定性恒定/时变 λ 的 renewal 与形式 QVI；没有完整随机到达问题 |

旧稿 main_BHH.tex 不是 V0 全部实现的替代：旧稿 auto vol（103）左端已经是订单总和，实际 V0-base（184–192）却为逐订单容量；V0-flex（352–357）才是总和。因此不能说“原实现与旧稿完全一致”。

依赖：四份 TeX 均无 input/include；自定义 theorem 环境和 proof 宏都在本文件，无外部宏文件。两份 main 的第 7 行引用“导出的条目.bib”，本地 rg 文件清单未找到任何 .bib，不能完整复核该引用或编译。两份 steady 未导入 bibliography。S-current19–25、358 提到 steady_state_check.py、steady_state_symbolic.py、bhh_tour_check.py，本地未找到；数值/拟合误差仅为稿件自述、未复现。S-current51/360 的 main_v2.tex 与实际 main_v2_BHH.tex 名称不一致；注释40–41 的 archive 路径与当前 S-linear 路径不同，版本关系待确认。

## 2. V1 的完整定义与代码映射

### 2.1 设置、变量与信息

V1:67–79：两城市、两方向；η_d 是 origin、η̄_d 是 destination。时间 0,…,T，单期 t0；τ 明定为正整数期数，物理干线时间 τt0。每单 l 的 (s_l,e_l,d_l,δ_l) 是 release、deadline、stop 数量和未服务罚金。每单位需求要求一处 pickup 和一处 drop-off；g 连续，订单类可拆分。N^k 是初始 HV，Nhat^k 是初始 AV，容量 M/Mhat。

f_k(q)=a_k q+b_k√q，a_k>0、b_k≥0。每辆 stem=2ρ_k；同 wave 的 v 辆车均分区域，共承担 f_k(q)，每辆 f_k(q)/v。车辆 x_{ij}^{kd}、y_{ij}^d、直送 w_{ij}^d 是整数；g_{ij}^{kl}、ghat、gtilde、z 非负连续；u、utilde 二元选择波内车数。domain（134、216）的 Z+ 是否含 0 应统一明确；wave link 允许所有 u=0，因此实现必须容许零车。

订单按期到达及滚动揭示不等于 stochastic optimization：稿内无到达分布、滤过、场景树或条件期望。V1 单窗口是已知活动订单的确定性 MILP。

### 2.2 强制换装

V1:82 的 AV 弧为 j=i+τ，local 弧为 0<j−i≤ceil((2ρ_k+f_k(M))/t0)，占用 i≤t<j。令 Δ=(j−i)t0、D=[Δ−2ρ_k]^+、h_k=f_k^{-1}，则

K_kij(v)=min{vM,h_k(vD)},

h_k(θ)=((√(b_k²+4a_kθ)−b_k)/(2a_k))²。

| V1 公式、位置 | 数学内容与现实现 |
|---|---|
| eq:obj 84 | Σδ_l z_l+ΣxΔc+Σyτt0 chat；base set_objective、flex175–214；flex 有额外 transfer cost |
| eq:num man veh 89 | 每城各方向 local 在途≤N_k；base set_constraints / flex228 |
| num auto veh 93；bal auto ± 97/101 | 总 AV 占用、逐城累计 departures≤初始+arrivals；flex271–318 可复用 |
| BHH 106；wave link 110 | Σ_l g≤Σ_v K(v)u_v，x=Σ_vvu_v，Σ_vu_v≤1；V0 无该表格二元结构 |
| auto vol 114 | Σ_l ghat≤Mhat*y；flex352–357 已有，base184–192 不符合 |
| violate 118；E137 | pickup start≥s_l，delivery end≤e_l；loader.generate_epsilon_sets128，base199–203、flex381–395 |
| transfer step 1/2 122/126 | 每单干线累计发出≤pickup 累计返回，delivery 累计出发≤干线累计到达；base207–261、flex398–463，后者 theta 非零会更严格 |
| unserved 130 | 两城人工总量分别=d−z；base267–277；flex 用 r_l 辅助拆分可等价 |
| domain 134 | 整数车辆、二元 wave、连续流；零值域须明确 |

本轮已独立核对代数：两 AV 库存式相加，departures−arrivals 等于在途数，故 AV 总量式冗余，V1:137 的解释成立。对每单在闭合终点 T，两城总流同为 d−z 且所有弧完成，累计两式夹逼干线总量=d−z；这依赖终点全部约束，不能直接套入有初始在途货和 extended window 的滚动子问题。

### 2.3 直送

V1:141 定义

ftilde_d(q,v)=2ρ_origin+f_origin(q)/v+τt0+f_dest(q/v)。

origin zoned、destination 全城散点，不是两端均 zoned。直送弧 τ≤j−i≤τ+ceil((2maxρ+f1(M)+f2(M))/t0)；Ktilde=max{q∈[0,vM]:ftilde_d(q,v)≤Δ}，无解约定 0。v=1,…,N1+N2。j−i=τ 是特殊零容量空 reposition，不因 2ρ 的货物服务公式删除。

m2:obj145–146 加 wΔctilde。m2:num man veh154 是 local occupation+cumulative outbound−cumulative inbound≤N_initial。m2:BHH/wave link172/176、direct BHH/direct wave link184/188 分别使用 local/direct 表。m2:violate direct196、219 加 direct 时间窗；m2:unserved208 将 direct 计入两城需求守恒；m2:return212 要求正反 direct 总车次相等，恢复终点 HV 库存。AV 不要求终点归位，不应擅自加上。

本轮已核对：终点归位是“车次平衡”，不是货量平衡；所有 direct 弧在 T 完成时，它与累计库存式共同恢复两城总 HV 数量，不要求车辆身份逐个原路配对。

## 3. 新旧数学差异

| 项目 | 旧稿 / V0 | V1 变化依据与域后果 |
|---|---|---|
| 需求单位 | main_BHH69 的 type-l number 含义不清 | V1:69 一单位=一个 stop；真实吨位/货件须先转换 |
| manual flow 城市索引 | 旧稿73、99、119 的 g 缺 city；base46 起已含 city/flow/order | V1:79、106、130 显式 g^{kl}；文稿修正不等于代码从零新增 |
| arc/time units | 旧稿76 j≤i+f(M)；loader85–89 int(f(M)) 未除 t0；state_dependent149 已除 | V1:82 ceil((2ρ+f(M))/t0)，增加 stem、统一单位；V0 各入口原本不同 |
| routing capacity | 旧稿99；base176、flex342 是 x*h(Δ) | V1:82、106、110 是 h(v[Δ−2ρ]^+) capped by vM，wave economies；与 V0 无统一可行域包含关系 |
| M cap | loader166–180 inverse 没 min(M,·) | V1:82 明确 vM；旧长弧可能超单车容量 |
| AV 容量 | 旧稿103 为总和；base184–192 逐单；flex352–357 总和 | V1:114/180 总和；base 要修，flex 可复用；不是新稿才首次提出总和 |
| direct service | 旧稿138 f1(q)+f2(q)+τ；flex78–88 合并 a/b inverse，372–373 乘 w | V1:141 方向相关、origin zoned、destination unzoned、stem、wave v；需替换单车系数 |
| manual relocation | 旧稿148 活动 arc 出入，时空不清；flex228–269 已累计出入 | V1:154/221 累计车辆库存；实际 flex 已有新稿这部分 |
| empty/terminal | flex68 仅 j≥i+τ+1，留正容量 direct；无 return 式 | V1:141 空 arc τ，212 正反车次相等并计费，改变有限期收益 |
| extra mechanisms | flex203–205 transfer cost；398–445 theta；523–543 ratio bounds | V1 无这些机制；精确 V1 模式需中性参数/明确扩展标签 |
| dynamic/stochastic | 已知订单滚动求解 | V1:226–252 仍 rolling MILP，不是随机 QVI，也无全日随机最优保证 |

代码算法对应：state_dependent_mip.py:136 的 candidate generation 与310–327 的旧 direct 系数需同步；bhh_priority_heuristic.py:65–95 用 existing_vehicles*capacity−load 和 ceil(required/capacity)，新波增车应使用 K(v_new)−K(v_old) 或查最小可行整数 v；paper_rolling_horizon.py:114 的 commit 需使 wave selector/vehicle/flow/history 口径一致。本轮未改代码。

### 3.1 独立核对的规模经济结论

这是模型内代数，不是实车 routing 定理：h 凸、h(0)=0，故 h(vD)/v 随 v 非减；K(v)/v=min{M,h(vD)/v} 非减，由此在定义域内 K(v1+v2)≥K(v1)+K(v2)。但是被线性 cap 截断后的 K 一般不再凸。b=0 或 D=0 时为线性/零容量，原文 more than proportionally 不能无条件解释成严格增。

相同 stem 且未 cap 时，v*h(D)≤h(vD)；与旧模型无 stem 的 v*h(Δ) 无一律大小关系。令 q=W、n=W/v，单车 local 时间 2ρ+f(W)/v=2ρ+(a+b/√W)n，正是 steady zonetour 形式。但共同形式不证明整体等价：V1 无 δ、sorting、lead-time cost、common timetable。

## 4. V1 的重要歧义与反例

1. **wave 索引**：71 说同城同方向同期开车共享 wave；106/110 却按 (i,j) 选择。允许同 i、不同 j 的多个 wave 各用全城面积参数 b。若是 start-wave，缺 aggregate stops/车数与各时长共享 zoning 关系；若是 per-arc wave，应收窄正文。不能擅自认二者等价。
2. **迁入 HV 与 local 上界**：154/221 允许迁入后 local；172/176 仍只 v≤初始 N_k。例 N1=1,N2=2，城2两车到城1后可有三车 local，表仍最多一车；N1=0 则迁入后仍没有 u。需确认 ownership 还是表上界。
3. **candidate chapter 仍旧式**：340 j≤i+f(M) 无 stem/t0；369 的 ftilde(M) 未定义 v/d；374 要同向订单，删掉无货回程空弧，与141/212冲突。303–311 只减 active direct occupation，不持续保存完成迁移后的库存；指示量 1_{ijd}^k 未定义。
4. **arc elimination419–426 仅条件性成立**：若 N_h(s) 是任何新方案都不能增加的真实剩余上界，0 当然排除跨期新占用。若它只是 inherited availability，新 direct 可以在 s 前调入车，0 未必排除 local：城1初始0、城2有1辆，可先迁入再 local。须证明 pruning 用上界而非仅既有供给；不能把这条条件性观察宣布为整个命题无条件错误。
5. **extension proposition429–444 未验证**：f1(M)+f2(M)+τ 整体除 t0，τ 在正文是期数；遗漏 stem、分腿取整；ftilde(M) 已非单参定义；hub 等待可以任意增加。必须区分“存在无等待完成路径”与“任意可行 option 都完成”，先给新模型明确上界后再证明。
6. **启发式可行性有手工反例**：525–527 用 C=h(Δ)，539–565 仍按固定 C 分货增车，遗漏 stem/M/K/u。取 f(q)=q、ρ=1、Δ=2，则 C=2，而新 K(v)=min(vM,h(v[2−2]^+))=0，任何正货违反正文。未运行实验。623–645 的单参数 direct inverse 同样旧式；输出无 u，也无212终点归位保证。686/690 的“feasible”不能直接成立。
7. **shortest-service 下界**：455 把 distribution 弧写为 direct 集；三段路径没有明确对全部可行三元组取最小；缺 release/正容量/方向筛选。空返零容量弧不得当载货最快 option。
8. **rolling 边界未闭合**：230–252 只 commit 起点在 control window 的任务；已 pickup 货物后续腿被重算时，需 hub/在途存货初始源项，不能当新订单重新 pickup 或消失。212 是每窗口还是全日归位？罚金是每窗预测还是实际到期结算？需要作者确认。
9. **stop 连续近似**：69 一单位一 stop，79/134 连续拆流，因此可把单客户分数拆到多 arc；只能是 continuum order-class 近似，不能称精确 customer routing。均分、空间均匀和小样本 BHH 精度需校准。

## 5. 两份稳态分析与 V1 的关系

S-linear49–78：恒定对称 Λ，独立均匀空间；balanced zones 同时返回；fleet 随负荷配置，忽略 queue；选 (W1,m,W2;n1,n2)，n_i≤min(M,W_i)。均值时间成本 ω 与固定 transfer Θ=2(ωθ+h)，无 deadline、finite sorting、δ。

S-current57–89：同样对称恒定需求和连续变量；common timetable W=ΛH 为 assumption；慢车=(1+δ)(2ρ+s(W)n)，μ=κΛ、κ>2，deadline L。这里 δ 是 routing imbalance，不是 V1 订单罚金 δ_l；L 是 deadline，不是 rolling extension；τ 是物理时间，而 V1 τ 是期数。

| 命题/公式 | 新旧变化与证据 | 当前科学状态 |
|---|---|---|
| consolidation lemma | S-linear82–90；current93–101，nested/整除下平均(max q−1)/(2Λ)，current增最大等待与一般下界 | 稿内有证明；本轮核对 closing-shipment 思路，在同序固定 stage 时长、无额外排队假设内成立；Poisson仅期望 |
| common wave | linear prop:wave101–114；current prop:common103–111 | 稿内有证明；独立核对提高各 wave 至 max 的比较代数，仅 κ=∞、nested、同 δ；one-by-one 比较另需 δ=0；不是有限 sorting 定理 |
| loads | linear104–105；current138–149 | 稿内有严格凸/KKT证明；固定 W、正载量、仅 M cap 的代数核对正确；n≤W 遗漏使全域声明待修正 |
| lead times | current119–122 | Tmean=γbar W/Λ+s[(1+δ)n1+n2/2]+Fbar；Tmax=γhat W/Λ+(1+δ)s(n1+n2)+Fhat；γbar=1/2+2/κ、γhat=1+2/κ；模型假定，不是由 V1 推导 |
| cost-bound wave | linear119–132；current164–188 | 稿内证明凸性、唯一最优、下界、比较静态、2/3渐近；本轮未完成独立全域证明，特别 n≤W 域与cap边界须重审 |
| regimes | current201–214 | 稿内附证明；未完整独立验证含边界/可行性全域的 threshold 声明 |
| full-load deadline | current219–235 | 稿内条件性证明，只在 deadline bind 且 n1=n2=M 的 regime；不能无条件外推成本不影响 wave |
| interior deadline | current239–240 | 原文明标数值观察、无比较静态证明，本轮未复现 |
| fleet/sorting | current243–245 | 平均 busy 数由守恒作代数解释；不等于固定 fleet 可行性/峰值保证 |
| direct benchmark | linear155–182、185–294；current247–259 | linear 的 zoned critical-distance monotonicity 明说无证明；whole-city 一组命题与 current hard-deadline benchmark 不同。current cost/time identities 可代数核对，不给最优比较定理 |

当前目标 G（128）为

c[2s(W)+2ρ/n1+2ρ/n2]+chatτ/min(W,Mhat)+ωTmean+2h，

P（132）要求 Tmax≤L、n_i≤M。S-current360 自己明确 time-expanded program 无 cost of lead time、sorting rate、timetable，必须先对齐再验证。V1:223 的 finite-horizon counterpart 是研究定位，不能当已证 equivalence。

### 5.1 需复核的稳态边界

- **n≤W**：current89 明说 n_i≤min(M,W)，P132 与loads138–139却只cap M。手工例 a=b=c=ρ=ω=1、δ=0、M≥2、W=1/16，s=5，n1=√(2/5)>W；取大L可使deadline松弛，仍违反 setting。linear107 有 provided loads do not exceed W，current缺对应限定。先选大波regime或补cap后重推，不能宣称全域已证。
- **AV 连续车数近似**：chatτ/min(W,Mhat) 在 W>Mhat 相当于车数 W/Mhat 可连续；整数波实际为 chatτ*ceil(W/Mhat)/W。批量连续的 approximation 可以保留，但不是 V1 整数精确式。
- **ratio**：current143 明说 neither load at capacity；上端 iff slack 还需 ω>0。ω=λ=0 的 loads 分母为零需满载极限；双满载可n1=n2，不代表 deadline 极紧。
- **Poisson 不给 hard bound**：current101 的平均可按期望解释，随机 fill time 无有限确定上界；120 的 W/Λ 不得直接当 Poisson deadline 保证。
- **fleet average vs peak**：244 平均忙车和 wave 用车量不保证重叠相位峰值可行；不能用平均数设 V1 N 后称模型对齐。
- **sorting utilization vs queue**：2/κ 只总平均负荷；两方向相位、共享 sorting 资源的排队/开始时间需明确，κ>2 单独不保证每波实际总 sorting 延迟都仅 W/μ。
- **数值不升级**：237“full-load 条件在所有算例也充分”、240 interior规律、358拟合b/δ均稿内数值自述，未复现，不是定理。

受控 S↔V1 验证至少要统一需求、空间分区、物理时间、service/deadline口径、lead-time/transfer/sorting、同步政策、峰值fleet、整数松弛和终点边界。当前不满足。

## 6. V2：本地 QVI 的实际内容与证明缺口

笔记138–176 是 qdot=Λ、dispatch 后q=0；180–230 定义 R=G−ωW/(2Λ)、C(q)=qR°(q)。246–443 的 complete-cycle shipment-weighted average 与 J(q)=ΛG°(q) 代数成立，故 argmin J=argmin G°。这是在 common-wave、cost-bound、deterministic constant Λ、renewal-separable、无跨周期fleet/sorting/pipeline作用下已证明的代数关系，不能证明随机到达最优或三段同步。

463–545 的形式 average-cost QVI：min{ωq−g+ΛV'(q), C(q)+V(0)−V(q)}=0。

650–706 的 finite-horizon 形式：min{V_t+λ(t)V_q+ωq, C(q)+V(t,0)−V(t,q)}=0。

对反复瞬时 reset 的成本最小化，两项符号相容；但没有 verification/existence/threshold theorem，所以状态是“形式控制方程，待验证”。711 起原文是“如果 optimal policy 具有 threshold structure”，不能改成已证边界存在。

| 构件 | 笔记实际给出 | 随机/完整网络仍缺 |
|---|---|---|
| state | q 或(t,q) | fleet位置/剩余任务、两hub库存、sorting；硬期限需age/deadline状态，通常不止oldest age |
| information | 已知确定性Λ或λ(t) | 概率空间、filtration、到达/mark、forecast、nonanticipativity |
| admissible controls | 全backlog reset的 repeated dispatch | stopping times、partial dispatch、composition/车辆/路径、no-Zeno、q=0无效干预限制 |
| intervention | MV=C(q)+V(0) | 非瞬时物理服务为何能预付且无剩余状态、renewal条件、λ变化时sorting参数定义 |
| objective | waiting integral ωq+C(q) | expectation、期限/丢单成本、平均/折现口径 |
| horizon/boundary | average g 或[0,T] | V(T,q)清仓/罚金；normalization、growth/transversality、verification假设 |
| threshold | constant renewal最优；时变边界条件式 | stochastic threshold、monotonicity、smooth fit未证 |

complete-cycle weighted average 是正确证明；若推广为任意日历时刻长期 average，还需累计时间趋无穷、排除干预聚集、控制末尾未完成周期。hard deadline与sorting/pipeline会破坏独立周期，不能直接复用。

候选接口而非已完成结果：若到达是单位 Poisson，continuation generator 应含 λ(t)[V(t,q+1)−V(t,q)]，而不是 λ(t)V_q；批量 marks 需jump-size积分。必须先由用户定义随机过程。本轮未证明任何新随机定理。

笔记807–901、966–1004明确区分 temporal stationarity 与 stage synchronization；finite sorting下common wave最优仍开放，S-current111/359同样承认 smaller collection waves 可有pipeline benefit。V1 wave容量、S common timetable、V2 intervention时刻自由属于三个层次。

## 7. 需要作者确认的问题与有限后续任务

1. V1-main是否实施目标、S-current是否分析目标、S-linear是否仅历史参照？是否要求本轮后同时加入sorting/lead-time，还是仅精确复现V1？
2. wave按start还是完整arc？同start不同end怎样共享区域与密度？
3. 迁入后允许local时，u表上界是否突破初始N_k？
4. V0 transfer time/cost/ratio如何处理？业务量到底为stops、shipments、pieces还是volume？
5. local/direct closed-tour完成与最后delivery是否同口径？是否加入稳态δ？
6. 空返仅τ、无stem是否业务本意？return每滚动窗口还是只全日结尾？
7. S-current补n≤W/正域/整数松弛说明后，哪些命题需重审？
8. rolling在途货、hub库存、未来腿撤回、deadline罚金如何结算？
9. V2先完善确定性nonstationary问题，还是直接Poisson/marked arrivals？terminal/admissibility是什么？
10. 缺失bibliography及校准/符号检验脚本能否提供？缺失期间保留未复现标识。

依赖顺序：先锁定上述合同；再制作容量表零边界/单位/上界验证规格与正文—算法一致性修订方案；再复核S完整域命题、设计共同机制下S↔V1验证；V2另立state/information/control/cost/boundary任务后，才研究verification与threshold。未在本轮执行这些任务，不将生产MIP结果当QVI证明，不将renewal代数当全网络routing定理。
