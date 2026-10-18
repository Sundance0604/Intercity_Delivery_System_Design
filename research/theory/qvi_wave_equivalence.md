# QVI 与最简 common-wave 模型：同参等价、全域验证与适用边界

2026-10-18。Theorist。仅新增本报告；未改 TeX、生产代码、历史数据，未编译、未运行求解器。

状态约定：**proved-in-report** 表示在本报告明确假设的约化模型中给出证明；**conditional** 表示应用于实际运输系统尚依赖额外假设；**open** 表示尚无本轮证明。下述证明不是对论文全部命题的认证。

## 1. 哪个版本是当前稿，trivial 应如何定义

`research/paper/steady_state_model.tex` 第27–41行记录当前稿在2026-10-01/02加入慢车因素、保留 sorting 和硬期限，并明确将 linear 稿称为 earlier draft。因此，以 **current 为主稿，linear 为简化参照**，不按文件名或磁盘修改时间反推版本。

共同来源对应：

| 数学对象 | linear 精确行号 | current 精确行号 |
|---|---|---|
| 恒定率、对称均匀空间 | 49–50 | 57–58 |
| zoned BHH tour，s(q)=a+b/√q | 53–61 | 61–69 |
| 按负荷配车、忽略车辆排队 | 65–66 | 77–78 |
| n_i≤min(M,q)，连续批量 | 75–78 | 89 |
| common-wave 成本的原始定义 | 94–99 | 113–130 |
| 无 sorting 时 common wave 比较 | 101–114 | 103–111 |
| sorting、硬期限及慢车带来的差异 | 不包含相应机制 | 81–86、115–120、131–149 |

本报告定义 **T0（trivial common-wave）** 为两稿的严格共同子模型：恒定确定性 fluid 到达；common wave；δ=0、κ=∞；不施加硬期限（或所比较的最优方案确实满足且不绑定期限）；连续 q、n；按负荷配车；服务成本跨周期可分。保留 n_i≤min(M,q)，不删掉 q 上界。θ、h 可非零，因为其常数/逐单成本不会破坏下面的等价。

**conditional：** 这是一维动态模型的定义范围，不是“current 全模型已经退化为 QVI”的结论。current 第132行的显示问题及第138–139行、第162行的负荷表达式只写 n≤M，遗漏第89行的 n≤W；linear 第104–107行也承认公式需满足 n≤W。本报告从原始 G 重新优化负荷，不直接继承这些缺域的闭式或全局凸性声明。

## 2. 同一参数、可行域与目标

取 Λ,ω,c,ρ,b,M>0，a≥0，Mhat≥M，chat,τ,θ,h≥0。时间统一为一种物理单位；c、chat 是货币/车辆/时间，ω 是货币/shipment/时间，q,n,M,Mhat 是 shipment（一个 shipment 各需一 pickup 和 delivery stop）。定义

\[
s(q)=a+b/\sqrt q,\qquad 0<n_i\le\min(M,q),\quad q>0.
\]

两稿在 T0 上的共同成本为

\[
\begin{aligned}
G(q;n_1,n_2)={}&c[2s(q)+2\rho/n_1+2\rho/n_2]
+\frac{\hat c\tau}{\min(q,\hat M)}\\
&+\omega\left[\frac{q}{2\Lambda}+s(q)(n_1+n_2/2)+3\rho+\tau+2\theta\right]+2h.
\end{aligned}
\]

由严格凸的一维负荷问题，**proved-in-report**：

\[
n_1(q)=\min\left(M,q,\sqrt{\frac{2c\rho}{\omega s(q)}}\right),\qquad
n_2(q)=\min\left(M,q,\sqrt{\frac{4c\rho}{\omega s(q)}}\right).
\]

令 G°(q)=min_n G(q;n)，R(q)=G°(q)−ωq/(2Λ)，C(q)=qR(q)。C 是一批货在 dispatch 后发生的广义服务成本，包含车辆时间成本和 dispatch 后 lead-time 成本。**没有再次计入 dispatch 前等待。** C 的单位为货币/batch。

**conditional（关键会计约化）：** dispatch 时预付整批未来服务成本，并只把“未发 backlog”清零。物理上的车队、在途货物、hub pipeline 并不瞬间归零。要把本模型当作真实系统，须假定这些服务任务互不争用有限资源、未来成本可按批可分、且没有随后会改变费用的随机信息。若按实际发生时刻记账，还须控制尚未发生的已预付成本与已发生的未记成本差额，使其为 o(T)；仅有稳态 fleet 平均 busy 数不构成该证明。

## 3. 确定性控制问题的完整定义

状态为 Q(t)≥0（未发 shipment fluid 量）。无随机输入，信息为已知常率及已发生动作；等待时 dQ/dt=Λ。动作是非预知的全量 dispatch：从 q>0 清到0，付 C(q)。无拒单、无部分发运、无车辆/年龄/站内库存等其他状态。

从 Q(0)=0 出发，一项 admissible 策略选择 q_k>0，dispatch 时刻 t_k=(q_1+⋯+q_k)/Λ，并要求 t_k↑∞（有限时段只有有限次动作，排除 Zeno）。在时间 T 内计入已 dispatch 的 C 和所有等待成本：

\[
A_\pi(T)=\sum_{t_k\le T}C(q_k)+\int_0^T\omega Q(t)\,dt,
\qquad \inf_\pi\liminf_{T\to\infty}A_\pi(T)/T.
\]

**proved-in-report（非 Zeno 充分条件）：** q_k≥ε>0 保证 t_k↑∞；重复任何正阈值满足。更一般地直接要求 Σq_k=∞ 即可。这里 C(q)≥4cρ，因为两个负荷均不超过 q；因此无限次 dispatch 在有限时间内会付出无限费用，不可能改善有限值最优解。但仍在 admissibility 中显式排除这类路径。

## 4. 周期恒等式与任意日历时域证明

**proved-in-report：** 每个完整周期时长 q/Λ，等待成本 ωq²/(2Λ)，所以

\[
J(q)=\frac{C(q)+\omega q^2/(2\Lambda)}{q/\Lambda}
=\Lambda G^\circ(q),\qquad
\operatorname*{argmin}_{q>0}J(q)=\operatorname*{argmin}_{q>0}G^\circ(q).
\]

目标是每方向的成本率；若双向对称同步，总成本率为2ΛG°，同一 argmin。

存在性不需要引用两稿的凸性命题：C 连续非负，J(q)≥4cρΛ/q+ωq/2，因此 J 在0及∞都发散，并在正紧区间达到最小值 g。最优阈值集合 B=argmin J 非空紧。本报告不在未审完分段负荷域之前声称该集合必为单点。

设 T 时完成 N 个周期，残余等待时长 r=T−t_N、当前 backlog Q(T)=Λr。精确分解为

\[
A_\pi(T)-gT
=\sum_{k=1}^N\frac{q_k}{\Lambda}[J(q_k)-g]
+\frac{\omega\Lambda r^2}{2}-gr
\ge-\frac{g^2}{2\omega\Lambda}.
\]

因此 **proved-in-report**：对任意 admissible 策略，完整日历时域的 liminf 平均成本≥g，**不只是在 completed-cycle 时刻成立**，也不需要事先假设残余周期有界。任取 b∈B 重复，r≤b/Λ，完整周期取等，故完整日历平均极限=g。有限初始 backlog 可一次 dispatch 后进入该策略，其有限费用不改变平均值。

这是对本地 QVI 笔记第246–361行仅在完成周期取加权平均的证明的补齐；其第385–445行的同解恒等式在上述域和会计约化下成立。

非唯一性须区分：即使静态 minimizer 唯一，允许任意有限个非最优 transient 周期仍然平均最优；不同 minimizer 的周期可以混合。也可让非最优周期的累计加权 excess 为 o(T)。所以结论是 **静态最优批量集合等于最优重复阈值集合**，不意味着所有平均最优策略逐期相同。

## 5. 全 q 域 average-cost QVI 与验证

干预算子 MV(q)=C(q)+V(0)，只在 q>0 可用。归一化 V(0)=0。最小化问题的正确符号为

\[
\min\{\omega q-g+\Lambda V'(q),\ C(q)-V(q)\}=0.
\]

两个分支均须非负。理由是等待一个短时段或立即 dispatch 都是可用动作，最优 V 不能高于任何一种动作的成本；等待展开得到 ΛV'+ωq−g≥0。把该式机械换成 max 而不同时改变两项的符号将出错。

定义

\[
F(q)=C(q)+\frac{\omega q^2}{2\Lambda}-\frac{gq}{\Lambda}
=\frac q\Lambda[J(q)-g]\ge0,
\quad m(q)=\inf_{z\ge q}F(z),
\]
\[
\boxed{V(q)=\frac{gq-\omega q^2/2}{\Lambda}+m(q).}
\]

**proved-in-report（全域构造）：** F 在(0,∞)连续且局部 Lipschitz，在0有有限右极限，在∞发散，tail 最小值达到。这里不声称 F 在0局部 Lipschitz；其 BHH 项可含√q。m 在(0,∞)非降且局部 Lipschitz，而在最小最优阈值之前恒为0，故所构造 V 在0仍为光滑多项式。m(q)≤F(q)，在严格不接触区 F(q)>m(q) 时 m 局部恒定。因此几乎处处

\[
C(q)-V(q)=F(q)-m(q)\ge0,
\quad \omega q-g+\Lambda V'(q)=\Lambda m'(q)\ge0,
\]

且至少一项为0。这给出带可验证绝对连续性质的 QVI 解；未在此额外宣称 PDE 解的唯一性或一般 viscosity 比较原理。

同一构造也是 first-dispatch 公式：

\[
V(q)=\inf_{z\ge q}\left\{C(z)+\frac{\omega(z^2-q^2)}{2\Lambda}-\frac{g(z-q)}\Lambda\right\}.
\]

从 q 等到实现 tail 最小值的 z 再 dispatch，然后从0采用最优重复阈值，实现该 reduced excess cost。初始0的首个接触点是 B 的最小元素；其余大状态的最优首动作由 tail 包络决定。

### 5.1 验证与横截条件

QVI 两个不等式沿任意策略积分并跨动作求和：

\[
A_\pi(T)-gT\ge V(Q(0))-V(Q(T)).
\]

若 V(Q(T))/T→0，直接得到平均成本下界。此处还能在有限 liminf 的子序列上**推导**所需横截性：

1. C≥0；选择 q≥1 的固定可行负荷 n_0=min(M,1)，s(q)≤a+b，得 C(q)≤Kq。小 q 时 C 有有限右极限4cρ+chatτ。因此 C(q)=O(1+q)。
2. first-dispatch 公式给 V(q)≤C(q)，且 V(q)≥−g²/(2ωΛ)，所以 |V(q)|≤K_0+K_1q。
3. 从0出发，当前未完成周期已经产生 ωQ(T)²/(2Λ) 的等待成本。任何 A(T_j)/T_j 有界的 T_j↑∞ 上，Q(T_j)=O(√T_j)，故 V(Q(T_j))/T_j→0。
4. 若 liminf A(T)/T 有限，取实现其极限的子序列即可验证≥g；若为∞结论显然。最优重复阈值 backlog 有界，横截性及等号均成立。

该证明无需把额外的任意策略全时刻 Q(T)/T→0 当作未证事实；它与第4节直接日历下界相互核对。对于在途物理费用，仍须另核第2节所述会计边界。

### 5.2 smooth fit、阈值上侧与非唯一性

令 b=min B。q≤b 时 m(q)=0，故 V(q)=(gq−ωq²/2)/Λ。若 C 在 b 可微且 b 为内点，则 F'(b)=0，给出

\[
V(b)=C(b),\quad V'(b-)=C'(b)=\frac{g-\omega b}{\Lambda},
\quad g=\Lambda C(b)/b+\omega b/2.
\]

**conditional：** AV capacity kink b=Mhat 等情形只能先用单侧导数的0∈∂F(b)，不可强制经典 smooth fit。V(q)=C(q) 对全部 q≥b 需要 F 在[b,∞)非降这一充分条件；仅由 b 最小化 J 不自动推出。若有更远的低 valley，已超 b 的初始状态可能还应继续等。tail 包络覆盖该情况。

因此“从0以 b 为阈值并每次清零”已全局平均最优；但“任意初始 q≥b 都立刻 dispatch”及“全状态唯一阈值策略”没有被简单周期恒等式自动证明。完整可达再生轨道上的结论和所有状态的最优首动作须分开。

## 6. 真随机到达：单位 Poisson 的精确特殊结果

**proved-in-report（限定阈值策略类）：** 若每次到达恰为一个 shipment，N(t) 是率Λ的 Poisson 过程，状态为整数未发数 n，仍为同一 C(n)、线性等待成本、全量 reset、无 marks/age/fleet。对整数阈值 B≥1，周期长度期望 B/Λ，各个状态0,…,B−1的停留时间期望均为1/Λ，故

\[
J_P(B)=\frac{\Lambda C(B)}B+\frac{\omega(B-1)}2
=J_D(B)-\omega/2.
\]

这里 J_D 是**同一整数阈值集合**上的 fluid 目标。因此这一策略类内 Poisson 和 fluid 的 argmin **严格相同**，最优成本率相差常数ω/2，而不是一般相同。这一特殊情形不能写成“随机性必然改变阈值”。同均值也不能把 Poisson generator 偷换成 ΛV'。

可核查离散 average-cost QVI 为

\[
\min\{\omega n-g_P+\Lambda[V_P(n+1)-V_P(n)],\ C(n)-V_P(n)\}=0,
\]

其中 n=0 禁止空干预。令

\[
F_P(n)=C(n)+\frac{\omega n(n-1)}{2\Lambda}-\frac{g_Pn}{\Lambda},\quad
m_P(n)=\min_{k\ge n}F_P(k),
\]
\[
V_P(n)=\frac{g_Pn-\omega n(n-1)/2}{\Lambda}+m_P(n).
\]

取 g_P=min_B J_P(B)，则 F_P≥0且在最优整数阈值接触0；m_P(n)=min(F_P(n),m_P(n+1))，故离散 QVI 的两项非负、至少一项为0；0态同样满足−g_P+ΛV_P(1)=0。**conditional：** 要据此将结论扩展到所有 adapted 随机停止/干预策略，须保证 stopped Dynkin/martingale 的可积性及期望横截条件，例如 E|V_P(N^π(T))|/T→0；本报告没有借整数阈值计算默默跳过这项随机 verification。

### 6.1 严格反例说明比较域的重要性

**proved-in-report（独立 setup-cost toy，非完整 BHH 系统）：** 取 Λ=ω=1、C(q)=9/8。continuous fluid 的唯一最优 q*=√(2C)=3/2，g=3/2。单位 Poisson 的整数阈值目标为 (9/8)/B+(B−1)/2，其唯一最优 B=2，g_P=17/16。不能要求3/2个不可拆 shipment 作为 Poisson 动作。若 fluid 也限制整数阈值，其最优同为2，成本25/16；两者正差1/2。反例反驳“只要参数和平均到达率相同，就自动得到同一连续解和同一目标值”，**不反驳同一整数动作域下已证明的阈值等价**。

一般 compound arrivals、随机 shipment marks、容量体积、随机服务时间、硬期限/年龄、非均匀率会引入 overshoot 或其他状态；上述常数差公式不自动成立。**open：** 这些机制下是否阈值最优、阈值如何变动以及如何与 current 的服务条件对齐。本轮不据均值Λ作替代证明。

单位 Poisson 在整数阈值 B>1 时，最老订单为凑批而等待后续到达，其等待没有确定性上界；B=1 则到达后立即发出，没有凑批等待。因而即使平均等待公式正确，也不能一般地直接保留 current 第85–86行的逐单硬期限保证。

## 7. 与现有 V1-F/RH 的关系，以及可进入论文的结论

**proved-in-report（模型差异识别）：** `main_v2_BHH.tex` 第67–82行是离散有限时域、固定初始车队、订单窗口/可拒单、弧与货流；第84行、第145–146行目标为车辆费用加未服务罚损，没有ω乘以实际 lead time。current 第360行已准确指出此未对齐。V1 distribution 在tour结束核交付，而稳态式平均为途中交付；整数车辆还不同于连续 W/n 和 AV min(q,Mhat) 装载近似。源码局部BHH相似不等于目标、可行域、状态或边界相同。

当前 V1-F 的独立 empty、滚动历史、城市迁移、终端政策解决的是另外的问题；RH 不是 QVI 的数值离散化。CFS 8单 smoke 只证运行，不证上述理论等价。

建议可写进论文的桥梁命题为：**在T0共同子模型及逐批可分会计约化下，允许 successive common-wave 大小任意变化不降低最优长期广义成本；重复静态最优 wave 达到全日历最优，其值和最优批量集合由 J=ΛG°给出。相应 QVI 可由全域 tail 包络验证。** 主文放周期解释及日历证明关键一行，附录放状态/动作/会计假设、tail 包络和 verification。该命题是在更宽的 successive-wave 政策类解释静态最优，不宣称有限 sorting 下 stage synchronization 内生最优。

建议写作依赖：先修 n≤W 与负荷公式域；统一单位、平均途中交付口径及逐批会计；再纳入上述限定命题。Poisson 用专门的小段列出整数阈值常数差和其边界。λ(t) 的有限期 QVI 需先定 terminal backlog 清理/罚损及服务是否跨T，再谈边界；时变阈值单调性和存在性仍 **open**。固定 fleet 与 sorting pipeline 的高维 QVI 应列扩展，不放入本轮已证命题。

本报告为本地原始公式的自包含推导，不新增文献或宣称理论创新性；公开文献定位由 Literature 独立完成。未执行数值验证，数值结果须引用专门 checks 产物，不能把本报告的解析证明当成全模型实验。
