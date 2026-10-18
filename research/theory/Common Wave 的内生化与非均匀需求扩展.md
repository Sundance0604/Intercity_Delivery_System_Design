# Common Wave 的内生化与非均匀需求扩展

## 从稳态批量优化到动态 dispatch boundary 的一个补充视角

> **说明**
>
> 这份笔记并不试图替代现有 steady-state model，也不认为现有模型中的 wave size \(W\) 是外生给定的。现有模型已经在 common timetable 的政策类中内生优化 \(W\)、\(n_1\) 和 \(n_2\)。
>
> 这里希望回答的是一个更窄、但可能具有理论意义的问题：
>
> **如果不预先假定 successive waves 必须具有相同大小，而允许运营者在每次 dispatch 时重新决定 batch size，那么现有 steady-state 最优 wave 是否仍能从一个更一般的 dispatch problem 中内生推出？**
>
> 在恒定需求、common-wave、cost-bound 且 renewal-separable 的情形下，答案是肯定的：动态 dispatch problem 的 stationary threshold 与现有 steady-state model 的最优 wave 完全一致。
>
> 在此基础上，当需求率随时间变化时，固定 wave 可以进一步推广为 time-dependent dispatch boundary。此时 dynamic formulation 不再只是对现有解的重新解释，而开始成为一种真正不同的调度建模方式。
>
> 本文暂不解决 finite sorting 下 collection、line-haul 与 distribution 是否仍应采用同一 wave 的问题。这个 stage-synchronization 问题在现有模型中已经被明确识别为 open issue，应作为下一步单独研究。

---

# 1. 现有 steady-state model 的基准

考虑三段同日达城际货运系统：

$$
\text{collection}
\longrightarrow
\text{line-haul}
\longrightarrow
\text{distribution}.
$$

在现有 steady-state formulation 中，三段采用 common timetable：

$$
W=\Lambda H,
$$

其中：

$$
\Lambda=\text{constant shipment arrival rate},
$$

$$
H=\text{common headway},
$$

$$
W=\text{shipments per wave}.
$$

城市端 zoned tour 的单位 stop routing time 为

$$
s(W)
=
a+\frac{b}{\sqrt W},
$$

因此

$$
s'(W)<0.
$$

这体现了 BHH aggregation effect：wave 越大，空间分区越细，单位 shipment 的 local-routing burden 越低。

在 common-wave 条件下，mean lead time 为

$$
\bar T(W;n_1,n_2)
=
\bar\gamma\frac W\Lambda
+
s(W)
\left[
(1+\delta)n_1+\frac{n_2}{2}
\right]
+
\bar F,
$$

其中

$$
\bar\gamma
=
\frac12+\frac2\kappa.
$$

generalized cost per shipment 为

$$
G(W;n_1,n_2)
=
c
\left[
2s(W)
+\frac{2\rho}{n_1}
+\frac{2\rho}{n_2}
\right]
+
\frac{\hat c\tau}{\min\{W,\hat M\}}
+
\omega\bar T(W;n_1,n_2)
+
2h.
$$

在 deadline 不绑定时，对 vehicle loads 优化以后定义

$$
G^\circ(W)
=
\min_{n_1,n_2}
G(W;n_1,n_2),
$$

于是现有 steady-state cost-bound wave 为

$$
\boxed{
W^\circ
\in
\arg\min_WG^\circ(W).
}
$$

因此需要强调：

> 当前问题并不是“把原来外生的 \(W\) 内生化”。原模型本来就在内生优化 \(W\)。
>
> 真正想放松的是更高一层的 policy restriction：为什么每一次 dispatch 都应重复使用同一个 stationary wave？

---

# 2. 从任意 dispatch sequence 出发

设 demand rate 恒定：

$$
\Lambda>0.
$$

暂时仍保留一个重要条件：

> 每一次 dispatch 都对应现有模型中的一个完整 common wave，即 collection、line-haul 与 distribution 对这一批 shipment 保持同步。

但是不再要求 successive waves 大小相同。

允许运营者选择任意 sequence：

$$
q_1,q_2,q_3,\ldots
$$

其中 \(q_k\) 是第 \(k\) 次 dispatch 的 batch size。

一次 dispatch 后，未发运 backlog 回到 0。下一轮等待期间：

$$
\dot Q(t)=\Lambda.
$$

因此，一个大小为 \(q\) 的 batch 需要

$$
T(q)
=
\frac q\Lambda
$$

才能积累完成。

这构成一个 renewal cycle。

---

# 3. 把 waiting cost 与 post-dispatch service cost 分开

现有模型中的

$$
\frac{W}{2\Lambda}
$$

是 shipment 在 dispatch 之前为等待 wave 凑齐所支付的平均 consolidation waiting。

因此定义

$$
R(W;n_1,n_2)
=
G(W;n_1,n_2)
-
\omega\frac{W}{2\Lambda},
$$

即把 dispatch 前的 consolidation waiting 从 generalized cost 中剥离。

进一步定义

$$
R^\circ(W)
=
\min_{n_1,n_2}R(W;n_1,n_2).
$$

于是

$$
\boxed{
G^\circ(W)
=
R^\circ(W)
+
\omega\frac{W}{2\Lambda}.
}
$$

如果当前 backlog 已经达到 \(q\)，现在立即 dispatch，那么这一整批货在 dispatch 之后产生的 generalized service cost 为

$$
\boxed{
\mathcal C(q)
=
qR^\circ(q).
}
$$

因此一个 renewal cycle 只需要处理两部分：

$$
\text{dispatch 前持续产生的 waiting cost}
$$

和

$$
\text{dispatch 时发生的 service/intervention cost}.
$$

---

# 4. Proposition 1：stationary wave 可以从任意 batch sequence 中推出

在一个大小为 \(q\) 的 cycle 中，

$$
Q(t)=\Lambda t,
\qquad
0\le t\le\frac q\Lambda.
$$

因此总 waiting cost 为

$$
C^{\mathrm{wait}}(q)
=
\int_0^{q/\Lambda}\omega\Lambda t\,dt
=
\frac{\omega q^2}{2\Lambda}.
$$

一个 cycle 的总成本为

$$
\mathcal C(q)
+
\frac{\omega q^2}{2\Lambda}.
$$

于是定义 batch size \(q\) 对应的长期 average cost rate：

$$
\boxed{
J(q)
=
\frac{
\mathcal C(q)+\omega q^2/(2\Lambda)
}{
q/\Lambda
}
=
\frac{\Lambda\mathcal C(q)}q
+
\frac{\omega q}{2}.
}
$$

## Proposition 1

对任意 successive dispatch sequence

$$
q_1,q_2,\ldots,
$$

长期平均 cost rate 不小于

$$
\min_{q>0}J(q).
$$

因此，如果

$$
q^*
\in
\arg\min_{q>0}J(q),
$$

那么 repeating policy

$$
\boxed{
q_1=q_2=\cdots=q^*
}
$$

是一个最优 stationary dispatch policy。

### Proof

前 \(N\) 个 renewal cycles 的平均 cost rate 为

$$
\begin{aligned}
\bar J_N
&=
\frac{
\sum_{k=1}^N
\left[
\mathcal C(q_k)
+
\omega q_k^2/(2\Lambda)
\right]
}{
\sum_{k=1}^N q_k/\Lambda
}
\\
&=
\frac{
\sum_{k=1}^N
q_kJ(q_k)
}{
\sum_{k=1}^Nq_k
}.
\end{aligned}
$$

右侧是 \(J(q_k)\) 的 shipment-weighted average，因此

$$
\bar J_N
\ge
\min_qJ(q).
$$

令所有 cycles 都采用 minimizer \(q^*\)，即可达到等号。

因此：

$$
\boxed{
\text{arbitrary successive batch sizes}
\Longrightarrow
\text{optimal stationary batch }q^*.
}
$$

恒定 demand 下，相应 headway 自动成为

$$
\boxed{
H^*
=
\frac{q^*}{\Lambda}.
}
$$

---

# 5. Proposition 2：动态 threshold 与现有最优 wave 完全相同

因为

$$
\mathcal C(q)
=
qR^\circ(q),
$$

所以

$$
J(q)
=
\Lambda R^\circ(q)
+
\frac{\omega q}{2}.
$$

又因为

$$
G^\circ(q)
=
R^\circ(q)
+
\frac{\omega q}{2\Lambda},
$$

因此

$$
\boxed{
J(q)
=
\Lambda G^\circ(q).
}
$$

由于 \(\Lambda>0\) 为常数，

$$
\arg\min_qJ(q)
=
\arg\min_qG^\circ(q).
$$

所以

$$
\boxed{
q^*
=
W^\circ.
}
$$

这一结果意味着：

> 在恒定需求、common-wave、cost-bound、renewal-separable 的条件下，从任意 successive batch sizes 出发得到的最优 stationary dispatch threshold，与现有 steady-state formulation 的最优 wave 完全一致。

因此 dynamic formulation 在这一特殊情形下**不会产生一个不同的最优解**。

它提供的是一个 policy-level interpretation：

$$
\boxed{
W^\circ
=
\text{optimal backlog threshold for dispatch}.
}
$$

换句话说，现有 steady-state model 可以理解为一个更一般 dispatch problem 的 stationary reduction。

---

# 6. QVI 表述：主要价值是解释，而不是重新计算 $W^\circ$

定义

$$
V(q)
$$

为 backlog 为 \(q\) 时的 relative value，

$$
g
$$

为长期 optimal average cost rate。

不 dispatch 时，

$$
\dot q=\Lambda,
$$

running waiting cost rate 为

$$
\omega q.
$$

因此 continuation equation 为

$$
\omega q-g+\Lambda V'(q)=0.
$$

如果立即 dispatch，则支付

$$
\mathcal C(q)
$$

并把 backlog reset 到 0。

所以可以写成 average-cost QVI：

$$
\boxed{
\min
\left\{
\omega q-g+\Lambda V'(q),
\;
\mathcal C(q)+V(0)-V(q)
\right\}
=0.
}
$$

其自由边界正是

$$
q^*=W^\circ.
$$

在这里，QVI 的价值主要是解释：

$$
\boxed{
\text{Continue}
\quad\text{vs}\quad
\text{Dispatch}.
}
$$

它并不会比直接求

$$
\min_WG^\circ(W)
$$

更简单。

因此，在 stationary environment 中，没有必要用 dynamic programming 替代现有 steady-state 求解方法。

“American option”可以作为直觉类比，但更准确的数学结构是 repeated or regenerative impulse control，因为每次 dispatch 后系统并未终止，而是 reset 后进入下一轮 cycle。

---

# 7. 为什么 nonstationary demand 才是真正的新扩展

现在考虑：

$$
\lambda=\lambda(t).
$$

如果上一轮 dispatch time 为 \(s\)，那么在时刻 \(t\) 的 backlog 是

$$
\boxed{
Q_s(t)
=
\int_s^t\lambda(u)\,du.
}
$$

如果在 $\tau$ dispatch，则本轮实际 wave size 为

$$
\boxed{
W(s,\tau)
=
\int_s^\tau\lambda(u)\,du.
}
$$

此时，恒定需求下的固定关系

$$
W=\Lambda H
$$

不再存在。

相同的 wave size 在一天中不同时间可能对应完全不同的 headway，而相同的 backlog 在不同 calendar time 也可能具有不同的 continuation value。

因此，nonstationary demand 下真正自然的决策对象不再是一个固定 \(W\)，而是：

$$
\boxed{
\text{何时 dispatch 当前 backlog}.
}
$$

---

# 8. 非均匀需求下的 waiting cost

从上一轮 dispatch time \(s\) 到当前时刻 \(t\)：

$$
Q_s(t)
=
\int_s^t\lambda(u)\,du.
$$

如果在 $\tau$ dispatch，则这一 cycle 的总 waiting cost 为

$$
\boxed{
C^{\mathrm{wait}}(s,\tau)
=
\omega
\int_s^\tau Q_s(t)\,dt.
}
$$

交换积分顺序后，

$$
\boxed{
C^{\mathrm{wait}}(s,\tau)
=
\omega
\int_s^\tau
(\tau-u)\lambda(u)\,du.
}
$$

这里不再需要使用“平均等待半个 headway”的 stationary approximation。

当

$$
\lambda(u)\equiv\Lambda
$$

时，

$$
C^{\mathrm{wait}}
=
\frac{\omega W^2}{2\Lambda},
$$

因此现有 steady-state waiting formula 自动成为这一表达式的特殊情形。

---

# 9. Time-dependent dispatch boundary

考虑 finite operating horizon：

$$
t\in[0,T].
$$

定义

$$
V(t,q)
$$

为当前时间 \(t\)、backlog 为 \(q\) 时的 minimum future cost。

如果继续等待，

$$
\dot q=\lambda(t),
$$

因此 continuation equation 为

$$
\boxed{
V_t(t,q)
+
\lambda(t)V_q(t,q)
+
\omega q
=
0.
}
$$

如果现在 dispatch，则支付

$$
\mathcal C(q)
$$

并 reset 到

$$
(t,0).
$$

因此 QVI 为

$$
\boxed{
\min
\left\{
V_t+\lambda(t)V_q+\omega q,
\;
\mathcal C(q)+V(t,0)-V(t,q)
\right\}
=
0.
}
$$

如果 optimal policy 具有 threshold structure，则存在一个 time-dependent boundary：

$$
\boxed{
q^*(t)
}
$$

使得

$$
q<q^*(t)
\quad\Longrightarrow\quad
\text{Continue},
$$

而

$$
q\ge q^*(t)
\quad\Longrightarrow\quad
\text{Dispatch}.
$$

如果上一轮 dispatch time 为 \(\tau_k\)，下一轮 dispatch time 即为

$$
\boxed{
\tau_{k+1}
=
\inf
\left\{
t>\tau_k:
\int_{\tau_k}^t\lambda(u)\,du
\ge q^*(t)
\right\}.
}
$$

于是

$$
W_{k+1}
=
\int_{\tau_k}^{\tau_{k+1}}
\lambda(u)\,du,
$$

$$
H_{k+1}
=
\tau_{k+1}-\tau_k.
$$

因此 wave size 与 headway 都不需要预先规定，而是成为 boundary hitting 的结果。

---

# 10. 这一扩展相对于直接 timetable optimization 的意义

在 nonstationary demand 下，如果直接优化 timetable，需要选择

$$
\tau_1,\tau_2,\ldots,\tau_N.
$$

改变一个 $\tau_k$ 不仅改变当前 wave size 和 waiting cost，也会改变下一 cycle 的起点，并进一步影响后续 schedule。

因此 schedule variables 具有天然的 intertemporal coupling。

dynamic formulation 则把这些未来影响压缩进

$$
V(t,q).
$$

每个状态下只需比较：

$$
\boxed{
\text{Continue}
\quad\text{vs}\quad
\text{Dispatch}.
}
$$

因此：

> 在恒定 demand 下，dynamic formulation 主要提供解释；
>
> 在 nonstationary demand 下，它才开始提供真正不同的建模结构。

---

# 11. 当前结果的边界：尚未解决 stage synchronization

这里需要特别区分两个问题。

第一类问题是：

$$
q_1,q_2,q_3,\ldots
$$

是否需要随 successive dispatches 改变。

本文已经证明，在恒定 demand、renewal-separable 的 common-wave 情形下，没有必要：

$$
\boxed{
q_1=q_2=\cdots=q^*
}
$$

是最优的。

第二类问题则是：

$$
W_1,\qquad m,\qquad W_2
$$

即 collection、line-haul 与 distribution 是否应该采用相同 wave。

这是不同的问题。

现有 steady-state model 已经证明，当 sorting 不构成约束时，nested waves 可以被 common wave weakly dominate。

但在 finite sorting rate 下，较小的 collection waves 可能使 origin sorting 更早开始，从而产生 pipelining benefit；现有模型本身也把这一问题列为 open issue。

因此当前 dynamic formulation **并没有证明**

$$
W_1=m=W_2.
$$

更准确地说，本文目前研究的是：

$$
\boxed{
\text{temporal stationarity}
}
$$

而不是

$$
\boxed{
\text{stage synchronization}.
}
$$

后者应作为下一步单独处理，而不应混入当前的等价性证明。

---

# 12. 关于 deadline、sorting workload 与其他 operating states

dynamic formulation 的一个潜在优点，是未来可以通过增加 state 来接入更多机制。

例如：

$$
A(t)
=
\text{oldest shipment age}
$$

可以用于描述 shipment-level deadline；

$$
Y(t)
=
\text{unfinished sorting workload}
$$

可以用于描述 finite sorting capacity；

更一般地也可以写成

$$
V(t,q,X).
$$

但这些目前应被理解为**可扩展接口，而不是已经完成的理论结果**。

特别是，一旦放弃 common wave，允许不同 stages 拥有独立 dispatch decisions，状态空间将不再只是一个 backlog \(q\)，而必须进一步记录各 stage 的 inventory、sorting workload 与 pipeline state。

因此本文暂不展开高维 QVI，也不声称已经解决这一问题。

---

# 13. 当前最值得保留的结论

目前这套 formulation 最核心的结果可以概括为：

$$
\boxed{
\begin{array}{c}
\text{arbitrary successive common-wave batch sizes}
\\
\Downarrow
\\
\text{optimal stationary threshold }q^*
\\
\Downarrow
\\
q^*=W^\circ
\end{array}}
$$

即：

> 现有 steady-state optimum 可以从一个更一般的 dispatch-policy class 中内生恢复。

进一步，当

$$
\Lambda
\longrightarrow
\lambda(t)
$$

后，stationary threshold 推广为

$$
\boxed{
q^*(t),
}
$$

而实际的 wave size 和 headway 由 boundary hitting 内生生成。

因此比较合适的理论关系是：

$$
\boxed{
\text{steady-state common-wave optimization}
\subset
\text{stationary dispatch-threshold formulation}
\subset
\text{time-dependent dispatch policy}.
}
$$

这并不是要替代现有 steady-state model。

恰恰相反：

> **现有模型提供了一个非常重要的、具有解析结构的 stationary special case；dynamic formulation 的作用，是解释这个 special case 为什么出现，并为 nonstationary demand 提供一个向外延伸的接口。**

---

# 14. 下一步值得单独讨论的问题

目前最自然的下一步并不是立即加入 fleet、weather、congestion 等更多状态，而是解决现有模型自身已经留下的一个具体 open issue：

$$
\boxed{
\text{finite sorting 下，common wave 是否仍然内生最优？}
}
$$

也就是从

$$
W_1=m=W_2
$$

进一步放松到

$$
W_1,\qquad m,\qquad W_2.
$$

如果能够证明在某些参数条件下三者重新合并，

$$
W_1^*=m^*=W_2^*,
$$

那么 common wave 就不再只是 timetable assumption，而可以被解释为一个更一般 stage-specific dispatch problem 的 endogenous synchronization regime。

相反，如果在另一些参数区间出现

$$
W_1^*<m^*
$$

或者其他非同步结构，那么也可以明确刻画 finite sorting 何时破坏 common wave。

我认为这一问题应与本文当前的 nonstationary extension 分开处理：前者研究 **stage synchronization**，后者研究 **temporal adaptation**。

这样可以避免模型在尚未解决基础结构之前过早扩展到高维动态系统。