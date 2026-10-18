# QVI–Wave 独立 toy 数值检查

日期：2026-10-18。状态：**NUMERICAL_OBSERVATION / 有界离散验证通过**。没有运行 Gurobi、CFS 或蒙特卡洛，没有修改或编译 TeX。

## 1. 比较对象与来源

本检查使用两份 steady 稿共有的最简限制：两城对称、每方向常率确定性流体到达、common wave、连续载荷、按负荷配车、忽略车辆排队；设 κ=∞、δ=0、L=∞，但保留固定 transfer 时间 θ 和费 h。要求 ω>0。它不包含 V1-F 的固定车队/离散弧、rolling horizon 信息限制，也不解决 finite sorting 下是否同步最优。

数学来源：

- `research/paper/steady_state_model_v1_linear.tex:75` 给出 n_i≤min(M,W_i)；95 的 GZfull 在 W1=m=W2=q 下定义目标。103 的 loads 公式遗漏的 q 截断在本检查中明确补入。
- `research/paper/steady_state_model.tex:89` 给出同一可行域；127 的 G 和 118–120 的 mean lead time，在上述限制下化为同一目标。131 的 program 及137/159的简式没有完整重复 n≤q，本检查不把该省略当作放松授权。
- `research/theory/Common Wave 的内生化与非均匀需求扩展.md` §3–§6 给出 intervention/holding 拆分及形式 average-cost QVI。本数值检查单独求离散 Bellman，不把原笔记的形式方程当作已证明的全网络 QVI。

修正后的比较目标为

\[
s(q)=a+b/\sqrt q,\quad 0<n_i\le\min(M,q),
\]
\[
G(q,n_1,n_2)=c[2s(q)+2\rho/n_1+2\rho/n_2]
+\frac{\hat c\tau}{\min(q,\hat M)}
+\omega\left[\frac q{2\Lambda}+s(q)(n_1+n_2/2)+3\rho+\tau+2\theta\right]+2h.
\]

动态服务成本定义为 C(q)=q[min_n G(q,n)−ωq/(2Λ)]，积压在等待时以 dq/dt=Λ 增长，dispatch 清零，等待费率为 ωq。一次 dispatch 后的运输时间成本被完整计入 C；此处清零只指未发运积压，不能解释为真实在途车队立即回归。renewal 可分性是建模假设。

单位：小时、shipment、货币；Λ 为每方向 shipment/小时，c/ĉ 为货币/车小时，ω 为货币/(shipment·小时)。G 是每shipment成本，g 是每方向成本率，比较必须使用 **g=ΛG**；双方向总成本率为2g，不改变最优 q。

## 2. 两条独立算法

脚本：[qvi_wave_consistency_check.py](../toy_models/qvi_wave_consistency_check.py)。

**稳态路径。** 对每个 q，用 scipy bounded scalar optimization 分别优化原式中 n1/n2，显式检查上界 min(M,q)；外层按几何分段搜索并保留 AV 容量 kink q=Mhat 及区间端点。这是数值优化结果，不在脚本中声称一个新的全域全局证明。

**动态路径。** 使用解析 clipped-load 最小服务费构造 C(q)，然后独立求状态网格上的平均成本 Bellman。它没有调用稳态的 argmin 或把稳态最优阈值传给动态求解器。两条成本路径另外在8个不同 q 探针逐点比较，最大绝对差≤5.69e−14。

q_i=iΔq、Δt=Δq/Λ。每次等待一步用精确流体 holding 面积，Bellman 为

\[
V_i=\min\{C(q_i),\ \omega(q_i+\Delta q/2)\Delta t-g\Delta t+V_{i+1}\}.
\]

q=0 禁止 dispatch，最大状态强制 dispatch。对每个 g 从上边界向后递推，再以70次二分求 V0(g)=0。实际执行策略是从0出发第一次到达的 dispatch 状态；没有预先假定全部大 q 都必须 dispatch，也没有使用全 tail 的 V=C 断言。

检查：离散 min residual、obstacle≥0、wait gap≥0、V0归一化、Δq收敛和上边界160→320的稳定性。最多6401个一维状态，有限17次 Bellman 求解；这不是高维 QVI 求解或连续 QVI 收敛定理。

## 3. 实际结果

公共参数：a=.02，b=.3，ρ=.2，c=10，ĉ=4，τ=2，ω=1，M=8，θ=.05，h=.1；下表仅改变 Λ 和 Mhat。

| 情形 | Λ | Mhat | 稳态最优 q | 内层(n1,n2) | Bellman Δq=.05 | 成本率绝对差 |
|---|---:|---:|---:|---|---:|---:|
| 小 wave，n≤q 活跃 | .1 | 20 | 1.973515470 | (q,q) | 1.95 | .0001383353 |
| AV 容量 kink | 2 | 8 | 8 | (5.632886142,7.966104098) | 8 | 浮点0 |
| 人工车容量活跃 | 50 | 20 | 61.304752812 | (8,8) | 61.3 | 1.383e−7 |

全部阈值误差≤.024；并不把离散阈值与连续阈值写成字面相等。细网格最大每shipment成本差约.001384（低需求例），其成本率差因Λ=.1为.0001384。三个例子在 qmax 从160扩大到320时细网格策略/成本不变。

离散 min residual 的输出为浮点0，obstacle 与 wait gap 非负；这是 Bellman 递推对应的**离散方程检查**，不能据此宣称连续 QVI 处处光滑、全域阈值唯一、生产调度最优或一般数值收敛已证明。三个主要例子的 |V0|≤1.1e−13。

## 4. 随机到达的有限补充与负例

**数值观察：单位 Poisson 特例不一定改变最优整数阈值。** 使用同一 C、同一参数和相同整数 action domain，另解 Poisson Bellman：等待一步的期望成本增量为 (ωq−g)/Λ，而流体整数网格为 (ω(q+1/2)−g)/Λ。两者均选择 B=20；流体 g=80.7035558774322，Poisson g=80.2035558774322，恰差 ω/2=.5。

与此对应的限定解析恒等式为 J_P(B)=ΛC(B)/B+ω(B−1)/2，J_D(B)=ΛC(B)/B+ωB/2。这解释该特例同 argmin，但这里只把计算结果标作数值观察；完整证明由理论报告提供。**连续流体 q≈20.4 与整数 B=20 是不同决策域**，不能用同整数域恒等式声称连续最优值也逐字相同。批量/带标记抵达、车辆状态、非线性 holding、deadline 和 pipeline 不能从该特例自动外推。

**有界负例：目标不匹配会导致不同解。** 同物理参数、同 C，仅遗漏等待积压费，在 qmax=80 的截断域内返回 q=80，扩大到160返回 q=160，成本率继续由60.1331降至56.9895；正确 holding 模型在同一网格选择 q=20.4。80与160是人工边界，不能称无界问题实际最优。解析上 C(q)/q 随 q 严格下降（2cs(q)严格下降，内层可行域扩大且各给定可行n的服务项下降，linehaul不增），所以删去 holding 后没有有限 q 的最小值。这是**换了目标**的示例，不是同模型的 QVI 失效。

若把 current 稿允许的 ω=0 与 L=∞ 同时用于此简化模型，也会失去防止无限增大 wave 的机制。本检查固定 ω=1，不泛化到该退化域。

## 5. 可复现记录与限度

命令：

```powershell
& 'C:/Users/32599/.conda/envs/pavane/python.exe' -B research/toy_models/qvi_wave_consistency_check.py
```

Python 3.11.15，numpy 1.26.4，scipy 1.15.3。最终脚本执行 **exit_code=0**，shell wall time=2.3600023秒，脚本内部计时2.0550秒。此前两次版本均exit0，内部计时1.9763和2.0855秒；第三次为增加runtime与source SHA256快照后最终复验。三次总内部时间约6.12秒，低于20秒预算，重复执行不增加不同case计数。脚本只输出stdout，不写实验文件。

SHA256：

- steady_state_model.tex：980CFAEF58AC3DD25712205A673599A74E52BE53BFE8D4DCCAC1B78065AC7083
- steady_state_model_v1_linear.tex：05F09EEBD5D3ADE1BB8B7022E460878A37C2A3C4566138852C62CC94F737DF80
- Common Wave笔记：A508B86D981F89CC57EAEDD58FE9071D9D884025C0370A36D99504EE39226CCA
- 最终脚本：901BD18CD811FB320962BA1303A76824F4466598294D30D84F6581B35EE989AF

本报告验证的只有限定 toy 模型的独立数值一致性。没有有限 sorting、hard deadline、固定车队、随机 marks、多阶段独立发车、真实 routing 或 rolling horizon 的一致性结论。CFS小样本结果不是本项 steady/QVI 证据。论文可把本测试作为理论桥梁的可复现 sanity check，不能替代核心定理证明或正式大实验。

### 最终原始 stdout

```json
{
  "scope": "deterministic fluid common-wave, no sorting, delta=0, no deadline; per direction",
  "cost_units": "time=hour, q=shipment; rate=currency/hour, G=currency/shipment",
  "runtime": {
    "python": "3.11.15 | packaged by Anaconda, Inc. | (main, Jun 11 2026, 15:12:53) [MSC v.1942 64 bit (AMD64)]",
    "numpy": "1.26.4",
    "scipy": "1.15.3"
  },
  "source_sha256": {
    "research/toy_models/qvi_wave_consistency_check.py": "901BD18CD811FB320962BA1303A76824F4466598294D30D84F6581B35EE989AF",
    "research/paper/steady_state_model.tex": "980CFAEF58AC3DD25712205A673599A74E52BE53BFE8D4DCCAC1B78065AC7083",
    "research/paper/steady_state_model_v1_linear.tex": "05F09EEBD5D3ADE1BB8B7022E460878A37C2A3C4566138852C62CC94F737DF80",
    "research/theory/Common Wave \u7684\u5185\u751f\u5316\u4e0e\u975e\u5747\u5300\u9700\u6c42\u6269\u5c55.md": "A508B86D981F89CC57EAEDD58FE9071D9D884025C0370A36D99504EE39226CCA"
  },
  "cases": {
    "small_wave_load_bounded_by_wave": {
      "parameters": {
        "arrival_rate": 0.1,
        "a": 0.02,
        "b": 0.3,
        "rho": 0.2,
        "c": 10.0,
        "c_auto": 4.0,
        "tau": 2.0,
        "omega": 1.0,
        "manual_capacity": 8.0,
        "auto_capacity": 20.0,
        "theta": 0.05,
        "handling": 0.1
      },
      "steady": {
        "q": 1.9735154703550035,
        "per_shipment": 26.23732489882575,
        "rate": 2.6237324898825753,
        "loads": [
          1.9735154703550035,
          1.9735154703550035
        ]
      },
      "fluid_bellman": [
        {
          "step": 0.2,
          "upper": 160.0,
          "threshold": 2.0,
          "rate": 2.623903679018718,
          "per_shipment": 26.23903679018718,
          "normalization_abs": 2.6645352591003757e-15,
          "discrete_qvi_max_abs": 0.0,
          "minimum_obstacle": 0.0,
          "minimum_wait_gap": 0.0,
          "forced_boundary": true
        },
        {
          "step": 0.1,
          "upper": 160.0,
          "threshold": 2.0,
          "rate": 2.623903679018718,
          "per_shipment": 26.23903679018718,
          "normalization_abs": 2.220446049250313e-15,
          "discrete_qvi_max_abs": 0.0,
          "minimum_obstacle": 0.0,
          "minimum_wait_gap": 0.0,
          "forced_boundary": true
        },
        {
          "step": 0.05,
          "upper": 160.0,
          "threshold": 1.9500000000000002,
          "rate": 2.623870825133441,
          "per_shipment": 26.23870825133441,
          "normalization_abs": 9.547918011776346e-15,
          "discrete_qvi_max_abs": 0.0,
          "minimum_obstacle": 0.0,
          "minimum_wait_gap": 0.0,
          "forced_boundary": true
        }
      ],
      "enlarged_boundary": {
        "step": 0.05,
        "upper": 320.0,
        "threshold": 1.9500000000000002,
        "rate": 2.623870825133441,
        "per_shipment": 26.23870825133441,
        "normalization_abs": 9.547918011776346e-15,
        "discrete_qvi_max_abs": 0.0,
        "minimum_obstacle": 0.0,
        "minimum_wait_gap": 0.0,
        "forced_boundary": true
      },
      "independent_cost_path_max_error": 5.684341886080802e-14
    },
    "linehaul_capacity_kink": {
      "parameters": {
        "arrival_rate": 2,
        "a": 0.02,
        "b": 0.3,
        "rho": 0.2,
        "c": 10.0,
        "c_auto": 4.0,
        "tau": 2.0,
        "omega": 1.0,
        "manual_capacity": 8.0,
        "auto_capacity": 8,
        "theta": 0.05,
        "handling": 0.1
      },
      "steady": {
        "q": 8,
        "per_shipment": 10.84580642358381,
        "rate": 21.69161284716762,
        "loads": [
          5.632886142441168,
          7.966104097844809
        ]
      },
      "fluid_bellman": [
        {
          "step": 0.2,
          "upper": 160.0,
          "threshold": 8.0,
          "rate": 21.69161284716762,
          "per_shipment": 10.84580642358381,
          "normalization_abs": 2.220446049250313e-15,
          "discrete_qvi_max_abs": 0.0,
          "minimum_obstacle": 0.0,
          "minimum_wait_gap": 0.0,
          "forced_boundary": true
        },
        {
          "step": 0.1,
          "upper": 160.0,
          "threshold": 8.0,
          "rate": 21.69161284716762,
          "per_shipment": 10.84580642358381,
          "normalization_abs": 1.1102230246251565e-15,
          "discrete_qvi_max_abs": 0.0,
          "minimum_obstacle": 0.0,
          "minimum_wait_gap": 0.0,
          "forced_boundary": true
        },
        {
          "step": 0.05,
          "upper": 160.0,
          "threshold": 8.0,
          "rate": 21.69161284716762,
          "per_shipment": 10.84580642358381,
          "normalization_abs": 2.4646951146678475e-14,
          "discrete_qvi_max_abs": 0.0,
          "minimum_obstacle": 0.0,
          "minimum_wait_gap": 0.0,
          "forced_boundary": true
        }
      ],
      "enlarged_boundary": {
        "step": 0.05,
        "upper": 320.0,
        "threshold": 8.0,
        "rate": 21.69161284716762,
        "per_shipment": 10.84580642358381,
        "normalization_abs": 2.4646951146678475e-14,
        "discrete_qvi_max_abs": 0.0,
        "minimum_obstacle": 0.0,
        "minimum_wait_gap": 0.0,
        "forced_boundary": true
      },
      "independent_cost_path_max_error": 2.842170943040401e-14
    },
    "large_wave_manual_capacity": {
      "parameters": {
        "arrival_rate": 50,
        "a": 0.02,
        "b": 0.3,
        "rho": 0.2,
        "c": 10.0,
        "c_auto": 4.0,
        "tau": 2.0,
        "omega": 1.0,
        "manual_capacity": 8.0,
        "auto_capacity": 20.0,
        "theta": 0.05,
        "handling": 0.1
      },
      "steady": {
        "q": 61.30475281174747,
        "per_shipment": 6.779142637744528,
        "rate": 338.9571318872264,
        "loads": [
          8.0,
          8.0
        ]
      },
      "fluid_bellman": [
        {
          "step": 0.2,
          "upper": 160.0,
          "threshold": 61.400000000000006,
          "rate": 338.9571873068187,
          "per_shipment": 6.7791437461363735,
          "normalization_abs": 1.9984014443252818e-15,
          "discrete_qvi_max_abs": 0.0,
          "minimum_obstacle": 0.0,
          "minimum_wait_gap": 0.0,
          "forced_boundary": true
        },
        {
          "step": 0.1,
          "upper": 160.0,
          "threshold": 61.300000000000004,
          "rate": 338.95713202551667,
          "per_shipment": 6.779142640510333,
          "normalization_abs": 8.815170815523743e-14,
          "discrete_qvi_max_abs": 0.0,
          "minimum_obstacle": 0.0,
          "minimum_wait_gap": 0.0,
          "forced_boundary": true
        },
        {
          "step": 0.05,
          "upper": 160.0,
          "threshold": 61.300000000000004,
          "rate": 338.95713202551656,
          "per_shipment": 6.779142640510331,
          "normalization_abs": 1.0702549957386509e-13,
          "discrete_qvi_max_abs": 0.0,
          "minimum_obstacle": 0.0,
          "minimum_wait_gap": 0.0,
          "forced_boundary": true
        }
      ],
      "enlarged_boundary": {
        "step": 0.05,
        "upper": 320.0,
        "threshold": 61.300000000000004,
        "rate": 338.95713202551656,
        "per_shipment": 6.779142640510331,
        "normalization_abs": 1.0702549957386509e-13,
        "discrete_qvi_max_abs": 0.0,
        "minimum_obstacle": 0.0,
        "minimum_wait_gap": 0.0,
        "forced_boundary": true
      },
      "independent_cost_path_max_error": 2.842170943040401e-14
    }
  },
  "passed": true,
  "unit_poisson_integer_comparison": {
    "parameters": {
      "arrival_rate": 10.0,
      "a": 0.02,
      "b": 0.3,
      "rho": 0.2,
      "c": 10.0,
      "c_auto": 4.0,
      "tau": 2.0,
      "omega": 1.0,
      "manual_capacity": 8.0,
      "auto_capacity": 20.0,
      "theta": 0.05,
      "handling": 0.1
    },
    "fluid_integer": {
      "step": 1,
      "upper": 160,
      "threshold": 20.0,
      "rate": 80.7035558774322,
      "per_shipment": 8.070355587743219,
      "normalization_abs": 2.4868995751603507e-14,
      "discrete_qvi_max_abs": 0.0,
      "minimum_obstacle": 0.0,
      "minimum_wait_gap": 0.0,
      "forced_boundary": true
    },
    "poisson": {
      "step": 1,
      "upper": 160,
      "threshold": 20.0,
      "rate": 80.2035558774322,
      "per_shipment": 8.02035558774322,
      "normalization_abs": 1.9539925233402755e-14,
      "discrete_qvi_max_abs": 0.0,
      "minimum_obstacle": 0.0,
      "minimum_wait_gap": 0.0,
      "forced_boundary": true
    },
    "rate_difference_expected": 0.5,
    "note": "Same integer action domain and same C; mean rates differ by constant omega/2."
  },
  "objective_mismatch_counterexample": {
    "parameters": {
      "arrival_rate": 10.0,
      "a": 0.02,
      "b": 0.3,
      "rho": 0.2,
      "c": 10.0,
      "c_auto": 4.0,
      "tau": 2.0,
      "omega": 1.0,
      "manual_capacity": 8.0,
      "auto_capacity": 20.0,
      "theta": 0.05,
      "handling": 0.1
    },
    "matched": {
      "step": 0.1,
      "upper": 80,
      "threshold": 20.400000000000002,
      "rate": 80.70005540739135,
      "per_shipment": 8.070005540739135,
      "normalization_abs": 5.440092820663267e-15,
      "discrete_qvi_max_abs": 0.0,
      "minimum_obstacle": 0.0,
      "minimum_wait_gap": 0.0,
      "forced_boundary": true
    },
    "holding_omitted": {
      "step": 0.1,
      "upper": 80,
      "threshold": 80.0,
      "rate": 60.13312629199987,
      "per_shipment": 6.013312629199987,
      "normalization_abs": 1.2931877790833823e-12,
      "discrete_qvi_max_abs": 0.0,
      "minimum_obstacle": 0.0,
      "minimum_wait_gap": 0.0,
      "forced_boundary": true
    },
    "holding_omitted_larger_boundary": {
      "step": 0.1,
      "upper": 160,
      "threshold": 160.0,
      "rate": 56.98946638440532,
      "per_shipment": 5.698946638440532,
      "normalization_abs": 1.3358203432289883e-11,
      "discrete_qvi_max_abs": 0.0,
      "minimum_obstacle": 0.0,
      "minimum_wait_gap": 0.0,
      "forced_boundary": true
    },
    "note": "Different objective, not QVI failure: removing accumulation holding pushes the bounded optimum to 80, then 160; these are artificial truncation boundaries, not unbounded finite optima."
  },
  "elapsed_sec": 2.055001500004437
}
```

