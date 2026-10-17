# Solution Approach 修订说明：分阶段候选网络与自适应经济启发式

日期：2026-08-20  
适用范围：`Intercity_Operation.pdf` 第 3.2--3.3 节、Algorithm 1、Algorithm 2 及其数值实验实现。

## 1. 修订动机

现有 Algorithm 2 先在 Algorithm 1 的剪枝网络上构造动态 BHH 优先级解，再将该解写入同一个削减 MILP 的 Gurobi MIP Start。修复后的配对实验表明：

- 200 单时，Algorithm 1 与 Algorithm 2 的最终成本均为 1837.03，但总时间由 1.59 秒增加到 2.41 秒；
- 500 单时，最终成本均为 5495.00，但总时间由 4.64 秒增加到 10.92 秒；
- 多数 100 单实验中两者的窗口 MIP 均达到 `MIP Gap = 0`，Algorithm 2 不能改善同一窗口模型的最优目标；
- 罚金为 10 的 100 单实验中，Algorithm 2 还会因为影响同成本窗口解的选择而改变后续提交状态，使平均滚动期成本从 2333.33 增至 2416.67。

因此，当前结果不是简单的参数失配。MIP Start 不改变可行域和目标函数；当 Algorithm 1 已能快速证明窗口最优时，Algorithm 2 只能增加构造、注入和 Gurobi 处理初解的开销。

## 2. 已识别问题

### 2.1 分阶段弧的开始时间被错误地使用同一窗口

现有实现把人工弧、自动驾驶弧和直送弧的开始时刻都限制在
`[t_h, t_h + H)`。实验参数为 `H = 8`、自动驾驶干线时间 `tau = 9`。
对一条完整换装路径

```text
origin manual -> automated line-haul -> destination manual
```

即使起点集货在窗口开始时立即完成，自动驾驶段到达时也已经越过
`T_start`，目的城市人工配送弧无法生成。延长完成时刻 `j` 并不能解决后续阶段的开始时刻 `i` 不在候选集中的问题。

该问题解释了 200/500 单实验中的退化现象：Algorithm 1 和 Algorithm 2
均返回 100% 直送，而精确模型选择换装运输。Algorithm 2 无法在缺少换装路径的候选网络上恢复被删除的选择。

### 2.2 Proposition 2 的实现条件不充分

扩展完成窗口要支持完整路径，需要同时满足：

1. 扩展长度覆盖最长的完整服务链；
2. 只有第一阶段的开始时刻受 `T_start` 限制；
3. 后续干线和末端配送阶段可在 `T_ext` 内开始；
4. 控制窗口只提交当前应执行的阶段，后续阶段可随状态更新重新优化。

只增加 `completion_end` 而不扩展下游弧的开始时刻，不能保证命题中的完整路径存在。

### 2.3 Algorithm 2 缺少经济停止条件

当前排序指标

```text
slack / (penalty * remaining_quantity + epsilon)
```

只描述紧迫度和未服务损失，没有比较“新增服务成本”和“拒单罚金”。路径选择又以最早完成时刻为主，因此只要路径可行，启发式就倾向于继续服务，即使新增车辆成本高于避免的罚金。

### 2.4 重复计算与组合枚举

- Algorithm 2 构造器生成一次候选网络，warm-start MIP 又生成一次相同网络；
- 每次分配后都为所有未完成订单重新枚举人工起点弧、自动驾驶弧和人工终点弧的组合；
- 当前实现给模型中的大量零变量也写入 MIP Start，增加 Python--Gurobi 边界调用和初解处理负担。

### 2.5 滚动期终端状态没有进入选择规则

当前启发式只看订单在当前窗口中的最早完成时间，没有评价直送造成的跨城人工车队转移、瓶颈时段车辆占用，以及控制窗口结束后的剩余可行性。即使两个窗口解具有相同目标值，MIP Start 也可能把 Gurobi 引导到未来状态更差的等价解。

## 3. 修订目标

本轮修订按以下优先级执行：

1. **正确性**：当 `H < tau` 但扩展窗口足够时，候选网络仍能包含完整换装路径；
2. **经济性**：启发式只分配具有正净收益的货量；
3. **零额外开销基线**：容易窗口不运行启发式，行为退化为 Algorithm 1；
4. **复用**：同一窗口的 Algorithm 1 网络只生成一次；
5. **可诊断性**：记录触发原因、构造成本、启发式净收益和被经济规则拒绝的订单；
6. **困难窗口增量价值**：只有在基线窗口显示求解困难时才启用部分 MIP Start 或后续邻域修复。

## 4. Algorithm 1 的修订

### 4.1 分阶段时间域

候选弧分成两种语义：

- 第一阶段弧：新订单的起点人工集货和直送任务，开始时刻必须位于 `T_start`；
- 下游阶段弧：自动驾驶干线和目的城市人工配送，开始时刻可位于 `T_ext`。

在现有按城市共享人工弧的模型结构下，采用以下兼容实现：

1. 人工弧和自动驾驶弧的候选开始时刻扩展到 `completion_end`；
2. 直送弧仍只允许从 `T_start` 开始；
3. 对每个订单显式关闭 `i >= start_end` 的起点城市人工货流变量；
4. 目的城市人工货流以及已经完成起点集货订单的自动驾驶货流可以使用扩展阶段弧；
5. 历史已提交弧继续无条件并入候选集并按原值固定。

### 4.2 扩展窗口诊断

每个窗口记录：

- `first_stage_start_end`；
- `downstream_start_end`；
- `extension_sufficient_for_transshipment`；
- 按订单统计的可行直送和换装路径数量。

若扩展长度小于理论最长链路时间，不静默宣称 Proposition 2 成立，而是在诊断中明确标记。

## 5. Algorithm 2 的修订

### 5.1 经济净收益

对订单 `l`、候选服务选项 `p` 和可分配货量 `q`，定义

```text
net_benefit(l, p, q)
  = penalty_l * q
  - incremental_vehicle_cost(p, q)
  - incremental_transfer_cost(p, q)
  - future_resource_cost(p, q).
```

新增车辆数按弧上的已有剩余容量计算，因此拼载到已发车辆的货量不会重复承担固定车辆成本。只有 `net_benefit > 0` 的选项可以进入构造解；否则保留为未服务货量。

### 5.2 经济后悔值排序

对每个订单保留净收益最高和次高的两个选项，定义

```text
regret_l = (best_net_benefit - max(0, second_net_benefit)) / (slack_l + 1).
```

优先处理后悔值高的订单；最早完成时间仅用于同分选择。这样可以优先保护“错过当前资源后没有近似替代方案”的订单，而不是无条件优先最早完成路径。

### 5.3 未来资源成本

第一版采用可解释的稀缺度代理：新增车辆覆盖的每个时段按照剩余车辆数反比计价；直送车辆还计入起点城市的终端稀缺度。该项只用于启发式排序，不改变原始 MILP 的会计成本。

后续若实验表明该代理有效，再用窗口 LP 松弛的容量约束对偶价格替代。

### 5.4 自适应触发

Algorithm 2 不再默认对每个窗口运行。触发条件使用上一窗口的真实 MIP 时间和当前平均剩余时间预算：

- 首个窗口直接运行 Algorithm 1，建立难度基线；
- 上一窗口 MIP 时间超过阈值，或平均剩余窗口预算低于阈值时，启用经济构造器；
- 容易窗口直接求解 Algorithm 1 的削减 MILP，不生成初解；
- 触发阈值作为 `RollingHorizonConfig` 参数保存并进入实验输出。

### 5.5 部分 MIP Start

触发后只给启发式非零决策、订单未服务量以及必要的汇总变量设置 Start；不再为所有未选择变量显式写零。Gurobi 可以补全其余变量，从而减少注入成本和过度引导。

### 5.6 受限邻域的启用条件

受限邻域不是本轮默认步骤。只有配对实验确认“经济初解明显优于 Gurobi 首个 incumbent，但完整 MIP 仍受限”后，才执行：固定高后悔值订单，释放低后悔值订单与瓶颈重叠弧，并在短预算内求解 restricted MIP。未达到证据门槛时保持关闭，避免再次出现无条件增加时间的问题。

## 6. 实现步骤

- [ ] 扩展人工和自动驾驶下游候选弧的开始时域；
- [ ] 加入新订单起点人工货流的 `T_start` 限制；
- [ ] 增加 `H < tau` 且换装可行的确定性回归测试；
- [ ] Algorithm 1 与 Algorithm 2 复用同一个候选网络；
- [ ] 加入增量成本、正净收益停止条件和经济后悔值排序；
- [ ] 加入未来资源稀缺度代理；
- [ ] 加入自适应触发和部分 MIP Start；
- [ ] 保存新诊断字段并更新结果字段文档；
- [ ] 在 `pavane` 环境运行单元测试；
- [ ] 运行同订单、同种子的 Algorithm 1/2 配对实验。

## 7. 验收标准

### 7.1 正确性

- 当 `prediction_horizon < travel_time_periods`、但完整服务链可在扩展窗口完成时，候选网络至少包含一条换装路径；
- 起点集货不能从 `T_start` 之后开始；
- 后续自动驾驶和末端配送可以从 `T_start` 之后开始；
- 两种算法均满足需求守恒、车辆容量、车辆时段占用和历史提交约束。

### 7.2 性能

- 未触发 Algorithm 2 时，其窗口模型、目标值和主要计时应与 Algorithm 1 等价；
- 触发时单独报告候选生成、启发式构造、MIP Start 注入和 MIP 求解时间；
- 不再仅以最终 `MIP Gap=0` 比较 MIP Start，而比较固定预算下的首次可行解时间和目标轨迹。

### 7.3 数值实验

至少报告：

- 滚动期总成本、未服务率和直送/换装比例；
- 总时间、最大窗口时间、启发式触发比例；
- 启发式初解目标、最终窗口目标和二者差值；
- 0.05、0.1、0.5、1、2 秒窗口预算下的 incumbent 轨迹；
- 经济停止、未来资源成本、自适应触发三个消融版本。

## 8. 论文表述调整

在新实验完成前，不再宣称 Algorithm 2 必然加速。建议将其定位为：

> An adaptive economic construction and partial warm-start procedure that is activated only for empirically difficult rolling-horizon windows.

Algorithm 1 的 Proposition 2 需要补充“下游阶段弧允许在扩展窗口开始”的条件。若仍坚持所有弧从 `T_start` 开始，则该命题在 `H < tau` 的实验设置下不成立。

## 9. Implementation status and initial re-evaluation (2026-08-20)

Completed: candidate-network boundary correction, first-stage decision bounds, network reuse, economic stopping, scarcity proxy cost, economic-regret ranking, nonzero partial MIP Start, adaptive triggering, and component-level timing. Restricted neighborhoods remain a later experiment and are not enabled by default without evidence from consistently difficult windows.

Regression tests: 20 passed, 2 subtests passed. Coverage includes a complete transshipment path when H < tau but the extension is sufficient, economic rejection under a low lost-demand penalty, and adaptive trigger skip/activate behavior.

In the 100-order paired experiment (three identical seeds, 120-second total limit), Algorithms 1 and 2 both had mean cost 968.67; mean runtimes were 1.157 s and 1.050 s. Algorithm 2 triggered no heuristic on these easy windows. Direct volume was zero and transshipment volumes were 900, 790, and 452, showing that the former 100% direct degeneration caused by the time-boundary defect has disappeared.

Under the 5-second strict-limit re-run (100 orders, seed 5201), both costs were 931. Algorithm 1 took 1.05 s and Algorithm 2 took 1.14 s; none of Algorithm 2's 12 windows triggered the heuristic. Compared with 1.80 s before trigger correction, the approximately 0.77 s erroneous heuristic overhead was removed.

These results support skipping the heuristic on easy windows, but do not yet establish a statistically significant advantage on difficult windows. The incumbent-trajectory and ablation experiments in Section 7.3 should precede any restricted-neighborhood implementation.
## 10. 压力测试补充（2026-08-20）

200/500/1000单、3个相同种子、每流程30秒的正式配对压力测试已经完成。完整设置和逐规模结果见 [Algorithm 2 配对压力测试报告](algorithm_2_stress_test_20260820.md)。

压力测试首次发现1000单窗口中启发式构造可消耗38.47秒并挤占全部MIP预算。现已增加默认1秒的Algorithm 2硬预算、候选枚举截止检查和0.5秒MIP保留时间；修复后同一实例的两组均能在状态9时保留可行incumbent。完整回归测试更新为 `23 passed, 2 subtests passed`。

正式矩阵中，200单和500单配对成本完全相同；1000单的A1+A2+MIP平均Gap从0.003022降到0.001563，但平均成本比A1+MIP高约0.030%，尚不能证明稳定收益。

500单种子6202的共同不可行已通过IIS和最小可行性松弛定位：跨窗提交保留了约4e-6的货流浮点尾差。最终方案保持正常窗口精确固定，只在状态3且无可行解时对已提交连续货流开放正负1e-5固定带并重求解一次；车辆及未提交历史仍精确固定。修复后500单三个种子的两组均完成12个窗口，平均成本同为2852.1036，可行率恢复为3/3。
