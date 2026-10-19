# BHH 城际配送论文：基于既有 CFS 实验的数值实验执行书

> **给 Codex / Experiment Agent 的任务书**  
> 研究稿基线：用户最后提供的 `Intercity_Operation_BHH (3)(1).pdf`，以其对应的**最新 LaTeX 源文件**为唯一数学规格；不得以旧 `main_v2_BHH.tex` 或旧 Python 实现替代。  
> 代码仓库：[`Sundance0604/Intercity_Delivery_System_Design`](https://github.com/Sundance0604/Intercity_Delivery_System_Design)，已检查分支 `research/qvi-multiagent`（2026-10-18）。  
> 目标：**复用已有 CFS 2022 PUMS 数据处理、历史实验、Gurobi/Rolling Horizon/候选筛选代码，在与新论文完全一致的 V1 模型上完成一套最小、可信、可复现的数值实验。**  
> 非目标：再次开发随机 QVI、增加新的大规模算法、追求过多参数组合、为得到显著结论修改数据、重写已完成的理论论文。

---

## 0. 先阅读仓库，禁止从零开始

### 0.1 现有真实资产（必须核对、尽量复用）

| 文件或目录 | 已有功能 | 本轮安排 |
|---|---|---|
| [`docs/cfs_2022_data_processing.md`](https://github.com/Sundance0604/Intercity_Delivery_System_Design/blob/research/qvi-multiagent/docs/cfs_2022_data_processing.md) | CFS PUMS 2022 字段、筛选、加权抽样、时间窗构造、SQLite | 复用；纠正论文的需求单位解释 |
| `intercity_delivery/data/cfs_processor.py`、`sqlite_store.py`、`cfs_catalog.py` | 大 CSV 分块转 SQLite、按城市对读取、订单生成、校准城际行程 | 复用，不重造新 ETL |
| `intercity_delivery/experiments/core.py`、`solvers.py` | 同订单多求解器、CSV/JSON 输出 | 复用接口，给 V1 新注册独立求解器 |
| `intercity_delivery/algorithms/paper_rolling_horizon.py` | 控制窗口、提交历史、继承在途任务 | 复用思想与接口；V1 必须核实 selector、成本账本及可行性 |
| `intercity_delivery/algorithms/full_window_mip.py` | 旧未剪枝 RH 的窗口求解 | 作为代码参考；旧母弧域不能直接用于剪枝公平对比 |
| `intercity_delivery/algorithms/state_dependent_mip.py` | 旧候选网络与剪枝 MILP | 迁移到 V1 容量及同一母弧域后再评价 |
| `intercity_delivery/algorithms/bhh_priority_heuristic.py` | 旧初解生成 | **不是主实验必需**；已有先导实验未证明收益 |
| `simulation/run_protocol.py` | 实验批次、manifest、preflight、日志、同订单求解 | 复用或扩展；旧 CLI 目前只识别 V0 solver |
| `simulation/analyze_stage_c.py`、`simulation/stage_c_20260819/` | 已有阶段 C 汇总 | 作为历史审计资料，不能混入 V1 正式统计 |
| `simulation/runs/FIX3/`、`FIX4/`、`C_*` | 过去运行的日志与结果 | 只读保留；不得覆盖或重新标成新版结果 |
| [`experiments/reports/model_code_gap.md`](https://github.com/Sundance0604/Intercity_Delivery_System_Design/blob/research/qvi-multiagent/experiments/reports/model_code_gap.md) | 新旧模型逐项差距、求解器含义、剪枝风险 | 作为 V1 实现及数据审计的起点 |
| [`coordination/MIGRATION_PLAN.md`](https://github.com/Sundance0604/Intercity_Delivery_System_Design/blob/research/qvi-multiagent/coordination/MIGRATION_PLAN.md) | G0–G9 迁移依赖与正确性验收 | 复用步骤，不重复规划一套冲突的工程流程 |

### 0.2 先明确三个已经确定的事实

1. 仓库历史程序称 **V0**；最新论文采用新的 **V1 wave-dependent capacity**。截至本次查看，仓库 README 和模型审计均记载 **V1 尚未迁移完成**。旧实验跑通不等于新论文已经有实验。
2. 2026-08-19 的 CFS 先导实验已经跑了 **LA–SF 双向城市对，100 / 200 / 500 单**、精确 MIP、Rolling Horizon 和候选剪枝。**无需从 0 重新证明旧代码能跑。**
3. 旧数值先导结论**不能直接作为论文结果**：既有审计发现 V0 与 V1 的车辆容量定义、母弧域及成本统计存在差异；尤其 Rolling Horizon 的末窗口目标、上下界不应直接当成全日执行成本与全局证书。

### 0.3 复用过去的结论，但标明其证据等级

[`simulation/stage_c_pilot_20260819.md`](https://github.com/Sundance0604/Intercity_Delivery_System_Design/blob/research/qvi-multiagent/simulation/stage_c_pilot_20260819.md) 的**历史报告值**：

- 城市对：Los Angeles–Long Beach (`06-348`) ↔ San Jose–San Francisco–Oakland (`06-488`)；
- 通过筛选的双向原始记录：**30,185**；代表距离：**424.8 miles**；按 60 分钟时段得到干线时间 **9 periods**；
- 100 单、3 个 seeds：在当时成本口径下，`penalty_lost=1` 使精确 MIP 与剪枝 RH 都出现服务/拒单权衡；
- 500 单、单一 seed：历史报表记载精确 MILP 约 **125.07 s**，剪枝 RH 约 **4.64 s**，RH 报告成本明显更高；
- 原先车队倍率 `0.75/1.0/1.25` 大都**不产生容量约束**，灵活模型在基准参数下多次**不选择 direct**；
- 旧 Algorithm 2（生成初解）在 200/500 单均**没有改善最终成本**，还增加运行时间；
- 更早阶段 B 的异常运行与无效批次不能用于论文：以 `simulation/fix_audit_20260819.md` 的有效性说明为准。

**要求：** 这些数字仅用于选择本轮实验区间与排查历史错误。正式论文每个结果都要由 V1 完整轨迹 **independent replay** 后重新计算，不能直接复制。

---

## 1. 论文的四个核心实验问题

| 编号 | 科学问题 | 对应对照 | 必须报告 |
|---|---|---|---|
| **RQ1** | BHH 式 wave pooling 与资源约束能否形成正确、可执行的调度？ | 微型算例 + V1 精确 MILP + 独立重算 | 公式残差、容量/时序/车辆守恒、可行率 |
| **RQ2** | 在信息逐步揭示的情况下，Rolling Horizon 能达到怎样的质量？ | V1 离线全知精确基准 vs V1 RH | 已执行总成本、未服务率、完成率、全知差距 |
| **RQ3** | 时间可行性剪枝是否提高运行效率？损失多少质量？ | **同一 V1 RH 母弧域**，dense vs pruned | 完整 wall-time、候选削减、目标差、可行率 |
| **RQ4** | Direct、Transfer 和 Empty Repositioning 在什么条件下有运营价值？ | 四种明确的运营机制 | 成本、准时/服务率、各运输方式占比、空驶率、车队利用率 |

后续工作仅围绕这四项展开。**QVI 的确定性平均成本证明已在论文附录，不强制再做一套随机控制数值实验。**

---

## 2. Step 0 — 冻结数学模型与代码版本（最高优先级）

**先做什么：**

1. 找到最终版 `(3)(1)` PDF 对应的全部 LaTeX `main.tex` 与 `sections/` 文件；生成 SHA-256；同历史 `main_v2_BHH.tex` 比较，确认哪个版本最新。
2. 列出最终论文中有限时域的服务容量、Direct 容量、成本函数、需求平衡、车辆库存、rolling-horizon 窗口公式。生成 `equation_to_code_matrix.md`：**公式 → 代码函数 → 单元测试 → 实验指标**。
3. 固定以下**会改变可行域**的定义，不要让 Codex 自动猜：
   - local wave 按完整 `(i,j,k,d)` 分组，还是同起点的不同结束弧共用空间分区？
   - 城市本地车队经 direct/empty 迁入后，wave 的 `v` 上限是**当前可用车数**还是**初始城内车数**？
   - direct 车辆的**起点、最后一个客户位置、目的城市可用节点**；所需 stem/回 hub 时间是否全部计入？
   - direct 与 independent empty 的区分：无载货 direct 不能暗中绕过“禁止空驶”实验开关；
   - 是否采用**全日终端车队自由分布**（最新稿允许的基准），或另设有成本的终端复位；**不能在每个滚动窗口强制回原城**；
   - `q` 与 `D_l` 的 stop-equivalent 单位、时间的分钟/时段、费用的币种/每车小时；
   - `s_l` 是否同时定义为订单揭示时刻，还是需要单独 `reveal_time`；
   - 同一波内 BHH routing work 的均分假设，和 `n=q/v` 的车辆载荷约束。
4. 更新 `MODEL_CONTRACT`，定稿后**冻结数学模型**。不能为求解更快悄悄修改公式或给模型加入旧代码的 `transfer_fee`、比例限制、强制 terminal return。
5. 保留全部 V0 求解器与旧测试；新模型用 **V1 新模块 + 新 solver ID + 隔离结果目录**，不要覆盖历史结果。

**为什么：** 2026-10-17 审计发现旧 Python 核心容量使用单车 `capacity × vehicle_count`，尚未实现 V1 的 `K(v)`/selector；空返、direct 回 hub、车辆迁入和弧域也有差距。

**交付/验收：** `V1_MODEL_FREEZE.md` 与 `equation_to_code_matrix.md`；影响可行域的未决问题必须标记 `BLOCKED`，不能进入正式实验。

---

## 3. Step 1 — CFS 2022 数据预检与固定样本

### 3.1 数据不必重新下载和转库

先查本机已有的 `data/cfs_2022_pums.sqlite`，并验证表、索引、city pair 列表及字段。仓库约定使用 `pavane` 环境，已有 `cfs_processor.py`、`sqlite_store.py`，不必重新处理原始 2.82GB CSV，除非缓存缺失或校验失败。

CFS PUMS 提供起终点的 CFS 区域、货重、货值、货类、运输方式、大圆距离、调查抽样权重等；**不提供**完整的动态订单发布时间、配送截止时间、精确客户经纬度和真实城市内停靠轨迹。

因此论文必须称 **CFS-calibrated synthetic operational instances**，不能把人为生成的 `earliest_start` / `latest_completion` 说成 CFS 实测时间。

### 3.2 沿用当前筛选与抽样口径

1. 默认选择 `MODE=111/112`，非出口、两个不同且明确的 CFS 都市区、重量和权重大于零、大圆距离至少 50 miles；
2. 从 `06-348` 与 `06-488` 开始，按两个方向分别抽样，并用 `WGT_FACTOR` 做加权无放回抽样；
3. 城际行驶距离和时长在**城市对层面**固定：由全体合格原始记录加权中位距离，乘路线绕行因子，再按固定速度及 `Δt` 取整；不得因 seed 变化而修改 `τ`；
4. 使用现有时间窗生成器；保存每个种子的原始 `SHIPMT_ID`（在许可和可导出范围内）、方向、距离、货重、构造时间窗及变换参数；
5. 测试同一个 `(city_pair, N_orders, seed)` 重复生成完全相同的订单；不同算法**复用同一个固定订单 JSON**；
6. 同时记录两方向货流差异，避免“各方向强制相同票数”掩盖真实样本比例；可以在基准设定中继续按方向均衡抽样，但必须披露为实验设计，并增加一个按观察分布抽样的稳健性情景。

### 3.3 **最需要解决：CFS 的 pounds 不是 BHH 的 customer stops**

旧处理器默认 `quantity = clip(round(SHIPMT_WGHT/50), 10, 300)`，并设人工车容量 1,000 个**旧模型货量单位**。这个转换对 V0 可以是方便的合成货量代理，**不能直接称为 1,000 个客户停靠点**。

V1 有两个可执行选项（**作者明确选一个**）：

- **路径 A：最小发表版（建议）**。保留 CFS OD、重量/货类分布作为需求输入；重新定义并披露一个**合成服务工作量/stop-equivalent 映射** `D_l = φ(SHIPMT_WGHT_l, ...; η)`。`η` 是外生仿真参数，做敏感性分析，不能声称 CFS 真实测量了 stops。`M, M_A, a_k, b_k` 全按相同等效工作量单位重新标定，确保它们的数值/量纲自洽。
- **路径 B：严格一票一服务任务的简化**。每个被抽取的 CFS shipment 只代表一个 pickup–delivery job（不是“真实一站”），令 `D_l=1 job-equivalent`，货重用于分层或另设容量限制；相应调整车容量与处理速度。这是另一种合成任务近似，不能将车辆物理重量容量直接和 job count 相加。如果额外增加重量维度会改变论文模型，**不要为了本轮发表另建二维容量模型**。

`DataContract` 至少保存：`source_weight_lb`、`model_quantity`、`model_unit`、`conversion_rule`、`seed`、`is_observed` 或 `is_synthetic`。**禁止在代码中把 pounds 变量直接改名 stops 了事。**

### 3.4 从单城市对开始，不必立即做六个城市对

- 主城市对：LA–SF（上述代码）；复用已有 SQLite；
- 有余力才增加 1 个较短线路、1 个较长线路，并保证双向样本充足；
- 城市间直线距离可以来自 CFS；**城内服务面积 `S_k`、hub–service-region stem `ρ_k` 不可从 CFS Area 编码直接推断为真实停靠位置**。需要独立公开来源或声明为合成参数。

**交付/验收：** `data_contract.json`、`cfs_profile.csv`、`orders/` 中每 seed 固定输入、`preflight.json`；数据字段来源和模型构造字段分开列示。

---

## 4. Step 2 — 独立实现并测试 V1 BHH 容量表

在不运行大规模 Gurobi 的情况下先开发纯函数。以最新版正文对应符号为准，核心局部容量是：

$$f_k(q)=a_kq+b_k\sqrt q,\quad a_k>0,\ b_k\ge0.$$

$$K^k_{ij}(v)=\min\left\{vM,\ f_k^{-1}\left(v[(j-i)\Delta t-2\rho_k]_+\right)\right\}.$$

**实施顺序：**

1. 写 `f_k(q)` 与稳定的反函数/单调二分求解，检查 `f(f^{-1}(z))≈z`；
2. 按 `(city, start, end, vehicle_count)` 预计算容量表，**不是** `v×单车容量`；
3. 使用 one-hot vehicle-count selector，允许 `v=0` 不开波，正载货波必须有正车数；
4. 加入 `q ≤ vM` 物理载量上限，stem time、整数时段及 `j≤T`；
5. direct 使用方向依赖的 `\widetilde K^d_{ij}(v)`：验证直接服务总时间、目的地车辆释放节点、不可载货的 `τ` 时段空行程；
6. 不能因为 `ρ=0` 就断言新模型退化为旧 `v×f^{-1}(B)`：BHH pooling 在 `b>0, v>1` 时仍不同。

**单元测试至少覆盖：** `q=0`、`v=0`、`v=1/2/3`、容量饱和、不饱和、无剩余可服务时间、`b=0`、双城非对称、`j=T`、同刻到达再出发、direct 无载货、车辆迁入后重用。

**一个重要的性质测试**（仅固定同城/方向/相同时段的 arc）：

$$K(v_1+v_2)\ge K(v_1)+K(v_2).$$

这是代码回归性质；它**不**意味着整个车队调度模型的目标函数一定随 `v` 单调下降。

**交付/验收：** `V1 capacity table`、容量测试全通过、每个表项能代回论文原始服务时间不等式；不合法零时长不得给正容量。

---

## 5. Step 3 — V1 全时域 MILP 与独立 replay 验证

**优先实现强制三阶段换装（V1-M），再实现 Direct/Empty 的 V1-F。**

- 订单流：collection → AV line-haul → distribution，满足 release/deadline、两级累计 precedence 和最终需求平衡；
- AV 载货量在同弧**跨所有订单共享**，不可每笔订单各使用一份车容量；
- local/direct 用 V1 的 one-hot wave 载货容量；
- 人工与自动车使用独立的**城市时空库存**，direct 和 empty 到达后改变人工车可用地；
- 成本单位统一为 money / vehicle-minute；若输入为 /vehicle-hour，转换一次 `/60`（不能再重蹈旧 60 倍错误）；
- 允许 partial rejection，且 rejected quantity 不得同时处于已取未送、干线在途或已交付状态；
- 对已存在或新加入的终端要求，必须按 Step 0 冻结的**真实物理终点**定义，不强迫每个算法窗口局部回原城。

**必须提供独立于 Gurobi 约束构造的 replay：**

1. 顺时间轴恢复每笔订单取货、干线、配送、direct 以及未服务数量；
2. 重算每个本地波和 direct 波的物理时间及载量约束；
3. 重算每个城市每个节点的车辆可用量，检查双向迁移、空返；
4. 重算货物优先时序及完工时限；
5. **从已执行的完整任务只计费一次**，重算车辆费用与真正未服务罚金；
6. 输出 `feasible/reasons`、`cost_replayed`、`cost_reported`、最大约束残差与完整性标记。

先做 6–10 个能手工枚举的小型例子；精确 MILP 应可求最优、replay 成本与求解器 objective 一致（数值容差内）。不通过则停止后续实验。

**交付/验收：** V1-M/V1-F solver + `independent_replay`；小算例 100% 校验通过，`total_cost_replayed` 是论文唯一认可的完整调度成本。

---

## 6. Step 4 — 将现有 Rolling Horizon 正确迁移到 V1

复用 `paper_rolling_horizon.py` 的时轴思路，使用最新论文：

$$b_t=\min(T,t+H),\quad c_t=\min(T,b_t+E),\quad a_t=\min(T,t+h).$$

其中 `H` 是起始预测窗，`E` 是允许下游完成的扩展窗，`h` 是滚动步长。**窗口末端不是运营终点。**

每个 `t` 执行：

1. 仅揭示当时应当已知的订单（如定义 `reveal_time=s_l`，需在 data contract 明确）；从持久 ledger 恢复全部已提交动作；
2. 保留当前城市空闲车辆、正在 local / direct / AV / empty 路途中的车辆及已安排的未来返回；
3. 保留已取未送的订单、hub 库存、已确定货运链、对订单的部分交付/已拒绝状态；
4. 固定历史**全部相关车辆/selector/货流变量**。已提交波不可在新窗口改变 vehicle count 或重新分区；
5. 在合法新弧域上解 MILP，必要时允许下游任务在 `b_t` 后、`c_t` 前启动与完工，且不超过真实 `T`；
6. 仅提交 `[t,a_t)` 内**开始的完整任务**（整个结束时间与货流也固定），未开始的计划下次可改；
7. 若窗口无 incumbent / 总预算耗尽：报告 `partial`，不生成虚构的完整成本或默认所有后续需求已服务；
8. 结束 `T` 时独立 replay，按最终完整交付量和运营任务重算成本。

**注意 LA–SF 基准已校准 `τ≈9` 个 60-min period。** 旧 `H=8,E=6,h=2` 是历史初始参数而非最优设置；必须验证 `H+E` 和每阶段候选链实际足以容纳“取货 + 干线 + 配送”，不允许因为窗口太短把 transfer 人为排除。

**交付/验收：** V1 `RH-dense` 在最小算例可从头到尾完成、所有窗口提交历史一致、终态 `completed_horizon=true`，且重放成本等于完整调度成本。

---

## 7. Step 5 — 将候选剪枝接到 V1，并建立公平对照

### 7.1 第一层：先做不会改变决策含义的时间支持筛选

参考最新论文 Appendix C 的结构：

- 起点取货候选 `P_l`：满足订单揭示/释放、正 local 容量和允许的 start window；
- 终点配送候选 `R_l`：满足完成 deadline、正 local 容量及 completion window；
- 干线候选 `Q_l`：必须存在可衔接的上游取货结束、下游配送开始；
- direct 候选：满足订单 release/deadline、positive capacity 和物理路线；
- **必须保留**已承诺的正货流、相关历史与已承诺车弧；不能因没有新货就把 AV、独立 empty 或无货车移动的母弧删掉。

如果只利用乐观的 `K(v_max)>0`，应区分“必要条件筛选”（潜在可证明安全）与“启发式/业务候选限制”（可能删掉最优解）。

### 7.2 第二层：修复旧代码最重要的公平性问题

旧 `full_window_mip.py` 和 `state_dependent_mip.py` 的**基础母弧域与延后下游允许范围曾不同**。因此不能直接拿 V0 的两个 solver 之差认定剪枝加速。

**本轮必须保证：**

- `RH-dense` 与 `RH-pruned` 相同的订单、物理模型、mother arcs、目标、window definitions、`H/h/E`、fleet、时间预算和 Gurobi 配置；
- 唯一区别是：`RH-pruned` 固定为 0 或不创建特定**货流变量**；
- 车辆与 selector 母域保持一致；相同一组历史承诺，才可单窗口一对一验证剪枝的增量影响；
- 不可基于“当前城市为零车”就删除未来 local candidate：本窗口可能发生 direct/empty **迁入**；
- 如果不能证明某条剪枝规则安全，则使用 `heuristic_pruning` 标签，并用小型穷举/精确解报告最优性损失。

### 7.3 三个核心剪枝验收指标

$$\text{Reduction}=1-\frac{\#\text{retained cargo candidates}}{\#\text{dense cargo candidates}},$$

$$\Delta C=\frac{C_{\rm RH-pruned}-C_{\rm RH-dense}}{\max\{\varepsilon,|C_{\rm RH-dense}|\}},$$

$$\text{Speedup}=\frac{t_{\rm RH-dense}^{\rm end\!\!-\!to\!\!-\!end}}{t_{\rm RH-pruned}^{\rm end\!\!-\!to\!\!-\!end}}.$$

三者必须**同时报告**；`Speedup` 用 data-prep/candidate build/model build/solve/replay 的全部 wall time，不只取 Gurobi `optimize()` 时间。

**交付/验收：** 可追踪剪枝原因、每窗口候选数/模型规模、dense/pruned 同域配对结果；不存在“因为缩短了可行路径而更快却误称剪枝无损”的情况。

---

## 8. Step 6 — 先导实验：旧运行复核 + V1 小规模正确性

### 8.1 历史复核（不是发表结果）

无需重跑全量 2026-08 数据，只须抽查 `FIX4`、`C_PEN100`、`C_SCALE500` 的 `manifest.json`、`preflight.json`、`run_summary.json`、日志及算法类型，核实旧记录对应 V0 和其成本定义。把过去失效批次与“尚无法重放验证”的结果标记清楚。

可在**用户本机确实存在 SQLite 时**，用旧 V0 入口做小规模再现（PowerShell；这是已有真实 CLI，不是 V1 命令）：

```powershell
conda activate pavane
python simulation/run_protocol.py `
  --stage stage_b `
  --batch-id V0_RECHECK `
  --database "data\cfs_2022_pums.sqlite" `
  --city-a 06-348 --city-b 06-488 `
  --order-counts 10,20 `
  --seeds 3001,3002 `
  --solvers flexible_direct_mip,flexible_direct_rolling,paper_candidate_mip `
  --penalty-lost 1 `
  --prediction-horizon 8 --rolling-step 2 --extension-horizon 6 `
  --time-limit 180
```

> **不要把该命令结果写成 V1 实验。** 若本机数据路径不同，先由 Codex 在已有 workspace 查实再修改参数；输出使用新 batch ID，不能覆盖现有运行目录。

### 8.2 V1 正确性先导（必须通过）

| 订单数 | seeds | 方法 | 目的 |
|---|---:|---|---|
| 手工玩具 2–5 | 固定设计 | V1 exact | 手工核实 wave/迁移/precedence/成本 |
| 10、20、40 | 各 5 | V1 offline exact、V1 RH-dense、V1 RH-pruned | 发现容量、时间窗、历史/剪枝口径错误 |

同一算例三方法使用**同一** CFS-derived orders 和相同成本、fleet 参数。精确 MILP 是 *perfect-information oracle*，只在已证明最优时称 “optimal objective”；限时场景同时报告 incumbent 与有效 bound。

**验收门槛：** 所有已标 completed 的解可通过 independent replay；离线全知最优成本不得高于任何**同模型同实例的可行完整** RH 执行成本（容差外违反即检查口径）；若 RH-pruned 比 dense 更快但 cost 更高，可如实报告，不必强行消除。

---

## 9. Step 7 — 正式运营机制实验（论文最重要）

分离 **freight-route choice** 与 **human-fleet mobility** 两种效应。建议在 V1-F 中明确实现以下四种策略，不能只用旧 `direct_ratio_min/max` 粗略代替：

| 机制 ID | 运货 direct | 独立 empty repositioning | 必须说明的限制 |
|---|---|---|---|
| `T0` | 不允许 | 不允许 | 所有承运货物经两 hub 换装 |
| `T1` | 不允许 | 允许 | 允许人工车空驶跨城，仍无运货直送 |
| `T2` | 允许 | 不允许 | **无货 direct 车弧也必须被关闭**，否则变相空返 |
| `T3` | 允许 | 允许 | 完整柔性机制，作为主要方案 |

**特别提醒：** 旧 V0 实现的 `transshipment_only` 还关闭 direct 和部分迁移；`direct_only` 用订单直送比例约束，并不天然等价于 `T2`。新实验必须用变量级开关让含义准确。

建议第一批：主城市对，100 单、至少 5 个配对 seeds，四机制；预算允许再加入 200 单和其他城市对。所有对照的订单、成本、fleet、终端定义保持相同。

报告：

- total/replayed cost、车辆运营费用、unserved penalty、已服务货量比例；
- direct 运货份额、AV line-haul 运货份额、empty trips 数量与成本；
- 两城市车辆利用率、车辆在途时间、平均/分位交付延迟（与模型按弧末交付的约定保持一致）；
- 完成率、任何不满足 deadline 的数量（理论应 0；若有则是 bug）；
- **与 `T0` 的配对差值**（每个 seed 对应一种成本差），而不是独立均值胡乱相减。

**不要强求 direct 必须被采用。** 先导结果中基准最优选择 0% direct，本身也是有意义的结论。若想解释其适用区间，再在 Step 10 扫描 `c_D/c_A`、sorting/transfer fee、stem、fleet 紧张度；所有变化都要标记为**敏感性情景**，不能偷偷改主基准直到算法“好看”。

---

## 10. Step 8 — Rolling Horizon 与剪枝的正式消融

| 组别 | 数学模型 | 信息 | 算法 | 比较目的 |
|---|---|---|---|---|
| `O` | V1-F | 全日订单已知 | Offline exact | 小规模全知参考，不是同信息公平算法 |
| `R0` | V1-F | 到时刻 t 已揭示 | RH-dense | 滚动优化性能 |
| `R1` | V1-F | 同 `R0` | RH-pruned | **剪枝增量** |
| `R2`（可选） | V1-F | 同 `R0` | RH-pruned + warm-start | 仅困难窗口需要时再比较 |

第一优先级是 `R0 vs R1`，因为它们最能客观回答“剪枝有没有用”。第二层才是 `O vs R0/R1` 的 information gap。

**正式规模建议：** `N_orders ∈ {20, 50, 100, 200, 500}`；每组至少 5 个固定 seeds。`O` 先对 `N≤100` 求证最优，较大算例可设置 time limit 并报告 gap，不能用只跑出 incumbent 的目标当全知最优值。

单实例保存：

- `total_wall_time_sec`、`preprocessing_sec`、`model_build_sec`、`gurobi_sec`、`replay_sec`；
- `window_time_mean / median / p95 / max` 与窗口超时率；
- `n_binary / n_continuous / n_constraints`、候选货流弧数量与删减率；
- `completed_horizon`、`solution_count`、各窗口 `gap`、最终 `replayed_cost`；
- `total_unserved_quantity / total_offered_quantity`；
- 如果状态 partial：只给已执行范围内的 partial 账单、未结货流及已完成窗口数，**不能把 partial cost 与 full-horizon cost 直接排序**。

**在线计算判据：**

$$\max_t t_{\mathrm{end-to-end},t}\le h\Delta t\times 60\text{ seconds},$$

且因 `Δt=60 min` 会使该判据很宽松，额外报告 **max-window ≤ 60 seconds** 的工程占比（这是经验目标，不是数学可行性条件）。

`R2` 仅在 R1 有明显困难窗口（如无可行 incumbent、窗口数值间隙过大或算力预算触顶）时启用；旧实验已显示旧 Algorithm 2 在简单窗口里不能自动带来加速，因此不作为主贡献。

---

## 11. Step 9 — 专门检查 BHH wave pooling 的作用（小型消融，建议保留）

对同一 local arc 的 V1 核心容量为：

$$K_{\mathrm{pool}}(v)=\min\{vM,f^{-1}(vB)\},\qquad B=[D-2\rho]_+.$$

设计一个**明确标记为比较模型的**“车辆各自面向全服务区、无跨车空间集中收益”基准：

$$K_{\mathrm{separate}}(v)=\min\{vM,v f^{-1}(B)\}.$$

这个消融**不是**声称全区独立巡回是真实运营的唯一替代，而是在相同其他约束下，将 cross-vehicle spatial pooling 机制去掉，以衡量新容量设计的边际影响。只对 local 容量做受控替换；若 direct 服务也改变，必须另列处理，不混杂比较。

最少展示小规模 `v=1,2,3,5` 的固定时段容量曲线，并在 1–2 个订单密度下比较运营成本/服务率。`b=0` 是一个有意义的退化算例，但**不要**把 `b=0` 的模型作为“真实 BHH 无效”的充分证据。

**交付：** 一张解释性的 capacity-vs-v 曲线 + 一张运营变化简表；如果赶时间，这一节可放 Appendix，不必新增新的理论命题。

---

## 12. Step 10 — 最少量、解释性强的敏感性分析

只选择真正影响决策的参数，不做全因子爆炸。

| 参数 | 优先测试水平/策略 | 试图回答的问题 |
|---|---|---|
| 订单密度 | N=50/100/200/500，其他城市参数不变 | 订单越密集，wave pooling 与 RH 是否更有效？ |
| 未服务罚金 | 先以 V1 成本单位重新校准，再做基准 ×`0.5/1/2` | 是否有实际的“服务 vs 拒单”权衡？ |
| 人工/自动车队 | **先找绑定区间**，参考旧实验尝试规模倍率 `0.25/0.5/0.75/1.0` | 车辆稀缺时 Direct/Empty 是否更有价值？ |
| Direct 成本 | 基准 `c_D ×{0.75,1,1.25}` | direct 从不采用是否仅因成本优势不足？ |
| Stem / line-haul | 选择可信的 `ρ_k` 范围；必要时改变 `τ` 或 transfer fee（单独情景） | 距离与换装连接成本是否改变运输模式？ |
| RH look-ahead | `H={8,12,16}`，E 从能形成完整三阶段候选链的范围选取，`h={1,2,4}` | 前瞻不足是否诱导次优 direct/高成本服务？ |
| BHH 系数 | `a,b` 的校准/扰动区间 | 对 routing approximation 的依赖有多强？ |

**校准要求：** 旧 `penalty_lost=1` 是 V0 + 旧货量映射的特定城市对先导水平；V1 若从 pounds proxy 切换到 stop-equivalents，**必须重新校准罚金、容量及 `a,b` 的量纲**。先选能观察到两种决策（至少部分服务又有少量拒单）的区间，不用夸大过拟合点。公开完整扫描范围和被排除的场景。

**可选的空间验证**：CFS 未提供精确停靠点，因此如要检查 BHH finite-sample 误差，应使用**另行生成的、明确标记 synthetic 的客户点/区域分区**，对比实际近似 TSP 长度与 `a q + b√q`；不能称为“用 CFS 精确位置验证 BHH”。如果时间紧，此项列为局限性 + 附录，不阻塞其余实验。

---

## 13. Step 11 — 统计分析、画图与论文写作

### 13.1 数据对齐规则

实验唯一键至少是：

`(model_revision, city_pair, sample_seed, N_orders, demand_mapping, scenario, H, h, E, pruning, cost_setting)`。

- **对同一个 seed 的配对差**做统计（平均、标准差、中位、置信区间）；不把不同订单样本混作配对；
- 每个模型的 `replayed_cost` 必须全部成本项、完整执行，或标 `partial`；
- 精确 MILP 的 gap/bound 只用于其所解的**有限时域模型**，滚动子窗口 gap 不得当全日 gap；
- 如果求解时间太快（如小于 1 s），增加问题规模或重复运行测 wall time，避免把噪声说成数量级加速；
- 数据只含一个城市对时，结论明确局限为该案例，不推广全美物流网络。

### 13.2 最小论文图表（严格按研究问题组织）

| 产出 | 内容 | 对应问题 |
|---|---|---|
| **Table 3** | CFS city-pair 资料、筛选、样本数、OD 比、距离、时间窗的观测/生成区分 | 数据可信度 |
| **Table 4** | V1 small-scale：exact / RH-dense / RH-pruned、完整成本/服务率/可行性/时间/删减率 | RQ1/RQ2/RQ3 |
| **Figure 3** | 订单数 vs end-to-end runtime，标注超时/未完成点 | RQ3 |
| **Figure 4** | 同 seed 下 `T0–T3` 的成本、拒单率、direct 份额、empty 次数 | RQ4 |
| **Figure 5** | `H` 与 `E` 对服务率、成本、最慢窗口时间的影响 | RQ2 |
| **Table 5** | pruning 候选数/变量/约束/时间/损失的 ablation | RQ3 |
| **Appendix Figure**（可选） | V1 的 pooled vs unpooled capacity 曲线 | BHH 机制 |

不要套用参考论文的乘客收益/社会福利指标：本研究的核心是**物流服务、成本、可行性与计算效率**。

### 13.3 Numerical Experiments 正文结构

```text
6. Numerical Experiments
6.1 CFS Data, Instance Construction and Calibration
6.2 Validation of the Wave-Dependent MILP
6.3 Operational Value of Direct Service and Fleet Repositioning
6.4 Rolling-Horizon Performance and Candidate Pruning
6.5 Sensitivity Analysis and Practical Implications
```

每节采用：**研究问题 → 公平比较的设置 → 表/图 → 对现象的机制解释 → 已知局限**。禁止只罗列若干曲线然后宣称算法具有普遍最优性。

---

## 14. 正式运行的记录与文件布局

沿用 `simulation/run_protocol.py` 的 `manifest.json` / `preflight.json` / `logs` / `results`；建议在**新的独立目录**保存 V1：

```text
simulation/v1_runs/
  <batch_id>/
    manifest.json
    preflight.json
    orders/
      <instance_id>.json
    raw/
      <instance_id>__<solver>.json
    logs/
      <instance_id>__<solver>.log
    validation/
      <instance_id>__<solver>__replay.json
    summary.csv
    summary.json
    figures/
```

`manifest.json` 最少写：Git commit + dirty status、PDF/TeX 来源 hash、`model_id=V1`、`solver_id`、CFS 数据库 size/mtime/hash（可选若哈希昂贵）、每个生成实例 hash、Python/Gurobi 版本、平台 CPU/RAM、Threads、MIPGap、TimeLimit、random seed、`Δt/ρ/a/b/M`、揭示规则、`cost_scope`、`bound_scope`、`completed_horizon`。

**严格区分状态：** `OPTIMAL / TIME_LIMIT_WITH_INCUMBENT / NO_INCUMBENT / PARTIAL_RH / INFEASIBLE / ERROR`。任何实验因失败缺失，必须计入分母；不可以只对成功解算均值。

**建议命令验收：** 迁移完成后新 `--solvers v1_exact,v1_rh_dense,v1_rh_pruned` 等名称由 Codex 实际注册，并将真实可运行命令写入 `simulation/v1_runs/README.md`。**现在旧 CLI 尚无这些入口，不能把上面这些名称误当作已存在的命令。**

---

## 15. Codex / 四 Agent 任务分工与停止条件

| 责任人 | 要做的事 | 不允许做的事 |
|---|---|---|
| **Experiment** | 数据接口、V1 容量/MILP/RH/剪枝代码与测试、固定样本、运行、导出图表 | 把旧 V0 结果重命名为 V1；未经预算许可跑全量实验 |
| **Theorist** | 对照最新 PDF 逐式检查容量、direct 端点、必要剪枝条件、模型单位 | 重新开启随机 QVI 或改变已冻结结论 |
| **Referee** | 独立复验一组订单、守恒、独立成本账本、母弧公平性和结果措辞 | 以“优化器返回 status=2”代替物理可行性审查 |
| **Literature** | 参考 Li & Liu (2021) 与 Banerjee et al. (2022) 的实验组织，校对正确引用 | 编造尚不存在的真实数据或运行成绩 |

### 必须暂停、先报给作者的红线

1. 最新 `main.tex` 和旧代码的模型定义未冻结；
2. CFS pounds 与 stops 混用，`M / a / b / penalty` 单位不一致；
3. direct 结束位置不明、空驶作为无货 direct 绕过开关；
4. RH 未保存所有继承任务、库存，或只是最后一个窗口的 objective；
5. dense 与 pruned 的 mother arcs 不同，却准备宣称是“纯剪枝加速”；
6. 任一完整解未通过独立 replay；
7. 操作数据不完整或正式求解所需本地 SQLite 不可用。

**遇到红线，不要凭经验悄悄补数学假设。只运行低成本测试，输出具体的阻塞位置、建议方案以及是否需要作者确认。**

---

## 16. 最小可投稿实验包（严格控制工作量）

如果目标是“尽快把论文发出去”，按以下优先级执行：

- [ ] **P0** 冻结最新版 V1 数学模型及 direct/stop-unit 约定；建立真实源码到新模型的映射。
- [ ] **P0** 复用 CFS SQLite 和已有订单抽样；整理 observed/simulated 元数据。
- [ ] **P0** 实现 V1 `K(v)`、V1-M、V1-F 与独立 replay；所有小算例通过。
- [ ] **P0** 正确迁移 V1 Rolling Horizon 与 same-domain pruning，生成 10/20/40 的配对消融结果。
- [ ] **P1** 用 LA–SF、100 单、至少 5 seeds 跑 `T0–T3` 运营机制。
- [ ] **P1** 用 20–500 单做 `RH-dense vs RH-pruned` 时间—质量曲线。
- [ ] **P1** 只做 2–3 个有解释力的敏感性维度：`H/E`、罚金/车队紧张度、direct 成本。
- [ ] **P1** 全部表格经 replay、参数 manifest 与 Referee 审核后写入 Section 6。
- [ ] **P2 可跳过** 更多城市对、启发式 warm-start/local branching、真实道路级 BHH 标定和随机 QVI 实验。

**交付包**：

- `V1_MODEL_FREEZE.md`
- `experiment_data_contract.json`
- `v1_validation_report.md`
- `v1_paired_results.csv` / `v1_paired_results.json`
- `v1_experiment_figures/`
- `numerical_experiments_draft.md`（英文、含图表定位、真实结果引用）
- `reproducibility_report.md`（命令、输入、版本、算力、全部失败情况）

### Codex 第一条执行指令

> 阅读本任务书及仓库现有的 `README.md`、`docs/cfs_2022_data_processing.md`、`docs/numerical_experiment_protocol.md`、`simulation/stage_c_pilot_20260819.md`、`experiments/reports/model_code_gap.md`、`coordination/MIGRATION_PLAN.md`。**不要从零写 CFS 数据读取和 Rolling Horizon。** 先盘点和冻结最新版 `(3)(1)` 稿对应的数学模型，并复核旧实验的有效性。按 Step 0–5 完成 V1 代码迁移与小型正确性测试，输出公式—实现矩阵和剩余阻塞；获得批准后再按 Step 6–11 执行正式实验。旧 V0 不动、旧日志不覆盖、凡是未经 replay 的结果不进入论文。

---

## 17. 主要依据（仓库内已有文件）

- [CFS 2022 数据处理说明](https://github.com/Sundance0604/Intercity_Delivery_System_Design/blob/research/qvi-multiagent/docs/cfs_2022_data_processing.md)
- [既有数值实验完整协议](https://github.com/Sundance0604/Intercity_Delivery_System_Design/blob/research/qvi-multiagent/docs/numerical_experiment_protocol.md)
- [2026-08-19 阶段 C 先导实测报告](https://github.com/Sundance0604/Intercity_Delivery_System_Design/blob/research/qvi-multiagent/simulation/stage_c_pilot_20260819.md)
- [旧问题修复审计 FIX3/FIX4](https://github.com/Sundance0604/Intercity_Delivery_System_Design/blob/research/qvi-multiagent/simulation/fix_audit_20260819.md)
- [模型—代码差距与公平对照风险](https://github.com/Sundance0604/Intercity_Delivery_System_Design/blob/research/qvi-multiagent/experiments/reports/model_code_gap.md)
- [V1 分阶段迁移计划](https://github.com/Sundance0604/Intercity_Delivery_System_Design/blob/research/qvi-multiagent/coordination/MIGRATION_PLAN.md)
- 用户上传的最终稿：`Intercity_Operation_BHH (3)(1).pdf`，以及 Li & Liu (2021), *Transportation Research Part E*, “Optimizing flexible one-to-two matching in ride-hailing systems with boundedly rational users”。
