# 模型—代码差距审计与迁移验证计划

日期：2026-10-17。负责人：Experiment。状态：静态审计完成；未运行 Python、测试、Gurobi、数值实验或生成结果。本文不声称任何数值最优性或理论创新。仅此报告由本 Agent 创建。

## 1. 证据与版本边界

已读 `AGENTS.md` 与初始 `coordination/MODEL_CONTRACT.md`（INITIAL / UNVERIFIED）。Git HEAD 为 `3b3c34b37df4f50d34ae6160bcd1303a696afd3b`。开始时工作树已有大量未跟踪 research、experiments、simulation 文件，以及 `.vscode/` 等；Git 提示 `.pytest_cache/` 无权限，因此不以“工作树干净”描述本次环境。TeX 也在未跟踪目录，HEAD 不能唯一标识本次模型输入，后续须另记录源文件散列。

审计输入：四份论文 TeX，`intercity_delivery/configuration.py`、两份 optimizer、全部 algorithms 分支、data loader/CFS 处理、experiments core/solvers、`main.py`、`simulation/run_protocol.py`、tests。不存在所假设的顶层 `config.py` / `experiment_core.py`：实际模块分别为 `configuration.py` / `experiments/core.py`。

四份 TeX 无 `\input` / `\include` 依赖；main 两份使用 `\addbibresource{导出的条目.bib}`，本地 paper 目录未找到该文件。宏为排版/证明环境，没有发现隐藏容量模型的外部宏。参考文献准确性由 Literature 独立审计。

编号约定：

- **V0-M**：当前 `base_optimizer.Optimizer` 的强制换装数学模型，以本地代码为准。
- **V0-F**：当前 `flexible_direct_optimizer.FlexibleDirectOptimizer` 的直送/换装共存模型，以本地代码为准。
- **V0-RH**：算法层对相应模型施加历史固定、可见订单、窗口/候选弧约束后的求解问题；不能与全域模型混称。
- **V1-M / V1-F**：`research/paper/main_v2_BHH.tex` 强制换装/灵活直送正文。旧稿 `main_BHH.tex` 提供比较，不代替 V0 证据。
- **S-linear**：`steady_state_model_v1_linear.tex`，均匀平衡定常需求、连续 wave/load、车辆按负荷配备、线性 lead-time 成本。
- **S-deadline**：`steady_state_model.tex`，另含有限 sorting、最慢车辆因素 δ、硬 lead-time 保证。
- **V2**：用户 QVI 研究；当前代码未实现随机到达强度、生成元、价值函数、干预算子或内生干预时刻。离线随机数据生成和固定步长 rolling horizon 均不等于 QVI。

## 2. 关键判断

不能把现有 Python 一概称为“旧模型”。V0-F 已有直送人工车空间迁移、共享 AV 容量、连续货流/未服务量、需求拆分、处理时间/费用以及比例约束；部分比旧稿明确，也与 V1 部分对应。但两份 optimizer 均没有 stem 参数、按 wave 车数预计算的容量表或 wave-size 二元选择，核心 BHH 规模机制尚未同步。

迁移须同时涵盖容量数据、完整 MIP、候选网络、历史固定、启发式增量容量、warm start、局部分支与结果解释。只替换 optimizer 的一个容量系数会留下错误的可行域和初解。

此外 V1 本文的求解章节仍保留旧单车容量与仅载货直送候选逻辑，不能把其算法伪代码作为已与新正文一致的规格。该文内冲突需交由用户确认。

## 3. 逐项数学—代码对应

以下代码行号按本轮读取版本；“对应”表示结构已找到，并非已运行验证。

| V1 公式/定义及 TeX 行号 | V0-M 实际实现 | V0-F 实际实现 | 差距与处理 |
|---|---|---|---|
| 69：d 为 customer stops，每单位一次取/送 | `data/loader.py:185–191` OrderBatch quantity，未携带 stop/weight 双单位 | 同左 | CFS 的 quantity 是重量归一化（见第 6 节），不能直接重标为 stops |
| 71、74：f^k(λ)=a^kλ+b^k√λ 及闭式逆，a>0、b≥0 | `data/loader.py:42–83` BHH/inverse | 共用 loader；另有 combined inverse | 基本逆可复用；加入参数校验与稳定求逆，a=0 只能在另行批准扩展中处理 |
| 71、82：每车 stem 2ρ^k，K(v)=min{vM,f^-1(v[((j−i)t0−2ρ)^+])} | `data/loader.py:166–180` 仅预计算单车 inverse(duration)；`models/base_optimizer.py:164–178` 用 x×coeff | `models/flexible_direct_optimizer.py:320–344` 同样 x×coeff | 核心未实现。必须建按 city/arc/v 的表并加二元选择；当前人工容量亦未 min M 截断 |
| 82：人工弧 0<j−i≤ceil((2ρ+f(M))/t0) | `loader.py:85–89` 使用 int(f(M))，未除 t0，无 stem | 完整 F 使用该 loader；paper reduced 自建 | 修正版独立生成器；不能改变 V0 生成行为后混用旧结果 |
| 79、134/216：x/y/w 非负整数，g/ĝ/ĝtilde/z 连续非负 | `base_optimizer.py:32–70` x/y integer、g continuous，但 z INTEGER | `flexible_direct_optimizer.py:90–173` x/y/w integer、流量和 r/q/z continuous | M 的 z 域不同；F 的额外 r/q 是可复用辅助量；u/ũ 全部缺失 |
| obj 84、m2:obj 145–146：罚损+人工+AV(+直送)服务时间成本 | `base_optimizer.py:72–88` | `flexible_direct_optimizer.py:175–215` | 四项结构可复用。代码用 hourly rates×t0/60；TeX称 unit-period cost 却又乘 t0（77、141），须统一量纲。F 额外 transfer_cost，V1等价模式应为0 |
| num man veh 89 | `base_optimizer.py:95–114` 活跃弧 sum x≤N^k | F 用含迁移表达式 | M 对应；端点 t=T 无活跃弧，不产生额外占用 |
| m2:num man veh 154：local active+累计直送出发−反向到达≤N^k | 不适用 | `flexible_direct_optimizer.py:228–269` | 已对应，可复用；不得漏掉直送完工后车辆留在目的城市 |
| num auto veh 93/159，bal auto ± 97/101/163/167 | `base_optimizer.py:118–163` | `flexible_direct_optimizer.py:271–318` | 在途总数和城市累计库存对应；τ>0 时 fleet active 为冗余但可保留 |
| BHH 106/172，wave link 110/176：x=Σv u，Σu≤1 | 无 u | 无 u | 新建 binary；v=0 通过全部 u=0 表示。不能用连续线性插值替代 |
| auto vol 114/180：Σ_l ĝ≤Mhat y | `base_optimizer.py:180–193` 对每笔订单分别 ĝ_l≤Mhat y | `flexible_direct_optimizer.py:346–360` 共享容量 | M 与 V1 明显不一致，多订单可重复使用同车容量；F 已对齐。两者不应视为同一 baseline |
| direct duration / Ktilde 141：origin zoned f(λ)/v + destination f(λ/v) + origin stem + τt0 | 无 | `flexible_direct_optimizer.py:50–88` 逆 f1(q)+f2(q)，min(capacity_direct)，再乘 w；`state_dependent_mip.py:310–328` 重复该系数 | 无 v、无方向、无 stem，与 V1 direct 不同；两城市不对称时新表必须按方向计算 |
| direct BHH 184、direct wave link 188 | 无 | `flexible_direct_optimizer.py:362–375` Σh≤direct_coeff×w | 新建 Ktilde(direction,arc,v) 与 ũ；V1使用 M，共享 manual capacity；代码另有 capacity_direct 是额外政策参数 |
| violate 118/192，violate direct 196、219 | `base_optimizer.py:195–204`；`loader.py:128–163` epsilon | `flexible_direct_optimizer.py:377–396` | 始发 i≥s、目的 j≤e 与直送 [i,j]⊂[s,e] 对应；方向稀疏变量应保留 |
| transfer step 1/2 122/126、200/204 | `base_optimizer.py:207–262` 按订单累计 | `flexible_direct_optimizer.py:398–464` 加每端 θ=transfer_time_periods，并等式 linehaul总量=r | F 仅 θ=0 时与 V1这些公式相符；额外 linehaul等式在完整有限域可行链下可作为加强约束，但需证明/检验窗口与历史扩展后仍等价 |
| unserved 130/208 | `base_optimizer.py:267–278` 两市各自 Σg=d−z | `flexible_direct_optimizer.py:466–521` origin r=destination r、direct Σh=q、r+q+z=d | F 的扩展变量表达可复用，需审核 partial-service 多段跨窗口历史 |
| return 212：Σw+=Σw−，221解释恢复人工车队 | 无 | `set_constraints:217–226` 及所有其下函数无该式 | 未实现；V1-F 与 V0-F 的可行域不同，回空成本当前可能低估 |
| 141：j−i=τ 的零货量回空弧 | 无 | `flexible_direct_optimizer.py:68–76` 从 τ+1 开始；paper generators 同样排除 τ 弧 | 未实现。空弧车辆变量必须允许、货量固定0；不能依赖存在订单才建弧 |

### 3.1 容量变化的方向不能一概判断

V1 比当前线性单车容量增加了 zone-wave 密度收益，同时加入 stem 和 vM 物理容量上限。新模型可能增加或减少特定弧的承运能力，不能事先声称目标必然改善。即使 ρ=0，v=1、需求不超过 M 时单车逆容量可相符，v>1 且 b>0 时 generally f^-1(vθ) 不等于 v f^-1(θ)。不能期待“ρ=0 即回归旧模型”。b=0、ρ=0、相同弧集、物理 cap 已一致、连续变量域相同等特殊条件下才可设计受控等价测试。

### 3.2 local wave 上界与人工迁移的冲突

V1 176 的 local u 枚举 v≤初始 N^k；154 允许反向直送迁入后当地可用人工车超过初始 N^k。当前 F 仅受动态库存约束，确可使用迁入车。严格照抄 V1 将额外限制某条 local wave≤N^k，即使当地有更多车辆。这不是实施时可以自行把上界改成总车队的“小修”，必须确认作者本意。多个相同开始、不同结束弧是否允许分别定义 wave，以及正文“同开始同方向”合并 wave 与 arc-wise wave 的关系，也须明示。

## 4. 求解流程与算法风险

### 4.1 真实入口与模型选择

`main.py:18,90,132,179` 解析参数、建立计划并调用实验套件；GUI 同用 registry。`experiments/core.py:473` 调 DataLoader 构建 DeliveryData；`:704,790–838` 先生成/载入相同 orders，再循环 solver；`:840–927` 输出汇总和详情。

`experiments/solvers.py` 实际映射：

| registry name | 类/入口 | 真实模型 |
|---|---|---|
| exact_mip | ExactMIPSolver.solve:73 | V0-M，全域已知订单 |
| rolling_horizon | RollingHorizonSolver.solve:150 → algorithms/rolling_horizon.py:59 | V0-M 历史固定与窗口约束 |
| flexible_direct_mip | FlexibleDirectMIPSolver.solve:194 | V0-F，全域已知订单 |
| flexible_direct_rolling | FlexibleDirectRollingSolver.solve:316 → FullWindowMIPApproach | V0-F + full-window候选，非候选裁剪 MIP 的“全域 exact” |
| paper_candidate_mip | PaperCandidateMIPSolver:422 → StateDependentMIPApproach | V0-F + reduced candidates |
| paper_priority_heuristic | PaperPriorityHeuristicSolver:436 → WarmStartedStateDependentMIPApproach | 启发式初解 + reduced MILP，非纯启发式最终结果 |
| paper_local_branching | PaperLocalBranchingSolver:450 → LocalBranchingStateDependentMIPApproach | 局部邻域阶段 + 移除邻域后的 reduced MILP |

### 4.2 弧域不一致与文内算法滞后

- `loader.py:85–89` 的最大人工弧长单位不一致；`state_dependent_mip.py:141–151` 却使用 ceil(f(M)/t0)。FullWindow 取 full_data 中的长弧，StateDependent 自建短弧，因此“剪枝比例”混入不同基础弧集。
- `full_window_mip.py:45–61` 限人工及 AV 新弧开始<start_end；`state_dependent_mip.py:153–181` 允许下游开始<completion_end。虽然 `ReducedFlexibleDirectOptimizer.build_model:330–345` 将 origin pickup 再限到 start_end，两者仍非同一弧域下的纯剪枝对照；reduced 甚至可能有 full-window 没有的下游弧。
- V1 文本 238–252 规定 start-window 的所有候选起始点，而当前 reduced 为 H<τ 保留下游延后起始。这是模型/算法规格的实质分歧，不可在迁移中悄悄决定。
- `state_dependent_mip.py:187–208` 直送必须有一笔满足时间窗的已知订单且起点城市 inherited availability 为正；`full_window_mip.py:67–78` 也按存在订单过滤。V1 回空弧不能按该规则删掉。
- `state_dependent_mip.py:47–79,164–181,203–206` 用已承诺历史计算可用车。对允许本窗口新增直送调入的 V1-F，history-only 某城市未来 availability=0 不足以证明未来 local arc 不可行；新调入车可改变未来库存。需独立证明的安全剪枝条件，或保守保留此类弧。不能把 V1 的 elimination proposition 自动应用到更广的 relocation 场景。
- V1 Algorithm/说明仍使用单车 C（例如 main_v2:525–565、623–645、688），candidate 参数未补 stem，且存在按订单筛直送弧。因此“严格同步正文”与“照抄算法段”会导向不同实现。

### 4.3 启发式、历史固定与初解必须联动

`bhh_priority_heuristic.py:65–119` 当前 residual=vehicles×capacity−load，新增车数按 ceil((q−residual)/capacity)；`:215–237` manual/direct capacity 仍旧逆函数；`:348–501` 用单车成本/容量评价净收益；`:503–586` 据此占用车队。V1 应使用 C(v)=K(v)：容量余量为 K(v)−L，增量车数为 min{m:K(v+m)≥L+q}，而非 q/C；direct 同理并按方向。成本和 stem 通过弧持续时间统一计费，新增回空的成本/车辆平衡必须计入；不能把只服务载货且不回空的构造解宣称为 V1 可行初解。

`base_optimizer.py:280–338`、`flexible_direct_optimizer.py:546–598` 提取/固定时间弧决策，当前没有 u/ũ；`paper_rolling_horizon.py:18–40,114–135` 只定义原有 six groups、对车辆整数做 round。迁移须选择“显式存并固定 selector”或“由固定 integer x/w 唯一重建 selector”；务必保存原波的组合及载货历史，不能新窗口将同一已发车波拆成不同 v。若显式加入 groups，selector 最后一维 v 不能被通用代码误认成 order_id。

`warm_started_mip.py:34–80` 给旧 groups 与 z/r/q 赋 Start；`local_branching_mip.py:136–158` 捕获旧 core variables。新增 binary 应完整赋值或明确让 solver completion，并做可行性验证；局部分支邻域距离应继续基于 x/y/w 还是包含 one-hot selectors，属于算法规格决定，不可机械双算同一车数变化。历史 flow 松弛 `state_dependent_mip.py:363–378` 只应限于明确记录的数值容差，不能松整数车/新 binary。

### 4.4 信息结构与结果字段

`paper_rolling_horizon.py:162–166` 用 earliest_start≤current_time 作为已知订单，没有独立 arrival_time；全域 exact 则事先拥有全部订单。`experiments/core.py:240–290` 的随机生成在求解前完成；不能据此认定模型使用 Poisson 到达或 stochastic control。若 s_l 并不等于 reveal_time，当前 rolling 信息结构不符合真实到达，须先确认数据定义。

`_PaperRollingSolver.solve` (`solvers.py:392–406`) 将最后窗口 best_bound/mip_gap 写进统一顶层字段，但 controller (`paper_rolling_horizon.py:233–275`) 总成本为最后窗口目标值，且若总预算先耗尽则只有部分订单已可见/部分时域完成。最后窗口最优不等于动态策略全域最优；窗口 bound 也不是滚动策略成本的全域 lower bound。只在 `completed=true`、全部应计订单进入核算且用已执行决策重新计算成本/罚损后，才能用于完成时域的策略比较。提前停止应标 incomplete，保留 partial cost，不能作为低成本胜出。

## 5. steady-state 与 V1/V2 的边界

S-linear 的 tour 式 `eq:zonetour:55–56` 与 V1 local wave 在 W=λ、n=λ/v 时结构相符；其 `eq:GZfull:95–97` 含 ω lead-time 与 Θ=2(ωθ+h)，按 shipment 归一化，fleet endogenous、不排队、平衡常率需求（49–76）。生产模型 fleet 固定，需求可拒绝且 finite horizon，无该目标；无法直接比较两个“total cost”。

S-deadline 在 `as:bhh:61–66` 增 slowest factor δ，`as:hub:81–82` sorting μ=κΛ、等待整波慢车，`eq:Tmean/Tmax:119–120`，`eq:G/P:128–132` 增 ω、硬 L。当前代码没有 μ、κ、δ、whole-wave barrier 或 lead-time 价格。仅设置 `transfer_time_periods` 是两端固定处理延迟，不等于 W/μ sorting；部分完成订单累计 transfer 约束也不等于整波等待最慢车。

因此本次建议 V1-M/F 优先迁移，S-linear/S-deadline 仅作为另列解析模型。若要与它们数值比较，须另授权设计匹配假设的桥接场景、单位/归一化及 warmup/terminal处理，不能直接复用旧实验并据图声称验证命题。V2 需要独立状态、到达/路由随机性、允许动作/干预时机、持有/拒绝/terminal cost 和 fleet恢复定义；本迁移计划不实施 QVI。

## 6. 数据与已有测试

`data/cfs_processor.py:528–534` 将 `SHIPMT_WGHT / pounds_per_model_unit` 四舍五入并上下截断为 quantity；`:584–585` 明示单位为 pounds。V1把 d_l 定义为 stops，CFS 无逐 customer-stop 映射，不能把一单位重量视为一停靠点并称实证校准。应选择合成 stops 校验或新增 stop_count/volume 双维数据 schema，后者是实质模型扩展须审批；保留原 CFS 输出及历史 benchmark。

`experiments/core.py:293–325,359–450` 有固定 seed 抽样、ID 重排及实时参数覆盖路径。迁移需保存原始行标识、抽样 seed、源数据散列、映射单位与被覆盖参数，避免把算法种子误作数据种子。CLI 与 GUI 的参数 metadata 会影响 solver applicability；新 ρ 等参数不能自动进入所有 V0 sensitivity 计划。

已有测试及实际覆盖边界（全部仅阅读）：

- `tests/test_flexible_direct_optimizer.py:31,44`：低直送成本、固定直送比例 exact/rolling；这些是 V0-F 特征，不能直接改期望值冒充 V1。
- `tests/test_paper_solution_approaches.py:27,42,60`：registry 运行、无 MIPGap 的窗口、真实订单重采样；不验证 V1 wave 或返空。
- `tests/test_regression_fixes.py:84,123,138,167,198,213,248,281,285,314`：扩展下游弧、服务经济性、adaptive trigger、local branching恢复、构造预算、flow dust、history retry、分钟转小时、warm start、full-window 长 travel。新模型必须保留这些行为回归，尤其最后两类窗口弧域还需统一。
- `tests/test_stage_c_protocol.py:31,36,70`：计划参数/机制边界/legacy ID；`test_cfs_data_processor.py`、`test_cfs_sqlite_gui_flow.py`：数据转换/SQLite/命名链路；不能提供 stop 语义的外部证据。

本轮未探测已安装版本及 Gurobi license、未导入项目包，避免生成缓存或调用求解器。因此测试能力/许可证状态保持 UNTESTED。

## 7. 最小侵入迁移任务（批准后，按依赖执行）

原则：保存 V0-M/F 类、registry 名称、默认参数与历史输出。新建 V1 专用 configuration/capacity/model/solver 路径，先取得静态与全域微型验证，再按次序接 RH、pruning、heuristic、local branching。明确 model_id、model_revision、algorithm_id，不覆盖“paper”旧名使旧脚本含义改变。下表是实施建议，不是已实施或已获运行预算。

| ID/依赖 | 实现方法/拟接触位置 | 验证 | 完成标准 |
|---|---|---|---|
| E0 无 | 用户确认第8节，冻结数学规格、stop单位、成本时间单位、V1-M/F与S/V2界限 | contract 与逐式追溯矩阵独立复核 | 所有会改变可行域的歧义有显式决定 |
| E1 E0 | 新增 V1Config 和独立纯容量/弧域模块；按 city/arc/v 和 direction/arc/v 预计算 K/Ktilde；按统一 base time 做闭式逆/二分及 bounds | 无 solver 的公式性质/边界测试 U1–U4 | 无 stem、cap、方向或时间单位遗漏；零/满容量与极端参数有效 |
| E2 E1 | 新增 V1MandatoryOptimizer；可复用 fleet/epsilon/transfer helper，但保留旧类；binary linking、共享AV、continuous z | 建模系数/域检查与 tiny enumeration U5/M1 | 每条 V1-M 公式可追溯至命名约束；独立验证器通过 |
| E3 E2 | 新增 V1FlexibleOptimizer：direction capacity、ũ、空返、终端返程等式；默认 θ=0、处理费=0、ratio=[0,1]、共享M/τ | asymmetric-city/imbalanced-demand/return cases M2 | 正/反空返均可行且计费，终端各城市人工fleet恢复，无重复货流 |
| E4 E3 | 新 registry 模型标签、结果 schema、独立约束残差及 cost recompute；旧名保持 | U6、golden V0 serialization/regression | 所有结果带模型/单位/完成状态/bound_scope；无窗口 bound 冒作全域 |
| E5 E4 | V1 RH 历史 selectors、revealed order IDs、固定/延后任务、终端义务、截止罚损账本；统一 start/ext约定 | M3 跨窗部分流、迁移在途、reveals、预算中止 | 完成成本来自 committed schedule、各订单只计一次；历史整数及selector不变 |
| E6 E5 | 统一 full/reduced 参考弧域，保护 empty returns，证明/保守化 relocation-aware pruning | M4 exhaustive tiny candidate/full comparison + referee proof review | reduced是full子集；安全删弧满足证明条件；不能证明的筛选标启发式 |
| E7 E6 | wave residual K(v)−load 和新增车查表；direct返程构造；warm start填selectors | M5 每次构造后校验全部约束；无法构造时回退MIP | 初解可行性可核查；不使用旧单车公式；预算停止不破坏状态 |
| E8 E7 | local branching incumbent含新变量，明确邻域度量，移除局部约束后解完整reduced | M6 恢复全 reduced feasible region检查 | 局部gap、窗口gap、全域gap分离；保留 incumbents可行 |
| E9 E8 | 新隔离运行目录、限定公平场景并提交明确资源预算 | 小规模 smoke先行，再申请长实验 | 代码审查/单测/微型MIP/回归通过且人类批准预算，才执行正式实验 |

## 8. 必须确认的数学/业务问题

1. 本次实施目标是否明确为 V1 main_v2 正文两模型，而 S-linear / S-deadline 单独保留解析分支？不得将 sorting/δ/ω 自动塞入 V1。
2. d、M、Mhat 是否统一为 stops？现有 CFS weight-units 数据如何合法映射？未解决前不能称真实数据已验证 V1。
3. t0、a/b、ρ、τ、c 的时间单位如何统一？建议实现规范显式区分 period index / minutes / hours，公式成本统一 cost_per_hour×hours；待用户确认。
4. V1-F local波上界 N^k 在迁入更多车时是否保留？以及相同开始/方向、不同终点弧是否是同一物理 wave？
5. 直送空返 τ 弧按正文为零capacity，但含正 stem 的 tilde-f(0,v)>τt0；是否约定空返独立 bypass local tour（建议明确特例，不能自行修稿）？
6. m2:return 是全域终端恢复条件。滚动子问题是否须立即恢复、只在 T 恢复，或引入 terminal imbalance obligation？短窗口强加 Σw+=Σw−会人为过度限制。
7. start-window 是否限制所有弧还是仅订单最初 pickup/direct？正文与现有 reduced 不同，需要唯一可验证定义。
8. s_l 是否同时表示揭示时刻？若不是，需独立 reveal_time，不能让 future订单提前被算法知道。
9. main_v2 未含 transfer θ、transfer fee、direct ratio、独立 capacity_direct/τ_direct；新默认是否严格禁用这些旧扩展，保留为另有标签的政策变体？

## 9. 测试、数值验证与回归设计（尚未执行）

### 9.1 纯函数/静态测试，无 Gurobi

| 测试 | 案例与独立判据 |
|---|---|
| U1 manual table | v=0→0；duration≤2ρ→0；0≤K≤vM；对duration/v非减；将 K 代回 f(K)≤v positive-tour-budget；单车不饱和闭式手算 |
| U2 economies | b>0、ρ=0、不饱和时比较 K(2) 与2K(1)，确认变化来自 wave；b=0 时线性边界，绝不预设跨饱和段严格超线性 |
| U3 direct table | 对两个方向分别代回 tilde-f；二分上下界/残差容差；a/b不对称时不错误复用同一表；λ=vM和无可行λ；empty-only τ特例显式测试 |
| U4 arcs/units | ceil临界、t0=60与30的物理时间一致映射、人工stem上界、τ空返边界、j=T、S(t)半开区间；full/reduced同一参考域 |
| U5 domains | u/ũ binary、x/y/w integer、流量/zcontinuous；v=1与all-u=0；缺订单方向不创建虚假流；cap参数正负/τ整数校验 |
| U6 result semantics | incompleted horizon不得给complete total；window bound不写全域；model_id/unit/schema保存；manual_usage注明是dispatch次数而非distinct vehicles |

### 9.2 需后续批准预算的微型数值验证

M1：T=4–8、每城1–2辆HV、0–2辆AV、1–3笔合成stops订单，手工/枚举核对最优目标与可行性，含 all-unserved、无车、窗不可行、分数需求与连续 z。两订单共享AV容量案例专门区分 V0-M per-order 与 V1 aggregated；此差异是预期模型变化，不能当成回归失败。

M2：单向订单必须付空返、两向负荷不平衡、无反向订单仍可返空、同时间到达后可再发、跨城迁入后local服务、T终端库存、非对称城市direct方向差异。对终端政策采用E0确认规格。

M3：跨窗口已揽收/干线在途/末端未送完、车已直送迁出迁入、订单晚于当前才revealed、deadline位于控制窗边界、预算在首窗/中途停止。独立账本核对收运/送达/拒绝不重复记账。

M4：对同一 tiny full-window模型枚举或最优求解，验证裁剪模型可行解属于完整模型；对宣称安全筛选，比较最优目标一致。特设 inherited当地0车但新调入可服务反例；只观察少量一致不足以证明 general safe pruning。

M5/M6：构造初解后独立重算 capacity/fleet/flow/time/terminal残差，验证 selector一致；warm start是否接受必须读日志，不能以赋 Start 数量代替；local branching结束后检查所有邻域约束已移除、完整reduced最优界域记录正确。

建议下一阶段单独申请的上限（并非本轮授权）：纯函数测试优先；micro MIP每例≤5秒、1线程、总≤60秒、仅tiny合成数据，禁止隐式全套旧30秒测试并行启动。许可证失败记为 unavailable，不能当测试通过。若仅验证可行性，可冻结手工 schedule直接评价，无须求最优。

### 9.3 回归与公平比较

- V0 原类/registry/旧配置/原始数据逐字保留；现有 tests 在单独批准的短预算下跑分层子集。新增V1预期使用独立fixture，不能修改旧结果来“通过”。
- 相同数据、reveals、fleet、时间/成本单位、总墙钟预算、线程/solver seed、tol、gap stop；先比较 V1 full-window vs reduced，再比warm/local，否则目标改善可能来自模型差别。
- exact clairvoyant与online RH需要不同 information label；全域clairvoyant最优值只有在同一V1可行域与同一订单集时才可作offline reference；窗口最优值不提供全域动态下界。
- 指标：独立重算total operational cost/penalty，served/unserved stops、on-time service、direct ratio、empty return trips/cost、manual/AV库存最小残差、最大容量/流守恒/窗违反、completed_horizon、wall time、构造/建模/求解分时、incumbent availability、bound_scope/optimality。normalized cost同时注明按offered或served stops；0 served时不除零。

## 10. 可复现性保护与审计限制

已有 `simulation/run_protocol.py:435–473` 拒绝非空批次目录，并记录Git、Python、platform、数据库大小/时间、可选SHA256和Gurobi版本；这套隔离结构可复用。迁移应补：四份TeX及相关配置/源代码散列（含untracked）、model_id/revision、数据单位映射、完整依赖版本、solver seed/threads/tolerances、UTC+本地时区、起止/退出码、license错误、完成时域状态、命令/参数、源数据与抽样ID、原始solver log与解释报告分离。历史 CSV/JSON/run目录均不改写。

本轮只读审计不能证明当前任何 solver 正常运行、模型数值正确、历史结果可信、或新迁移会改善目标。重要差异依据源代码和公式直接识别；算法安全性与模型等价性未获得完整证明，必须经 Referee 及用户批准后推进 E0–E9。
