# V1-F / Rolling Horizon / 剪枝独立实施审查

2026-10-18。Referee 在实施专员交付后审查；本审查未修改生产代码、论文、实验结果，未调用求解器。

**结论：ACCEPT，限于已披露的 V1-F 工程模型、RH/保守剪枝实现及小样本 CFS 可运行性。** 没有发现阻断本轮交付的错误。此结论不认证原论文逐字等价、一般剪枝最优性、算法性能优势、全规模实验或 QVI。

## 证据与交付判断

| 交付 / 主张 | 判断 | 核查依据与范围 |
|---|---|---|
| V1-F 方向相关 direct 容量 | ACCEPT | `research/paper/main_v2_BHH.tex:141` 与 `intercity_delivery/v1/flexible_capacity.py:10`。展开后线性项为 `(a_o+a_d)q/v`，平方根项为 `(b_o/v+b_d/sqrt(v))*sqrt(q)`；闭式逆与原单调式一致。tau 长度载货容量单独为零；完整母弧保持。 |
| 迁入库存与 local selector | ACCEPT，属于显式源模型修正 | `flexible_optimizer.py:33,123` 使用两城人工总车数作为 selector 上界，实际可用量由累计 direct+empty 出入和半开 local 占用约束。这允许初始零车城市在车辆迁入后服务，不能称为原文 `v<=N_city` 的逐字实现。 |
| 无窗口复位、独立 empty、绝对历史 | ACCEPT | `flexible_optimizer.py:192,223`、`flexible_rolling.py`：历史开始事件和隐式零固定，过去任务不被当前 completion cut 截断；所有车辆母弧不依赖订单。默认无真实终点目标，full 可显式指定；RH 拒绝尚无尾部可达性保障的终点目标。 |
| 真实货流剪枝 | ACCEPT，范围为必要时间支持筛选 | `pruning.py:19,63–97` 与 optimizer 的 sparse cargo 建模：按释放/期限、正容量、三段时间支持删 cargo 变量；既有 pickup/AV 分别作为后继路径种子，保留历史非零变量。车辆/selector 母弧及独立 empty 保留，不按历史零库存删除未来迁入服务。 |
| 原 f 独立核验与 RH 状态 | ACCEPT | `flexible_validation.py:23,148` 直接检查原 f 时长不等式、物理容量、逐时库存、逐单因果和费用，未用 optimizer 的 K 表自证。`flexible_runner.py:91,183` 将 partial 的全日成本与 RH 全局 bound/gap 设为空；完成 RH 仍不标全局 optimal。 |
| CFS 单位、来源、tau 与混跑隔离 | ACCEPT，执行代理假设 | `adapter.py`、`experiments/core.py:463,495`：真实 CFS 一记录作为一 shipment-stop；原重量/原数量保留在 provenance。转换发生在该 solver 的数据构造前，汇总需求同单位；V0 使用原对象。processed JSON 的 travel 建议仅映射到 V1 局部配置，显式 travel 灵敏度保留。 |
| CLI / registry 集成 | ACCEPT，静态与纯测试范围 | 三个新 solver 已登记；旧默认集合保持。专用 CFS smoke 的实际数值调用通过同一 V1 runner；本轮没有另行运行正常 CLI 的持久化实验套件，因此不声称该套件写盘路径已实跑。 |
| CFS full/RH/pruned RH 可运行 | ACCEPT，8 条样本 | 下述保存结果已由 Referee 独立重放；不是全规模研究实验。 |

## 独立执行与保存证据

Referee 执行 69 项纯测试，exit 0：`FlexiblePureTests`、`ControllerTests`、`PruningTests`、`FlexibleCheckerTests`、`FlexibleRunnerPureTests` 和 `test_v1f_integration`。这不含任何真实优化调用。PI 最终报告 73 项纯测试全部通过（0.537 秒，包含旧入口的 4 项 integration 回归；在此前 71 项口径上另有 options 与 sparse export 各一项）。该最终运行由 PI 执行，Referee 未重复运行；69 与 73 不应相加。

数值依据为 [保存的完整 CFS 输入及非零计划](../../experiments/reports/v1_f_cfs_smoke_results.json)。Referee 从 JSON 重建 `V1FConfig` / `V1Order`，逐份再次运行独立原 f validator，并传入保存的总成本核对。三份重放均通过；所有记录的 runtime source SHA256 与当前文件一致，processed CFS 源文件 SHA256 一致。非零导出没有阈值舍入，省略的合法变量按零解释。

| solver | 完成与证书范围 | 总成本 | 服务 / 需求 | 方式 | 拒绝量 |
|---|---|---:|---:|---|---:|
| `v1_flexible_mip` | full OPTIMAL；该配置下 bound=1100、gap=0 | 1100 | 8 / 8 | direct 6、transshipment 2 | 0 |
| `v1_flexible_rolling` | 实际 T=24 完成，12 控制窗；无全日 bound/gap | 2025 | 7 / 8 | direct 2、transshipment 5 | 1 |
| `v1_flexible_pruned_rolling` | 同上 | 2025 | 7 / 8 | direct 2、transshipment 5 | 1 |

样本 seed=4101、两向各四条；路线 `06-348` ↔ `06-488`、tau=9、t0=60 分钟。RH 为 H=8、step=2、extension=6；同一批订单，但 full 离线信息与 RH 按 release 揭示的可用信息不同。高拒绝罚金及运力/容量参数用于执行检查，未校准为实证成本。

按保存的每窗实际 `cargo_variable_count` 求和，未剪枝 8265、剪枝 397，差 7868（约 95.20%）；与 `pruning.removed` 合计一致。这是跨 12 个分别重建模型的 cargo 变量数，不能称为单个模型全部变量削减率。该样本两种 RH 成本和服务量相同，不推出一般剪枝等价或加速。保存的三 run 总耗时约 0.953 秒，仅记录执行事实，不作性能比较。

PI 报告 7 个微型求解用例先跑 7 次，其中一个断言要求独立 empty 必须非零，因 tau 长度的零货 direct 与 empty 同价而失败；成本/完成检查仍通过。随后只在测试配置中令 empty 更便宜，定向复跑通过成本 2.5；累计 8 次微型 run。Referee 核对了对应测试及模型允许这两类物理移动的来源，没有自行重跑 solver，也不把最初断言失败写成模型不可行。

## 按严重性记录的发现与边界

| 严重性 | 证据 / 文件 | 影响 | 补救 | 状态 |
|---|---|---|---|---|
| LOW | `experiments/solvers.py:26` 原 `SolverResult.status: Optional[int]`；RH runner 实际返回 `COMPLETED` / `PARTIAL_*` 字符串 | 原静态类型契约不准确；运行透传/序列化及原数值结果不受影响 | PI 已增加 `Union` 导入并改为 `Optional[Union[int, str]]`；Referee 定向读源码确认 | CLOSED |
| 范围限制 | 源 TeX local selector/return 与 runtime 合同；`flexible_runner.py` metadata | 该实现包含迁入上界修正、独立 empty、可选真实终点，以及兼容 transfer/ratio 扩展 | 保持显式披露；若主张“源论文原式完全等价”，需另行数学对齐，届时该主张应 MAJOR REVISION | 已披露，受控 |
| 范围限制 | `pruning.py` 明确 `global_safety_proof=False`；仅一个 CFS 样本及有限微例 | 当前证据支持实现和样本可行性，不能认证一般无损剪枝或算法优越性 | 扩展到研究性实验或论文定理时再提供对应证明/系统测试；本轮无需大型实验 | 不作为已完成主张 |
| 范围限制 | offline full 与 release-revealed RH 的信息集不同 | 1100 与 2025 的差异不是公平的纯算法优劣估计，也不是可用的在线全局 gap | 不排名、不计算 RH 全局 gap；后续比较须另设计同信息基线 | 已按此报告 |

本轮有可重放的输入、计划、独立可行性与成本证据，进展超出了未审查文本。对 S / 随机订单 / 内生 wave / QVI 的推导、证明或文献创新性，本次没有产生新证据，也未授予接受结论。
