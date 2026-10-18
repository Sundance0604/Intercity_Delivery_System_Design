# 独立空车调拨实现审查

日期：2026-10-18。Reviewer：Referee。范围：已交付的 `intercity_delivery/repositioning/` 六个文件、四份 `tests/test_repositioning_*.py`、`main.py`、`intercity_delivery/configuration.py`、`intercity_delivery/experiments/solvers.py` 的接入差异。参照 `AGENTS.md`、`MODEL_CONTRACT.md`、`VEHICLE_REPOSITIONING_SPEC.md` 及继承的 `flexible_direct_optimizer.py`。只读审查实现；本审查没有执行求解器或修改实现。

**结论：对限定的、显式选择的 V0-F + 独立 empty 原型交付建议 ACCEPT。对“完整 V1 已迁移”“全货物可行性已有独立验证”“RH 具有全局最优性证书”的主张均不予接受；当前实现也没有作这些主张。** 没有发现需要撤回本次两项车辆政策实现的阻断错误。以下开放项限制证据强度和对外结果解释。

## 交付与执行证据

PI 告知已执行 18 项 ledger、4 项入口、6 项 full 微型求解及 6 项 rolling 测试，共 34 项通过；真实 registry rolling 用例耗时约 0.018 秒，rolling 总预算 2 秒，full 各求解至多 2 秒、单线程。Reviewer 查看了用例源码，没有重新执行，也没有独立核验完整 stdout。通过数量是 PI 提供的执行证据，不是 Reviewer 新产生的实验结果。真实数据、规模、随机重复、性能及统计优势均未测试。

真实 rolling 用例验证订单晚揭示、empty 调入、后续 direct 出发及完成、0.05 连续货量、执行总成本 3、完成状态和空的顶层全局 bound/gap。它没有覆盖三段换装链。

## 按严重性排列的发现

| 严重性 | 证据 / 文件位置 | 影响 | 修复或处理 | 状态 |
|---|---|---|---|---|
| 中（已修复） | 原 `rolling.py:100` 的末窗状态透传问题；现 `runner.py:21–28` 及 `test_partial_report_has_no_complete_total_or_global_bound` | 原先一窗最优后预算耗尽可把 partial 标为 OPTIMAL；现未完成 rolling 且末窗 status=2 时顶层明确映射 TIME_LIMIT=9，并在 detail.algorithm.last_window_status 保留原值。其他不可行/错误状态不被覆盖 | Reviewer 已只读核实该分支与 stub 同时断言顶层9、末窗2；PI 执行该无求解器 stub 后记录结果。完整 RH 的末窗最优仍不等于全局最优，必须结合 bound_scope、completed 解读；CSV 增加显式 completed 等列仍为可选后续改进 | FIXED_STATIC_ACCEPT；修复后测试待 PI 记录 |
| 中 | `tests/test_repositioning_rolling.py:test_registry_rolling_empty_migration_then_cross_window_delivery`、`optimizer.py:117–125`、继承模型 FD-(11)–FD-(17) | 真实 rolling 回归只走 direct，不能据此宣称已实测 pickup→AV→destination 的跨窗货物延续与换装费用。静态代码保留全部既有订单/历史和累计时序，未发现删除义务，但缺直接执行反例测试 | 后续限定预算增加强制换装、至少两次控制提交的微例；固定/核对 pickup、AV、末端的绝对时刻、容量与时间先后、处理费、最终 r 与配送量，不把账本的目的端求和当独立货物证明 | OPEN，缩窄测试覆盖表述即可接受本次限定交付 |
| 低 | `MODEL_CONTRACT.md`、`VEHICLE_REPOSITIONING_SPEC.md` 在审查开始时仍写生产未开始、默认项未决定 | 与新模块已经存在的状态不一致，会误导后续 agent；完整 V1 未实施仍属实 | PI 完成本轮综合时明确登记：已实现的是隔离 V0-F 扩展；fallback 默认的来源、批准范围和真实终点限制；完整 V1/QVI 仍开放 | PENDING_PI_SYNTHESIS |

## 逐项判断

| Claim / deliverable | 判断 | 证据与成立范围 |
|---|---|---|
| rolling 计算窗不强制城市复位 | ACCEPT | `optimizer.py:66–79` 唯一可选终点等式指向 cfg.T；`configure_event_window` 无城市分布等式。rolling 在构造前拒绝 terminal_target，避免无尾部可达性证明仍承诺真实关停目标 |
| 独立、无订单 empty 网络与单独费用 | ACCEPT | `optimizer.py:20–45` 仅由 T、正整数空行时长及两方向生成 empty_arcs/e，e 不进入继承模型货量/容量/服务比例；费用加在完整父目标上，按 period_hours 乘整段时长。无订单强制 empty 微例直接覆盖候选与库存 |
| 库存与绝对历史正确接通 | ACCEPT，限静态检查及已列微例 | `_city_idle` 计 w+e 的累计出入，local 只扣占用；`configure_event_window:108–116` 对所有过去开始的变量精确固定，包括隐式零；`rolling.py:91–95` 只增量提交当前控制区，完整历史保留。`ledger.py:56–96` 完成迁移不会从账本消失、同刻到达可用、在途不能提前使用 |
| partial 费用及 bound 没有冒充完整结论 | ACCEPT，status 修复静态接受 | `runner.py:_result` partial total_cost/unserved_rate 为 None；rolling 顶层 best_bound/mip_gap 为 None，每窗 bound 标记 window_local_only；账本按已承诺任务整段计费，未声称 elapsed-only 成本。未完成 rolling 的末窗 OPTIMAL 已映射顶层 TIME_LIMIT，原始状态另外保留 |
| 完整模式真实 T 的显式目标 | ACCEPT | target 默认 None，校验总人工车数守恒；所有新任务均在物理母域 T 内结束，T 时库存等式计 w+e。账本再核对所有任务完工且目标相符。rolling hard target 尚未实现并主动拒绝，不能声称已有持续可达性控制 |
| 有限完成域与跨窗义务 | ACCEPT，限所选域 | 新任务必须 j≤completion_end；过去已出发任务在设置新边界前先固定，因此不能截短。controller 的 completion_end 单调不减，已接受计划的完成域不会后退。可在有限扩展中完成后续 AV/末端，不能推广为允许任意越 cut 新任务 |
| 独立账本验证全部模型可行性 | REJECT，如提出此主张 | `ledger.py:257` 明确排除货物因果、容量、服务窗及比例政策；已有测试甚至故意用没有完整货物链的末端货流检验核算。ledger_valid 只表明标明范围内的车辆账本/标量/核算通过，不能充当完整 MILP verifier |
| 旧默认行为兼容 | ACCEPT，限接入静态证据 | registry 增加新名字，旧 key/class 保持；CLI 默认 exact_mip 保持；新 rolling extension 元数据单独接通；没有把 e 注入旧 warm start/local branching/heuristic。旧大型回归未运行，不作全库兼容性实测保证 |
| 完整 V1、BHH wave 或 V2/QVI 已实现 | REJECT，如提出此主张 | 继承 V0-F 的容量/货流/目标结构；`runner.py:55` 明确标记 not V1 wave MILP。没有新增 QVI 状态、随机生成元或 intervention 理论 |
| “每辆 w 必有正货”或“e 等于所有实际空驶” | REJECT，如提出此主张 | 容量只有上界，w 可零货；`ledger.py` 分别报告服务车次与正货弧数。实现没有擅加 epsilon 配载门槛，该语义符合现有修正规格 |

## 尚不能得出的结论

本审查没有证明正式 V1 公式与代码等价，没有推导或审查新的 QVI 证明，没有校验文献创新性，也没有生成正式数值比较。没有全局在线最优性、跨样本政策优劣或工程规模性能证据。旧 baseline/TeX 的前后散列检查由 PI 综合阶段提供，本审查不替代该检查。

本次 verified knowledge 是独立 e 与窗口无复位机制的具体实现、有限微例和可审查的明确边界；生产规模准备度、三段换装跨窗回归仍是后续事项。partial 状态修复已静态接受，修复后 stub 执行由 PI 记录；独立执行状态的更多 CSV 列可另作界面增强。
