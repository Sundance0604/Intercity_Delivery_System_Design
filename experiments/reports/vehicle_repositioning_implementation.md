# 独立空车调拨实施与有限验证

2026-10-18，PI综合。分支`research/qvi-multiagent`；开始HEAD为`f26b4c96907082c2ebb7b0104263831e7100aa26`，当前含未提交工作；HEAD不能唯一标识本地输入。用户已批准实施，未授权本轮提交或推送。

## 实施范围

`intercity_delivery/repositioning/`新增轻量初始化、config、ledger、optimizer、rolling、runner。两位Experiment分别负责政策/账本和模型/控制/结果，PI整合入口。模型继承实际V0-F货流与容量，加订单无关empty整数变量、全母弧域、费用及累计人工库存。计算窗无车辆复位；所有历史（包括已完成迁移和在途）按绝对时间固定，控制区间仅提交已发起任务。

默认empty时长沿用direct时长，小时费用沿用cost_direct，terminal_target=None；API可覆写，metadata披露来源。完整模式可设真实T目标；rolling暂显式拒绝目标，避免未经验证的终点可达性。新任务须在完成窗内结束，不截断历史。成本独立重算，partial完整总成本为空；rolling全局bound/gap为空，窗口值留在detail。partial最后窗OPTIMAL不能当全程完成，顶层改TIME_LIMIT并保留last_window_status。

三处原文件仅作入口接入：`main.py`新增可选名称；`intercity_delivery/configuration.py`标注direct/extension参数适用；`intercity_delivery/experiments/solvers.py`注册与结果适配。旧默认exact_mip和旧类映射保留，GUI读取注册表。

## 实际验证

运行环境：`C:/Users/32599/.conda/envs/pavane/python.exe`，Gurobi10.0.3，许可证可用。未安装依赖。预算见[方案](../specs/vehicle_repositioning_micro_validation.md)：总solver≤60秒、单线程、仅微型合成实例，无真实数据/大型实验。

| 命令（均使用上述Python，带-B） | 实际结果 |
|---|---|
| `-m unittest discover -s tests -p test_repositioning_ledger.py -v` | 18项PASS，测试约0.002秒，exit0 |
| `-m unittest tests.test_repositioning_integration -v` | 4项PASS，约0.739秒，exit0 |
| `-m unittest tests.test_repositioning_optimizer tests.test_repositioning_rolling -v`（添加真实RH例前） | 11项PASS，约0.062秒，exit0；含6次完整微型测试，每次solver限制2秒 |
| `-m unittest tests.test_repositioning_rolling.RepositioningRollingTests.test_registry_rolling_empty_migration_then_cross_window_delivery -v` | 1项PASS，约0.018秒，exit0；真实rolling整次预算2秒 |
| `-m unittest tests.test_repositioning_rolling.RepositioningRollingTests.test_partial_report_has_no_complete_total_or_global_bound -v`（status修复后） | 定向重复1项PASS，约0.001秒，exit0，无求解 |
| `main.py --cli --solver flexible_direct_empty_rolling --dry-run --seeds 1` | exit0，正确生成计划，无求解 |

共34个不同用例通过；不是35个不同测试。测试耗时包含建模/断言，不是精确求解器CPU累计统计；相关进程均小于1秒，远低于预算。未扩大至旧全套求解测试。

核心证据：无订单也可强制空调拨且不回归；已完成迁移支持下一窗本地任务；在途车不可提前使用；真实T目标可用独立空车恢复；0.05货量保留；完整目标与独立核算一致并可JSON导出。真实RH例T5、初始HV=(1,0)、AV=0，t1揭示city2→1需求0.05；先empty迁入再direct，最终交付0.05、手算费用3，首窗不见未来订单。相同成本存在时点并列最优，不硬编码early-dispatch规则。

## 独立审查与限制

[Referee报告](../../coordination/reviews/vehicle_repositioning_implementation_review.md)限定ACCEPT，未发现车辆机制阻断问题。partial status问题已修复，经Referee静态复核及PI定向测试通过。

仍未解决：真实RH测试覆盖empty→direct，未覆盖pickup→AV→destination跨窗换装链；ledger仅检查车辆库存、标量输入与成本，不独立验证全套货流因果、BHH容量和服务窗。rolling真实终点目标、任意跨完成边界新任务、empty剪枝/启发式/warm start/local branching均未实现。V1 wave/stem容量、稳态与随机QVI尚未迁移或证明。不对历史实验作新旧模型公平比较或普遍最优性声明。

## 文件和后续

新增6个模块文件（含__init__）及4个`test_repositioning_*.py`；上述3处入口修改；新增本报告、微型预算方案和实施审查；更新README及协调合同/主张/任务/进度/计划/车辆规格。保留此前未提交的规格/toy及无关untracked输入，未改历史结果或TeX，未提交推送。

下一有限任务：先补跨窗换装链微例和独立cargo验算，再按业务需要实现真实终点安全尾计划；完整V1继续按D1–D6确认后的路线推进。详见[MIGRATION_PLAN第8节](../../coordination/MIGRATION_PLAN.md)。

收尾核对：实施前后21份原models/algorithms/data及四TeX的SHA-256全部一致；11份导航/协调/实施文档的本地链接无缺失；`git diff --check`通过，仅现有LF/CRLF提示。Git状态前后已读取；既有`.pytest_cache`权限警告保留，未尝试清理。
