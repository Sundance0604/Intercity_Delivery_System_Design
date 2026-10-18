# V1-F、rolling horizon 与剪枝：实现及 CFS 执行检查

2026-10-18。分支 `research/qvi-multiagent`。负责人 PI；模型、控制/剪枝、独立核验、数据/入口由四个 Experiment 分工，Theorist只读核公式，Referee在具体产物交付后独立审查。

**本轮交付完成：完整 V1-F、release-revealed rolling horizon、实际稀疏货流剪枝及三个注册入口均已实现；同组8条CFS订单的三个运行全部完成并通过独立核验。** [独立审查](../../coordination/reviews/v1_f_runtime_review.md)限定ACCEPT；[完整输入和非零结果](v1_f_cfs_smoke_results.json)可重放。

## 1. 实现与源模型的关系

继承作者确认D1–D6-A：逐起止期/城市/方向wave、连续stop量、分钟与车小时费率、tour结束交付、同刻交接、整数时间窗。数学来源为 `research/paper/main_v2_BHH.tex`，并遵守用户两项评审：计算窗口不复位；empty独立于真实订单。详细API和工程选择见[运行合同](../../coordination/V1_F_RUNTIME_CONTRACT.md)。

- local容量为 `K(v)=min(vM, f^-1(v[Δ−2ρ]^+))`，one-hot选择联合wave车数。迁入一致性修正：selector上界用两城总人工车数，城市实际可用量由动态库存限制，避免初始零车城市永远无法用迁入车。
- direct遵循源141行 `2ρ_o+f_o(q)/v+τt0+f_d(q/v)≤Δ`，等价逆式系数 `A=(a_o+a_d)/v, B=b_o/v+b_d/√v`，另限 `q≤vM`；τ长度弧载货容量为零。
- 独立 `empty_vehicles` 有完整母弧、费用与累计库存，无订单也可创建。源保留的零货direct弧同样可能被选作车辆移动，不能以一次解中e=0断言独立empty未实现；载货由h变量表达。
- 真实期末默认无目标；full支持显式目标，RH拒绝非空目标，等待全程尾部可达性机制。每个计算窗口均无城市回归约束。
- RH保留所有过去开始的vehicle/selector/cargo事件及隐式零，任务不因新窗口截断。pickup→AV→配送可跨控制窗口；订单只在释放后可见，到期或已经pickup的订单不会从历史删除。
- 剪枝实际稀疏创建g_manual/g_auto/h_direct：释放/期限、正容量、三腿时间支持，历史pickup和AV为后续种子。保留所有车辆/selector母弧，不用history-only零库存删除未来迁入任务。
- 独立核验器直接代回原f，不用模型K表自证；检查数值域、车辆库存、共享容量、逐单货流、时窗、需求和成本。partial不给完整成本，RH不给全日bound/gap。full的OPTIMAL只针对所披露工程模型。

这不是对源文逐字等价、任意实例无损剪枝、稳态或QVI的认证；transfer/ratio是显式兼容扩展，默认0费用/延迟、比例[0,1]。

## 2. CFS 输入和运行事实

复用已有 `data/cfs_processed_sqlite/cfs_model_orders.json`，来源2022 CFS PUMS，源SHA-256：
`e5cc5fb58711dcdea4fa36fa2169ff8f4f74eaac8ef93d4c9128644263eae911`。

城市对06-348 Los Angeles ↔06-488 San Francisco；seed4101，正反各4条，保留原SHIPMT_ID、重量、权重、旧数量和旧合成窗口。显式一shipment一stop是运行代理，未校准客户停靠点，重量/权重未作新需求。普通core按solver复制数量和配置，V0保持原重量语义；V1从JSON来源采用τ建议，显式travel敏感性覆盖保留并披露。SQLite只读索引抽样亦可复用，但本轮实际三次求解用已处理JSON。

参数：T=24，t0=60分钟，τ=9；人工车每城2辆、AV每城1辆；M=4、M_auto=8；a=.05、b=.1、ρ=5分钟；人工/AV/direct费率20/15/25每小时；罚损1000/stop，专为正服务检查设置。空驶时间和费率回退τ/direct；没有终点目标。RH H=8、step=2、extension=6。

| 入口 | 状态 | 服务 | direct / 换装 | 拒绝 | 成本 | 独立核验 |
|---|---|---:|---|---:|---:|---|
| v1_flexible_mip | OPTIMAL；full bound1100，gap0 | 8 | 6 / 2 | 0 | 1100 | PASS |
| v1_flexible_rolling | COMPLETED；T24，12窗 | 7 | 2 / 5 | 1 | 2025 | PASS |
| v1_flexible_pruned_rolling | COMPLETED；T24，12窗 | 7 | 2 / 5 | 1 | 2025 | PASS |

full模型1622个变量。两种RH的12个分别重建模型，实际货流变量累计8265→397，删除7868，约95.20%；与剪枝统计一致。这个比例不包含全部车辆/selector变量，也不是一个模型的总变量削减率。两种RH在此样本成本和服务量相同，不证明一般等价。full离线信息与RH在线信息不同，不能直接据1100/2025计算在线全局gap。

三run总墙钟约0.953秒，各runner分别约0.188/0.422/0.343秒，仅记录执行事实，不用微小样本宣称加速。empty计数均为0；独立empty的正例由手算微例另行验证。

## 3. 检查、失败记录与预算

- 最终73个纯测试一次通过（约0.537秒）：模型7、控制9、剪枝9、独立核验与runner33、数据/入口11、旧empty入口回归4。它们不调用求解器。
- 七个不同真实微型用例分批通过，累计实际8次run：连续direct拆分成本53；empty后反向direct成本3.5；迁入零初始库存城市后local/换装成本5；无订单保留已承诺empty成本1；未剪枝/剪枝三腿跨窗成本各3；隐藏未来订单后empty→direct成本2.5。
- 初轮7次中最后一个测试只因“e必须非零”断言失败，成本3和完成均通过；原因是τ长度零货direct与empty同价的最优解退化。仅将测试empty费率调为.5、手算改2.5后定向复验通过，没有缩生产可行域来迎合断言。保留此失败事实，不把初轮记为全通过。
- 本轮实际CFS三run，不重跑大实验。批准上限为3×15秒 + 最多10×2秒=65秒，实际微例8run及CFS3run，按各run上限合计61秒，真实各进程均小于2秒；不冒称精确solver累计CPU时间。
- Referee另行纯测试，并从保存JSON独立重建三方案、代回原f/库存/货流/成本全部通过，核对runtime/source散列一致。未重复求解。
- Referee唯一LOW为旧SolverResult.status整数注解与新RH字符串状态不符；PI改为Optional[Union[int,str]]，不改变结果或求解。
- 21份原models/algorithms/data及四TeX来源前后SHA-256一致；25份相关Python AST有效；git diff --check通过（现有LF/CRLF提醒）。已有untracked文件和.pytest_cache权限警告保留。

保存结果为非零schedule，所有十组名保留、精确零省略、无容差舍弃或小量舍入；遗漏合法键按0重建。包含全部配置、订单、source records、版本/输入/来源散列、验证结果、窗口和剪枝统计。专用脚本在可选输出创建检查后才求解；默认stdout，PI通过可用文件工具保存，未覆盖旧结果。

## 4. 本轮文件职责与变更范围

| 文件 | 本轮变更目的 |
|---|---|
| v1/flexible_configuration.py | F参数、信息集、empty回退、终点配置检查 |
| v1/flexible_capacity.py | 方向相关direct逆容量、direct/empty母域 |
| v1/flexible_optimizer.py | 十组变量、联合wave、共享容量、迁入库存、历史固定及稀疏货流 |
| v1/flexible_rolling.py | release揭示、绝对历史、提交、总预算、完成/部分状态 |
| v1/pruning.py | 真正删除无时间支持的cargo候选，留历史及车辆母弧 |
| v1/flexible_validation.py | 原式独立全约束/成本核验 |
| v1/flexible_runner.py | 完整与RH统一运行、证书范围和provenance |
| v1/adapter.py | CFS/旧配置/结果显式映射 |
| tests/test_v1_flexible.py、test_v1_flexible_rolling.py | 纯检查与7个手算真实微例 |
| tests/test_v1_pruning.py | 时间支持、历史后继、迁入保护 |
| tests/test_v1_flexible_validation.py | 原f/库存/货流/数值域与状态负例 |
| tests/test_v1f_integration.py | 来源、数量口径、tau、混跑隔离与导出 |
| experiments/specs/run_v1f_cfs_smoke.py | 有界预检/三run/完整证据输出 |
| intercity_delivery/configuration.py | 显式V1两城stem字段和新入口参数适用 |
| intercity_delivery/experiments/solvers.py | 三个可选V1-F入口及状态类型 |
| intercity_delivery/experiments/core.py | 按solver隔离CFS数量/配置、metadata和汇总需求 |
| main.py | CLI可选三个新名称，旧默认不变 |
| coordination/V1_F_RUNTIME_CONTRACT.md | 本轮源式修正、API、窗口与有限预算 |
| coordination/reviews/v1_f_runtime_review.md | Referee独立审查 |
| 本报告与v1_f_cfs_smoke_results.json | 实施范围与可重放数值证据 |
| README及MODEL_CONTRACT/CLAIMS/TASKS/PROGRESS/MIGRATION_PLAN | 当前状态、入口、文件结构与剩余依赖 |

表中v1路径均以 `intercity_delivery/` 为根。没有修改旧优化器/算法/数据处理器、原测试、论文、依赖或历史结果；没有提交/推送Git。普通CLI/GUI的注册和转换以纯测试核对，本轮没有另行执行普通CLI持久化写盘套件。

## 5. 复用方式和后续范围

在项目根目录使用：
```powershell
python -B experiments/specs/run_v1f_cfs_smoke.py
python -B experiments/specs/run_v1f_cfs_smoke.py --run
```

第一条不求解；第二条会再次消耗三次小run并输出完整JSON，可加 `--output 新路径.json` 保存。已有结果足以复核本轮完成标准，无需先重新计算。

本轮用户指定范围已经完成。若后续需要旧Algorithm2/3级别优化，依次迁移V1-aware可行初解/warm start，再迁移local branching，并以独立validator和原/新同信息基线验收。真实期末hard目标的RH尾计划是独立任务。正式大规模CFS、单位校准、稳态和随机QVI需各自定义与预算；不在后台自动继续。

