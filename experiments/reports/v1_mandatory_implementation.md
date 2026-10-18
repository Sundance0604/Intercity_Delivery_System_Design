# 已确认V1-M完整基准实施记录

2026-10-18。PI综合，两位Experiment分别拥有模型/配置/runner和独立验证器；Theorist只读核对源式和手算oracle；Referee在交付后审查。分支`research/qvi-multiagent`，HEAD=`f26b4c96907082c2ebb7b0104263831e7100aa26`，源文件及实现含未提交内容，不能只用HEAD复现。

## 1. 交付范围与未决定义

已经实现作者确认profile下的完整有限时域强制换装MILP，覆盖M01–M15。当前model_id为`v1_mandatory_arc_wave_v1`，不是原V0或V1-F/RH。入口为`intercity_delivery.v1.runner.run_mandatory(config, orders, profile, time_limit)`，没有改旧registry或CLI默认。最初未确认原型的旧model_id与记录保留为历史。

六个profile字段仍必须显式传入，支持offline、arc-wave、连续stop量/零车合法、分钟+每车小时费率、tour-end交付/同刻交接、全域内整数订单窗。作者现已明确回复“采用这组定义”，冻结D1–D6-A，编号`V1M-D1-D6-A-2026-10-18`。来源和API见[基准profile](../../coordination/V1_M_IMPLEMENTATION_PROFILE.md)。新结果为`author_confirmed_profile`，原8次未确认运行保留当时状态，不回写历史。源71与逐arc公式的文字张力仍在原稿，作者选择不等于TeX已经修改。

## 2. 模块与公式追溯

| 模块 | 实现与源依据 |
|---|---|
| `v1/configuration.py` | 六字段无默认profile、独立配置/订单、finite/type/城市键/整数车队及订单窗验证；输入限制属于明确条件合同 |
| 已有`v1/capacity.py,network.py` | M01–M02：源74/82的f/h/K与local/AV母弧，保留零容量弧 |
| `v1/mandatory_optimizer.py` | M03–M05：人工/AV半开占用、两hub库存，含t=T；M06–M07：K(v)与binary one-hot、车数联结；M08：同弧共享AV；M09–M12：释放/截止、逐单两段累计时序、两端需求等式；M13连续g/ĝ/z与非负整数车数；M14分钟/小时只换算一次 |
| `v1/validation.py` | 不导入Gurobi或K表，从原f(load)≤v·max(Δ−2rho,0)及load≤vM独立核容量，重建母弧/库存/时序/域/需求及三项成本；AV总流另核闭合域夹逼结果 |
| `v1/runner.py` | M15完整六组导出，含零值、小数与z单元素tuple；独立验证、input/source SHA256、版本/seed/tolerance、incumbent/全域bound/gap与无解状态 |

源首要依据：`research/paper/main_v2_BHH.tex:67–137`，SHA256=`500BE1ADA854317BE79C358E0FF1269786DD4C7EA8E1102B501354EFF04A3457`。M15是实现验收要求，不是额外源约束。源有限全域、无起始货物存量时，T时两段时序给D≤A≤P，两端需求式给P=D=d−z，故AV总货流A相等；不把这个夹逼移用于带跨界存量的RH。

## 3. 实际执行

运行环境为`C:/Users/32599/.conda/envs/pavane/python.exe`，Gurobi10.0.3。资源上限见[微型方案](../specs/v1_mandatory_micro_validation.md)：最多10run，每run整次2秒、Threads1；初轮8run，实际T3、至多2单、每城至多2HV/总AV1；Seed0、FeasibilityTol/IntFeasTol=1e-8。独立validator采用绝对默认1e-6，整数域容差不超过1e-6。

PI先执行validator36+既有kernel22：58项PASS，suite0.058秒。随后运行：

```powershell
& 'C:/Users/32599/.conda/envs/pavane/python.exe' -B experiments/specs/run_v1_mandatory_micro_validation.py
```

unittest结果：**74 tests，0.500秒，OK**，包含配置8、微型MILP8、validator36、既有kernel22，无skip或测试失败。8run均取得OPTIMAL、完整六组计划、独立核验通过，输入/source hash及严格JSON序列化断言也通过。

| 微型case | 已通过的手算目标断言 | 关键证据 |
|---|---:|---|
| F2 两单共享AV | 0.5 | 总未服务.5，非指定各单分配 |
| F3 连续未服务 | 0.5 | z=.5，不错误取整 |
| F4 单位与同刻交接 | 12 | 人工10+AV2，三腿分别0→1/1→2/2→3 |
| F4 deadline=2 | 100 | 无法完成三腿，全拒单 |
| F5 零车/小量/τ>T | 0.1 | z=.05，另含零需求订单，空selector/AV域不报错 |
| 联合wave | 5 | 两城各u2，服务全1；h(2)=1 |
| F1 stem零容量 | 100 | 保留源母弧但不可载正货 |
| 反向订单需无货AV方向 | 4 | 无正向订单仍保留0→1正向AV空驶，随后1→2反向运货 |

联合wave案例与“只把容量替成x·h(1)”的手算值`5+100(sqrt(5)−2)`形成差别；没有运行旧优化器或宣称整V0模型公平比较/性能优势。各微例的最优含义限Gurobi求解容差及当前profile。

初轮新增52个不同测试（8配置+8真实求解+36独立validator），74总数中22是已有工具回归。收尾再新增4个runner纯mock，最终不同用例78个、本轮新增56个；不是一次性重跑78个。最后运行配置/validator/kernel/runner共70纯例PASS（0.043秒、进程0.211秒），确认后两次真实定向复验见下节。实际solver CPU累计未单独精确计量；0.500秒是初轮suite墙钟，不把20秒预算上限写成实际耗时。

## 4. 日志保存异常与证据范围

首版执行器在全部74项通过后尝试以Python创建JSON文件，workspace写入被拒，故外围进程退出1：`PermissionError`。这是证据文件创建失败，不能将进程称为exit0，也不是求解或断言失败。那一次内存中的8个完整raw payload未保存，不重建冒充原始计划。

PI用允许的文件编辑接口保存[断言证据JSON](v1_mandatory_micro_assertions.json)：明确记录74 OK、各case通过的目标断言、外围exit1和raw未捕获，并含实现指纹。执行器已修为默认stdout；显式`--output`在求解前独占创建/检查文件，避免权限错误浪费预算。修复后`--help` smoke exit0；另用已有README作为输出负例，验证求解前退出2且文件SHA不变，外层检查exit0、solver0次。没有为日志故障重复整套求解。

复现程序现将完整payload JSON写stdout（或指定新的可写输出路径），不可覆盖既有证据。新增输出机制没有修改MIP/validator数学实现。

作者确认及最终metadata更新后，PI使用预算内剩余2个run保存了[完整raw payload](v1_mandatory_confirmed_micro_results.json)：cost_and_handoffs=12、joint_wave=5，均OPTIMAL、独立schedule_valid、author_confirmed_profile与decision_id一致。命令进程exit0、约0.395秒，单run含建模/验证/导出分别约0.156和0.047秒；每run仍限2秒/单线程/Seed0。共8个不同真实用例累计执行10run，没有超过20秒理论预算上限。没有重新求解初轮全部8例，也不将两份raw称为初轮丢失的8份计划。

`tests/test_v1_runner_status.py`用4个纯mock核对：建模耗尽预算不optimize；限时无incumbent保留bound但无cost/valid；依赖不可用不伪造计划；独立验证失败抛错且dispose。Referee亲跑这4项PASS（0.005秒），并从保存输入与schedule纯重放两份raw结果，分别重算12和5；未追加solver。

## 5. 限制与审查

支持明确声明的浮点/数值范围，溢出显式失败，基础h下溢遵循float；validator的valid为数值容差内给定完整计划核验，不是BHH路径精确性或全域理论证明。没有RH/随机到达、direct或人工empty、稳态、QVI、真实数据规模、统计比较。V1-M的AV无终端复位，人工仅城内闭环，因此没有把计算窗车辆回归或manual empty偷偷加入本基准。

独立审查[Referee报告](../../coordination/reviews/v1_mandatory_review.md)最终限定ACCEPT。作者确认直接来自人类回复，独立审查核验实现而不替代该决定。不得扩大为源论文全语义已统一、V1-F/RH已迁移或QVI成果。下一步是独立实验入口/schema接入，再按F/RH各自规格迁移；不擅自替换现有默认入口。

## 6. 文件范围与收尾核对

本阶段新增生产文件为`v1/configuration.py`、`mandatory_optimizer.py`、`validation.py`、`runner.py`；测试为`test_v1_mandatory.py`、`test_v1_validation.py`、`test_v1_runner_status.py`。另新增基准profile、微型方案/执行脚本、本报告、两份JSON证据及独立review；更新README与共享MODEL_CONTRACT/CLAIMS/TASKS/PROGRESS/MIGRATION_PLAN、原V1-M决策规格。保留前阶段未提交文件；旧3处CLI/配置/registry变更来自V0-F-empty阶段，本阶段没有再改它们。

最终检查：21份原models/algorithms/data及四TeX的SHA256与本阶段开始一致；21份相关Markdown本地链接无缺失；12份V1源码/测试/执行器AST有效；两份最终raw中的8类source指纹与当前文件逐项一致；`git diff --check`通过。仅既有LF/CRLF和`.pytest_cache`权限提醒，未清理。未改旧测试、原模型/算法、历史数据或论文，未安装依赖、提交或推送Git。
