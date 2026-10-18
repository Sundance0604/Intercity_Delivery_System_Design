# V1-M 已确认基准接口与语义边界

2026-10-18。作者已明确回复“采用这组定义”，确认本文件六项定义（V1_M_DECISION_SPEC的D1–D6-A）。决定ID：`V1M-D1-D6-A-2026-10-18`。此前条件原型升级为已确认的offline V1-M基准；旧运行记录保留当时的未确认状态，不改写历史。此次决定不覆盖V1-F/RH、稳态或QVI，也不修改源TeX。

## 1. 实现的明确条件

本基准只实现作者确认的profile：offline已知订单、逐(i,j,city,flow) arc-wave、连续stop量与非负整数车数、分钟/每车小时费用、tour-end交付且同刻交接、0≤s≤e≤T整数窗的合成输入。调用者仍必须显式传入profile。源main_v2:71的同start语言与82/106/110的逐arc索引张力仍留在原稿；作者现选择逐arc作为本基准定义，不声称原稿措辞已同步。输入限制是已确认的首版合同，不冒称源文原先已逐一规定。

## 2. 共享API（文件所有权互斥）

Experiment A拥有`intercity_delivery/v1/configuration.py`、`mandatory_optimizer.py`、`runner.py`、`tests/test_v1_mandatory.py`；Experiment B拥有`validation.py`、`tests/test_v1_validation.py`。PI只管本文、协调/README及必要适配；Referee只写独立审查。各Agent不得回滚他人修改。

`configuration.py`使用标准库，提供：

- `V1MProfile(information, wave, demand, units, service, input_domain)`：六字段均无默认，唯一首版组合分别为`offline`、`arc`、`continuous_stops_zero_allowed`、`minutes_hourly`、`tour_end_same_time`、`integer_windows_within_horizon`。
- `V1MConfig(T, t0_minutes, tau_periods, N_manual, N_auto, a, b, rho, M, M_auto, c_manual_hour, c_auto_hour)`：五个城市参数映射N_manual/N_auto/a/b/rho的key严格为1、2；所有数字finite，整数车队、a/M正性等采用显式条件合同。
- `V1Order(order_id, flow, quantity, earliest_start, latest_completion, penalty)`：order_id非空字符串、方向+/-，列表中ID不得重复，货量/罚金非负有限。
- `validate_instance(config, orders, profile)`：验证输入，不读旧DeliveryConfig、不自动换单位、不裁整订单窗。返回值不作为跨Agent协议，调用后仍用传入对象。

优化器入口：`V1MandatoryOptimizer(config, orders, profile).build_model()`。完整schedule的键名统一为`x_manual`=(i,j,city,flow)、`y_auto`=(i,j,flow)、`u_wave`=(i,j,city,flow,v)、`g_manual`=(i,j,city,flow,order_id)、`g_auto`=(i,j,flow,order_id)、`z_unserved`=(order_id,)。dict tuple键或字符串tuple键在JSON导出后可由literal_eval安全还原。保留所有非零连续量，不使用0.1展示阈值。

独立验证器：`validate_schedule(config, orders, profile, schedule, *, tolerance=1e-6, reported_objective=None)`，标准库，无Gurobi依赖，不调用优化器表达式或容量表；从原f与每车tour时间不等式直接核K可行性，逐单、逐t检查所有源约束、域、时间窗和成本。失败抛ValueError；成功返回JSON-safe证据和费用项。不把profile通过等同作者确认。

runner：`run_mandatory(config, orders, profile, time_limit, output_flag=0)`，Threads=1，扣除建模时间的整次预算。完整输出x/y/u/g/ĝ/z、索引、输入/source SHA256、profile/语义状态、费用、退出状态、incumbent/bound/gap与验证结果；无incumbent成本为空。只有独立验证通过才可标schedule_valid，不把限时可行叫最优。

## 3. 有界测试授权

PI批准首批最多10个微型run，每个整次≤2秒、Threads=1，总solver上限20秒；测试仅T≤8、≤3订单、每城0–2HV、总AV≤2，合成输入，不调用旧全套solver测试。生产Agent编写测试但初次由PI统一运行，以计数预算。纯validator/配置负例可自行运行。不运行真实数据、大型实验、TeX编译，不提交或推送。

优先实现源M01–M15：one-hot、联合wave容量、共享AV容量、两hub库存、两个累计货流前置、两端需求相等、连续z、费用一次分钟到小时换算、完整导出。F1–F5手算oracle加zero fleet/demand、τ>T、源母弧零容量、同刻交接、stem和wave差异、非法/篡改负例。测试通过仅为这个显式profile的有限证据。
