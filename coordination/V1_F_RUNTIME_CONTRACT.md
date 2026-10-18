# V1-F、Rolling Horizon与剪枝的实施接口

2026-10-18。用户要求一次交付V1-F、RH和剪枝，并只验证CFS可运行。继承已确认D1–D6-A的arc-wave/连续量/分钟与小时费率/同刻交接；RH按release揭示订单，已知过期订单与承诺保留。沿用独立empty、计算窗不复位、默认真实终点目标为空的已用政策。源TeX不修改。

## 模型与必要修正

Local用源K(v)，direct用main_v2:141的方向相关式`2rho_o+f_o(q)/v+tau*t0+f_d(q/v)`。其等价系数为`A=(a_o+a_d)/v`、`B=b_o/v+b_d/sqrt(v)`，可调用稳定逆函数，不必数值二分。direct母弧为`tau<=j-i<=tau+ceil((2maxrho+f1(M)+f2(M))/t0)`；tau期载货容量强制0，empty另设独立弧/变量/费用。

V1-F local selector上界采用两城总人工车数，不沿用原文N_city截断，因迁入车辆也能服务。动态库存计direct+empty累计迁移与local半开占用，真正限制城市可用量。此处是迁入一致性修正，结果披露`selector_bound=global_manual_fleet`。没有每窗回归。full可显式真实T目标；rolling非空目标暂拒绝，避免承诺未经验证的尾部可达性。

## 共享API

模型Agent拥有`flexible_configuration.py`、`flexible_capacity.py`、`flexible_optimizer.py`、`tests/test_v1_flexible.py`。控制Agent拥有`flexible_rolling.py`、`pruning.py`、`tests/test_v1_flexible_rolling.py`及pruning测试。验证Agent拥有`flexible_validation.py`、`flexible_runner.py`、`tests/test_v1_flexible_validation.py`。CFS/入口Agent拥有`adapter.py`、`tests/test_v1f_integration.py`、`experiments/specs/run_v1f_cfs_smoke.py`及main/configuration/solvers/core的必要新增适配。PI管协调/预算/最终执行，Referee在交付后审查。所有人不是独自在代码库，不覆盖他人修改。

- `V1FConfig(V1MConfig)`继承所有基础字段，增加必填`c_direct_hour`，可选`empty_tau_periods=None,c_empty_hour=None,terminal_target=None,transfer_time_periods=0,transfer_cost_per_unit=0,direct_ratio_min=0,direct_ratio_max=1`。empty回退tau和direct费率，metadata披露。transfer和ratio是显式兼容扩展，不冒充源正文。
- 使用既有`V1Order`。`flexible_profile(information="offline")`返回既有`V1MProfile`，F允许information=`offline`或`release_revealed`；其余五字段同M。`validate_f_instance(config,orders,profile)`通过临时offline副本调用基础验证，再检查F字段，不改实际profile。
- `direct_capacity(config,i,j,flow,vehicles)`、`direct_arcs(config)`、`empty_arcs(config)`为纯函数。
- `V1FlexibleOptimizer(config,orders,profile,candidates=None).build_model()`；属性`model,groups`，方法`extract_schedule()`、`configure_window(current_time,start_end,completion_end,committed)`。
- 十组schedule：既有`x_manual,y_auto,u_wave,g_manual,g_auto,z_unserved`，新增`w_direct=(i,j,flow)`、`u_direct=(i,j,flow,v)`、`empty_vehicles=(i,j,flow)`、`h_direct=(i,j,flow,order_id)`。z键仍(order_id,)；其余都有绝对(i,j)头。完整结果可含零值；commit只取当前控制区间开始的非零事件，z不提交。历史各组包括selector精确固定，隐式零同样固定，不能丢已完成迁移。
- 窗口新origin-local/direct/empty开始<start_end；AV/destination-local可在extension启动；所有新任务结束<=completion_end。过去已启动任务先固定不截断。配置终点目标在rolling中显式拒绝。订单参数不随窗口重置。
- `candidates`若非None，为dict group_name -> set(tuple keys)，只用于`g_manual/g_auto/h_direct`货流组；未列组视为不剪；模型按允许货流创建变量。保留所有车辆母弧及selector，避免删空调拨；按零可用货流进一步缩local x/u是可选优化，不在本轮必需。
- `prune_candidates(config,orders,current_time,start_end,completion_end,committed)`返回`{"candidates":dict,"statistics":dict}`。保守按释放/期限、正容量与三腿最短时间下界筛货流；历史所有非零键必须纳入；保留已提交pickup/AV的后续路径（不能只从新pickup出发找路径）。不得用history-only城市idle删未来可迁入任务，不宣称有限比较证明全域剪枝安全。
- `RollingOptions(prediction_horizon=8,rolling_step=2,extension_horizon=6,prune=False)`；`solve_flexible_rolling(config,orders,options,time_limit,profile=None,output_flag=0)`返回raw：`schedule,windows,current_time,completed,status,elapsed_sec,message,profile`。controller负责总预算、单线程Seed0、release可见性、绝对历史提交、pruning统计；不自行宣称全局bound或以末窗目标当总成本。
- `validate_flexible_schedule(config,orders,profile,schedule,*,completed=True,completed_horizon=None,tolerance=1e-6,reported_objective=None)`标准库，独立原f容量/库存/订单/成本检查，失败ValueError。partial允许hub/在途，延期最终平衡/ratio/未来可达；费用按已发起整段任务一次计费，transfer按实际交付；completed才有全日total。
- `flexible_runner.run_flexible(config,orders,profile,time_limit,output_flag=0)`与`run_flexible_rolling(config,orders,options,time_limit,profile=None,output_flag=0)`返回统一dict：`model_id,status,completed,total_cost,best_bound,mip_gap,elapsed_sec,schedule,validation,windows,metadata`，加served/unserved/direct/empty指标。`schedule`为JSON字符串tuple键；full gap范围是这个profile全域，RH顶层bound/gap=None；partial total=None且状态不得OPTIMAL冒充完成。full和RH都做独立验证，metadata记录来源/单位/策略/参数和版本。

## CFS及预算

小样本优先复用SQLite只读OD加权抽样与旧合成时间窗，原SHIPMT_ID和重量/权重保留。一条CFS记录作为一单位shipment-stop，是本次运行验证的显式代理，不声称校准真实客户stop。旧V0重量数量保持原语义。若在共用core转新solver订单，必须同时更新该solver数据、总需求和单位metadata，不能只在solver内部偷换数量。

PI批准最多3个CFS run（full/RH/pruned RH），同一seed和样本8–12条、T<=24，每run整次<=15秒/单线程；另最多10个微型run，每run<=2秒/单线程，总solver上限65秒。Agent不自行solver，由PI统筹；纯测试可跑。不运行大实验、重建数据库、改历史/TeX或提交Git。CFS成功标准：实际CFS来源可追溯、得到可行incumbent或RH完成、独立核验通过且有正服务；若预算中断则如实partial，不能报全流程通过。
