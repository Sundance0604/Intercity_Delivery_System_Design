# V1 公式—实现—核验矩阵

2026-10-18。范围：实得21页 `Intercity_Operation_BHH (3).pdf` 的有限期与RH式(1)–(26)，以及Appendix C时间支持过滤和独立replay；对应当前 `research/paper/final_manuscript/`。以TeX label和Python函数作为稳定定位，不猜测PDF缺失版本或源码行号。

**状态：FROZEN_AUTHOR_CONFIRMED_SYNTHETIC_PROFILE；静态核对完成。** 作者已确认单位 `synthetic_job_equivalent`（`V1P0-JOB-UNIT-2026-10-18`）、原式及城市抽象可用节点的direct解释（`V1P0-DIRECT-ENDPOINT-2026-10-18`），以及T2主值m=0.1/敏感性0.05/0.1/0.2（`V1P0-T2-MINLOAD-2026-10-18`）。本矩阵列出真实存在的测试定位；运行结果以实施/审查和后续实际求解报告为准，不预报通过。

路径缩写：SC=`research/paper/final_manuscript/sections/service_capacities.tex`；OM=`.../sections/optimization_model.tex`；SM=`.../sections/solution_method.tex`；IM=`.../sections/implementation_details.tex`；V1=`intercity_delivery/v1/`；Tests=`tests/`。省略号只在此缩写说明表示同一 `research/paper/final_manuscript` 目录。全部Python符号以文件内实际定义为准。

## 1. PDF式(1)–(26)

| PDF式 | TeX文件与label | 实现定位 | 独立核验/检查定位 | 已有测试定位或明确边界 |
|---|---|---|---|---|
| 1 | SC `mod:local-function`，f与F | V1 `capacity.py::bhh_time`, `bhh_inverse` | V1 `flexible_validation.py::validate_flexible_schedule` 内原f计算 | Tests `test_v1_kernel.py::test_hand_calculated_f_h`, `test_inverse_substitution_original_formula`, `test_inverse_linear` |
| 2 | SC `eq:wave-geometry`，均衡分区几何 | `capacity.py::local_capacity` 采用f(q)/v；不模拟客户坐标 | 原f检查不检验几何假设是否真实 | Tests `test_v1_kernel.py::test_units_covariance`；没有真实空间拟合验收，b仍为输入 |
| 3 | SC `mod:local-arcs` | V1 `network.py::local_arcs`；F优化器 `__init__` | F validator重新算local arc bound | Tests `test_v1_kernel.py::test_local_hand_arcs_retain_zero_capacity`, `test_ceil_float_neighbors` |
| 4 | SC `mod:movement-arcs`，AV/empty | V1 `network.py::av_arcs`, `flexible_capacity.py::empty_arcs` | F validator检查AV duration/empty duration | Tests `test_v1_kernel.py::test_av_hand_arcs_and_horizon_boundary`；`test_v1_flexible_validation.py::test_empty_physical_duration` |
| 5 | SC `mod:direct-arcs` | V1 `flexible_capacity.py::direct_arcs` | F validator重算direct_limit | Tests `test_v1_flexible.py::test_mother_arcs_do_not_need_orders`；母域不依赖订单 |
| 6 | SC `mod:local-capacity` | V1 `capacity.py::local_capacity`；F optimizer `__init__`预计算 | F validator原f budget及vM检查 | Tests `test_v1_kernel.py::test_cap_boundary_hand_oracle`, `test_linear_capacity_subtracts_stem`, `test_capacity_never_exceeds_cap` |
| 7 | SC `eq:analysis-superadditive` | 模型性质，不增加MILP约束 | 证明见 `sections/proofs.tex`；replay不证明定理 | Tests `test_v1_mandatory.py::test_06_joint_wave_strictly_changes_capacity` 是有限算例，不能代替证明或推出严格普遍性 |
| 8 | SC `mod:direct-duration` | V1 `flexible_capacity.py::direct_capacity` | F validator直接计算 `2rho_o+f_o(q)/v+tau*Delta t+f_r(q/v)` | Tests `test_v1_flexible_validation.py::test_direct_original_formula_and_hour_cost`, `test_directional_origin_stem`；作者确认车辆在j进入目的城市抽象可用节点，不增物理hub连接项 |
| 9 | SC `mod:direct-coefficients` | 同 `direct_capacity`，alpha/beta | 不调用direct容量表，回核式8 | Tests `test_v1_flexible.py::test_direction_and_origin_stem` |
| 10 | SC `mod:direct-capacity` | 同 `direct_capacity`，稳定逆+vM截断 | F validator核原式和tau期零容量 | Tests `test_v1_flexible_validation.py::test_zero_capacity_tau_arc`, `test_direct_physical_capacity` |
| 11 | OM `mod:objective` | V1 `flexible_optimizer.py::V1FlexibleOptimizer.build_model`；`cost_*`项 | F validator `task_cost` 与penalty/transfer重算；`reported_objective`比对 | Tests `test_v1_flexible_validation.py::test_direct_original_formula_and_hour_cost`, `test_complete_transfer_cost`, `test_partial_inflight_whole_task_cost` |
| 12 | OM `mod:local-links` | 同 `build_model`，`local_link_*`、one-hot、local_cargo | F validator校selector、wave load、原f；N取总人工车队 | Tests `test_v1_flexible_validation.py::test_local_joint_load_not_per_order`, `test_migrated_vehicle_selector_uses_total_fleet` |
| 13 | OM `mod:direct-links` | 同 `build_model`，w/u_direct及direct_cargo | F validator同一link循环的direct分支 | Tests `test_v1_flexible.py::test_01_continuous_direct_split`；原式只有上界，T2最低载量为额外政策 |
| 14 | OM `mod:auto-capacity` | 同 `build_model`，auto_cargo聚合约束 | F validator `shared AV capacity` | Tests `test_v1_flexible_validation.py::test_shared_av_capacity` |
| 15 | OM `mod:human-inventory` | 同 `build_model`，`manual_stock_{city}_{t}` | F validator每城每时刻 `N+inc-out-busy` | Tests `test_v1_flexible_validation.py::test_manual_city_inventory`, `test_migrated_vehicle_selector_uses_total_fleet` |
| 16 | OM `mod:auto-inventory` | 同 `build_model`，`auto_stock_{city}_{t}` | F validator每城AV累计到达/出发 | Tests `test_v1_flexible_validation.py::test_av_city_inventory` |
| 17 | OM `mod:auto-busy` | 同 `build_model`，`auto_busy_{t}` | F validator两城库存隐含全车队在途上界 | Tests `test_v1_validation.py::test_av_occupancy`, `test_av_wrong_hub_despite_sufficient_total_fleet`；前者是M核验器测试，F依靠两城库存逻辑 |
| 18 | OM `mod:local-windows` | 同 `build_model`，local cargo非法窗UB=0 | F validator pickup release、delivery deadline | Tests `test_v1_flexible_validation.py::test_time_window_release_and_deadline` |
| 19 | OM `mod:direct-windows` | 同 `build_model`，direct非法窗UB=0 | F validator同一窗口检查覆盖direct | 同上；连续量不等于逐件不可拆分 |
| 20 | OM `mod:pickup-before-auto` | 同 `build_model`，order_auto与order_local累计约束 | F validator `pickup before AV` | Tests `test_v1_flexible_validation.py::test_cargo_causality`, `test_handling_delay_on_both_hubs`；正文默认delay=0 |
| 21 | OM `mod:auto-before-delivery` | 同 `build_model`，destination累计约束 | F validator `AV before delivery` | 同上；同刻允许交接 |
| 22 | OM `mod:demand` | 同 `build_model`，两端local+direct+z=D | F validator completed stage balance/demand balance；partial延期 | Tests `test_v1_flexible_validation.py::test_unserved_balance`, `test_partial_uninitiated_or_unserved_forbidden` |
| 23 | OM `mod:optional-terminal` | 同 `build_model`，非空 `terminal_target` 下 `true_terminal_*` | F validator完整T目标检查；RH入口拒非空目标 | Tests `test_v1_flexible_validation.py::test_explicit_terminal_target_and_no_default_reset`；`test_v1_flexible_rolling.py::test_terminal_target_rejected_before_build`；默认free terminal |
| 24 | SM `alg:window` | V1 `flexible_rolling.py::solve_flexible_rolling`, `RollingOptions.validate` | runner/replay核completed_horizon；rolling保存各窗范围 | Tests `test_v1_flexible_rolling.py::test_options_validate`, `test_total_budget_deducts_build_and_prior_windows` |
| 25 | SM `alg:origin-cut` | V1 `flexible_optimizer.py::V1FlexibleOptimizer.configure_window` | 历史先固定、origin/direct/empty用start_end | Tests `test_v1_flexible.py::test_history_implicit_zeros_and_new_bounds_without_solver`, `test_04_committed_empty_is_retained_with_no_orders` |
| 26 | SM `alg:downstream-cut` | 同 `configure_window`，AV/destination用completion_end | 历史整段固定，不在新窗端截断 | Tests `test_v1_flexible_rolling.py::test_unpruned_three_leg_cross_window`, `test_commit_keeps_all_history_types_and_tiny_cargo` |

## 2. Appendix C时间支持、replay及信息口径

| 来源 | 实现/核验定位 | 测试定位与范围 |
|---|---|---|
| IM `alg:time-bounds`，pickup最早完成/配送最迟开始 | V1 `pruning.py::prune_candidates` 的 `earliest_pickup`、`latest_delivery_start`；已承诺pickup播种后继 | Tests `test_v1_pruning.py::test_committed_pickup_seeds_av_without_new_pickup`, `test_real_removals_release_deadline_and_positive_capacity` |
| IM `alg:auto-support`，AV时间链 | 同函数 `av_supported`；只删cargo，不删vehicle/selector | Tests `test_v1_pruning.py::test_three_leg_chain_respects_both_transfer_delays`, `test_impossible_three_leg_deadline_prunes_transshipment` |
| IM `alg:delivery-support`，含历史AV | 同函数 `earliest_av_arrival`、历史cargo逐键保留 | Tests `test_v1_pruning.py::test_committed_av_seeds_delivery_without_new_pickup_or_av`, `test_history_retained_after_deadline_and_tiny_quantity` |
| IM direct支持及未来迁入 | 同函数 `positive_direct`，使用全人工车数而非历史idle | Tests `test_v1_pruning.py::test_city_with_no_initial_hv_can_receive_future_vehicle`, `test_no_manual_or_auto_fleet_does_not_create_cargo_support` |
| IM `alg:local-replay` 与direct原式 | V1 `flexible_validation.py::validate_flexible_schedule`；原f不等式与vM分别核对 | Tests `test_v1_flexible_validation.py::test_local_joint_load_not_per_order`, `test_direct_physical_capacity`, `test_overflow_in_original_f_is_not_feasibility` |
| IM independent schedule replay | 同函数完整10组域、整数/binary、归一化键、库存、货流、成本 | Tests `test_v1_flexible_validation.py::test_numeric_domains_nan_inf_bool_noninteger`, `test_duplicate_normalized_key`, `test_tolerance_validated_and_absolute` |
| SM release信息/历史保留 | V1 `flexible_rolling.py::revealed_orders`, `commit_schedule`, `solve_flexible_rolling` | Tests `test_v1_flexible_rolling.py::test_revealed_orders_retains_expired_and_hides_future`, `test_commit_rejects_nonfinite_negative_and_fractional_vehicle` |
| IM全域/窗口证书与partial | V1 `flexible_runner.py::run_flexible`, `run_flexible_rolling`, `_metadata`；partial total与RH顶层bound/gap不作全日最优证书 | Tests `test_v1_flexible_validation.py::test_full_no_incumbent_never_fabricates_cost`, `test_rh_partial_inflight_no_global_bound`, `test_rh_complete_reconstructs_z_without_global_optimality`, `test_partial_has_no_reported_full_objective` |
| cargo-only剪枝对照边界 | `V1FlexibleOptimizer._allowed`只认三货流组，母弧由 `__init__`独立生成；`prune_candidates`记录实际数量 | Tests `test_v1_flexible_rolling.py::test_pruned_three_leg_has_same_hand_cost_and_real_removal` 为有限算例，不是一般无损性或加速证明 |
| CFS单位与输入构造 | V1 `adapter.py::load_cfs_stop_orders`, `adapt_order_batches`；loader赋每条source shipment为1；adapter本身保持caller量；新 `experiment_inputs.py::load_cfs_job_orders` 绑定已确认合同与metadata | Tests `test_v1f_integration.py::test_fixed_seed_balanced_stop_sample_preserves_source`, `test_missing_source_records_rejected`, `test_conversion_exposes_stem_and_rejects_inconsistent_physics`；作者确认单位synthetic_job_equivalent，旧函数/测试名中的stop不提供真实stop校准；新单位检查见 `test_v1_p0_source_and_units.py` |

## 3. P0新增机制的源码定位与作者政策

四机制不是PDF原有的四套独立公式。T0限制已在OM正文出现；T1/T2是为拆分运营效应设置的比较政策。当前实施位置为 `flexible_configuration.py::V1FConfig` 的 `service_mechanism/min_direct_load_per_vehicle`、`validate_f_instance`，`flexible_optimizer.py::V1FlexibleOptimizer._apply_mechanism_bounds`、`build_model` 的 `direct_minimum_load_*`，以及 `flexible_validation.py::validate_flexible_schedule` 的机制禁用与最低载量检查。

- T0：w/u_direct/h和empty均为0；T1：w/u_direct/h为0、允许empty；T2：empty为0、最低平均direct载量m=0.1；T3：原式、最低载量默认0。
- T2主值0.1和敏感性0.05/0.1/0.2是作者已确认的新增政策，单位synthetic_job_equivalent/车，不能写成式(13)本来就保证正载货。`sum h>=m*w`不证明逐车均非空；独立empty效应用T2与同m的T3匹配比较。
- 机制测试的函数名称和本轮运行结果以实施Agent交付和独立审查报告为准。本文记录静态定位，不预报通过数。
- 源码可接受transfer delay/fee/direct-share/terminal等扩展；PDF核心对应配置必须明确使用0/0/[0,1]/free terminal。开启扩展后仍可核验，不代表与核心可行域完全相同。

## 4. 映射结论的边界

该矩阵支持“有限期模型、RH与既有代码有可检查的逐式对应”。它不证明PDF由当前源编译、全部软件正确、一般剪枝无损、现实车辆物理端点已经校准、CFS提供客户坐标或需求stop数，也不把有限期MILP/RH称为稳态或随机QVI的数值验证。
