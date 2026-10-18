# MIGRATION_PLAN — 模型迁移依赖与当前交付

2026-10-17。负责人：Principal Investigator。

**2026-10-18更新：V1-F、RH、货流剪枝与三个实验入口已实现，CFS小样本执行检查完成。** V1-M已完成。旧阶段保留历史，最新交付与剩余依赖见文末；warm start/local branching、真实终点RH尾计划、大型实验和QVI尚未完成。

依据：[模型合同](MODEL_CONTRACT.md)、[差异索引](MODEL_DIFF.md)、[代码报告](../experiments/reports/model_code_gap.md)、[独立审查](reviews/model_alignment_review.md)。

## 1. 范围与保护措施

- 保留V0-M/F/RH原类、registry名、默认参数、测试期望与历史运行目录。V0差异写明，不顺手修复或重构。
- 建议另建V1配置、纯容量表、两种optimizer和新solver标识。复用已核实结构；baseline重构另获授权。新参数不得自动进入V0敏感性方案。
- V1仅实现作者确认后的main_v2正文。S-current/S-linear验证桥梁和随机V2独立排队，不默认加入sorting、δ、ω或QVI。
- 使用隔离结果目录；记录model_id/revision、algorithm_id、输入散列、数据单位、信息集、seed、依赖/solver配置、completed_horizon、cost_scope、bound_scope、退出状态和原始日志。
- 先用合成stops。真实CFS不能重标为stops；stop/volume双维模型、映射及历史结果重算另行审批，不覆盖旧数据。

## 2. G0：作者需确认的定义

以下保留原始决策清单；未标为作者已确认的项目仍为 **OPEN**，选项/建议不视为默认决定。最新Q4已有两项确认，见该行与车辆调拨分支。

| 编号 | 需要确认的问题 | 影响 |
|---|---|---|
| Q1 | 是否先迁移main_v2正文V1-M/F，S-current独立作解析目标、S-linear历史参照？sorting/ω/δ是否另阶段处理？ | 冻结实施模型，避免三稿混成新问题 |
| Q2 | wave按同城市/方向/开始期定义，还是按完整(i,j,k,d)？同起点不同终点能否各用独立全城分区？ | 变量和容量耦合；R1 |
| Q3 | direct迁入后库存超过初始N^k时，local单波是否仍限N^k？迁入车能否做local？ | u枚举域；不可自行改成总车队 |
| Q4 | 最新作者已确认计算窗不强制复位，空调拨独立且无订单也能决策。真实T_op目标、空行时长/费率、边界状态仍待定义 | 已确认部分见[车辆修正规格](VEHICLE_REPOSITIONING_SPEC.md)，不再重复询问窗口是否回归；不自动要求全日或AV复位 |
| Q5 | d/M/Mhat是否按stops；连续流是否为订单类近似？CFS如何取得stop映射？s_l是否同时是揭示时刻？ | 数据和信息集；合成stops仅为先行验证建议 |
| Q6 | period index、分钟物理时间、每小时费率是否为统一规范？ρ/a/b单位是什么？正文整数域是否含0？ | 弧、表、目标量纲与零车状态 |
| Q7 | start-window限所有腿还是仅首次pickup/direct？control/start/ext半开端点是什么？completion保证存在路径还是任意option？ | 母弧域、off-by-one、延后任务与证明；R8 |
| Q8 | 精确V1是否将旧F的θ/fee设0、ratio设[0,1]，direct用共同M/τ？旧扩展是否仅另立标签保留？ | 不让额外约束/成本改变V1 |
| Q9 | local tour返回与最后delivery时点如何对应？硬deadline指近似模型确定界、真实逐单保证还是分位数？ | V1与S期限标准；平均stem/拟合δ不能自动认证逐单保证 |
| Q10 | S-current的n≤W是全域约束还是仅研究不触界的大波域？deadline等号、ω=0、不可行/重根如何限定？ | 稳态命题重审；可与V1核心实施分离 |
| Q11 | 正式Bib、校准/符号检查脚本、参数seed和原始记录能否提供？ | 引文与数值复现；缺失期间保持未核实 |
| Q12 | V2先完善确定性时变模型还是另建Poisson/marked到达？terminal/admissibility是什么？ | 独立理论任务，非本轮迁移内容 |

G0完成标准：对批准范围内影响可行域/目标的Q1–Q9有书面决定并更新公式、数据、窗口追溯矩阵；Q10–Q12可以明确延期。源TeX修订仍需另行授权。Q2/Q3/Q4/Q5/Q6/Q7未解决时，不标记“精确实现V1”。

状态更新：上表“均OPEN”仅指初次审计时；现Q4的两项作者决定已被记录，不再视作未知。其余项不因该意见自动采用推荐方案。新修正将e从w拆出，因此真实恢复目标若启用也须使用w+e，而不是保留旧Σw+=Σw−原式。

## 3. 按依赖顺序实施

主链：`G0 → G1 → G2 → G3 → G4 → G5 → G6 → G7 → G8 → G9`。仅批准强制换装时，可在G2与G4的对应结果验证后交付，不默认推进direct/RH。

| 阶段/依赖 | 实现方法及拟接触位置 | 验证方式 | 完成标准 |
|---|---|---|---|
| G0 无 | 冻结批准后的模型、单位、信息/终点规格；正文—算法冲突列表 | Referee核对作者决定与公式映射 | 无未记录的可行域选择，批准范围明确 |
| G1 G0 | 独立V1Config及capacity/arc纯函数模块；闭式h、K(city,arc,v)、K̃(direction,arc,v)二分；统一占用集 | U1–U4代回公式/边界/性质；不调solver | cap/stem/方向/单位正确，空返不载货；不用连续插值代替整数选择 |
| G2 G1 | 新V1MandatoryOptimizer：x/y/u/g/ĝ/z、one-hot、共享AV、库存/时序/需求；复用helper不改旧类 | 公式—命名约束/变量域检查，独立schedule验证器，微型M1 | 每项V1-M公式有实现/测试，连续z和共享容量正确 |
| G3 G2 | 新V1FlexibleOptimizer：K̃/ũ、HV迁移、空返、全日人工复位，中性化未授权扩展 | M2非对称城市/无反向货/空返/迁入local/terminal | 正反车次与库存守恒、返程收费；Q2/Q3正确落实 |
| G4 G2/G3相应范围 | 新registry/结果schema；独立重算schedule成本、served、罚损；cost/bound分域 | U6及V0配置/序列化回归 | 旧名/默认不变，partial不能当complete，窗口界不冒作global |
| G5 G4 | V1-RH历史selectors/车数/流；hub/在途初始源项、reveal ledger、到期罚损与全日terminal义务 | M3跨窗部分服务/迁移/reveal/预算中止 | 无重复pickup或记账，已承诺整数/binary不变；完成成本可复算 |
| G6 G5 | full/reduced同母弧域；保护空返；证明或保守化relocation-aware筛选 | M4小域对照及Referee安全性审查 | reduced为full子集；安全剪枝有证明，否则显式标heuristic |
| G7 G6 | 启发式改用K(v)−load与最小可行新增整数车数；填u/ũ；构造返程和warm start | M5每步全约束残差、Start接受日志；失败明确回退 | 初解满足新容量/库存/终点，不沿用单车ceil公式 |
| G8 G7 | local branching定义车数/selector邻域，避免双计；结束移除邻域恢复reduced | M6邻域恢复/预算停止/可行incumbent/bound_scope | 保留可行解，local/window/global证书不混用 |
| G9 G8 | 新隔离运行目录，公平数据/信息/时间/线程/seed/tolerance；先smoke后正式 | 分层回归、代码审查、指标/来源检查 | 通过前置验收且用户批准正式预算后才运行；旧结果不改写 |

G4的完整成本从相应口径全部已执行决策核算；预算中止只报告覆盖范围和partial，不补造全日成本。clairvoyant exact与online RH的信息优势单列，只有数学/数据/信息口径一致的有效界才能作为相应reference。

拟议文件落点（**尚未创建**）：G1使用`intercity_delivery/v1/configuration.py`、`capacity.py`、`network.py`；G2/G3使用同目录`mandatory_optimizer.py`、`flexible_optimizer.py`及`validation.py`；G5–G8使用独立V1滚动/候选/启发式模块。G4只在获批后向现有`experiments/solvers.py`、`experiments/core.py`及CLI/GUI增加新模型入口，保留原名称/分派/默认行为。新增测试拟放`tests/test_v1_capacities.py`、`test_v1_models.py`、`test_v1_rolling.py`；不改旧测试期望。实施时由Experiment唯一拥有新生产/测试文件，Theorist与Referee只提供公式及审查，PI管理协调文件；需要拆分并行编码时先明确互不重叠的文件所有权。

## 4. 必要测试与数值验证设计（未执行）

详细用例见代码报告第9节；以下采用独立公式/枚举/残差判据，非照抄实现。

| 用例 | 独立判据 |
|---|---|
| U1 local表 | 代回f(K)≤v[Δ−2ρ]^+；v=0、零tour budget、M cap、b=0、饱和边界；不假设K凸/全域严格超线性 |
| U2 wave机制 | 未饱和b>0例区分h(vD)和v h(D)；分别控制stem/M，ρ=0不代表一般回归V0 |
| U3 direct表 | 两向非对称、f̃残差/二分包络、满载/无可行货量、τ期空返；不把空弧当最快载货路径 |
| U4 时间/弧 | ceil临界、j=T、到达同刻再出发、半开占用；不同t0下同物理场景；full/reduced同域 |
| U5 数学模型 | binary/integer/continuous域、全u=0、无车/无需求/分数需求；多单AV不重复用容量 |
| U6 结果 | incomplete中止、cost/bound_scope、独立成本、model_id/单位/信息标签、V0默认保持 |
| M1–M2 微型完整模型 | 手工/枚举oracle；1–3笔合成stops、每城1–2HV；all-unserved、不可行窗、回空、非对称、terminal |
| M3 滚动 | pickup已做但后续未做、hub/在途状态、late reveal、末窗deadline、短窗返程义务、中途预算耗尽 |
| M4 剪枝 | inherited当地车均占用但当窗新调入可支撑local；同母域full/reduced比较；有限例一致不替代一般证明 |
| M5–M6 构造/邻域 | 所有容量/库存/时序/terminal残差；selector一致；warm start接受证据；邻域移除恢复 |

回归：先纯函数/静态建模，再微型MIP，再RH/启发式。保留V0旧测试期望，V1用独立fixture；base逐订单AV与V1共享容量造成的最优值差异是预期模型差异，不应修改旧期望掩盖。

预算建议（**尚未授权**）：纯函数先行；micro MIP每例≤5秒、1线程、总≤60秒、仅tiny合成数据。实施阶段提交确切命令/用例数/资源上限后批准；不自动跑旧全套求解测试。license失败记unavailable，不记通过。大型Gurobi、真实数据、长实验另申请预算。

公平比较记录：相同订单/抽样ID/信息揭示、数学约束、fleet、单位、seed、线程/容差/停止准则和总墙钟。指标含独立重算operational cost/penalty、served/unserved stops、准时率、空返车次/成本、direct份额、库存/容量/流守恒/窗残差、completed_horizon、incumbent、bound_scope。按offered或served归一化须标分母，零服务不除零。

## 5. 独立研究分支

| 分支 | 依赖 | 交付与验收 |
|---|---|---|
| S1 稳态数学边界 | Q10，Theorist后Referee | 重审n≤W、active sets、可行性、deadline临界/重根；明确命题域，改源稿另授权 |
| S2 V1↔S桥梁 | Q1/Q9、S1、V1基本验证 | 统一需求、峰值fleet、服务时点、目标归一化、sorting/δ/timetable、整数/首末期；匹配后才设计stationary验证 |
| L1 引文/校准 | Q11 | 正式Bib、具体篇页/公式、脚本/seed/参数/原始输出；不以未搜到同类工作认定创新 |
| QV1 V2形式化 | Q12 | state/filtration/control/intervention/cost/horizon/boundary/no-Zeno合同；选随机过程后建立生成元 |
| QV2 V2验证 | QV1 | 明确existence/verification/threshold命题与独立证明；不用MIP/玩具观察代替理论 |

## 6. 初期批准建议（历史，现况见第10节）

G0-M的[决策规格包](V1_M_DECISION_SPEC.md)已准备并经[Referee接受](reviews/v1_m_decision_spec_review.md)，该接受仅针对文档准备。D1–D6均OPEN：offline强制换装范围、arc-wave解释、单位、连续/零域、tour-end服务、受限合成输入。作者确认与代码实施授权仍PENDING。

限定M路线细分为：`G0-M-PREP（已完成）→ G0-M-CONFIRM（待作者）→ G1-M（待批准）→ G2及对应G4-M（待批准）`。G1-M只含local容量/完整母弧域和配置，不把原G1中的direct K̃模块一并实施。V1-F的迁入/空返、RH reveal/start/ext、稳态和V2可延期，但相关成果不得提前宣称完成。

建议确认D1–D6后，批准**G1-M与G2及对应G4-M的独立实现**，微型求解预算单独明确；直送仍待Q2/Q3/Q4等确认再推进G3。这只是建议，未执行。每阶段交付可审查diff、公式追溯矩阵、真实测试结果及局限，再推进依赖阶段。

初轮模型审计交付后，用户已授权将10份审计文档及重写README提交推送（93b2dcb、f26b4c9）；该历史授权不等于新代码迁移或昂贵求解授权。当前G0-M止于规格、审查和协调文档更新，没有继续生产实现或启动实验。

## 7. 最新作者评审后的车辆调拨分支

当前新增有界交付为[VEHICLE_REPOSITIONING_SPEC](VEHICLE_REPOSITIONING_SPEC.md)与独立车辆账本检查，回应“不每窗复位、独立空调拨”。不取消G0-M成果，也不把此前D1–D6当作已批准。

后续实施顺序调整为：确认真实终点模式/空行参数/有限cut策略 → 独立事件账与empty母弧域 → 新变体e变量、费用和库存 → full/reduced与历史提交 → 完整结果核算 → 启发式、warm start、local branching → 获批的微型MILP验证。只有已确认的两项政策可直接进入规格；其余定义不得用生产默认值代决。

生产接入不能被简化为“删终端等式”：该等式当前根本不存在；所有构造器、vehicle group、history、候选、费用和结果都须接通e。历史已完成跨城迁移不可消失；history-only零库存不构成未来永无车的上界。原稿修订仍未实施。

## 8. 已批准的V0-F独立空车分支及后续依赖

已交付：显式政策/事件账→独立empty弧与变量/费用/累计库存→绝对历史与滚动提交→完整结果核算→可选注册入口→有界微型测试→Referee独立审查。见[实施报告](../experiments/reports/vehicle_repositioning_implementation.md)。默认空行时间/费率沿用direct且披露，真实目标为空；完整模式可设目标，rolling明确拒绝。新任务必须在完成窗内结束，保留全部历史，不实施任意跨cut新任务。

| 顺序 | 下一有限任务 | 方法 | 验证/完成标准 |
|---|---|---|---|
| 1（已完成） | 换装链跨窗回归与独立货流检查 | pickup、AV、destination跨控制窗；独立重放hub/容量/期限 | 31项新增用例PASS、手算费用7.2、NaN修复、审查ACCEPT；见第9节 |
| 2 | 如需rolling真实目标，补全程可达性保障 | 给每次提交建立真实T可行尾计划或等价安全条件 | 短窗不能提前破坏终点可达；不可达输入显式失败；Referee复核 |
| 3 | 需要时接入剪枝/启发式/邻域 | 在empty完整母域验证安全上界和完整历史，单独接入 | 同域微例对照、候选安全证据、成本核算与复现日志 |
| 4 | 另行推进完整V1 | 确认G0-M的D1–D6后实施G1-M/G2-M | wave/stem/容量与公式逐项对齐；不能复用本次34项测试宣称完成 |

本次仅有限合成测试，总solver预算60秒；大型实验、真实数据及网络发布继续单独批准。实施不产生新研究定理。

## 9. 连续两步的阶段记录（历史）

第一步独立cargo核验/真实跨窗链完成并经审查；随后第二步实现V1 f/h/K与母弧纯内核。两步实际预算见[方案](../experiments/specs/two_step_alignment_validation.md)，不引用上一轮60秒建议扩张权限。当前累计原扩展34+货流31+纯工具22=87个不同用例分批通过，见PROGRESS及两份实施报告。

G1的capacity/network基础部分已交付，完整配置、订单域和优化器仍依赖[V1-M D1–D6](V1_M_DECISION_SPEC.md)：尤其D2波次索引冲突、D3 stop/费用单位和D1信息集不能由工具调用隐式选定。下一次完整MIP实施应先记录这些决定，再复用纯工具组装one-hot/共享AV/逐单因果及完整导出验证；不得把当前V0-F cargo checker直接当V1容量验证器。真实终点安全尾计划仍是另一个可选任务，本轮未实施。

## 10. 已确认V1-M交付与后续依赖

作者现确认D1–D6-A（`V1M-D1-D6-A-2026-10-18`），不再等待重复确认这些相同定义。完整configuration/mandatory_optimizer/validation/runner及M01–M15、完整六组计划/来源指纹和微型验收已交付，见[实施报告](../experiments/reports/v1_mandatory_implementation.md)与[独立审查](reviews/v1_mandatory_review.md)。验证先后为初轮74项、最终70纯例和2次真实复验，8个不同真实用例共10run，20秒上限内；原始日志缺口按报告保留。

下一有限任务为G4-M实验入口与适配：给V1-M单独的solver名、显式配置/profile与结果schema，保留V0默认；核对单位/订单来源，未转换的CFS重量不得当stops。验证以dry-run与当前微型输入为主，不自行扩大正式实验。V1-F的K̃/迁入selector/empty与V1-RH的信息、历史、可达性须分别落实新容量，不直接复用V0-F-empty当作V1。无新的大型求解或Git发布授权。

## 11. 最新交付：V1-F / RH / 剪枝

用户本轮要求的实现与CFS小样本可运行检查已完成，见[运行合同](V1_F_RUNTIME_CONTRACT.md)、[实施报告](../experiments/reports/v1_f_runtime_smoke.md)、[原始结果](../experiments/reports/v1_f_cfs_smoke_results.json)及[审查](reviews/v1_f_runtime_review.md)。

| 阶段 | 实现与验证 | 当前验收 |
|---|---|---|
| G1-F/G3 | direction direct逆、global fleet selector、独立empty及迁入库存，手算/原f核验 | DONE，明确源式修正与终点默认 |
| G4-F | 三registry/CLI入口、solver局部数据/τ/汇总、完整/partial状态成本 | DONE，纯适配测试；普通CLI写盘未另实跑 |
| G5 | 绝对历史、release、跨窗换装/迁移、整次预算 | DONE，无计算窗复位；hard真实终点目标暂拒绝 |
| G6必要筛选 | 稀疏货流，留所有车辆母弧和历史后继 | DONE，CFS实际8265→397；一般无损证明未完成 |
| 有界CFS检查 | 同8单、三run、独立原f重放 | DONE，1100/8、2025/7、2025/7；非正式大样本G9 |
| G7→G8 | V1可行初解/warm start→local branching | 未实施，非本轮CFS可运行前置 |
| RH真实终点尾计划 | 每次承诺维护真实T目标可达，不能仅最后窗补约束 | 独立OPEN，无目标默认运行不受阻 |
| 正式G9及S/V2 | 单位校准、同信息比较与独立理论定义 | 未启动，不重复旧大规模实验 |

如后续要求G7：在V1容量/库存上增量构造方案并填one-hot，独立核验后提交Start；不可沿用单车ceil代替K(v)，失败回退现有MIP。完成标准为方案可行、成本可重算和Start接受日志。G8依赖G7，兼容车数/selector邻域、移除限制并保留incumbent，不能把邻域bound当全域bound。真实终点尾计划和正式实验各自需要定义/预算，不作为本轮未完成事项自动推进。
