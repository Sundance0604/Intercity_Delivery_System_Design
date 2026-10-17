# MIGRATION_PLAN — 条件化代码迁移计划，等待批准

2026-10-17。负责人：Principal Investigator。

**只完成审计与设计，未批准或启动代码迁移。** Referee接受静态审计，源模型/算法同步为MAJOR REVISION。下列实现路径、参数和阶段均为建议，不表示已有模块或已解决数学歧义。先由作者确认G0，再由用户批准具体实施范围及测试预算。

依据：[模型合同](MODEL_CONTRACT.md)、[差异索引](MODEL_DIFF.md)、[代码报告](../experiments/reports/model_code_gap.md)、[独立审查](reviews/model_alignment_review.md)。

## 1. 范围与保护措施

- 保留V0-M/F/RH原类、registry名、默认参数、测试期望与历史运行目录。V0差异写明，不顺手修复或重构。
- 建议另建V1配置、纯容量表、两种optimizer和新solver标识。复用已核实结构；baseline重构另获授权。新参数不得自动进入V0敏感性方案。
- V1仅实现作者确认后的main_v2正文。S-current/S-linear验证桥梁和随机V2独立排队，不默认加入sorting、δ、ω或QVI。
- 使用隔离结果目录；记录model_id/revision、algorithm_id、输入散列、数据单位、信息集、seed、依赖/solver配置、completed_horizon、cost_scope、bound_scope、退出状态和原始日志。
- 先用合成stops。真实CFS不能重标为stops；stop/volume双维模型、映射及历史结果重算另行审批，不覆盖旧数据。

## 2. G0：作者需确认的定义

所有回答当前为 **OPEN**；选项/建议不视为默认决定。

| 编号 | 需要确认的问题 | 影响 |
|---|---|---|
| Q1 | 是否先迁移main_v2正文V1-M/F，S-current独立作解析目标、S-linear历史参照？sorting/ω/δ是否另阶段处理？ | 冻结实施模型，避免三稿混成新问题 |
| Q2 | wave按同城市/方向/开始期定义，还是按完整(i,j,k,d)？同起点不同终点能否各用独立全城分区？ | 变量和容量耦合；R1 |
| Q3 | direct迁入后库存超过初始N^k时，local单波是否仍限N^k？迁入车能否做local？ | u枚举域；不可自行改成总车队 |
| Q4 | τ期空返是否跳过stem/tours、仅干线计费？人工复位是否仅在全日T，rolling怎样保存未来返程义务？ | 空返/成本/terminal账本；不自动要求AV复位 |
| Q5 | d/M/Mhat是否按stops；连续流是否为订单类近似？CFS如何取得stop映射？s_l是否同时是揭示时刻？ | 数据和信息集；合成stops仅为先行验证建议 |
| Q6 | period index、分钟物理时间、每小时费率是否为统一规范？ρ/a/b单位是什么？正文整数域是否含0？ | 弧、表、目标量纲与零车状态 |
| Q7 | start-window限所有腿还是仅首次pickup/direct？control/start/ext半开端点是什么？completion保证存在路径还是任意option？ | 母弧域、off-by-one、延后任务与证明；R8 |
| Q8 | 精确V1是否将旧F的θ/fee设0、ratio设[0,1]，direct用共同M/τ？旧扩展是否仅另立标签保留？ | 不让额外约束/成本改变V1 |
| Q9 | local tour返回与最后delivery时点如何对应？硬deadline指近似模型确定界、真实逐单保证还是分位数？ | V1与S期限标准；平均stem/拟合δ不能自动认证逐单保证 |
| Q10 | S-current的n≤W是全域约束还是仅研究不触界的大波域？deadline等号、ω=0、不可行/重根如何限定？ | 稳态命题重审；可与V1核心实施分离 |
| Q11 | 正式Bib、校准/符号检查脚本、参数seed和原始记录能否提供？ | 引文与数值复现；缺失期间保持未核实 |
| Q12 | V2先完善确定性时变模型还是另建Poisson/marked到达？terminal/admissibility是什么？ | 独立理论任务，非本轮迁移内容 |

G0完成标准：对批准范围内影响可行域/目标的Q1–Q9有书面决定并更新公式、数据、窗口追溯矩阵；Q10–Q12可以明确延期。源TeX修订仍需另行授权。Q2/Q3/Q4/Q5/Q6/Q7未解决时，不标记“精确实现V1”。

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

## 6. 下一步批准建议

建议先确认G0，再批准**G1–G2独立V1-M实现及相应G4结果验证**；直送待Q2/Q3/Q4确认后推进G3。这只是建议，未执行。每阶段交付可审查diff、公式追溯矩阵、实际测试结果及局限，再推进有依赖阶段。

本轮止于四份专员/审查报告与六份协调文档。实际迁移等待用户确认重要歧义并批准具体范围；没有自动提交、推送、外部发布或昂贵求解授权。
