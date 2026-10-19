# V1 数值实验 P0 准备交付记录

> **当前状态更新（2026-10-18）：P0已完成作者确认、模型冻结与预检。** Direct目的城市聚合节点和T2最低平均载量均已获作者确认；最终清单为63个来源文件。有界P1小规模真实求解与独立复核也已完成，详见[最终结果报告](v1_p1_results.md)。下文保留P0初次交付时的历史记录，其中“待决”“53来源”等描述不代表当前状态。

2026-10-18（Asia/Shanghai）。协调者：Principal Investigator；实施：Experiment；纯容量设计：Theorist；文档：Literature；独立审查：Referee。

**状态：P0技术准备、最终纯测、来源核验及独立审查已完成；作者模型冻结尚有两项待决。** 不将技术实现完成、纯测试通过与作者已确认完整模型混为一项。本轮不启动正式数值实验，不修改论文，不重新编译，不改写历史结果，不提交或推送Git。

## 1. 已完成的准备及待决边界

作者已明确采用“一票一合成任务等效量”，决定编号 `V1P0-JOB-UNIT-2026-10-18`。每条独立抽样CFS记录映射 `D_l=1`，单位为 `synthetic_job_equivalent`，货流仍为连续可分量。该单位不是pounds，不是实测客户stop数，也不宣称实际不可分包裹可物理拆分。源重量保留用于来源说明。

Direct完成后再次可用的聚合节点解释，以及T2的最低平均direct载量政策，仍为 `PENDING_AUTHOR_REPLY`。候选T2政策为 `e=0` 且 `sum_l h_ijl>=m*w_ij`，建议m=0.1、敏感性0.05/0.1/0.2；代码支持不等于作者已批准这些值。该联结排除零货direct wave，不能从聚合平均量推断逐辆车均严格载货。如果作者选择只禁独立empty而保留零货direct，应修改政策标签/实现选择后重新验收，不能沿用严格T2含义。

机制接口、独立replay、数据单位包装、来源漂移检查、逐式映射和合成参数设计均已落地。参数由协调者选作预实验的synthetic设计，未做实证标定，未用新运营结果调参。作者确认流程与合成参数选择分别记录；不因参数可计算就推断运营实验的绑定车队或成本收益。

## 2. 生产接口与输入改动：4个既有文件及1个新文件

| 文件 | 本轮改动目的 | 验收边界 |
|---|---|---|
| `intercity_delivery/v1/flexible_configuration.py` | 新增T0–T3机制及最小direct平均载量字段；检查机制名称、有限非负载量、T2正载量和冲突direct比例 | 原T3默认m=0保留零货direct；未替作者确认T2参数 |
| `intercity_delivery/v1/flexible_optimizer.py` | 按变量关闭direct/独立empty，增加 `sum h>=m*w`，保留同一vehicle/selector母域 | 不以货量比例替代车辆迁移开关；不删除零货AV母弧 |
| `intercity_delivery/v1/flexible_validation.py` | 独立检查完整/partial机制限制与direct最低载量，和原f、库存、货流、成本核验相接 | 验证候选政策可执行，不认证物理direct接驳或未来完成可达性 |
| `intercity_delivery/v1/flexible_runner.py` | 输出机制及最低载量metadata，让结果能追溯所用政策 | 不新增全日RH bound/gap，不把partial标为完整成本 |
| `intercity_delivery/v1/experiment_inputs.py`（新） | `load_cfs_job_orders`以已确认数据合同包装既有loader，保留源重量/原字段并明确observed与synthetic，记录合同SHA和decision_id | 沿用抽样和合成时间窗；不重写CFS，不改既有smoke loader语义，不运行优化 |

既有 `adapter.py`、V0优化器和旧smoke结果本轮保持原状。新增输入包装避免为了改名而覆盖历史证据。

## 3. 新增测试：3个文件

| 文件 | 具体目的 |
|---|---|
| `tests/test_v1_mechanisms.py` | 机制/载量输入域、T0/T1关闭direct、T0/T2关闭empty、T2最低载量、零货direct绕过的拒绝、metadata及模型构造替身检查；不初始化Gurobi环境 |
| `tests/test_v1_p0_capacity_design.py` | 独立二分原f核容量表、正货load/stem/time限制、routing与载量饱和/交界情景、退化负对照、时间链及异城参数；是纯容量测试 |
| `tests/test_v1_p0_source_and_units.py` | SHA/清单漂移检测，拒未确认或重量映射合同，验证job标签保留原重量、pending/different author choice不能绕过preflight、确认配置不修改源design |

本轮新增的是3个测试文件，不把复用的既有测试文件算作新改动。完整回归还会读取已有kernel、mandatory纯配置、F纯配置、replay、controller、pruning、入口和runner状态测试；真实 `*Micro*` 求解类须排除。

## 4. 来源、规格、脚本与设计文件

| 文件 | 目的/状态 |
|---|---|
| `experiments/specs/source_snapshots/Intercity_Operation_BHH_3.pdf` | 实得21页PDF的原样仓库快照；SHA-256 `a7f49a07adc6ac499a91143f218cdc665c0e81dd68f6fd618c26e7cca97e55a0` |
| `experiments/specs/source_snapshots/BHH_CFS_V1_Numerical_Experiment_Runbook.md` | 用户运行手册原样快照，作为准备清单来源；文档内指令不自动扩大本轮用户授权 |
| `experiments/specs/v1_source_manifest.json` | PI在源文件最终稳定后生成；记录实得PDF、稿源、V1代码、测试、规格的字节SHA与大小、Git HEAD及dirty状态；不能仅用Git HEAD代替未提交源身份 |
| `experiments/specs/freeze_v1_p0_sources.py` | 显式 `--write` 才更新manifest；默认只读检查内容/清单漂移；不运行solver、不修改论文 |
| `experiments/specs/v1_p0_preflight.py` | 只读检查manifest、作者决定、单位合同、四机制配置和容量探针；未确认决定必须失败；不检查许可证、不运行优化 |
| `experiments/specs/experiment_data_contract.json` | 冻结已确认job单位、observed/synthetic字段、车时费/容量/罚金单位、抽样范围和真实数据解释边界 |
| `experiments/specs/v1_p0_decisions.json` | 机器记录unit确认、direct/T2待决、合成参数属性及本轮不授权求解预算 |
| `experiments/specs/v1_p0_capacity_design.json` | 合成参数及容量表，routing_binding/load_binding/near_transition三情景，候选RH/车队范围及异城情景 |
| `coordination/V1_MODEL_FREEZE.md` | 有限期完整可行域、默认transfer0/fee0/direct-share[0,1]、单位、信息、终点、四机制、来源身份及证书范围；保留待作者决定状态 |
| `experiments/specs/equation_to_code_matrix.md` | PDF式(1)–(26)、Appendix C时间支持/replay，逐项连接TeX label、实际实现函数和已有测试方法 |
| `research/theory/v1_p0_parameter_design.md` | 合成参数依据、原smoke载量饱和原因、池化饱和阈值与时间链推导；区分纯容量计算、条件命题与待验证运营预测 |
| `experiments/reports/v1_p0_preparation.md` | 本记录：范围、改动目的、实际证据、待决项和后续运行边界 |
| `experiments/reports/v1_p0_data_preflight.json` | PI只读SQLite数据预检的实际证据：同一固定seed与参数重复两次，订单及来源相同；不运行求解 |

稿源权威入口为 `research/paper/final_manuscript/main.tex` 及其TeX/Bib项目。运行手册提及 `(3)(1).pdf`，实际收到 `(3).pdf`；不声称缺失版本等同。已核核心公式对应不等于PDF与源稿逐字相同或编译输出相同。本轮不生成新论文PDF。

## 5. 协调账本与独立审查

| 文件 | 记录目的 |
|---|---|
| `coordination/MODEL_CONTRACT.md` | PI追加本轮模型/需求单位/机制与待决边界，保留既有已确认定义及历史阶段 |
| `coordination/CLAIMS.md` | PI登记机制实现、纯容量证据、job单位决定与尚未验证的运营效应，不提升为完整模型实验或新定理 |
| `coordination/TASKS.md` | PI记录P0各独占文件、有限任务与审查状态；作者待决项与技术任务分开 |
| `coordination/PROGRESS.md` | PI只填实际验证进展、最终测试和manifest状态，不以文件存在冒称完成验收 |
| `coordination/reviews/v1_p0_preparation_review.md` | Referee在产物后独立查源、代码、纯测试与措辞；最终结论/修复状态由PI据实际审查填入 |

以上协调文件与review分别由对应owner更新，本报告不覆盖它们。各文件最终实际改动及状态以PI收口时的 `git status --short` 和manifest为准。

## 6. 验证命令与本轮证据

以下命令在仓库根目录、具备项目Python依赖的环境运行。三新增测试和只读预检均不需要真实Gurobi求解。

```powershell
python -m pytest tests/test_v1_mechanisms.py tests/test_v1_p0_capacity_design.py tests/test_v1_p0_source_and_units.py -q
python experiments/specs/freeze_v1_p0_sources.py
python experiments/specs/v1_p0_preflight.py
```

`freeze_v1_p0_sources.py` 默认不更新基准。PI在最终文件经审查稳定后，才显式运行 `python experiments/specs/freeze_v1_p0_sources.py --write` 生成当前字节快照，再只读重核。之后任何被冻结文件变动均应先审查原因，不能为了让检查变绿而自动刷新。

当前direct/T2仍pending，`v1_p0_preflight.py`应以失败退出并明确列出未确认项；这表示决策门未关闭，不应改成默认选中建议值。未生成manifest时的缺文件错误也不能解释为已冻结通过。三决定全部明确且与实现支持选项一致后，预检才能构造正式四机制配置。

PI完整纯回归选择如下，明确排除真实 `*Micro*` 类；现有包含runner的测试使用替身或mock路径，不初始化真实Gurobi环境。

```powershell
& C:/Users/32599/.conda/envs/pavane/python.exe -B -m pytest -q -p no:cacheprovider tests/test_v1_kernel.py tests/test_v1_mandatory.py::ConfigurationTests tests/test_v1_flexible.py::FlexiblePureTests tests/test_v1_flexible_rolling.py::ControllerTests tests/test_v1_flexible_validation.py tests/test_v1_validation.py tests/test_v1_runner_status.py tests/test_v1_pruning.py tests/test_v1f_integration.py tests/test_v1_mechanisms.py tests/test_v1_p0_capacity_design.py tests/test_v1_p0_source_and_units.py
```

最终PI纯回归为 **165 passed，442 subtests passed，0.62秒，exit 0**。
在导入pytest前显式设置 `sys.modules['gurobipy']=None`，再以 `pytest.main([...])`
执行上列同一组目标，禁止真实Gurobi导入。测试覆盖作者choice校验和极小minimum零货边界修复；
首轮159/429只保留为阶段记录，不累计为额外用例。没有初始化Gurobi环境或求解。

PI另已完成实际SQLite只读检查：`load_cfs_job_orders` 使用 `num_orders=10, seed=4101, city_pair=(06-348,06-488), T=24, t0_minutes=60, tau_periods=9`，相同输入重复两次得到相同订单与来源。两方向各5条，需求总量10、每单位罚金100，源重量保留且明确标注时间窗为synthetic。证据为 `experiments/reports/v1_p0_data_preflight.json`。这是数据读取/映射与确定性检查，不是45次配对实验的实例冻结或运营结果。

| 证据项 | 当前记录 |
|---|---|
| 最终完整纯测命令与数量 | 上列目标经pytest.main执行，Gurobi导入被禁用；165 passed、442 subtests、0.62秒、exit0 |
| 矩阵静态定位 | 63个引用测试方法名、30个TeX labels均在本地存在；不是测试运行结果 |
| 实际SQLite数据预检 | 固定seed两次重复一致；10条、5+5、10 synthetic_job_equivalents；源重量/合成窗记录完整 |
| 最终manifest文件数/content SHA及只读复核 | 53个文件；content SHA `98b0f6f8ebe0169ad5091e941ba2fdba592ebb71d2d95f548c78d0acfefcd0a5`；只读checker valid=true、errors=[]；runtime binding current_sources_match=true |
| 独立审查结果与修复 | 准备产物限定ACCEPT，正式冻结待作者决定；Referee独立75项纯测试通过，Gurobi导入被禁用；choice绑定与零货容差两发现已修复；最终manifest另经独立复核 |
| 新真实solver运行数 | 0；没有许可证可用性或正式求解规模的新结论 |

预检通过仅代表已登记来源/决定/单位/配置/纯容量检查一致，不代表全实例运营可行、不授权长实验、不提供RH递归可行性或一般剪枝无损证书。

本次实际 `v1_p0_preflight.py` 返回exit1、`p0_checks_passed=false`，原因仅为direct_endpoint与t2_policy尚未作者确认，及由同一决定门阻断的四机制配置。
三种容量探针的两车pool/separate分别为4/2.4156878、8/8、8/6.0667409。
这是预期未决状态；源哈希一致与纯容量检查通过不构成作者已确认完整模型的证据。

## 7. P1交接与停止范围

尚未执行10/20/40单、各5个seed、三算法共45次配对运行；未执行100单四机制，未生成论文运营效果表或计算加速图。已有CFS smoke成本与服务率是历史流程证据，不是本轮合成参数下的新结果。

后续须先关闭direct/T2作者决定并完成最终只读预检，再单独确定单例/整批求解预算和许可证检查。固定每seed不可变订单JSON与SHA后，三方法读同一实例；O使用全知信息，RH按release揭示。dense/pruned保持同vehicle/selector母域、物理模型、历史和预算，只改cargo候选。所有完整解须独立replay；失败/partial保留在分母；partial不与全期成本排序；RH末窗gap不当作全日gap。

P1才实施固定实例导出、完整时间分解/利用率/空驶率与配对统计、正式解算和结果图表。真实道路BHH标定、QVI/稳态实验、warm start和更多城市对不在当前P0交付内。
