# V1 有预算小规模真实求解记录

2026-10-18（Asia/Shanghai）。Batch：`P1_SMALL_20261018_01`。负责人：Principal Investigator；结果整理：Literature；独立审查：Referee。

**状态：SMALL_SCALE_COMPLETED / LIMITED_ACCEPT。** 75个批量运行全部完成且独立replay通过；统计、图表及Referee独立全量审查完成。加上5个micro，共80个顶层attempt、413次optimize，累计求解编排墙钟136.108秒。10/20单剪枝与dense成本相同；40单剪枝平均更快但平均成本更高，不能称为无损。10单机制样本未识别direct/独立empty/minimum的运营效应。

本批规模仅为10/20/40单，是有限预算的预实验。它不等于完成运行手册全部P1、正式大规模论文实验、真实道路BHH校准或投稿验收。

## 1. 冻结模型与证据身份

作者profile状态为 `FROZEN_AUTHOR_CONFIRMED_SYNTHETIC_PROFILE`。三项作者决定为：

| 决定 | 冻结内容 |
|---|---|
| `V1P0-JOB-UNIT-2026-10-18` | 每条抽样CFS shipment映射1个 `synthetic_job_equivalent`，货流仍连续可分；源重量不作为需求或额外容量约束 |
| `V1P0-DIRECT-ENDPOINT-2026-10-18` | 保留原direct聚合时长式，车辆在完成节点j进入目的城市抽象可用节点，并可从j衔接后续任务 |
| `V1P0-T2-MINLOAD-2026-10-18` | T2禁止独立empty，direct满足 `sum_l h_ijl>=m*w_ij`；主值m=.1，敏感性.05/.1/.2 |

最低平均载量是wave层面的约束，不保证每辆车严格非空。数学上，它同时排除零货和平均载量低于m的正货wave；例如w=1、q=.05在m=.1下不可行，已由micro核验。因此不能将其解释为仅改变q=0时的可行性。

冻结来源manifest覆盖 **63个文件**，内容SHA-256为 `5c8b233fbc50cda4d23383f404915027e2b8d63de5faa03a30f6703063b1c6c4`。这标识登记的源字节，不证明实得 `(3).pdf` 等同缺失 `(3)(1).pdf`，也不证明该PDF由当前TeX编译。论文正文、附件PDF和历史结果不因本批运行改写。

PI最终纯回归记录为 **195 passed，454 subtests passed，0.70秒**。它是纯测试证据，不计作真实optimizer运行。源冻结、预检及决定详情见 [V1_MODEL_FREEZE.md](../../coordination/V1_MODEL_FREEZE.md)、[v1_source_manifest.json](../specs/v1_source_manifest.json)、[V1_P1_BOUNDED_PROTOCOL.md](../specs/V1_P1_BOUNDED_PROTOCOL.md) 与 [v1_p1_budget.json](../specs/v1_p1_budget.json)。

## 2. 数据构造与参数

运行目录为 [P1_SMALL_20261018_01](../../simulation/v1_runs/P1_SMALL_20261018_01/)。在比较求解前固定 **15个不可变实例**：N=10/20/40，seeds=4101/4102/4103/4104/4105。每个实例保存原订单、来源记录、参数、信息规则和SHA；同N同seed的所有算法/机制读取同一实例。相同seed不同N不自动视为相同订单集或嵌套实例。

输入prepare实际墙钟为 **26.39秒**。本城市对符合当前筛选的记录共 **30,185条**，两个方向为 **15,177与15,008条**。样本两方向平衡是实验构造，不能解释成原总体OD平衡。该计数对应本次查询口径，不概括全部CFS货运总体。

| 项目 | 本批固定设置 |
|---|---|
| 城市对 | `06-348↔06-488`，LA–SF案例 |
| 需求 | 每条源shipment提供1个合成job-equivalent，N分别10/20/40；数量可连续拆分 |
| 时间 | T=24、每期60分钟、tau=9期；release/deadline为沿用生成程序的合成窗 |
| Local参数 | 两城a=5分钟/等效量、b=20分钟/等效量平方根、rho=5分钟 |
| 单车容量 | 人工M=4，AV M_A=8，单位均为 `synthetic_job_equivalent` |
| 初始车队 | 每城人工6、AV2 |
| 小时车费 | local/AV/direct/empty为20/15/25/25，货币/车小时；只除60一次 |
| 拒单罚金 | 100货币/未服务等效量 |
| RH | H=12、h=2、E=12，按release揭示，绝对历史保留 |
| 核心扩展默认 | transfer delay=0、transfer fee=0、direct比例边界[0,1]、真实终点城市分配自由 |
| 环境 | Python 3.11.15、Gurobi 10.0.3、Threads=1、solver seed=0 |

上述a/b/rho/容量/费用/车队/罚金是预先登记的合成机制设计，不是由CFS实测校准的城市运营参数。源重量、区域、商品等观测字段和合成需求/时间窗分别记录。

tau=9沿用既有LA–SF recommendation。事后[全城市对只读校准核对](../../simulation/v1_runs/P1_SMALL_20261018_01/corridor_calibration_audit.json)已用30,185条eligible记录及WGT_FACTOR加权中位数复核：代表性大圆距离354 miles，circuity因子1.2给出模型路线距离424.8 miles，假设车速50 mph对应8.496小时，1小时一期向上取整为9。该核对与预定tau一致，未改变冻结输入，未新增solver运行；circuity/车速与换算里程是模型构造，不能写成观测实际路线或实际行程时间。输入metadata中的 `explicit calibration` 所带 `representative_route_miles=0` 是固定tau接口的placeholder，**不能报告为真实路线距离0**；原metadata保留，由单独audit解释。

## 3. 预算、阶段与运行分母

预算上限为80个顶层独立attempt、累计求解编排墙钟600秒。累计计入输入prepare、5个micro及执行session，包含worker启动、建模、solver和run内replay；人类等待及后续只读复验/统计另报。Micro和机制/敏感性逻辑预算每run3秒，pilot每run5秒；RH整次共用预算，worker硬保护15秒。顶层attempt与逐窗 `optimize()` 调用数分别统计，不把一次RH的多窗误计为多次独立样本。

| 阶段 | 计划顶层attempt | 已知状态/后续填写 |
|---|---:|---|
| A 手算真实micro | 5 | 5个case均满足预期；共8次optimize调用，详见第4节 |
| B 三算法pilot | 45 | 14个full OPTIMAL、1个full限时可行incumbent、30个RH COMPLETED；45/45完整可行 |
| C 四机制 | 新增15；另复用B的5个T3-full | 新增15个均OPTIMAL、完整可行；T3复用不增加运行或样本数 |
| D T2 minimum敏感性 | 新增10 | m=.05/.2各5个seed；10个均OPTIMAL、完整可行 |
| E matched T3 | 新增5 | T3/m=.1，5个均OPTIMAL、完整可行 |
| 总计 | 80 | 80个顶层attempt，批量75/75完整可行；无partial/error/未运行 |

批量状态为44 OPTIMAL、1 TIME_LIMIT、30 COMPLETED；唯一TIME_LIMIT拥有完整可行incumbent，不归为无解或partial。总墙钟由prepare26.390秒、micro0.375秒和执行session109.343秒组成，共136.108秒，低于600秒。后续只读全量replay另耗12.375秒，不计作新增求解或隐去的运行次数。未删失败seed、自动重试或追加规模。

唯一限时行 `n40_s4103_T3_m0_full` 的runner墙钟5.234秒、parent进程墙钟5.390秒，记录 `budget_exceeded=true`。这超过5秒逻辑预算；硬worker保护15秒未触发。因求解停止和后续核算开销，不能把此轮记作全部单例均严格满足5秒端到端上限。总墙钟与单例逻辑预算分别披露，不以总预算有余掩盖超时。

## 4. 已完成的真实micro门

五个micro是预登记的小型手算核对，实际外层墙钟 **0.375秒**，顶层5个case、内部共8次optimize。三例返回预期不可行状态（solver status=3、无incumbent），两例获得可手核完整结果，成本分别5和4；两可行例通过独立replay。这里记录solver不可行状态，不声称独立重建了不可行性证书。预期不可行例是正反例验收的一部分，不是本批正式实例的三次未知失败。

| Micro | 检查内容 | 已确定结果 |
|---|---|---|
| 1 | T2强制每车.05货量、minimum=.1 | 预期不可行，真实返回不可行 |
| 2 | Direct完成后同刻开始目的城市local车辆连续性任务 | 成本5，与手算及独立replay一致 |
| 3 | 将上述local启动提前一期 | 预期不可行，真实返回不可行 |
| 4 | T0下强制direct | 预期不可行，真实返回不可行 |
| 5 | 跨窗direct执行与费用 | 成本4，与手算及独立replay一致 |

第2例local货量为零，是车辆连续性核对，不能描述为真实载货换装链。五例不单独证明AV共享容量、正empty最优采用、自然最优解中minimum绑定频率、所有RH递归可行性或一般模型正确性。Micro证据为 [gate_evidence.json](../../simulation/v1_micro/P1_SMALL_20261018_01/gate_evidence.json)、[raw.json](../../simulation/v1_micro/P1_SMALL_20261018_01/raw.json) 与 [execution.json](../../simulation/v1_micro/P1_SMALL_20261018_01/execution.json)，实施入口为 [run_v1_p1_micro.py](../specs/run_v1_p1_micro.py)。

## 5. 比较方法与结果表位置

O-full事前知道全日订单；RH-dense/pruned只知道release已揭示订单。O与RH的成本差包含信息与滚动决策限制，不能据此断言同信息算法优劣。Full限时有incumbent时保存有效bound/gap，只有已证明最优时称optimal；RH窗口gap不是全日gap。

Dense/pruned保持同物理模型、车辆与selector母域、窗口、预算和揭示规则，只改变货流候选。不同运行的历史可能分叉，因此整个trajectory对比不能自动当作相同历史下的单窗严格对照。时间支持候选减少不是一般无损定理，也不预先保证成本相等或更快。

以下每行5个seed全部完成并独立核验；成本为货币，服务率按连续货量，时间为parent进程端到端墙钟均值，包含worker启动及run内阶段，不含共享prepare26.39秒。

| N | 方法 | 平均replayed成本 | 平均服务率 | 平均进程墙钟（秒） |
|---:|---|---:|---:|---:|
| 10 | O-full | 688.0000 | 78.0000% | 0.6622 |
| 10 | RH-dense | 780.0000 | 80.0000% | 1.6720 |
| 10 | RH-pruned | 780.0000 | 80.0000% | 1.3876 |
| 20 | O-full | 1071.0000 | 88.0000% | 1.1842 |
| 20 | RH-dense | 1280.8387 | 87.5081% | 2.0408 |
| 20 | RH-pruned | 1280.8387 | 87.5081% | 1.4842 |
| 40 | O-full | 1669.8387 | 93.7540% | 2.9404 |
| 40 | RH-dense | 2399.3547 | 86.0161% | 2.7314 |
| 40 | RH-pruned | 2509.0677 | 87.6983% | 1.6810 |

40单full均值包含一个限时incumbent，因此不是五个全知最优值的均值。该行cost=1910，global bound=1893.9115018068374，gap=.00842329748333123（约0.84233%）；对它只能给 `1893.9115<=C*<=1910` 的该有限期模型证书。其他14个pilot full及新增30个机制/敏感性full均OPTIMAL。全部RH顶层global bound/gap保持空值，其窗口证书不能升级为全日gap。10单RH服务率比full略高并不违反full成本最优，因为目标同时包含运营费用和拒单，未最大化服务率。

配对成本差定义为pruned减dense，时间比为逐seed `parent_wall_dense/parent_wall_pruned` 的平均；后者不是两组均值之比。

| N | 平均配对成本差 | 95% Student-t区间 | 平均进程时间比 | 95% Student-t区间 |
|---:|---:|---|---:|---|
| 10 | 0.0000 | [0.0000, 0.0000] | 1.2049 | [1.1667, 1.2432] |
| 20 | 0.0000 | [0.0000, 0.0000] | 1.3748 | [1.2910, 1.4586] |
| 40 | +109.7130 | [-204.6042, 424.0303] | 1.6256 | [1.4961, 1.7550] |

40单五个成本差依seed4101–4105为约0、+562.4200、0、-13.8548、0；中位数0。该样本观察到剪枝平均更快且平均成本增加，服务率亦改变，不能称为无损加速。宽区间与单seed大差须一并报告；区间含0不抹去已经发生的成本损失，也不证明期望成本无差。历史分叉、求解退化选择与有限预算共同作用的具体归因未单独验证。

所有15个dense/pruned实例的车辆/selector母域签名一致；沿pruned实际trajectory统计的货流候选删减率平均为95.7216%（N10）、95.7104%（N20）、95.9290%（N40），全样本范围94.3223%–96.2091%。这是该trajectory候选域的缩减，母域相等不等于两算法拥有相同历史约束；不据此证明任意实例剪枝无损。

[三算法图](../../simulation/v1_runs/P1_SMALL_20261018_01/figures/pilot_runtime_cost_completion.png)左面板用进程时间**中位数**，上表用均值；中面板为平均完整成本，右面板75批量中pilot各组完成率为1。两种时间统计不可混称。每窗phase及候选/变量/约束明细见第8节CSV。

## 6. 机制归因与指标定义

T0关闭direct与独立人工empty；T1只开放独立empty；T2开放带minimum的direct并禁止独立empty；T3原式默认minimum=0，允许零货direct与独立empty。T2(.1)与T3(0)同时改变两项政策，不能把它们全部差异归因于独立empty。

独立empty可用性的匹配比较为T3(.1)对T2(.1)；minimum政策比较为T3(.1)对T3(0)；T2(.05/.1/.2)检查同一机制对minimum的敏感性。匹配只固定所列政策，不能由聚合约束 `q>=m*w` 推断逐车路线均非空。

N10的T0、T1、T2(.1)、T3(0)、T2(.05)、T2(.2)、T3(.1)这7组各5个seed，所有full均OPTIMAL；各组平均成本均688，逐seed成本依次825、630、665、630、690。机制相对T0、matched-empty、T3 minimum与T2 minimum敏感性的逐seed成本差全部0，配对描述区间均[0,0]。

已逐组检查这35行：货运direct share、direct成本、独立empty车次、零货direct车次全部为0。因此此处的零差是本组稀疏小样本在这些政策下均未采用相关移动，**未识别direct/empty/minimum的运营收益或敏感性**；不能推广为这些机制在更紧张车队、不平衡或更大规模下无效。没有为追求显著效果追加调参或更换样本。[机制与minimum图](../../simulation/v1_runs/P1_SMALL_20261018_01/figures/mechanism_minimum_load.png)保留该零差结果。

交付按任务末尾计时，连续货量服务率不自动等于整票履约率。费用报告local/AV/direct/independent empty/penalty/transfer及replayed total。独立e的车次/成本不能称为全部人工空行，零货direct应单列。

人工全车队利用率按 `sum (j-i)Delta t*(x+w+e)/(N*T*Delta t)`，AV同理用y和A，计忙车时间而非任务数。N40 full/dense/pruned的平均人工忙时比例分别13.3333%/17.7083%/20.4861%，AV为37.5000%/45.0000%/43.1250%；这些是本批全车队车时比率，不是城市动态利用率或资源约束必然绑定的证明。逐run busy time、lead time与成本分项见summary CSV；未提供城市动态可用车时分母，不另造城市利用率。亚秒求解时间存在运行噪声，时间比同时报告进程端到端口径及不含共享输入prepare的范围。

## 7. 小样本统计口径

同N同seed做完整配对差，定义“比较对象成本减参考成本”，负值表示比较对象成本低。保留每seed原值与差值，汇总mean、sample SD、median和完整配对数n；未完成行保留原计划分母。主95%区间为配对差的Student-t区间 `mean(d) ± t_(.975,n-1)*sd(d)/sqrt(n)`，仅n>=2时计算。n=1列单值，n=0不做成本推断。相对差若使用，明示epsilon和近零参考情形。

每组最多5 seeds，区间的近似独立/常态差值假设未由小样本自动验证，仅作本案例的描述。算法执行顺序固定为full/dense/pruned，每个seed仅计时一次，没有随机化顺序或重复计时。因此时间比t区间描述的是这5个seed之间的差异，不能当作固定实例计时随机误差的置信区间，也不能排除启动/缓存/固定顺序偏差。T3复用不增加独立观测数；三规模和多个政策属于探索性比较，不作未经校正的显著性筛选。complete-pair样本被失败筛选时须披露，不能据其平均推广至全部计划或全美物流网络；本轮没有缺失完整配对。

## 8. 全量重放、数据链接与结论边界

事后[全量独立replay](../../simulation/v1_runs/P1_SMALL_20261018_01/analyses/replay_20261018T153338441272Z.json)逐个检查75个保存方案，75/75 PASSED，耗时12.375秒，无solver调用；核原f容量、direct式、货流/库存/窗、机制与费用。跨检查最大残差为 `4.214684873293864e-7`，小于核验容差；该最大值混合不同单位，不能解释成单一物理误差。纯核验通过不提供一般软件正确性或物理route实证保证。

可追溯产物如下。CSV保存逐run与逐窗明细，JSON保存摘要和配对统计，不以报告中四舍五入值替代原始精度。

- [全运行表 CSV](../../simulation/v1_runs/P1_SMALL_20261018_01/tables/v1_run_summary.csv)：75行实际运行，状态、成本分项、货量、忙时、证书与phase。
- [配对结果 CSV](../../simulation/v1_runs/P1_SMALL_20261018_01/tables/v1_paired_results.csv) 与 [汇总 JSON](../../simulation/v1_runs/P1_SMALL_20261018_01/tables/v1_paired_results.json)：80行配对差值、三算法均值、运行分母及域检查；复用75个实际attempt，80行不代表新增80次实验。
- [逐窗表 CSV](../../simulation/v1_runs/P1_SMALL_20261018_01/tables/v1_window_summary.csv)：RH子窗证书、phase、模型规模与母域签名。
- [完整analysis JSON](../../simulation/v1_runs/P1_SMALL_20261018_01/analyses/20261018T153421101497Z.json)：组状态、配对Student-t统计与选择规则，SHA-256 `3c3a0b9d8fdefea2c96eb316f8cbd7ca78b5ebbdbbd9cb53d56f8758abfaa7b2`。
- [Batch manifest](../../simulation/v1_runs/P1_SMALL_20261018_01/manifest.json) 与 [计划行](../../simulation/v1_runs/P1_SMALL_20261018_01/planned_jobs.json)：运行身份、参数、实例及计划分母。
- [Corridor audit](../../simulation/v1_runs/P1_SMALL_20261018_01/corridor_calibration_audit.json)：解释placeholder并独立核对固定tau，不改变输入。
- [三算法图](../../simulation/v1_runs/P1_SMALL_20261018_01/figures/pilot_runtime_cost_completion.png) 与 [机制图](../../simulation/v1_runs/P1_SMALL_20261018_01/figures/mechanism_minimum_load.png)：标准静态科学图，展示实际结果及零差，不用于一般性证明。

Batch内独立派生脚本 [export_tables.py](../../simulation/v1_runs/P1_SMALL_20261018_01/export_tables.py) 内容SHA-256 `06e9852af1ac5db29475e17c2d28e73709f0e14a087da75398fc0ac770f9793e`；统计与绘图不产生新solver运行。主要表输出字节身份如下，供复核使用，不取代源manifest和raw本身。

| 文件 | SHA-256 |
|---|---|
| `tables/v1_run_summary.csv` | `8dfe30a04153dc6ac1621ba854e442c108ac2f4093840255a2675f49d826bd33` |
| `tables/v1_window_summary.csv` | `8b8a0399ea5dd7fb2eb1e5ad318049e8a3909b188d36a4b24469e3e002f781ab` |
| `tables/v1_paired_results.csv` | `b2ef40ff2b21c7d36126072a68ee83f66ec294661bf8f7a85af890e37026311e` |
| `tables/v1_paired_results.json` | `a9796cf9eee05593729c7db4bf6570438e0e69e672f5e279c4198bc98558735f` |

## 9. P1文件职责与改动范围

| 文件/产物 | 本轮用途 |
|---|---|
| `intercity_delivery/v1/flexible_runner.py` | Full/RH阶段计时、调用与结果metadata，保留证书/状态范围 |
| `intercity_delivery/v1/flexible_rolling.py` | PhaseTimer、车辆/selector母域签名、逐窗与整run optimize调用数；RH预算不按窗口重新充值 |
| `intercity_delivery/v1/flexible_validation.py` | 原f独立replay并导出各检查残差、比较次数与单位，便于全量复核 |
| `intercity_delivery/v1/experiment_metrics.py` | 完整结果成本/服务/busy-time/lead-time汇总和配对Student-t描述统计 |
| `experiments/specs/run_v1_p1.py` | 独占写入不可变输入与job，预算/worker/执行session、只读replay和analysis；复用同ID而非重复求解 |
| `experiments/specs/run_v1_p1_micro.py` | 五个真实micro的手算与预期不可行门，保留top-level和optimize计数 |
| `experiments/specs/plot_v1_p1.py` | 从已完成记录绘制标准静态图；不运行solver、不替换失败点 |
| `tests/test_v1_direct_continuity.py` | Direct抽象城市节点与同刻衔接的纯验证边界 |
| `tests/test_v1_p1_instrumentation.py` | 阶段计时不双计、partial/异常时间、残差单位与母域签名 |
| `tests/test_v1_p1_harness.py` | 计划数量/复用、策略配置、输入防漂移、micro/source门、预算/timeout及无cost的失败记录 |
| `tests/test_v1_p0_capacity_design.py` | 已有P0参数测试由作者pending阶段更新为已确认合成profile，保持容量与时间支持的限定范围 |
| `experiments/specs/v1_p0_decisions.json`、`v1_source_manifest.json` | 三决定正式确认及冻结后的源字节身份，批manifest内嵌对应快照 |
| `experiments/specs/v1_p1_budget.json`、`V1_P1_BOUNDED_PROTOCOL.md` | 80顶层attempt/600秒、micro门、固定规模/seed、信息/成本/统计/停止规则 |
| `coordination/V1_MODEL_FREEZE.md`、`experiments/specs/equation_to_code_matrix.md` | 作者确认的数学/单位/终点/机制与准确源码映射 |
| `research/theory/v1_direct_endpoint_contract.md` | 城市抽象可用节点和原direct公式的限定解释，不补造物理hub行程 |
| `research/theory/v1_p0_parameter_design.md`、`experiments/specs/v1_p0_capacity_design.json` | 冻结合成参数和时间链，更新作者endpoint语义，不将纯容量检查当运营实验 |
| `simulation/v1_micro/P1_SMALL_20261018_01/` | 五例真实micro的gate/raw/execution证据 |
| `simulation/v1_runs/P1_SMALL_20261018_01/inputs/`、`jobs/`、`sessions/` | 15个不可变输入、75个job的raw/执行日志及会话预算账本；逐run文件按run ID归档，不逐个重命名 |
| Batch `export_tables.py`、`tables/`、`analyses/`、`figures/`、`corridor_calibration_audit.json` | 派生表/统计/重放/图和事后只读tau校核，源身份与输出SHA见第8节 |
| `coordination/CLAIMS.md`、`MODEL_CONTRACT.md`、`TASKS.md`、`PROGRESS.md` | PI据实际证据登记范围、计数、失败/限时、有限结论与后续任务 |
| `coordination/reviews/v1_p1_validation_review.md` | 专业产物与全量结果之后独立审查，限定ACCEPT已交付 |

本轮论文源未修改、未编译。[Referee最终报告](../../coordination/reviews/v1_p1_validation_review.md)给出**限定ACCEPT**：审查者独立读取并用原f重放全部75个raw，核对source/order/config/raw SHA、15组母域配对（180个对齐窗口）、阶段计时、16组配对统计/区间、全OD只读校准、CSV/PNG与派生脚本self-hash，均与保存证据一致；其复核未新增求解。该接受限于本批可复现的小规模观测，不接受一般无损剪枝、已证direct/empty运营收益或广泛统计优越性；单例5秒逻辑预算超出继续保留披露。

本批是给定合成参数、LA–SF CFS-derived输入和预算下的**小规模完整有限期模型实验**，不是纯容量toy。它给出了完整执行、剪枝时间/质量取舍和该N10机制样本零差的证据；不验证真实物理hub接驳、不证明一般剪枝无损、不证明QVI/稳态与有限期MILP等价，也不支持全国推广。100–500单、更紧张车队或不同城市的完整P1论文级实验尚未执行；后续不得仅以本批图表声称已经完成这些研究。
