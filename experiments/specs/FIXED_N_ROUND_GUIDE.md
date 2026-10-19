# 固定订单数的四方向实验：运行与解释

2026-10-19（Asia/Shanghai）。本轮以作者确认的“一轮固定订单数，依次运行四个方向”为组织原则。本文是新入口的使用说明，不是新的实验结果。原始运行手册中的多规模建议改为由使用者逐轮指定 N，不自动扩展订单数量。

## 1. 一轮是什么

一轮由固定订单数 N、固定 seed 列表、不可变输入、预先声明的方案和累计预算组成。每个 seed 的订单只生成一次，四个方向读取同一订单 JSON；比较过程中不重新抽样。N 可以是不小于2的任意整数，奇数时两方向分别取向上/向下的一半；seed 需为不同的非负整数，默认4101–4105。下一轮改变 N 时使用新的 batch 名称。相同 seed、不同 N 不自动构成嵌套样本，也不代表原先订单全部保留。

实验入口为 [run_fixed_n.py](../run_fixed_n.py)。模型与单位沿用 [MODEL_CONTRACT.md](../../coordination/MODEL_CONTRACT.md) 和 [V1_MODEL_FREEZE.md](../../coordination/V1_MODEL_FREEZE.md)：每条 CFS shipment 对应一个连续可分的合成任务等效量，重量仅作来源追踪；Direct 完成后进入目的城市抽象可用库存。该城市节点假设不提供真实道路或 hub 接驳的实证验证。

## 2. 四个方向与共享基线

每个 seed 预先登记 **14 个唯一求解任务**。多个表中出现的基线引用同一 run，不重复求解，也不增加独立样本数。

| 顺序 | 方向 | 本方向使用的方案 | 新增唯一任务/seed |
|---|---|---|---:|
| D1 | 运营机制 | full：T0/m=0、T1/m=0、T2/m=0.1、T3/m=0、T3/m=0.1 | 5 |
| D2 | RH 剪枝 | RH-dense、RH-pruned，均为 T3/m=0；共享 D1 的 full T3/m=0 参考 | 2 |
| D3 | 参数敏感性 | full T2/m=0.05、0.2；full T3/m=0 的 Direct 费率 ×0.75、×1.25；RH-pruned H=8、16 | 6 |
| D4 | BHH pooling 消融 | full T3/m=0 的 local-separate；共享 D1 的 local-pooled full | 1 |
| 合计 | 按 D1→D2→D3→D4 执行 | 例如 5 seeds 对应 70 个唯一任务 | **14** |

T0 无载货 Direct、无独立 empty；T1 无载货 Direct、允许独立 empty；T2 允许 Direct、关闭独立 empty；T3 允许两者。T2 采用每条 Direct 弧的最低平均载量政策 `q >= m*w`，并非逐车严格非空载保证；正货但平均载量小于 m 的 wave 也被排除。T3/m=0 保留源模型的零货 Direct 特例。因此识别独立 empty 的增量作用时比较 **T2/m=0.1 与 T3/m=0.1**，不能直接用 T2/m=0.1 对 T3/m=0 的差值归因 empty。

RH 基线 H=12、E=12、步长 h=2；H 敏感性只改为 8 或 16，E 和 h 保持不变。增加 H 不会提前揭示未来订单，会同时改变任务开始域和 `min(T,t+H+E)` 的完成域；固定 E 指延长长度固定，并非绝对完成端点固定。Direct 费率敏感性只改 Direct 成本，独立 empty 费率显式固定为基准值25，不同时改需求或车队。T3/m=0 中零货 Direct 也使用该费率，因此这是 Direct 弧价格敏感性，不限于载货价格。其余参数保持冻结基线，不在本轮自动寻找有利参数。

local pooling 消融只将 local 容量替换为 `min(v*M, v*f_inverse(B))`，其中 `B=max(Delta-2*rho,0)`，Delta 为以分钟计的任务时长；原 pooled 为 `min(v*M, f_inverse(v*B))`。Direct 容量式、车辆库存、目标、订单和其他约束保持一致。除原全模型核验外，separate 另以原函数检查 `q<=v*M` 及 `f(q/v)<=B`（v>0），不能仅通过更宽松的 pooled 容量检查。local-separate 是受控比较模型，不是用 CFS 测量得到的另一种真实运输系统。它不宣称同时关闭 Direct 中的空间机制。

## 3. 使用顺序

以下命令均在仓库根目录执行。`python` 应指向已安装本项目依赖与可用求解器的环境。真实运行前先查看计划；计划不启动求解器。

```powershell
python experiments/run_fixed_n.py --orders 40 --seeds 4101,4102,4103,4104,4105 --batch FIXED_N40_A --plan
```

准备阶段生成不可变输入、方案和来源身份；已有输入可用 `--input-batch P1_SMALL_20261018_01` 从对应 N/seed 的保存订单中复用，不复用旧求解结果作为新版本的结果。没有指定复用批次时从 `--source` 的 SQLite 读取，默认 `data/cfs_2022_pums.sqlite`。下例显式设置累计预算 600 秒、每 run 逻辑预算 5 秒、worker 硬超时 15 秒；这只是可修改的命令示例，不是对所有 N/seed 数量的预算保证。

```powershell
python experiments/run_fixed_n.py --orders 40 --seeds 4101,4102,4103,4104,4105 --batch FIXED_N40_A --input-batch P1_SMALL_20261018_01 --budget-sec 600 --per-run-sec 5 --worker-timeout-sec 15 --prepare
python experiments/run_fixed_n.py --orders 40 --seeds 4101,4102,4103,4104,4105 --batch FIXED_N40_A --input-batch P1_SMALL_20261018_01 --budget-sec 600 --per-run-sec 5 --worker-timeout-sec 15 --run
```

若不需要单独审查准备文件，可直接 `--run`：目标批次不存在时先准备，存在时核对冻结身份并继续尚未尝试的任务。只有 `--run` 必须显式提供 `--budget-sec`；`--prepare` 可以不带总预算，首次 `--run` 才将累计执行预算绑定到 `execution_budget.json`。省略动作时使用只读计划模式。`--design` 可提供与 [default_design.json](../fixed_n/default_design.json) 同 schema 的有限敏感性设置；该设计在准备时冻结，不在同一 batch 中边跑边改。查看完整选项以实际 `--help` 为准。

```powershell
python experiments/run_fixed_n.py --help
python experiments/run_fixed_n.py --orders 40 --seeds 4101,4102,4103,4104,4105 --batch FIXED_N40_A --replay
python experiments/run_fixed_n.py --orders 40 --seeds 4101,4102,4103,4104,4105 --batch FIXED_N40_A --analyze
```

更改 N、seed 列表、预算或实验设置时，先检查其是否改变既有 batch 的冻结身份。需要新的设计或新的求解尝试时创建新 batch；不能用相同名称覆盖原始结果。程序不自动增加 N，不自动遍历多个规模，也不启动后台研究循环。

## 4. 预算、继续执行和停止

`--budget-sec` 是显式累计的整轮执行预算，不是每次调用脚本重新获得的预算。准备和各次执行 session 累计计入；worker 启动、建模、求解和 run 内核验均属于执行开销。单次 RH 的所有窗口共享 `--per-run-sec`，不能解释为每个窗口各有该时间。worker 硬超时是进程保护，也受整轮剩余预算限制，不代替逻辑求解预算。

继续运行只处理尚未尝试的任务。已登记 attempt 的失败、超时、无 incumbent 或中断任务保留原记录，不自动重试、不重写旧 raw。来源身份不一致、独立核验失败或其他正确性错误应先停止并审查；普通限时结果如实记录。预算不足的任务保留在计划分母中，不补零成本，也不以成功样本替换。

系统强制终止控制器时可能留下独占 `run.lock`，脚本会阻止继续执行；它不自动删除旧锁，也不承诺任意强制中断后自动恢复。需保留现场进行检查，不能手工移除锁后假定累计耗时仍完整。正常完成的再次执行会跳过所有已登记任务。

新入口、独立 pooling 比较实现和相关测试采用补充来源冻结；原 [v1_source_manifest.json](v1_source_manifest.json) 的 63 项来源保持原样。来源 hash 记录版本身份，不构成模型正确性证明。若新实现需修正，须创建可追溯的新版本和批次，不刷新 hash 掩盖原批次变化。

## 5. 输出怎么读

保存目录使用 `simulation/fixed_n_runs/<batch>/`；输入、计划、预算、每任务 raw/状态、执行 session 和分析产物共同保留。`--run` 完成执行后自动重放并导出分析，后处理时间单列；`--replay` 和 `--analyze` 也可单独调用，各写新的时间戳分析目录。完整分析导出 `runs.csv`、`windows.csv`、`paired.csv`、四个方向 CSV、`summary.json` 和 `summary.md`。先检查计划任务数、实际尝试数、完整可行数、错误和预算未运行数，再解释成本或时间。复用的基线只算一次求解和一次观测。

成本排名只使用全日完成且独立核验通过的结果。`partial`、无 incumbent、错误、worker timeout 和预算未运行不进入完整成本排名，仍进入状态分母。`OPTIMAL` 指求解器设定容差内最优；限时 full 只能报告 incumbent、有效 bound 和 gap，不写成精确最优。RH 的单窗 bound/gap 不提供全日证书。

四方向优先报告同 seed 的配对差值和有效配对数。成本差定义为比较对象减参考；相同目标系数下负值表示比较对象成本低。费率敏感性改变了目标系数，其成本变化是情景响应，不能解释为同一目标下的算法质量排名。全知 full 与 RH 的信息集不同：前者已知全日订单，后者按 release 揭示；两者差值同时含信息与滚动决策限制。RH-dense 与 RH-pruned 的完整历史可能分叉，端到端比较不能替代相同历史下的单窗剪枝诊断。候选数减少本身不证明无损或加速。

机制报告成本及分项、服务/拒单货量、Direct 货流、独立 empty、零货 Direct；连续货量比例不能直接称为整票履约率。pooling 比较报告受控容量差、运行成本/服务变化，不据零差值断言 BHH 机制无效，也不从一个固定 N 外推全部规模。所有系数、车队、费用和时间窗仍为披露的合成设计。

## 6. 验证与研究结果的边界

已通过23项新增纯测试。功能批次 [FIXED_N_SMOKE_20261019_01](../../simulation/fixed_n_runs/FIXED_N_SMOKE_20261019_01/) 固定N=4、seed4101，顺序完成14个真实任务（10个full OPTIMAL、4个RH COMPLETED，共58次optimize），全部通过独立重放。首次prepare+执行30.593秒、后处理1.688秒；再次执行新增attempt=0，14份raw的SHA保持不变。该验证只用于检查四阶段编排、输入共享、预算、记录、核验与恢复行为，不作为机制收益、算法加速或论文规模结果。独立复核见 [fixed_n_round_review.md](../../coordination/reviews/fixed_n_round_review.md)。

研究设计依据：[原始运行手册](source_snapshots/BHH_CFS_V1_Numerical_Experiment_Runbook.md) Step 7–10、[有界 P1 协议](V1_P1_BOUNDED_PROTOCOL.md) 和 [固定 N 数学/比较合同](../../research/theory/fixed_n_round_contract.md)。既有 [P1 小规模报告](../reports/v1_p1_results.md) 与 [N40 后续机制报告](../reports/v1_followup_20261019_results.md) 保留各自旧设计与版本，不因新入口而改写历史。
