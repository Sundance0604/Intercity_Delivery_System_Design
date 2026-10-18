# V0-F-EMPTY 跨控制窗换装链独立验证

时间：2026-10-18 09:04（Asia/Shanghai）。执行：Experiment 子 Agent；PI 负责接入 runner 与独立审查。

## 结果与证据等级

独立货流检查器已实现，真实 registry rolling 微例通过 pickup→AV→destination 三个控制窗。该结果是**确定性合成微例与单元验证**，不提供一般数学定理、真实数据性能结论、V1 对齐证明或全局最优证书。

新文件：`intercity_delivery/repositioning/cargo_validation.py`、`tests/test_repositioning_cargo.py`、本报告。子 Agent 未修改原优化代码、旧测试、历史数据或论文；PI 在 `runner._result` 接入 `detail.cargo_validation`，属于单独所有权。

## 独立检查范围

API：`validate_cargo(config, data, decisions, completed_horizon=None, completed=False, tolerance=1e-6, *, order_summaries=None)`。非法计划抛 `ValueError`，通过时返回 JSON 字典。

检查器不导入 Gurobi，也不读取优化器表达式。它直接读取订单、配置、实际弧域及全部已发起 commitments，按绝对事件逐订单重放：

- 始发 hub 库存为已完成 pickup 减已出发 AV；可用库存另计处理延迟 θ。
- 目的 hub 库存为已到达 AV 减已出发末端配送；另计目的处理延迟 θ。
- 每笔订单独立守恒，禁止以其他订单库存弥补缺货；保留小量和跨 cut 在途任务。
- local 共享容量独立反算 `a q+b sqrt(q)=Δ`；保留实际 V0-F 无 `capacity_manual` 截断的公式。AV 为同弧跨订单共享容量。direct 用两端合并服务时间反函数与 `capacity_direct` 截断。
- pickup/direct 的释放时间、delivery/direct 的交付期限、启动数量与需求预约上限。
- 完整执行要求 pickup、AV、delivery 总量一致，检查总体 direct-ratio 政策；如提供模型 r/q/z summaries，逐笔核对。

partial 执行允许 hub 库存和在途货物，保留未来完成及 hub ready 事件；明确 defer 最终链路平衡、r/q/z 等式、全局 direct ratio。完整执行未提供 summaries 时只核对执行货流并计算 residual，不宣称已检查未输入的 z。

车辆位置、可用性、空车物理时间归 `ledger.py` 检查；本检查器不能替代车辆账本。也不检查 V1 stem/联合 wave 容量，不能验证概率信息结构、一般滚动可达性或真实运营可行性。

## 微例与手算

来源：内存构造 `OrderBatch`，无随机采样、外部 benchmark 或历史数据。T=5，t0=60min，HV 初始各城市 1 辆，AV 初始 city1 为 1、city2 为 0；AV τ=1，θ=0。订单 city1→city2，quantity=0.05，release=0，deadline=3，penalty=1000/单位。local `a1=a2=60min/单位`、b1=b2=0；AV capacity=1。合法设置 `direct_ratio_max=0` 禁用载货 direct，未破坏原始变量或求解器。

通过新 registry `flexible_direct_empty_rolling`；prediction_horizon=1、step=1、extension=2。唯一可按期完整服务的最短链为 pickup[0,1]、AV[1,2]、delivery[2,3]，三个发车决策在三个不同控制窗提交，历史货流需在后续窗持续存在。

人工小时成本 2、AV 小时成本 3、换装费 4/单位，手算完整费用为 `2×1+3×1+2×1+4×0.05=7.2`。断言实际 cost=7.2、served=0.05、direct=0、transshipment=0.05、unserved=0、completed=True、全局 best_bound=None；cargo replay 另核对 t=1 AV 在途、t=2 配送在途、t=3 已交付。

只运行过 **1 个真实 rolling run**，整个 run 时间预算 2s、Threads=1；T5/step1 实际完成 5 个微型窗口 MILP。不能把一个 rolling run 误记成只有一次 MILP optimize。Gurobi 随机种子未单独覆盖，沿用默认值；该数据构造无随机种子。

## 命令、环境及原始输出摘要

Python：`C:/Users/32599/.conda/envs/pavane/python.exe`，3.11.15（Anaconda，MSC v1942，AMD64）；Gurobi 10.0.3。Git HEAD=`f26b4c96907082c2ebb7b0104263831e7100aa26`，工作树已有其他未提交文件，因此 HEAD 不是完整输入指纹。未安装依赖，未提交或推送。

命令：

```powershell
& 'C:/Users/32599/.conda/envs/pavane/python.exe' -B -m unittest tests.test_repositioning_cargo -v
& 'C:/Users/32599/.conda/envs/pavane/python.exe' -B -m unittest tests.test_repositioning_cargo.CargoValidationTests -q
git diff --check
```

第一次完整检查（21 纯测试 + 1 真实 rolling，exit 0，进程 0.162s）：

```text
Ran 22 tests in 0.048s
OK
```

之后补充在途/处理时间及 Referee 指出的数值输入边界测试，最新纯测试（30 项，exit 0）：

```text
Ran 30 tests in 0.003s
OK
```

本文件现有 31 项不同测试（30纯+1真实）。真实 rolling 部分只运行上述一次；之后新增 `detail.cargo_validation.valid` 与 served 集成断言，按 PI 指示不重复求解，交由 PI 最终验证。开发中首次纯测试有 1 次断言失败：跨订单借货用例在预期 causality 之前被完整链路 imbalance 拒绝；将该用例改为 partial 检查以单独触发 causality，后续通过。不是生产求解失败。

PI 回报：接入 runner 后曾运行 56 项（原34+本文件当时22），全部通过，suite 0.329s、进程0.545s；该轮含6 full+2 rolling run。此为 PI 独立回归记录，不计作本子任务新增不同测试或子 Agent 求解次数。后来新增纯测试后合计应为65项，待PI最终记录。

最终 PI 执行集合补记：上述56项整suite一次；补4例后的 cargo26项suite一次，0.055s；最新纯30项一次，0.003s、进程0.126s。均PASS。因此当前31个不同新增测试及原34合计65个不同测试均已执行过，但未声称最后版本一次性跑完65项。新换装场景真实 rolling 共3整次（子Agent1+PI2），各5窗口，总15个微型窗口MILP；每整次2s cap，计入主Agent批准预算。旧6 full及旧empty→direct rolling兼容回归单独为最多14s cap，不能混作新场景求解次数。最新 pure-only 改动未增加求解。PI确认Referee静态关闭NaN阻断后已连续启动第二步。

**最终计数（取代上段截至交付时的3整次/15窗历史计数）**：PI为验证NaN修复后的最终runner，另定向执行 `CargoRollingMicroTests`，1项PASS，suite0.051s、进程0.158s、exit0。至此新换装RH累计4整次（子Agent1+PI3），各5窗口，共20个微型窗口MILP；每整次2s cap，共8s预算已用完。旧6 full+旧direct RH兼容回归另14s cap，两部分理论上限合计22s，实际进程时间总和远低于该上限。最终30 pure+1 real均在最终代码通过；不再执行求解。本补记只更新报告，不修改代码或测试。

Referee 复现初版 scalar NaN 假阳性后，检查器增加配置有限性/符号/整数校验、严格 bool completed、非 bool tolerance、订单有限且有序的实数时间阈值、服务容量中间值 overflow 拒绝。支持服务 a>0、b>=0，与实际 loader 域一致；不再静默接受 a=0 特例。有限分数订单窗口保留 V0 阈值语义，不强制取整。修复后只运行纯测试，未增加 solver run。

无 license failure、solver warning 或非零求解退出；`git diff --check` exit0，仅输出已有 tracked 文档 LF/CRLF 提醒。仓库初始 `git status` 提示 `.pytest_cache` 不可访问；未读取或修改该缓存。

交付时 SHA256：

| 文件 | SHA256 |
|---|---|
| models/flexible_direct_optimizer.py | 590C805DC65ED23A6689FD65583D63ABE864124221614FF56862A65B7F701DF2 |
| repositioning/cargo_validation.py | 6391E279203CB82BBD674D49D010E34357E9F1553EAF695972AB680533739CED |
| tests/test_repositioning_cargo.py | EEC26D4B9E57E4F16F4BE4B6C7132B5F8DE1DB6975BAF294892DCD4C5959A8BB |

## 有意破坏与缺口

负例覆盖 pickup 未完成先发 AV、AV 未到先配送、两端处理延迟、跨单借货、local/AV/direct 跨订单重复占容量、释放前发车、过期交付、pickup+direct 重复预约需求、完整期末残留货流、篡改 r summary、错误订单/方向、NaN/负数/非整数车辆及未来未发起承诺。正例覆盖完整与 partial、在途 pickup/AV、direct-only、1e-5 小量、合法处理延迟、序列化 keys，以及 V0-F 不额外引入 V1 manual M 截断。

没有发现必须修改生产优化器的 bug；Referee 所查假阳性属于本次新 checker 的输入边界，已在上述域内修复。完整有限时域货流与分项容量在本例可重算；多订单、双向大量滚动、非零处理延迟的真实求解、提前终止后恢复、真实数据规模表现尚无实验覆盖。检查器采用绝对 tolerance=1e-6，属于数值容差，不能提供任意微小负库存的精确算术证书。
