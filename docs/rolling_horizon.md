# Rolling Horizon 实现与修改说明

本文档说明项目中 Rolling Horizon 算法的实现方式、涉及文件、运行方法和当前版本边界。

## 1. 实现目标

Rolling Horizon 对外仍是一次求解器调用：

```python
solver.solve(config, data, orders_tuple, time_limit)
```

求解器内部把规划期拆成多个重叠预测窗口。每轮优化一个长度为
`prediction_horizon` 的区间，但只提交前 `rolling_step` 个时段开始的决策。
窗口后段仅用于预测，下一轮允许重新优化。

默认参数如下：

```python
RollingHorizonConfig(
    prediction_horizon=8,
    rolling_step=2,
)
```

当 `T=24` 时，共会依次执行 12 个滚动窗口：

```text
优化 [0,8]，提交 [0,2)
优化 [2,10]，提交 [2,4)
...
优化 [22,24]，提交 [22,24)
```

## 2. 设计选择

当前版本保留完整的全局时间索引，每个窗口都复用 `Optimizer` 构建原数学模型：

1. 出发时间早于当前时刻的变量固定为已经提交的历史值。
2. 当前预测区间内的变量允许重新优化。
3. 预测区间之外、或完成时间超过窗口末端的弧固定为 0。
4. 只保存本轮执行区间内开始的决策。
5. 进入下一窗口后，之前保存的决策成为固定历史。

这种实现会让历史弧继续出现在原有累计约束中。因此，跨窗口在途车辆、已经承运的
货量和换装时序都由原模型自动继承，不需要另写一套容易与原约束不一致的状态方程。

## 3. 修改文件

### `config.py`

新增独立的 `RollingHorizonConfig`：

- `prediction_horizon`：每轮向未来优化多少个业务时段。
- `rolling_step`：每轮真正执行并固定多少个业务时段。

该配置与 `DeliveryConfig` 分离，并作为“算法参数”由灵敏度系统独立动态加载。
算法参数规格只交给 Rolling Horizon，精确 MIP 自动跳过。

### `optimizer.py`

新增三个公共方法：

- `decision_variable_groups()`：统一返回四组带时间弧的变量。
- `configure_rolling_window()`：固定历史并关闭预测区间外的弧。
- `extract_committed_decisions()`：提取执行区间内开始的非零决策。

同时修正自动驾驶车辆城市库存平衡约束的符号。正确含义是“累计出发量不能超过
初始车辆与累计到达车辆之和”：

```python
N_auto[1] + negative_arrivals - positive_departures >= 0
N_auto[2] + positive_arrivals - negative_departures >= 0
```

### `rolling_horizon.py`

原来的重复模型代码替换为：

- `CommittedDecisions`：保存已经执行的 `x_manual`、`y_auto`、
  `g_manual` 和 `g_auto`。
- `RollingHorizonController`：控制窗口推进、剩余时间预算、求解和历史提交。
- `RollingHorizonOutcome`：向求解器接口层返回统一结果。

传入的 `time_limit` 是整个 Rolling Horizon 过程共享的总预算，而不是每个窗口各用
一次完整预算。

### `solvers.py`

删除占位实现，注册正式的 `RollingHorizonSolver`。它将控制器结果转换为项目统一的
`SolverResult`，因此 GUI、CSV 和 JSON 输出链路无需单独修改。

Rolling Horizon 是一系列局部优化结果，`Best_Bound` 和 `MIP_Gap` 不具有与完整 MIP
相同的全局含义，因此当前输出为留空；各窗口的状态、目标值和窗口 MIP Gap 保存在
JSON 的 `detail.windows` 中。

### `main.py`

CLI 与 GUI 共用三类动态参数。可以通过 `--list-parameters` 查看算法参数，并使用
`--level algorithm.<字段名>=JSON` 覆盖灵敏度水平。

## 4. 运行方法

使用默认预测区间与步长：

```bash
conda activate pavane
python main.py --cli --scenario quick --solver rolling_horizon --time-limit 60
```

同时比较完整 MIP 和 Rolling Horizon：

```bash
python main.py --cli --scenario quick --solver all --time-limit 60
```

也可以在 GUI 中勾选 `Rolling Horizon`。

若要调整预测区间，可以在 GUI 的“算法参数”标签页修改，或通过 CLI：

```bash
python main.py --cli --scenario sensitivity --solver rolling_horizon \
  --level "algorithm.prediction_horizon=[6,8,10]"
```

也可以在代码中显式传入：

```python
algorithm_config = RollingHorizonConfig(
    prediction_horizon=10,
    rolling_step=2,
)
controller = RollingHorizonController(config, data, algorithm_config)
```

必须满足：

```text
0 < rolling_step <= prediction_horizon
```

预测区间还应足够覆盖一条完整服务链。当前模型至少需要考虑始发端人工服务、城际运输
和目的端人工服务；区间过短时，模型会因看不到完整服务链而倾向于拒单。

## 5. 结果说明

`detail.algorithm` 记录：

- 实际使用的预测区间和滚动步长。
- 已推进到的时刻。
- 是否因总时间预算耗尽而提前停止。

`detail.windows` 记录每个窗口的：

- 窗口起点、终点和提交终点。
- Gurobi 状态码。
- 求解时间和解数量。
- 有可行解时的窗口目标值与 MIP Gap。

最终 `Total_Cost` 来自最后一次一致模型：其中历史决策已经固定，尚未进入历史的决策
是最后窗口的计划。它是 Rolling Horizon 策略成本，不是完整规划期的全局下界或全局
最优性证明。

## 6. 当前版本边界

当前实现是确定性 Rolling Horizon：

- 所有订单从第一轮开始已知，预测区间限制的是可作出的运输计划范围。
- 每轮仍创建完整变量集合，再通过上下界关闭窗口外变量；这优先保证与原模型一致，
  但模型创建开销尚未缩小。
- 不模拟新订单随机到达、车辆故障或行驶时间扰动。

后续如果需要在线随机场景，可以在保持控制器接口不变的情况下增加订单可见时刻，并把
窗口外订单从当前信息集中隐藏。如果需要进一步提升大算例速度，可以让 `DataLoader`
直接生成窗口弧集合，但必须继续保留已提交跨界弧对车辆和货物流平衡的影响。

## 7. 验证记录

本次修改完成了以下验证：

1. 所有修改文件通过 `py_compile` 语法检查。
2. 默认快速算例 `T=24` 成功完成 12 个窗口。
3. 高未服务惩罚小算例 `T=16` 成功完成 8 个窗口，并产生非零运输决策：
   自动车使用量为 2、人工车使用量为 4、未服务量为 0。
