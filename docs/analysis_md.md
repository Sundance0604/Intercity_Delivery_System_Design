# 仿真实验结果分析指令

当用户引用本文件（`@analysis_md`）时，你的任务是分析 `results/` 目录中的 JSON 详细结果文件，对同一批次的算例进行横向比较，并将分析结论写入一个结构化的 Markdown 报告。

## 操作步骤

### 第 1 步：扫描并分组

扫描 `results/` 下所有 `detail_*.json` 文件。文件名格式举例：

```
detail_BASE_N20_S1001_exact_mip_20260626_182742.json
detail_exp_B_100_20260114_200118.json
```

**分组规则**：

| 分组维度 | 规则 | 用途 |
|---------|------|------|
| 按时间戳 | 文件名中 `YYYYMMDD_HHMMSS` 相同的归为一批 | 同批次横向比较 |
| 按灵敏度参数 | JSON 中 `sensitivity_parameter` 字段相同 | 灵敏度分析 |
| 按 Exp_ID | 去掉 solver 后缀和时间戳后相同的 | 不同求解器对比 |

### 第 2 步：单文件解析

对每个 JSON 提取以下信息（字段含义详见 `docs/result_fields.md`）：

**元数据**：`scenario`、`experiment_id`、`sensitivity_parameter`、`sensitivity_value`、`sensitivity_level`
（旧版 JSON 可能缺少顶层元数据，需从文件名推断；`seed`、`buffer_range`、`large_order_prob` 等生成参数在 `generation_parameters` 中）

**配置**（`model_parameters`、`algorithm_parameters`、`order_parameters`）：
- 模型参数：`N_manual`、`N_auto`、`cost_manual`、`cost_auto`、`penalty_lost`、`capacity_manual`、`capacity_auto`、`T`、`t_0`、`travel_time_periods`
- 算法参数：`prediction_horizon`、`rolling_step`
- 订单参数：`num_orders`、`buffer_range`、`large_order_prob`、`small_quantity_range`、`large_quantity_range`

**订单** (`orders` 字段)：
- 订单总数、总需求量
- 正/反向订单数（`flow` = "+" / "-"）
- 平均时间窗（`latest_completion - earliest_start`）
- 大订单数（`quantity > 50`）

**解** (`solution` 字段)：
- `y_auto`：自动驾驶行程数 = `len(y_auto)`，列出具体的 `(出发时间, 到达时间, 方向)` 行程
- `z_unserved`：未服务批次数 = `len(z_unserved)`，未服务总货量 = sum(z_unserved)
- 服务率 = `1 - 未服务总货量 / 总需求量`

### 第 3 步：同批次横向比较

对同一时间戳内的所有算例：
1. **订单结构差异**：总需求、大订单比例、方向分布、时间窗分布
2. **解质量差异**：服务率、未服务量、Auto 行程数、未服务批次数
3. **配置差异**：是否同一批次内有不同的车辆数/成本参数
4. **统计汇总**：均值、最小值、最大值、极差

### 第 4 步：诊断预警

根据 `docs/result_fields.md` 第 4 节的指导，识别以下异常：

| 异常模式 | 判断条件 | 含义 | 建议 |
|---------|---------|------|------|
| 全部未服务 | 服务率 ≈ 0 且 auto行程=0 | penalty_lost < 运输成本 | 提高 penalty_lost 或降低运输成本 |
| 部分批次异常 | 同批不同seed间服务率差异大 | 订单随机性影响大 | 检查seed敏感度 |
| Auto 闲置 | N_auto > 0 但 y_auto_count = 0 且服务率低 | 自动车在当前成本下无竞争力 | 降低 cost_auto |
| 时间窗过紧 | 平均时间窗很小且服务率低 | 订单时间窗过于苛刻 | 调整 buffer_range |

## 报告结构

输出到 `analysis/result_analysis_YYYYMMDD_HHMMSS.md`，必须包含以下章节：

### 1. 报告头部

```markdown
# 仿真实验结果分析报告

**生成时间**: 2026-06-26 21:00:00
**分析算例数**: N 个 JSON 文件
**涉及批次**: M 个实验批次
**数据来源**: results/ 目录
```

### 2. 数据集概览

- 场景分布统计表（每个 `scenario` 的算例数）
- 灵敏度参数分布统计表（每个 `sensitivity_parameter` 的算例数，含水平数）
- 求解器分布统计表
- 批次时间线（按时间倒序列出所有批次）

### 3. 按批次分析（核心章节）

对每个批次（按时间倒序），依次输出：

#### 3.x.1 批次概览

```markdown
### 批次 X: YYYYMMDD_HHMMSS

**场景**: quick | **灵敏度参数**: model.cost_auto（如为灵敏度实验） | **算例数**: N | **求解器**: exact_mip
```

#### 3.x.2 算例结果总览表

```markdown
| Exp_ID | Seed | 订单数 | 总需求 | 大订单数 | 平均时间窗 | 服务率 | 未服务量 | Auto行程数 | 未服务批次数 |
|--------|------|--------|--------|---------|-----------|--------|---------|-----------|------------|
| `BASE_N20_S1001` | 1001 | 20 | 804 | 4 | 7.3 | 0.0% | 804 | 0 | 20 |
```

#### 3.x.3 求解配置参数表

```markdown
| Exp_ID | N_manual(1/2) | N_auto(1/2) | cost_manual | cost_auto | penalty_lost | capacity_auto |
|--------|---------------|--------------|-------------|-----------|-------------|---------------|
```

#### 3.x.4 批次统计汇总表

```markdown
| 指标 | 均值 | 最小值 | 最大值 | 极差 |
|------|------|--------|--------|------|
| 总需求 | 804 | 804 | 804 | 0 |
| 服务率 | 0.0% | 0.0% | 0.0% | — |
| 未服务总量 | 804 | 804 | 804 | 0 |
| Auto行程数 | 0 | 0 | 0 | 0 |
| 未服务批次数 | 20 | 20 | 20 | 0 |
```

#### 3.x.5 订单结构对比

```markdown
| Exp_ID | 正向(+) | 反向(-) | 平均时间窗 | 大订单比例 | 平均每单需求 |
|--------|---------|---------|-----------|-----------|-------------|
```

#### 3.x.6 批次诊断

用引用块（`>`）输出诊断结论，指出：
- 该批次整体表现如何
- 是否存在异常算例
- 如果在灵敏度分析场景，参数变化对结果的影响趋势

### 4. 灵敏度参数横向对比

按 `Sensitivity_Parameter` + `Sensitivity_Level` 字段聚合（不再依赖 `Scenario` 名称），常见参数映射：

| `Sensitivity_Parameter` | 横轴参数 | 推荐纵轴 |
|------------------------|---------|---------|
| `model.N_auto` | N_auto | 服务率、Auto行程数、未服务量 |
| `model.cost_auto` | cost_auto | 服务率、Auto行程数 |
| `model.N_manual` | N_manual | 服务率、未服务量 |
| `order.buffer_range` | buffer_max | 服务率、未服务量 |
| `order.large_order_prob` | large_order_prob | 服务率、未服务量、总需求 |

灵敏度算例编号示例：`SENS_MODEL_COST_AUTO_L2_S3001` = 测试 `model.cost_auto` 第 2 水平，seed=3001。

为每个灵敏度场景输出聚合表：

```markdown
| 参数水平 | 算例数(seed数) | 平均服务率 | 平均未服务量 | 平均Auto行程 | 平均总需求 |
|---------|--------------|-----------|-------------|------------|-----------|
```

如果某参数水平有多个 seed，取均值，并注明 seed 数。

### 5. 全局诊断与建议

```markdown
## 全局诊断与建议

| 指标 | 数量 | 占比 |
|------|------|------|
| 完全未服务算例 (服务率<1%) | X | X% |
| 完全服务算例 (服务率>99%) | Y | Y% |
| Auto行程为0的算例 | Z | Z% |

### 综合建议
- ...
- ...
```

对于占比异常的指标，给出具体建议，引用 `docs/result_fields.md` 对应章节。

## 格式要求

1. **数值精度**：服务率用百分比保留 1 位小数（`73.2%`），需求量保留整数，小数保留 2 位
2. **表格对齐**：所有 Markdown 表格正确对齐
3. **Exp_ID 用反引号**：如 `` `BASE_N20_S1001` ``
4. **诊断用引用块**：`>` 开头
5. **百分比符号不可省略**：服务率和大订单比例必须带 `%`
6. **文件引用**：指代 JSON 文件时用文件名（不含路径），如 `detail_BASE_N20_S1001_exact_mip_20260626_182742.json`

## 注意事项

- **只读 results/ 目录**：只分析结果 JSON，不读源代码
- **兼容新旧格式**：旧版 JSON 缺少顶层字段时从文件名推断
- **无数据时说明**：如果某个分析维度没有数据（如没有灵敏度场景），明确写"（无数据）"而非留空
- **先读 1-2 个样本**：在批量分析前，先读 1-2 个 JSON 文件确认结构，再批量处理
- **报告保存后告知**：输出保存路径，并口头总结 3-5 条最重要的发现
