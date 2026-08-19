# 可视化仿真实验界面操作说明

## 1. 启动

```bash
conda activate pavane
python main.py
```

## 2. 界面布局

界面分为三个横向区域：

1. 顶部控制栏：实验场景、求解方式、种子数、时间限制和运行按钮。
2. 中部参数区：模型参数、算法参数、订单参数三个宽幅标签页。
3. 底部输出区：实验计划预览和运行日志左右并排。

参数区占据窗口主要空间。模型参数较多时会自动分为左右两组，并支持滚动。

## 3. 实验场景

| 场景 | 含义 |
|---|---|
| `quick` | 使用三类配置的基准值生成一个快速连通性算例 |
| `sensitivity` | 对三类配置的全部字段执行单因素灵敏度分析 |

## 4. 求解方式

| 求解器 | 适用参数 |
|---|---|
| 精确 MIP | 模型参数、订单参数 |
| Rolling Horizon | 模型参数、算法参数、订单参数 |
| 直送-换装协同 MIP | 模型参数、订单参数，包括直送专属参数 |
| 直送-换装协同 Rolling Horizon | 模型参数、算法参数、订单参数 |

算法参数规格只运行 Rolling Horizon 类求解器；直送专属参数只运行两个直送模型
求解器。预览中的“实际求解次数”已经扣除不适用组合。

### 4.1 两种论文 Solution Approach

两种解法都运行在论文专用 Rolling Horizon 控制器中：

| GUI 名称 | 内部名称 | 窗口内处理 |
|---|---|---|
| Rolling Horizon：剪枝 | paper_candidate_mip | 生成状态相关候选网络并求解剪枝后的 MIP |
| Rolling Horizon：剪枝 + 生成解 | paper_priority_heuristic | 动态优先级构造 MIP Start，再求解相同的剪枝 MIP |

### 4.2 真实 CFS SQLite 与城市对

1. 选择“真实数据（CFS SQLite）”。
2. 点击“浏览”选择数据库；GUI 在后台读取 shipments 表。
3. 界面显示实际列名及 SQL 类型。
4. 先选择城市 1；城市 2 只列出与城市 1 双向均有记录的都市区。
5. 界面显示两个方向的原始记录数。
6. 运行时根据订单数和随机种子从该城市对直接抽样，不需要预生成 JSON。

## 5. 三类动态参数

参数由 `intercity_delivery/configuration.py` 中三个 dataclass 自动加载：

- 模型参数：`DeliveryConfig`，参数键为 `model.<字段名>`。
- 算法参数：`RollingHorizonConfig`，参数键为 `algorithm.<字段名>`。
- 订单参数：`OrderGenerationConfig`，参数键为 `order.<字段名>`。

新增 dataclass 字段后，界面会自动出现对应输入项，无需修改 GUI 代码。

每个输入框填写灵敏度水平的 JSON 数组：

| 类型 | 示例 |
|---|---|
| 整数 | `[12,24,36]` |
| 小数 | `[0.15,0.3,0.45]` |
| 区间 | `[[0,2],[0,5],[0,8]]` |
| 城市字典 | `[{"1":10,"2":10},{"1":30,"2":30}]` |

每个参数旁边会显示基准值。灵敏度分析每次只改变一个参数，其余参数使用基准值。

## 6. 推荐流程

1. 勾选 `quick` 和需要的求解器，运行一次环境检查。
2. 勾选 `sensitivity`，先将种子数设为 `1`。
3. 调整三个标签页中的参数水平并查看预览。
4. 确认规格数和实际求解次数合理后开始试跑。
5. 正式实验再把种子数增加到 `3`、`5` 或更高。

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| 参数无法解析 | 检查是否为合法 JSON 数组 |
| 实际求解次数少于规格数×求解器数 | 精确 MIP 自动跳过算法参数规格，属于正常行为 |
| Rolling Horizon 未完成全部窗口 | 总时间预算可能耗尽，检查 JSON 的 `detail.windows` |
| 全部订单未服务 | 检查未服务惩罚与车辆成本量纲 |
| 界面运行时短暂无日志 | SQLite 可能正在加载城市目录，或 Gurobi 正在处理当前窗口 |
| 城市 2 列表为空 | 所选城市 1 在 SQLite 中没有双向都市区记录 |

## 8. CLI 等价入口

CLI 与 GUI 共用同一实验核心：

```bash
python main.py --list-parameters
python main.py --cli --scenario sensitivity --solver all --seeds 1 --dry-run
python main.py --cli --scenario sensitivity --solver rolling_horizon \
  --level "algorithm.prediction_horizon=[6,8,10]"
```
