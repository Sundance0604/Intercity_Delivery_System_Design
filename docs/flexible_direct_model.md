# 直送—换装协同模型实现说明

## 1. 文件与接口

核心模型为 `flexible_direct_optimizer.py`，提供两个统一求解器：

```text
flexible_direct_mip
flexible_direct_rolling
```

程序化构建：

```python
from flexible_direct_optimizer import FlexibleDirectOptimizer

optimizer = FlexibleDirectOptimizer(config, data).build_model()
optimizer.model.setParam("TimeLimit", 60)
optimizer.model.optimize()
```

也可以通过统一求解器接口调用：

```python
result = SOLVER_REGISTRY["flexible_direct_mip"].solve(
    config, data, orders_tuple, 60, rolling_config
)
```

## 2. 业务假设

1. 换装方式为“始发城市人工集货—自动驾驶干线—目的城市人工配送”。
2. 直送方式由同一人工驾驶车辆完成取货、跨城运输和最终配送。
3. 直送车辆来自城市人工车队，抵达后进入目的城市车辆库存。
4. 订单批次货量允许拆分成直送、换装和未服务三部分。
5. 直送比例按已服务货量计算；默认上下界 `[0,1]`，由模型内生决定。

## 3. 新增参数

参数位于 `DeliveryConfig`，且灵敏度规格只会交给两个直送求解器：

| 参数 | 含义 |
|---|---|
| `direct_travel_time_periods` | 人工直送跨城时间 |
| `capacity_direct` | 直送车辆容量 |
| `cost_direct` | 直送车辆单位时间成本 |
| `transfer_time_periods` | 两端换装处理时间 |
| `transfer_cost_per_unit` | 单位换装货量处理成本 |
| `direct_ratio_min` | 已服务货量中的最低直送比例 |
| `direct_ratio_max` | 已服务货量中的最高直送比例 |

## 4. 变量

| 变量 | 含义 |
|---|---|
| `x_manual` | 城市内集货/配送人工车辆 |
| `y_auto` | 自动驾驶干线车辆 |
| `w_direct` | 人工跨城直送车辆 |
| `g_manual` | 换装货量的城市人工运输部分 |
| `g_auto` | 换装货量的自动干线部分 |
| `h_direct` | 直送弧上的订单货量 |
| `r_transshipment` | 每笔订单的换装货量 |
| `q_direct` | 每笔订单的直送货量 |
| `z_unserved` | 每笔订单的未服务货量 |

## 5. 约束组

代码中每一组约束均有业务注释和独立名称：

| 编号 | 含义 |
|---|---|
| FD-(2) | 人工城市任务与直送任务共享车队，并跟踪跨城后的车辆位置 |
| FD-(3)–(5) | 自动车在途规模与两城市库存平衡 |
| FD-(6) | 城市人工服务弧总容量 |
| FD-(7) | 自动车干线弧总容量 |
| FD-(8) | 直送弧总容量 |
| FD-(9)–(10) | 换装链与直送链时间窗 |
| FD-(11)–(13) | 始发换装、干线、目的换装时序及干线货量守恒 |
| FD-(14)–(17) | 始发/目的人工货量、直送货量与订单需求拆分 |
| FD-(18)–(19) | 总体直送比例上下界 |

直送单车容量按“始发城市服务时间 + 人工跨城时间 + 目的城市服务时间不超过直送弧
持续时间”预计算。

## 6. CLI 和 GUI

```bash
python main.py --cli --scenario quick --solver flexible_direct_mip --time-limit 60
python main.py --cli --scenario quick --solver flexible_direct_rolling --time-limit 60
```

固定 50% 直送比例：

```bash
python main.py --cli --scenario sensitivity --solver flexible_direct_mip \
  --level "model.direct_ratio_min=[0.5]" \
  --level "model.direct_ratio_max=[0.5]"
```

GUI 会自动显示两个新求解器和七个直送模型参数。

CSV 和 JSON 统一输出 `Direct_Ratio`、`Direct_Volume` 和
`Transshipment_Volume`，便于直接绘制机制比较和比例敏感性曲线。

## 7. 测试

```bash
python -m unittest tests.test_flexible_direct_optimizer -v
```

测试覆盖低直送成本产生非零直送，以及完整 MIP、Rolling Horizon 都能满足固定 50%
直送比例。

## 8. 当前边界

- 当前按货量允许订单拆分；订单只能选择一种方式时，应增加订单级二元变量。
- 始发和目的换装暂时共用同一个处理时间参数。
- 直送成本、换装成本和处理时间需要使用企业数据校准。
- 直送 Rolling Horizon 当前属于所有订单预先已知的确定性滚动优化。
