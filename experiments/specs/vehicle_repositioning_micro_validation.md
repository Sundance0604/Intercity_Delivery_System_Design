# 独立空车调拨变体：有界实施与验证计划

日期：2026-10-18。用户已授权执行并在额度恢复后继续；本计划限定本次检查范围，不启动大型实验。

## 模型与实现边界

- 模型标识：`v0_f_explicit_empty_v1`，沿用V0-F货物/容量结构，另加独立empty网络、计费和人工库存；不是完整V1 wave/stem迁移。
- 默认空行时长/费率显式沿用`DeliveryConfig.direct_travel_time_periods`和`cost_direct`，API可单独覆盖，结果记录来源。
- 默认真实终点目标为None；任何计算窗口均无人工复位要求。完整时域API可选真实T目标；rolling暂拒绝显式target，防止伪装已具备终点可达性保障。
- 统一不按历史库存删弧；新任务仍受有限完成域约束，已承诺任务保留绝对历史，不宣称支持任意越cut新任务。
- 新registry名称`flexible_direct_empty_mip`、`flexible_direct_empty_rolling`为显式选择；旧名称、旧类及CLI默认/`all`展开不变。

## 代码所有权

| Owner | 唯一写入范围 |
|---|---|
| Experiment-model | 新包optimizer.py/rolling.py/runner.py/__init__.py及optimizer/rolling新测试 |
| Experiment-ledger | 新包config.py/ledger.py及ledger新测试 |
| PI | registry/CLI/参数适用名的增量接入、integration新测试、协调与说明文档 |
| Referee | 专业交付后新增独立实施审查，其他源文件只读 |

## 运行预算与顺序

1. 纯标准库账本测试和无solver的registry/CLI测试优先；仅运行新增有意义检查，禁用字节码/pytest缓存。
2. 微型Gurobi仅合成小实例：T≤8、少量订单/车辆；各测试求解上限2秒、Threads=1，单次总solver预算不超过60秒。新rolling测试的整个run预算也限定2秒；不执行旧全套30秒求解用例。
3. 先编译/导入，再统一运行新optimizer/rolling测试；失败修正后只复测相关组，记录新增求解次数与剩余预算。达预算立即停止，不自动扩大规模。
4. 运行环境：`C:/Users/32599/.conda/envs/pavane/python.exe`，已探测Gurobi 10.0.3许可证可用。许可证失败标unavailable，不能作通过。

## 必要验收

- 无订单也存在两方向empty弧，库存不足禁止发车；在途不能提前复用，到达时可用。
- empty费用只计一次，不产生货量、处理费或direct share；小额连续量不按0.1丢弃。
- 计算窗口结束允许异地/在途；完成迁移在多窗后仍保留；历史empty车数整数固定，不作货流松弛。
- 完整时域可选真实target，与默认无target行为区分；rolling显式target拒绝并解释安全尾部未支持。
- 控制窗仅提交实际开始的任务；未revealed订单不提前优化服务；partial总目标为空，窗口bound不作全域bound。
- 完成策略的成本从实际提交账本独立核算，完整模型目标与独立核算比较；结果可JSON保存并含配置来源/模型范围。
- 旧optimizer、算法、数据和论文散列保持一致；接入文件只增新名称/适用元数据，不改变旧分派。

本计划不授权真实数据校准、长Gurobi实验、旧结果覆盖、论文更改、Git提交或推送。实际执行记录见后续验证报告。
