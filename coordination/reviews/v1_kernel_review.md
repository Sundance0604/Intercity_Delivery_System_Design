# V1 纯数学内核独立审查

2026-10-18。Owner：Referee。对象：`intercity_delivery/v1/{__init__,capacity,network}.py`、`tests/test_v1_kernel.py`，参照 `research/theory/v1_kernel_contract.md` 与 `research/paper/main_v2_BHH.tex` 实际源码。只修改本审查文件，不修改实现、论文或试验结果，不运行求解器。

**建议 ACCEPT：限定为源公式和母弧的隔离纯工具。** 未发现公式、stem/cap、离散母弧或半开占用的阻断错误。没有完整 V1-M/F、wave 分组决策、RH 策略或 QVI 实现证据；本审查也不批准 D1–D6 的完整模型解释。

## 严重性、证据、影响与处理

| 严重性 | 证据 / 路径 | 影响 | 处理 | 状态 |
|---|---|---|---|---|
| 数值限制，非阻断 | `capacity.py` 模块声明及 `bhh_inverse` 的 float 平方 | 极小正 θ 的数学 h(θ)>0，但浮点结果可能归零；Reviewer 实测 `bhh_inverse(1e-200,1,1)==0.0`。不应将返回0推为数学上的零需求或无容量定理 | 已声明 underflow 遵循 Python float；使用者须选择可解析的参数/货量范围，需要更小量时另选高精度实现。未来可加固定 underflow 例明确该接口行为 | ACCEPTED_LIMITATION |
| 数值限制，非阻断 | `capacity.py:_finite`、`local_capacity`、`network.py:local_arcs` | 即使输入都有限，有限 K 或有限母弧有时也会因中间量溢出被拒绝。例如 K 的 cap 已饱和时，当前仍可能先拒绝巨大的 v×tour budget；并不保证计算所有数学可表示结果 | 已明确采用“中间值超支持范围则 ValueError”的合同；可以未来优化 cap 先判定或比例缩放，但不能声称当前支持任意有限输入 | ACCEPTED_LIMITATION |
| 范围限制，非阻断 | 仅有纯函数与显式参数，无订单、配置默认或 solver | K 表本身不选择 wave 分组、不决定迁入上界、费用单位或数据 stop 语义 | 保持与 D1–D6、完整模型组装分离 | ACCEPTED_LIMITATION |

## 逐项核对

| Claim / deliverable | 判断 | 依据 |
|---|---|---|
| 原 f/h 公式 | ACCEPT，限公开浮点域 | TeX:71、74 的 f(q)=aq+b√q、逆函数被正确实现。逆使用 `theta/(hypot(b,2sqrt(a)sqrt(theta))/2+b/2)` 后平方，代数等价，避免相减取消；b=0与theta=0单独处理 |
| K(v)、stem与cap | ACCEPT | TeX:82 的 `min(vM,h(v[max(duration−2rho,0)]))` 匹配；先减一次每车 stem 再乘 v，没有套旧 x×h(duration)。手算 cap 切换、线性 b=0、零 tour budget、K≤vM 有独立预期 |
| local 母弧 | ACCEPT | 按 TeX:82 `ceil((2rho+f(M))/t0)` 约束 j−i，保留零容量弧；没有订单过滤，未以 floor 或 epsilon 削减界。`nextafter` 的上下邻界测试可识别错误宽容取整 |
| AV 母弧与占用 | ACCEPT | τ 是期数，不再次除 t0；弧在0…T内，允许j=T；活动判定 i≤t<j，包含到达时释放并启动下一任务的同刻边界 |
| 独立 oracle 与数值测试 | ACCEPT，有限样例 | 手算 f/h 和 K；直接代入源 f；250位 Decimal 用原相减式核对小 theta；单位协变；ceil 邻界；overflow/domain负例。测试不只重复调用被测实现比较自己 |
| 工具域扩展 | ACCEPT，明示扩展 | v=0→K=0、T=0→空弧集；a>0、b≥0、rho≥0、M/t0>0与整数期数被检查；拒绝bool。工具扩展不自动裁决完整 MILP 的整数零值/selector规则 |
| 隔离与旧基线兼容 | ACCEPT，限静态依赖证据 | `__init__.py` 只导入本模块；独立子进程断言没有载入 Gurobi、旧 configuration 或 models。Reviewer 搜索 `intercity_delivery` 与 `main.py` 未见生产代码导入本 v1 工具，旧容量尚未替换 |
| 完整 V1 对齐或创新理论已完成 | REJECT，如提出此主张 | 未实现 one-hot变量/联结、完整货流、车辆约束、目标、RH、QVI；纯工具测试无法证明这些性质。没有新增文献核验或理论创新证明 |

## Reviewer 实际执行

```powershell
& 'C:/Users/32599/.conda/envs/pavane/python.exe' -B -m unittest tests.test_v1_kernel -v
```

Reviewer 实际输出：**22 tests，0.035秒，OK，exit0**；工具进程约0.215秒。包含独立子进程的依赖隔离断言，没有建模或优化。另用一次纯 Python 查询确认上述 underflow 例输出0.0。没有添加实验结果文件。

这只是有限输入 oracle 与防线验证，没有穷尽浮点域、一般误差界或百万弧规模/内存评估。尤其不能从单位协变有限样例宣称任意极端单位缩放都不受浮点范围影响。

## 实施报告证据

`experiments/reports/v1_kernel_implementation.md` 已交付，Reviewer 已完成只读核对。API、源公式定位、数值域/下溢/中间溢出声明、ceil 语义、隔离边界与实际代码及测试相符。实施者自述22项测试0.073秒，PI回报另行22项0.043秒；这两次执行分别标明来源，没有冒充 Reviewer 的0.035秒执行。本报告直接依据已交付代码、源合同及 Reviewer 亲自执行的22项纯测试，最终建议 **ACCEPT**。

实施报告创建时的“独立审查待完成”是交付顺序状态，现由本报告完成审查闭环；PI 可在综合文档中更新其状态。没有发现需修改实现或追加求解的事项。源文件/受保护基线的最终散列和本地链接总检查由 PI 收尾提供，本审查不预先代报其结果。
