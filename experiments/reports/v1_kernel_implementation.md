# V1 纯公式与母弧内核实施记录

日期：2026-10-18（Asia/Shanghai）。Owner：Experiment。状态：**IMPLEMENTED / 22 PURE TESTS PASS / INDEPENDENT REVIEW PENDING**。本报告记录有限纯工具实施，不是完整 V1 优化模型或数值实验。

## 来源与范围

已读 [AGENTS](../../AGENTS.md)、[MODEL_CONTRACT](../../coordination/MODEL_CONTRACT.md)、[纯内核合同](../../research/theory/v1_kernel_contract.md) 和本地 [main_v2_BHH.tex](../../research/paper/main_v2_BHH.tex):67–112。直接依据为71行的 f、74行逆函数、82行 local/AV 母弧、逐弧 K 和半开占用；106/110行说明 K 在完整模型中的 selector 用途，但本轮没有建立 selector 或 wave 决策。

输入源码 SHA-256：`500BE1ADA854317BE79C358E0FF1269786DD4C7EA8E1102B501354EFF04A3457`。运行时 Git HEAD：`f26b4c96907082c2ebb7b0104263831e7100aa26`；工作树含既有未提交更改，HEAD 不是本轮源码的完整快照。

新增文件：

- [v1/__init__.py](../../intercity_delivery/v1/__init__.py)：七个公开纯函数的轻量入口。
- [capacity.py](../../intercity_delivery/v1/capacity.py)：`bhh_time(q,a,b)`、`bhh_inverse(theta,a,b)`、`local_capacity(duration,vehicles,a,b,rho,capacity)`。
- [network.py](../../intercity_delivery/v1/network.py)：`local_arcs(horizon,t0,a,b,rho,capacity)`、`av_arcs(horizon,tau_periods)`、`is_active(arc,time)`、`active_arcs(arcs,time)`。
- [test_v1_kernel.py](../../tests/test_v1_kernel.py)：22项标准库单元测试，无求解器运行。
- 本报告。

没有更改旧模型、旧容量公式、旧网络、旧 RH、配置或入口，也没有修改 TeX、历史数据或提交 Git。未采用 D1–D6 的未确认选择；没有订单、费用、城市/方向策略、全局 wave 分组、优化器或 QVI。此接口不把逐弧容量自动复制到多个结束时刻的共同车辆组。

## 数值与接口语义

输入实数限于 `numbers.Real` 且可转为有限 Python float，拒绝 bool；a、M、t0严格为正，其余实参数非负。vehicles/horizon/time/弧端点必须是非 bool 非负整数，tau_periods严格正整数。工具域扩展 K(0)=0、horizon=0空母弧；不宣称原文已明确要求这些扩展。所有参数先验证，再执行零车快捷返回。

调用者显式统一 duration、rho、a、b、theta、t0 的物理时间单位。AV 的 tau 参数是期数。没有从旧配置猜分钟/小时，也没有 stops/重量转换。

逆函数使用有理化根 `theta/(hypot(b,2*sqrt(a)*sqrt(theta))/2+b/2)` 再平方，theta=0及b=0另行处理。避免小theta与b相减的消减误差。所有已计算中间值与结果检查有限性；不支持的中间动态范围显式 ValueError，即使某个等价的更复杂计算可能给出有限最终值。本轮没有任意精度承诺；下溢沿用 Python float 规则，极小结果可为0。

local弧上界严格按实际float比值求ceil，不做epsilon、round、isclose边界修正。返回有序tuple；零容量弧保留。active_arcs验证所有输入弧并返回有序去重tuple，按 i≤t<j占用；同刻到达释放后可启动另一条弧。纯接口不自动裁剪查询时间到某个horizon。

## 已执行验证

环境：Windows PowerShell；`C:/Users/32599/.conda/envs/pavane/python.exe`，Python **3.11.15**；只用标准库。确定性手算输入，无外部数据或随机种子；无CPU/GPU/Gurobi作业预算消耗。

命令：

```powershell
& 'C:/Users/32599/.conda/envs/pavane/python.exe' -B -m unittest tests.test_v1_kernel -v
```

时间：2026-10-18约09:08（UTC+08:00）。退出码0；测试时间0.073秒，工具进程约0.180秒。原始末尾输出：

```text
Ran 22 tests in 0.073s

OK
```

主Agent另行独立重跑同一22项测试：PASS，测试时间0.043秒、进程0.243秒；此为主Agent回报的独立复核，未与上述实施者执行记录合并。已交Referee，审查结论待其独立产出。

覆盖：

- f(0/1/4)、h(0/2/6)及b=0手算值；逆函数结果直接代回源 f 式。
- theta=1e-10、1e-20、1e-100，以250位 Decimal 的原始未有理化表达式作独立oracle，防止同函数自证。
- 零tour预算、零车扩展、cap未绑定/恰绑定/严格绑定、线性stem扣除、容量不超过vM。
- local完整手算母弧、零容量弧保留、非整数ceil、3及其上下nextafter邻域；AV恰在T、tau>T及零horizon。
- 半开占用、同刻释放/启动、重复调用和排序去重；所有时间系数统一乘60或0.01的单位协变。
- 非法域、NaN/Inf/bool、分数车辆/期数、M=0、中间乘积/和/逆结果/网络比值溢出明确报错。
- 独立子进程检查导入v1不导入gurobipy、旧configuration或models。

`git diff --check`退出0，存在既有LF→CRLF提示；初末git status已检查。原有`.pytest_cache`目录权限告警与本测试无关；未扫描/清理该目录。没有依赖安装、许可证调用、解状态或MIP gap可报告。

## 结论与边界

上述测试支持“给定参数下的V1 f/h/K与正文母弧工具已实现并通过有限核验”。不验证 BHH 渐近近似精度、不验证完整 MILP 可行域或最优性，也不验证跨窗货流、费用、同start共同wave、迁入selector上界、RH真实终点可达性或QVI。独立Referee审查由主Agent在本交付后安排；本报告不代替该审查。
