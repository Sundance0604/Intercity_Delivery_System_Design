# V0-F 独立货物核验审查

2026-10-18。Owner：Referee。范围：`intercity_delivery/repositioning/cargo_validation.py`、`runner.py` 的新增接入、`tests/test_repositioning_cargo.py`，并参照 `data/loader.py`、`models/flexible_direct_optimizer.py`。本次只写本审查，不修改实现、不运行求解器。

## 初审与阻断项

**最终建议 ACCEPT：限定 V0-F 独立货物重放与真实三窗微例；初审 NaN 假阳性已修复并通过 PI 运行的纯负例。** 初审曾为 MAJOR REVISION，修复复核见末节。

| 严重性 | 证据 / 路径 | 影响 | 修复 | 状态 |
|---|---|---|---|---|
| 高（已修复的核验器假阳性） | 初版 `cargo_validation.py:_inverse`、AV/shared capacity 比较没有检查配置和计算容量 finite；修复后入口及 `_inverse` | `NaN` 参与 `amount > capacity` 比较返回 False，核验器曾可把无有效容量上界的计划标 valid | 已增加时间、服务系数、容量、比例的 finite/type/定义域检查；逆函数判别式、分子、结果与溢出显式报错；新增配置、控制标量和极端运算负例 | FIXED_ACCEPT；PI最新30项纯测试PASS，exit0 |
| 限制 | `validate_cargo` 只接收已提交事件；partial 的 full stage balance、r/q/z、全局 direct ratio 明确 deferred | partial stock 合法不说明残余期限内必可完成，也不证明在线政策全局可行/最优 | 保留范围声明；不得将 partial valid 当未来可达性证书 | ACCEPTED_LIMITATION |

Reviewer 已实际运行一次**纯 Python、无求解器**反例：使用 `cargo_fixture(2)`，令两城 service_a=20、capacity_auto=NaN；pickup/AV/delivery 均为 2，AV 车数为1。人工一期间容量3，不会先触发人工超载；初版 `validate_cargo(..., completed=True)['valid']` 实际输出 `True`。该反例验证的是错误配置绕过容量检查，不声称正常模型求解器会接受 NaN。

## 数学与测试证据判断

| 交付 / Claim | 判断 | 证据与范围 |
|---|---|---|
| V0-F 容量独立重算，没有替换成 V1 | ACCEPT，限合法数值域 | 人工容量是 `(j-i)t0` 代入 f 逆再乘 x；AV 同弧按所有订单总货量比较 capacity_auto×y；direct 使用两城 a/b 之和、扣 τ 后反算并与 capacity_direct 取 min。与 V0-F 原约束相符。没有偷偷加入 V1 stem、wave selector 或人工 M 截断；专用人工容量反例明确检查最后一点 |
| 逐单货物因果、processing 与共享容量 | ACCEPT，附数值输入修复要求 | 每个订单单独重放 pickup ready、AV departed/ready、destination departed；比较 j+theta≤t，包含同刻交接。订单身份不能互借；不同订单同弧容量先汇总。origin/destination 负例和 processing 两端负例分别触发目标错误；测试通过只表明列明的有限用例 |
| full 余额与 r/q/z 核对 | ACCEPT | 检查 pickup+direct≤d，pickup=AV=delivery；全域弧完成≤T，最终 residual=d-delivered。可选 order_summaries 分别与 pickup、direct、residual 对比。runner full 实际传模型 r/q/z；RH 仅重放事件，不伪造 summary |
| cut 与 partial 状态 | ACCEPT，限重放语义 | 拒绝尚未启动的正量承诺；已启动任务保留真实 future completion，当前 stock 只包含 cut 前事件；明列 in-progress、hub、ready、AV in-transit。新调整的 partial AV 测试确实在两期 AV 行程中间取 cut，不再把“已到达”误名为在途 |
| 知识集 | ACCEPT，组合证据 | cargo validator 用订单 release 检查 pickup/direct，不负责生成实时信息集；实际限制来自 rolling.visible_data 的 s≤current，且保留已知过期订单。runner 事后用完整 data 输出离线核验，不据此声称未来订单对控制器可见 |
| 真实三控制窗换装链 | ACCEPT，限微例 | 新真实 registry 测试 direct_ratio_max=0、单订单0.05、H=step=1、ext=2、deadline3，迫使 pickup(0,1)、AV(1,2)、delivery(2,3) 分别在三个控制窗提交；断言处理费0.2与车辆费用7合计7.2，最终 transshipment0.05、无 global bound，事件状态和 runner.detail.cargo_validation 接通。这关闭此前审查中“没有真实三段跨窗回归”的具体缺口，不证明所有配置 |
| 完整独立可行性证书 | REJECT，如作此扩张 | 本模块明确排除车可用性，须与独立车辆账本合用；partial 不验证未来可达性；数值容差为1e-6。只对给定 V0-F 输入/域的已提交事件作核验，不覆盖 V1 与任意第三方母弧/容量表定义 |

PI 告知最新 cargo 26 项通过（约0.055秒），加原34项共60个不同用例；此前完整回归56项通过。Reviewer 核读测试源码，没有重跑这些求解；数量与耗时属于 PI 提供的执行证据。进一步新增配置负例后的测试结果另记，不预先声称通过。

## 后续纯 V1 工具的独立前置判断

只读核对 `research/theory/v1_kernel_contract.md` 与 `research/paper/main_v2_BHH.tex:71–82,106–110`：f、逆函数、`min(vM,h(v[(j-i)t0-2rho]^+))`、local ceil 母弧、固定 τ AV 母弧和半开占用均有源文依据。非减 K(v)/v 论证没有偷换为全域严格凸性。合同明确费用/单位不推断、wave 分组与迁入 selector 上界不冻结；零车/零时域是明示工具扩展。

**对先实现隔离、显式参数的 f/h/K、母弧和占用纯工具，建议 ACCEPT（仅合同/实施范围）。** 这不要求先裁决 D1–D6，因为工具不组装完整可行域、不接入现有 V0-F、不作 wave 分组政策选择。仍须独立测试数值稳定、ceil 邻界和无单位偷换；本判断不是工具实现或其测试已经通过的声明。

## 修复后复核

Reviewer 已只读核实修复后的 `validate_cargo` 和新增3项纯测试方法：

- `test_nonfinite_invalid_config_never_certifies_capacity` 覆盖最初反例与其他不合法配置；
- `test_validation_control_scalars_are_strict` 拒绝 bool/NaN 等 tolerance 与非 bool completed；
- `test_inverse_overflow_rejected_instead_of_nonfinite_capacity` 拒绝支持数值范围之外的逆运算，不静默提供 NaN 容量。

该修复不改变合法 V0-F 公式，a>0 的域与现有 loader 反函数除以 a 的要求相符。静态层面原反例会在 capacity_auto 的 `_number` 检查处被拒绝。后续又补有限且有序的订单窗检查，并保留源实现允许有限实数阈值的语义，不擅改成整数订单窗。Reviewer 已阅读相关两项正负测试。

PI 最新实际执行 `CargoValidationTests` **30项PASS、exit0、测试0.003秒、进程0.126秒**，包含配置NaN/Inf、运算溢出与订单窗口负例；没有追加求解器。Reviewer 不重复执行。唯一阻断项关闭，可以推进有明确隔离边界的第二步纯工具实施。

Experiment 报告 `experiments/reports/repositioning_cargo_validation.md` 现已交付并读过：模型范围、partial deferred、真实微例和手算成本均准确；报告中的“本人只运行一次”是 Experiment 自身范围，不能读成包括 PI 的全项目只运行一次。审查时报告的26/60数量与hash尚未更新到这3项修复测试，需最终补记，属于文档证据同步，不构成新的数学阻断项。第二步纯工具的 ACCEPT 仅是实施边界许可，仍需它自己交付后的独立审查。
