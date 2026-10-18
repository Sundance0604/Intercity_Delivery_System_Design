# V1-M 最小决策规格包独立审查

日期：2026-10-17（Asia/Shanghai）。Owner：Referee。

**规格准备：ACCEPT。作者确认：PENDING。实施授权：PENDING。**

审查对象为`coordination/V1_M_DECISION_SPEC.md`。直接读取AGENTS、MODEL_CONTRACT、MIGRATION_PLAN、前轮review；逐行核对`research/paper/main_v2_BHH.tex:67–137`与实际`intercity_delivery/models/base_optimizer.py`、`data/loader.py`、`configuration.py`、`experiments/solvers.py`。仅手算，不运行Python、测试、solver、实验或联网。本审查接受有边界的静态规格准备，不接受“已经迁移/测试通过/模型创新得到证明”等无证据结论。

## Findings

### R1 [Major / 作者决策依赖] 选项待批，不等于阻断规格准备

- **Evidence**：规格第1、3、8节；源67–71、77、82、134。D1–D6均OPEN。
- **Impact**：offline全给定订单、arc-wave、分钟/小时、非负车辆域及受限合成输入属于明确待批解释或范围。不能由Referee替作者采用。源同start语言与逐arc公式的张力仍存在。
- **Remedy**：作者逐项书面决定并批准实施范围/预算后再冻结；D2-A若获批，登记为正文逐arc公式解释，不登记为原稿已改。
- **Status**：披露充分；作者确认与实施授权仍PENDING。**Verdict：ACCEPT（规格披露）；PENDING（确认/授权）。**

### R2 [Minor / 已解决] 输入域和F5参数需完整

- **Evidence**：初稿D6未逐项列明正t0/整数fleet等；经反馈，PI已在D6明确有限参数、t0/a/M/Mhat正值、b/rho/d/费用/罚金非负、两类N非负整数及整数s/e，拒绝静默clip/round；F5明确继承F4分钟配置并全车队置0。
- **Impact**：未来验证器可依据单一清单拒绝t0=0、分数fleet等未定义输入。M/Mhat>0及整数时间窗是待批受限基准，不可冒称源文全部输入域。
- **Remedy**：保留待批标签；实施按批准版本测试边界。
- **Status**：小修已解决。**Verdict：ACCEPT（输入合同准备）；采用政策仍待作者确认。**

### R3 [Verified] 公式—源码矩阵准确

- **Evidence**：源74逆函数，82弧域/半开占用/K，84目标，89/93车队，97/101 AV库存，106/110波容量与one-hot，114共享AV，118/137时间窗，122/126累计前置关系，130两端需求，134域，均准确映射M01–M14。实际`base_optimizer.py:32–69,72–88,95–278`、`loader.py:42–83,85–104,128–180`、`configuration.py:161–189,245–250`。
- **Impact**：旧单车容量乘x、无u、整数z和逐订单AV容量不能原样复用为V1。旧循环多为range(T)，新规范按0…T核验，不能仅复制循环。实际实验代码在`intercity_delivery/experiments/`，规格路径纠正正确。
- **Remedy**：未来逐式实现/验收，重点检查t=T、空selector、同刻到达再发。旧逆函数可参考不代表旧弧域/stem/cap正确。
- **Status**：静态核验完成，无实现或运行证据。**Verdict：ACCEPT（映射）。**

### R4 [Verified] 单位与零车/全拒单条件正确

- **Evidence**：规格第3–5节；源84/106/110/130/134；实际`base_optimizer.py:79,83`及`configuration.py:245–250`。
- **Impact**：D3采用分钟/小时则c_source=c_hour/60，车辆费只换算一次；不得源费率数值不变又双重换算。D4非负整数域下N=0导致空selector、x=0及当地货流0。全拒单全部车/流/u为0、z=d可行，依赖非负需求/初始fleet且无强制服务/非零历史等额外约束；不推出全拒单最优。
- **Remedy**：保留条件，不给AV擅加人工终端归位；独立核算单位及目标三项。
- **Status**：手工代入成立。**Verdict：ACCEPT（限定代数）。**

### R5 [Verified] F1–F5手工预期成立

- **Evidence**：规格第6节；源82/89–134。
- **Impact / checks**：F1 Delta=2rho使K=0，空弧未被禁止。F2/F3正货必须三条至少一期的腿，T=3时AV只能(1,2)；一AV且Mhat=1给服务总量≤1，两端local各容量2。F2最少拒0.5，给出的对半服务可行但非唯一；F3服务1/拒0.5，最优罚金0.5。F4车辆费2×0.5×10+0.5×4=12；分钟重标仍12，单位服务费12低于罚金100，故全服务最优；e=2时三正时长腿无法完工，全拒100。F5无车且d=0.05，z=0.05、成本0.1。D2的等load/等tour-time示意亦手算正确，只比较具体规则，不比较所有start-wave定义。
- **Remedy**：未来用这些值作独立oracle，不强求F2唯一分配；不得把手算或赋Start当solver可行性证据。
- **Status**：仅验收设计，未执行测试或求解。**Verdict：ACCEPT（设计）。**

### R6 [Verified] 完整导出要求必要且不是源新增约束

- **Evidence**：实际`intercity_delivery/experiments/solvers.py:114–118`仅导出y/z并筛v.X>0.1；规格M15和第7节。
- **Impact**：若V1复用旧导出规则，F5的0.05会遗漏；当前ExactMIPSolver配套V0-M的z为integer，此处不是V0运行中发现的小量丢失。缺x/u/g/ghat无法重算V1可行性。
- **Remedy**：新V1完整导出变量索引/数值/容差并独立核算。若稀疏省零，明确缺键=0，不能吞合法正小量；不改V0出口以免扩大范围。
- **Status**：静态缺口核实；新导出未实现。**Verdict：ACCEPT（缺口/补救设计）。**

### R7 [Verified] direct/RH/S/V2延期合理

- **Evidence**：规格第7–8节；MIGRATION_PLAN及前轮R1/R4/R8。
- **Impact**：mandatory人工车不跨城，无manual direct空返；全域模型没有start/ext窗口，因此V1-F迁入上界/人工恢复和RH reveal/history/terminal义务可延期。AV空间库存与同刻交接仍须保留。offline参考不等价于在线策略的信息集。
- **Remedy**：获批后只推进G1及G2/对应结果层；不得宣称整个G0或G3–G9完成。
- **Status**：范围收敛可接受。**Verdict：ACCEPT（限定路线）。**

## 保护与停止

开始Git状态已有大量untracked文件及`.pytest_cache`权限警告，不称工作树干净。唯一可写路径为本报告，其他文件只读。报告完成后再次查看Git状态；并行PI的授权编辑不归为Referee成果。不改论文/代码/历史结果，不commit。审查到此停止，交付PI综合。
