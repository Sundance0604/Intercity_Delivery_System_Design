# V1 P0 合成参数设计（预注册冻结 v1）

日期：2026-10-18。对应机器参数：`experiments/specs/v1_p0_capacity_design.json`。
状态：**frozen_synthetic_parameter_design_v1；PI 在 P0 授权范围内预注册冻结合成基线**。
一票一合成任务等效单位已由作者确认，决定编号
`V1P0-JOB-UNIT-2026-10-18`。本文不授予正式实验或求解预算。

## 来源和范围

已读 `coordination/MODEL_CONTRACT.md`，特别是第 8、9、12 节。
容量依据 `research/paper/final_manuscript/sections/service_capacities.tex`
的 `mod:local-capacity`、`mod:direct-duration`；并用
`intercity_delivery/v1/capacity.py` 的纯函数核验。本设计不修改生产代码，
不改变 direct 端点，不运行 Gurobi，不研究随机 QVI。

**数值观察（纯容量计算）**：既有 smoke 的 a=.05、b=.1、rho=5、M=4、
60 分钟 period 给 f(M)=.4，而最短 local 弧预算 B=50。
故所有正整数 v 的 pooled 与 separate 容量都是 4v，不能观察容量 pooling。
下列参数是合成机制辨识设计，并非 CFS 实测校准。

## 单位与冻结基线

一个 model unit 定义为一个合成 pickup–delivery job-equivalent；每个单位
在取货和配送端各产生一次等效服务工作。需求连续可分仍沿用论文订单类约定，
不把实际不可分 shipment 宣称为可拆分实物。CFS 重量仅作来源/分层属性，
不直接等于工作量。作者已选择每条源 shipment 映射为一个合成任务等效量；
该确认不把下述数值系数提升为实证校准。

- 时间：period=60 min，tau=9 periods，T=24 periods。
- rho=5 min；M=4 job-equivalents/人工车；M_A=8 job-equivalents/AV。
- a=5 min/job-equivalent；基线 b=20 min/sqrt(job-equivalent)。
- 合成人工/AV/direct 费率为 20/15/25 currency/vehicle-hour；empty 暂同 direct。
- 冻结拒单罚金 100 currency/job-equivalent；50/100/200 为预定敏感性范围。
- 冻结基线车队每城人工 6、AV 2；4/6/8 人工车每城仅是计划敏感性范围，
  未经运营求解，不能称为已经产生资源瓶颈的车队设置。
  容量图 v=5 需要可用至少 5 车；它不代表模型额外配置车辆。

**数值观察（手算）**：基线一单位货、两端各一个 1-period local 任务、
一辆 9-period AV 的单链车辆成本为 2*20+9*15=175；若 local 和 AV 均满载，
该成本可由多单位分摊。p=100 处于两者之间只是先导探针，
不保证最优解一定同时服务与拒单。direct、可用车辆、期限与其他订单均可能改变选择。
这些费率、罚金、车辆数没有实证标定，不可直接进入论文作为现实估计。

## 三种一小时 local 容量情景

固定 a=5、rho=5、M=4、D=60，从而 B=50；只改变 b。
冻结主基线为 near_transition。对照仅替换 local 容量；不同时替换 direct 公式。

|情景|b|f(M)|机制|
|---|---:|---:|---|
|routing_binding|40|100|v=1,2,3,5 下 pooled 仍由 routing 时间限制|
|load_binding|10|40|所有 v>=1 pooled/separate 均为 vM，零差异负对照|
|near_transition|20|60|v=1 有 routing 限制，v>=2 pooled 达 vM，separate 尚未达 cap|

容量值保存在 JSON，不以结果图宣称完成运营实验。

**已证命题（上述 f 和固定弧假设）**：
`K_pool(v)=vM` 当且仅当 `aM+b*sqrt(M)/sqrt(v)<=B`。
证明：f 严格递增，`f^{-1}(vB)>=vM` 当且仅当 `f(vM)<=vB`，两边除以 v。
separate 达 cap 当且仅当 `f(M)<=B`。
当 B>aM、b>0 时，pooled 的连续临界车辆数是
`v_star=(b*sqrt(M)/(B-aM))^2`；整数饱和阈值取其向上整数。
本设计三情景依次为 v_star=64/9、4/9、16/9，故对应整数阈值 8、1、2。
这解释 v=5 在 routing_binding 下仍未饱和；该标签并不声称 v>=8 时仍受 routing 限制。

**待验证预测**：near_transition/routing_binding 的容量差在需求足够、车队或期限
绑定时可能改善运营成本或服务率；容量差本身不保证最优调度出现严格改善。
两算法相同成本不反驳容量公式，也可能是需求不足或其他瓶颈。

## 时间链与异城扩展

**已证命题（只论时间支持）**：三个情景均有一车一单位 local 时长
`ceil((2rho+f(1))/60)=1`，故 collection→AV→distribution 最短完整链为
1+9+1=11 periods。一车满载 4 单位时 routing_binding/near_transition 的 local
时长为 2 periods，故其完整链为 2+9+2=13 periods；load_binding 的满载 local
仍为 1 period，完整链为 11 periods。最长 13-period 时间需求不构成车辆库存或全实例可行性证明。
基线 direct 若采用原聚合时间式，一车一单位为 600 min=10 periods；
这只是式内计算；作者后续确认采用目的城市级抽象可用节点，不将该计算解释为已建模全部真实hub接驳。

冻结的 P1 先导 RH 设计为 H=12,h=2,E=12；对未触及真实终点的窗口，最晚 origin start
i=b_t-1 的 13-period 链满足 i+13=b_t+12=c_t。
若真实 T 截断或 deadline 更早，仍需逐订单逐链检查，不能靠 H+E 宣称全部链存在。
旧 H=8,E=6 虽有 H+E=14，但最晚 i=b_t-1 的 13-period 链会超过 c_t；
它不能覆盖所有新 origin 起点。此处登记 P1 先导设计，不改既有生产 RH 默认。

异城待审情景保留 city1=(a=5,b=20,rho=5)，city2=(a=6,b=24,rho=8)。
city2 一单位 local 为 46 min，满载 local 为 88 min，因而相同 11/13-period
时间链仍可用。必须逐城算 K，direct 必须按方向使用 origin 的 stem 与 zoned 项、
destination 的未分区项；不以平均系数替代。该情景在 JSON 中单列，不混入三情景的单变量对照。

## 核验边界

`tests/test_v1_p0_capacity_design.py` 独立用二分原 f 不等式核对容量表，
核对饱和机制、手算锚点、所有图点的原时间/载量约束、链时长和异城区别。
这是纯容量数学检查，不是完整模型实验、routing 实测或对全部 RH 实例的可行性证明。
参数已预注册冻结，仍为合成机制设计，不升级为实证校准。
作者后续已明确确认T2最低平均装载m=.1及.05/.1/.2敏感性，见`V1P0-T2-MINLOAD-2026-10-18`；
该平均约束不证明逐车非空。此确认不改变原容量曲线或合成参数的证据等级。
正式运行前仍需端点冻结、固定需求与样本、独立 replay、运营量级预算和审查。
