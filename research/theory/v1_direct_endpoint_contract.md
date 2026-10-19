# Direct 完成节点与后续服务合同

2026-10-18；作者确认后的有限解释。依据 `coordination/MODEL_CONTRACT.md`
及 `research/paper/final_manuscript/sections/service_capacities.tex` 的
`mod:direct-duration`、`optimization_model.tex` 的人工城市库存约束。

## 已确认定义

保留原 direct 时间公式，不补项：

`Psi_d(q,v)=2 rho_origin+f_origin(q)/v+tau Delta_t+f_destination(q/v)`。

direct 弧 `(i,j,d)` 在 `i` 从出发城市可用库存扣除 v 辆，在 `j` 加入目的城市的
**抽象可用库存**。该城市状态没有记录最后一个客户的物理坐标。
原库存约束中的 departure 条件是 `i<=t`，arrival 条件是 `j<=t`；
local 占用为半开区间 `i<=t<j`。因此在 j 完成的车辆可以在同一节点 j 开始
目的城市的 local、反向 direct 或 empty；若目的城市此前没有其他可用车辆，
在 j-1 开始上述任务会违反库存。相同节点的货流交接仍须满足独立的订单先后约束。

这是聚合模型的再派约定，不宣称真实 hub-to-hub 直送，也不宣称最后客户至 hub
的实际接驳为零。如果真实实施需要额外 last-customer-to-hub 距离/时间，必须另建
位置/接驳状态或显式新服务公式，重新确认容量和时间；不能自行在当前公式上补 rho。
本合同不把数学可用库存升级为真实道路车辆定位证明。

## T2 附加实验政策

作者采用每弧 `sum_ell h_ij^ell >= 0.1 w_ij`，同时关闭 independent empty e。
`.05/.1/.2` 保留为最低平均装载敏感性水平；不是已经完成的求解结果。
该下界单位为 job-equivalent/vehicle，必须与连续可分需求的同一单位保持一致。
它是每波每车**平均**最低装载，不是逐车非空证明。其正性足以排除整弧零货 w，
但不能依据聚合货流还原每辆实际车的货物分配。

## 有限命题与证据

**已证命题（城市库存与上述半开占用假设）**：单辆车从城市 1 于 i 出发，
于 j 到城市 2，若城市 2 初始库存为零且没有其他流入，则城市 2 在 j 前
不能由该车开始任务，在 j 开始一项一车任务不违反人工库存。
证明：t<j 时 incoming=0，目的城市任何 departure 或 local occupancy=1
使库存为 -1；t=j 时 incoming=1，减去一项任务为零。后续任务须分别满足
各自母弧域、容量、订单和其他库存约束；本命题不证明这些附加条件自动成立。

`tests/test_v1_direct_continuity.py` 用独立原 f validator 核验：
载货 direct→载货 destination-local、载货 direct→反向载货 direct、
载货 direct→empty；同刻 j 合法、j-1 非法；以及先前窗口留下的完整 direct
任务在未来完成后的库存恢复和一次计费。
测试使用 60-min periods、tau=1、a=60、b=rho=0 的可手算微型输入，
不是 P1 tau=9 的运营实验。continuation 直接复制已承诺绝对任务账本后增添任务，
只认证账本/replay 层，不声称运行了真实 RH 控制器或其历史提交逻辑。

**数值观察（纯构造核验）**：首例两个 local 各 1 货币单位、AV 1、
三小时 direct 6，总成本 9；往返两段三小时 direct 成本 12；direct 后一小时
empty 成本 8。partial 的已启动 direct 整段计费 6，在途时尚未记为交付；
完成后加一小时 local，完整成本 7，不重复收取先前 direct 费用。
所有完整测试仍须通过订单平衡和时限检查。没有运行 Gurobi 或全数据仿真。
