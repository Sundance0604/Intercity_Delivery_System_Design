# N40完整批次独立审查记录与中断范围

2026-10-19（Asia/Shanghai）。整理人：Principal Investigator。此文件整理Referee在本次任务中实际返回的核验消息，**不是Referee交付的最终审查报告，也不授予最终ACCEPT**。

## 已由Referee报告完成

对象：simulation/fixed_n_runs/FIXED_N40_20261019_01及其20261019T012741615782Z分析。审查未调用求解器，gurobipy导入已禁用。

- 70/70保存方案逐个独立执行原完整f核验，通过。
- 对940个separate local wave，从原始g_manual/x_manual独立核aq+b√(vq)<=vB及q<=vM，通过，最大额外残差约6.395e−13。
- raw/record/request/worker claim、订单/config/profile、原63及suite来源身份检查通过。
- 41 OPTIMAL、9 TIME_LIMIT、20 RH COMPLETED；290次optimize、240个RH窗一致。
- 60对对齐dense/pruned窗口的车辆/selector母域键签名相同；不包含历史或UB的等价保证。
- phase分区差0；累计prepare+执行181.313秒；恰9个full限时软预算超出，所有worker低于15秒硬限。
- 独立报告的各组成本/服务均值与自动摘要一致，含pooled1669.8387、separate1798.3435且后者4个限时。

## 未完成

Referee在独立复算75组配对、15组统计/上下界及最终报告prose阶段遭遇“workspace is out of credits”，任务中断。报告整理的Literature任务也因同一额度限制中断。两者均未交付可标为最终验收的文件。

协调者随后完成结果报告及独立于汇总公式的统计计算：850项数值比较通过，证据见[coordinator_statistics_audit.json](../../simulation/fixed_n_runs/FIXED_N40_20261019_01/coordinator_statistics_audit.json)。该复算不是独立Referee终审。执行与保存结果已完成；完整统计/结论及prose终审仍OPEN。后续有可用额度时可只读完成，无需新增求解或改写原raw。
