# 条件V1-M微型验收方案

2026-10-18。PI批准：最多10个微型run，每整次≤2秒、Threads=1，总solver预算上限20秒。初次最多8个run，保留2次用于失败后的定向验证。仅合成T≤8、≤3订单、每城0–2人工车、AV总数≤2；不执行旧全套求解、真实数据、规模比较、TeX编译或安装依赖。

定义/API见[实现profile](../../coordination/V1_M_IMPLEMENTATION_PROFILE.md)。首轮8run时作者答复尚未到达，历史结果标`conditional_not_author_confirmed`。作者随后明确采用D1–D6-A，最终结果改为`author_confirmed_profile`并带decision_id，剩余2run用于新状态定向复验；此变更不改变数学约束。显式profile无默认。

两位Experiment分别拥有模型/配置/runner和独立validator，生产Agent不自行求解，PI统一计数执行。纯输入/validator测试不需要许可证或solver预算。Referee在交付后审查，不额外求解。

验收预期：

- F1容量边界：源式正部导致零tour预算容量为0，正车零货弧不得被validator错误拒绝。
- F2/F3：共享AV容量为1，两单各.75或一单1.5，零车辆费/单位罚金1，最优总未服务量.5；不要求两单分配唯一。
- F4：三腿各一车、t0=a=30分钟，人工10/小时、AV4/小时，服务1的费用12；deadline缩到2时无三腿路径，全拒单罚100。
- F5：零车队、需求.05、单位罚金2，连续z=.05、费用.1，完整导出保留小量。
- Wave：T3/tau1/t0=1分钟，两城各2HV、origin1AV、a=b=M=Mauto=1/rho0，需求1/窗[0,3]/罚100，人工和AV均60每车小时。唯一载货链0→1→2→3，两端u2时h(2)=1，最优成本5。只与手算的旧线性容量替换作解释，不运行旧优化器或冒称整V0模型公平比较。
- 无订单、tau>T、非零stem及非法输入尽量用纯测试和结构检查；实际solver run名单以已交付测试代码为准，不超过上述总预算。

独立validator由原f(load)≤v·max(Δ−2rho,0)和load≤vM核联合容量，独立重建弧域/库存/时序/窗口/需求/费用。one-hot、AV共享超载、跨单借货、错误整数域、NaN/Inf、输出objective篡改等用定向负例。报告记录每条命令/测试数/实际run数/退出码/时间、版本与source/input指纹，不把有限通过升级为一般等价或性能证明。

执行已完成：首轮8run+最后2run恰好用满10个run配额，每run2秒上限；无需追加求解。执行器`run_v1_mandatory_micro_validation.py`现默认stdout，显式文件在求解前独占预检；避免首版74项通过后文件写入失败的日志问题。再次运行该完整执行器会产生8个新run，不属于本轮已用完的20秒预算；复核保存计划可只调用纯validator而不求解。
