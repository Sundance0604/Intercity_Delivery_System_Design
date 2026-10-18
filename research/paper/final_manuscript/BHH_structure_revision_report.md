# BHH 结构修订报告

2026-10-18。状态：**结构修订完成，独立审查限定 ACCEPT；由作者自行编译。**

## 改了什么，保留了什么

当前入口仍为 [main.tex](main.tex)，标题改为 *Wave-Based Intercity Freight Scheduling with Fleet Circulation*。修订前核对了原标题、17条文献及核心公式，修改的是 final_manuscript 新稿。

| 状态 | 对应任务 | 完成内容 |
|---|---|---|
| DONE | P0 版本及依赖 | 核对新稿入口，记录原标签与迁移去向，保留已有工作区修改 |
| DONE | P1 主线 | 服务系统与容量机制 → 有限期 MILP 与 rolling horizon → 独立稳态解释 |
| DONE | P1 理论位置 | 容量命题紧接容量定义；载荷与重复发车结果留正文；完整 QVI、Poisson 比较及证明移入附录B |
| DONE | P1 BHH解释 | 用均衡面积与货量解释 f(q)/v 与 f(q/v) 的区别；b仍为输入系数，不宣称已有面积或速度标定 |
| DONE | P1 实施方法 | 核心滚动步骤留正文；完整 cargo filtering 和 replay 规则移附录C |
| DONE | P1 文献 | 阅读 Banerjee 原PDF正文与参考文献，新增准确比较；17旧条目保留，合计18条 |
| DONE | P2 精简 | 摘要单段187词；引言约752→512词；综述约1252→503词；两项主要贡献；结论压为三段 |
| DONE | P2 图表 | 两幅原创任务/几何与跨窗时间线示意；主符号表和模型差异表 |
| NOT_APPLICABLE | 可选输入输出表 | 章节衔接已说明输入输出，为控制篇幅未再加第三张表 |
| NOT_APPLICABLE | 实验、模型扩展 | 本轮未运行实验、添加结果或启动新理论；未迁入固定分区、无等待或日终回库规则 |
| REQUIRES_AUTHOR_CONFIRMATION | P0 Direct路径 | 原公式、容量反解和最大弧长保留，物理连接仍未确认，见下表 |
| NOT_APPLICABLE | 本机最终编译 | 作者已明确自行编译；不继续尝试本机编译，不交付未经验证的新PDF |

正文粗略英语词数约 **6700→4479，减少33%**；含附录约7784→6542。使用同一源码统计规则，去掉注释、显示公式及主要交叉引用命令；这不是期刊排版字数或页数。精简主要来自重复论述、技术内容移附录和综述收束，没有删去已有结论。

## Banerjee 论文怎样用于本稿

读取用户指定的22页原PDF，并视觉查看PDF第6、11页。其关键链条是“单车区域面积 → 车队规模指南 → 区域划分”（印刷1336、1339页）。本稿借鉴的是结果进入下一步决策的组织方式，保留自己的有限车队调度目标。

| 参照证据 | 本稿采用方式 |
|---|---|
| 印刷1331页／PDF6：setup、连接、BHH区域内路由、逐单服务分解 | 补解释性几何式与任务示意，说明系数和物理含义 |
| 印刷1333页／PDF8：no-wait指dispatch后的车辆闲置 | 不据此删除订单积累或要求连续发车 |
| 印刷1336页／PDF11：面积函数供fleet sizing及partition使用 | 将容量表、MILP和执行写成连续主线 |
| 印刷1330、1339页／PDF5、14：单车固定zone、日终回库与方法层次 | 不迁入本稿，不把车队规模指南写成随机运营保证 |

未读取不在该PDF内的 Online Appendices，也未根据其二手介绍增加其他文献。新增参考文献信息：Banerjee, Dipayan; Erera, Alan L.; Toriello, Alejandro (2022), *Transportation Science* 56(5), 1327–1347, DOI 10.1287/trsc.2022.1125。

## 物理语义未决项

### Direct 路径账本

| 环节 | 已有表达 | 尚待确认 |
|---|---|---|
| 出发 | 从起点城市可用车队扣除 | 实际位置是否为hub |
| 原点收集 | 分区路由及2rho_origin | 收集后是否回hub／入口 |
| 跨城 | tau乘Delta t；AV使用同一tau | 确切端点、是否覆盖连接段 |
| 目的派送 | f_destination(q/v)，全区域非分区 | 入口到服务区域的stem计在哪里 |
| 再次可用 | 任务结束计入目的城市库存 | 是目的hub还是服务区域 |
| 后续local | local从hub出发 | 两任务之间是否仍有未计定位时间 |

状态为 **REQUIRES_AUTHOR_CONFIRMATION**。未通过添加目的stem、重定义tau或画确定路线来代替作者决定。保留原Direct零货tau期特例和独立empty变量。上一轮报告中的条件解释仍适用，但并未成为模型假设。

### Transfer-only 的比较含义

已明确 w=e=h=0 表示 **transfer-only operation with city-dedicated human-driven fleets**：既禁止Direct货运，也关闭人工空车跨城。此项是对既有可行域的文字澄清，状态 DONE；没有重新决定是否允许empty。若作者希望“只关闭Direct货运、保留empty”，须另行明确模型选择。当前不能将同时开放两项决策的收益全部归因于Direct货运。

## 公式、引用和独立审查

- 本轮移动前后比对：原53个显示数学块、5项命题/定理、2项假设与5份proof环境均保留，去除空白后无内容删除；新增1个BHH均衡分区解释式。未新增研究定理。
- 最终静态检查：11个TeX文件、10个input全部存在；78个标签无重复，68处交叉引用全部解析；18条Bib均引用，无缺失key；括号和环境配对通过。
- [BHH_source_and_claim_map.md](BHH_source_and_claim_map.md)逐项记录结论依据、旧标签与新位置；编号由LaTeX生成，未机械写死。
- Literature负责标题、摘要、引言、综述及文献；Theorist核查几何解释、Direct与transfer-only含义；Referee在产出后独立审查。
- Referee发现初版时间线把既有任务终点画成j>c_t，与固定H、E下c_t单调不减不符。现已改为 **i<t<b_t<j≤c_t≤T**，只说明跨origin-admission边界；复核通过。
- Referee最终意见：**本轮结构精简限定 ACCEPT**。没有发现必须补回的实质段落；Direct仍待确认，实际排版未验收。

## 编译与交付范围

修订前和重排后曾调用现有内置编译工具，均在准备阶段返回 BLOCKED：

~~~text
kind: compile-failed
message: Could not prepare the LaTeX preview
exec failed: unsupported operation: windows unelevated restricted-token sandbox cannot enforce split writable root sets directly; refusing to run unsandboxed
~~~

这不是已定位的TeX语法错误。随后作者明确“本机无法编译latex，我会自行编译”，因此最终本机编译标记 NOT_APPLICABLE，不再重试。**没有新PDF、成功编译日志或公式无溢出的视觉验收结论。** 两幅示意图直接使用LaTeX picture，未增加绘图库依赖。作者以本目录为根、main.tex为主文件，沿用LaTeX与BibTeX流程即可检查；现有旧PDF不代表本次修订。

改动包括本目录main、全部章节、references.bib和README；新增service_capacities、dispatch_verification、implementation_details三个章节文件及两份结构报告。协调文档MODEL_CONTRACT、CLAIMS、TASKS、PROGRESS追加当前状态。旧BHH_revision_report保留为历史记录。生产代码、旧论文、实验数据未作本轮编辑；未提交或推送Git。

