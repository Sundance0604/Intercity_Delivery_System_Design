# BHH 论文修订报告

日期：2026-10-18。修订对象：本目录的英文新稿，主文件为 [main.tex](main.tex)，标题为 *Wave-Based Intercity Freight Scheduling: A Mixed-Integer Model and a Dispatch Interpretation*。

**结论：已完成本轮能够确认的保守修订；尚不能宣称正式投稿验收全部通过。** Direct 路线的物理定义仍需作者确认，最终编译仍受环境阻碍。两项均未被用未经证实的文字或公式替代。

## 1. 任务来源、边界与审查流程

完整读取了用户指定的任务书。可读取的仓库文件中未定位到同名任务书，因此使用明确提供的附件原件：D:\download\BHH_Codex_Revision_Task.md。其 SHA-256 为：

~~~text
52d41751268fafbf7ca73299f7df690b1f12a7e2776bad31b2f639df595418c0
~~~

先读取 AGENTS.md、MODEL_CONTRACT、当前 Git 状态和新稿，再对照旧稿及相关实现。修订仅发生在 research/paper/final_manuscript/。旧稿、生产代码、历史数据未作本轮修改；未运行求解器、数值实验或新的理论研究，未提交或推送 Git。

Literature 负责引言、综述和文献库；Theorist 只读核对现有公式、证明和 Direct 物理解释；上述产出完成后，Referee 独立只读审查最终源稿。主 Agent 负责其余保守文字修订、检查与综合。各 Agent 没有并发写同一文件。数学审查属于现有结论核验，没有增加新命题。

## 2. 按优先级完成的修改

### P0：模型解释、主线和贡献

- **Direct 式（6）**：核对旧稿、稳态稿及实现后，确认现有材料不足以判定是否确实漏项。因此保持持续时间、系数、容量及最大弧长公式不变；将“因此得到”改为明确采用的聚合近似，避免声称已有完整物理推导。未擅自把目的城市 stem 加入公式。详见第3节。
- **有限期模型与解析基准的关系**：在数学分析开头增加集中说明。两者共享路由时间原语，但目标函数、车队约束、信息、交付记账与策略类不同。未建立松弛、等价、稳态极限或 rolling-horizon 最优性证书；没有向 MILP 增加等待成本参数。
- **摘要、引言及贡献**：主问题收束为有限期、固定车队的两城运输调度；三项贡献分别对应 wave 容量与车辆流模型、容量及条件载荷结构、受限基准中的静态—动态一致性与验证。RH、过滤和通常的 QVI 工具没有被单独包装成新算法突破。

### P1：文献、符号、数学边界和实施方法

- 保留原有12条经典参考文献，补充5篇经出版商、作者或机构主来源核实的相关研究。现有17条全部被引用。综述说明相关工作处理的问题与本稿的具体联系，不作未经证明的“首次”或全面缺口断言。
- 区分 wave 总货量、平均单车载荷与车辆数；修正将车辆数量称为 wave size 的易混文字。
- 说明 stop equivalent、分钟、离散时间期、每车分钟费率及拒单罚金的单位；小时费率只在实现入口转换一次。有限期目标仅包括车辆运行与拒单，不把解析基准中的等待或 lead-time 成本混入其中。
- 明确 local tour 的 hub 闭环，以及 direct 完成时计入目的城市库存是当前模型的记账约定。后者没有被当成物理接驳已经解释完毕。
- 核对路由逆函数、容量取正部与载重上限、库存、同刻交接、货流先后关系和期末需求平衡。未发现可确认、需在本轮改动公式的错误。
- common-wave 被明确称为预设策略类；没有声称独立阶段自然同步最优。原载荷上界、正性条件、可分离成本、日历余项和 QVI 验证条件均保留。
- 将 excess-cost 函数 F 与路由逆函数 F_k 的符号说明放在定理外，避免混淆。QVI 的 min 写法解释为非负残差及至少一项为零的互补条件，没有更改方程。
- Poisson 比较仍只涉及相同的正整数阈值，不推广为两个模型具有一般相同最优策略。
- RH 和 cargo filtering 作为既有实施流程描述；没有增加速度保证、一般无损性、递归可行性或全局在线最优性主张。
- 讨论按“已成立结果—运营含义—范围限制—结论”重组，结论为两个自然段。集中保留必要边界，减少重复的防御性说明。

### P2：编辑与文档

- 统一专业英文、术语与模型记账表述；保留 article 模板及空作者字段，未虚构姓名或机构。
- 保留既有章节与标签，没有增加数值章节、实验计划、新示意图或期刊专用模板。
- 更新本目录 README 的文献数量、当前审查入口和实际编译状态，区分历史审查与本轮最终源稿。

### 本轮文件清单

| 文件 | 改动 |
|---|---|
| main.tex | 摘要和编译说明注释 |
| sections/introduction.tex | 有限期主问题、三项贡献、模型关系 |
| sections/related_work.tex | 综述定位与5篇相关研究 |
| references.bib | 增加已核实的5条文献，保留原12条 |
| sections/optimization_model.tex | 变量、单位、库存记账和近似的文字解释 |
| sections/mathematical_analysis.tex | 模型关系、策略类和符号说明 |
| sections/solution_method.tex | 明确既有实施方法的角色 |
| sections/discussion.tex | 结果、运营解释、限制与结论重组 |
| README.md | 当前状态与报告入口 |
| BHH_revision_report.md | 新建本报告 |

sections/proofs.tex 未修改。其他旧论文及研究文件未因本次任务改写。

## 3. Direct 物理定义：保留公式，等待一个作者决定

### 3.1 当前公式及证据

源码顺序推计式（6）为：

\[
\Psi_d(q,v)=2\rho_{o(d)}
+\frac{f_{o(d)}(q)}{v}
+\tau\Delta t+f_{r(d)}(q/v).
\]

对应标签 mod:direct-duration；式（7）、（8）分别给出系数和容量，式（4）给出 Direct 最大弧长。式（6）含起点城市的双向 stem，未显式列出目的城市 stem。

证据：

- [当前模型](sections/optimization_model.tex) 的 Orders, time, and service technology 小节同时定义 hub-to-hub AV、local hub 闭环和 direct 的城市库存回补；Direct 持续时间随后使用同一个 tau。
- ../main_v2_BHH.tex 中 hub/公路入口、local 闭环和 direct 服务说明保留同类表达，但没有足够明确的 direct 目的端接驳及车辆最终位置定义。
- ../main_BHH.tex 及 ../steady_state_model.tex、../steady_state_model_v1_linear.tex 不能独立消除该歧义。
- intercity_delivery/v1/flexible_capacity.py 及 flexible_validation.py 复现这一时间表达，可证明实现遵循表达式，不能证明物理路线完整。

因此，本轮**不把“可能漏项”写成“已证明错误”**。原式的代数反解与给定持续时间相符，争议在持续时间的物理定义。

### 3.2 两种条件解释，均未直接实施

**解释 A：作者能够确认现有 direct 路径及 tau 的定义已经覆盖所需目的接驳和车辆再次可用的定位。**

只有在该定义与共享 tau 的 AV hub-to-hub 语义也一致时，才可以保留现有式（6）及全部系数、容量和弧长界。现有正文不足以确认此解释，不能直接声称“tau 已经包含目的 stem”。

**解释 B：tau 严格为 hub-to-hub 时间，direct 的目的配送从目的 hub 出发并返回该 hub。**

在这一明确闭环假设下，一种待作者确认的持续时间是：

\[
\Psi_d^{\mathrm{closed}}(q,v)
=2\rho_{o(d)}+\frac{f_{o(d)}(q)}{v}
+\tau\Delta t+2\rho_{r(d)}+f_{r(d)}(q/v).
\]

此时应相应使用：

\[
B_{ij}^{d,\mathrm{closed}}
=(j-i-\tau)\Delta t-2\rho_{o(d)}-2\rho_{r(d)},
\]

而 alpha、beta 的载荷系数不变，容量需用此 B 重新反解并截断于 vM；相应弧长界应协调为：

\[
L_D^{\mathrm{closed}}
=\tau+\left\lceil
\frac{2\rho_1+2\rho_2+f_1(M)+f_2(M)}{\Delta t}
\right\rceil .
\]

这些仅为条件性影响说明，没有写入论文、代码或容量表。若车辆停在服务区域而非 hub，增加单个目的 stem 也不能自动解决后续任务定位与库存可用性问题。

**请作者确认的一个问题：一趟 direct 的实际起点、起点收集后是否回 hub/入口、tau 的计时起终点，以及最后配送后车辆是在目的 hub 立即可再派还是留在服务区分别是什么；据此，现有的 2rho_origin 是否完整，目的 stem 是否已计入且与 AV 共享 tau 的含义一致？**

确认后应统一修改持续时间、反解容量、弧长及相关实现/核验，不能仅孤立补一个 rho。本次未启动这种模型修改。

## 4. 公式与引用检查

### 4.1 与修订前源码的比对

新稿目录尚未作为独立已跟踪文件逐项出现在普通 Git diff 中，因此另外保存了修订前源码文本及哈希用于逐文件比较。

- **53个显示数学块**：去除排版空白后与修订前一致。
- **3个命题、2个定理**：陈述环境去除排版空白后一致；不是通过削弱条件来消除问题。
- **完整 proofs.tex**：SHA-256 与修订前一致。
- 主输入顺序及公式编号顺序未改变。以下为源码计数推计，并非已编译 PDF 的编号核验：

| 源码推计编号 | 核对内容 |
|---|---|
| (6)–(8) | Direct 持续时间、系数与容量，保持原式 |
| (9) | 有限期车辆与拒单目标，未增加等待成本 |
| (20) | 需求平衡，保持原式 |
| (22)–(25) | 局部容量、逆函数及规模经济 |
| (26)–(30) | 载荷域、广义成本、条件最优载荷及发车成本 |
| (31)–(33) | 平均成本、静态—动态一致性及日历余项 |
| (34)–(38) | QVI、尾包络、相对值及验证 |
| (39) | 共同正整数阈值的 Poisson 比较 |

现有 Prop. 4.1、Prop. 4.2、Thm. 4.1、Thm. 4.2、Prop. 4.3 的假设和证明引用均经过独立核对。容量结论不被推广成凸性；条件载荷结论不被推广成全模型同步最优；静态—动态一致性不被推广到 finite-fleet MILP。

### 4.2 静态结构结果

主 Agent 与 Referee 独立检查结果一致：

| 项目 | 结果 |
|---|---|
| TeX 源文件 | 8个 |
| input | 7个，全部目标存在 |
| labels | 68个，无重复 |
| ref/eqref 等交叉引用出现次数 | 62处，无缺失目标 |
| Bib 条目 | 17条，无重复 key、缺失 cite key 或未引用条目 |
| 花括号与 begin/end 环境 | 静态配对通过 |
| 本轮源码尾随空白 | 未发现 |
| Git diff --check | 退出码0；仅提示既有文件未来 LF/CRLF 转换 |

这些检查不替代 TeX/BibTeX 编译，也不能证明无溢出、字体问题或版面缺陷。

### 4.3 新增文献及核实边界

| Key | 正式发表信息 | 用于本稿的具体联系 |
|---|---|---|
| Klapp2018 | Klapp, Erera and Toriello (2018), *The Dynamic Dispatch Waves Problem for same-day delivery*, EJOR 271(2), 519–534；DOI 10.1016/j.ejor.2018.05.032 | dispatch/wait 与发车波决策 |
| Scherr2020 | Scherr, Hewitt, Neumann Saavedra and Mattfeld (2020), *Dynamic discretization discovery for the service network design problem with mixed autonomous fleets*, TR-B 141, 164–195；DOI 10.1016/j.trb.2020.09.009 | 混合自动化车队、两层网络与时间离散 |
| He2022 | He, Boland, Nemhauser and Savelsbergh (2022), *An exact algorithm for the service network design problem with hub capacity constraints*, Networks 80(4), 572–596；DOI 10.1002/net.22128 | hub 容量与服务网络设计 |
| Wei2023 | Wei, Çetinkaya and Cline (2023), *Inbound replenishment and outbound dispatch decisions under hybrid shipment consolidation policies: An analytical model and comparison*, TR-E 175, 103135；DOI 10.1016/j.tre.2023.103135 | 集并策略的成本、等待与库存 |
| Lee2024 | Lee, Dalmeijer, Van Hentenryck and Zhang (2024), *Optimizing Autonomous Transfer Hub Networks: Quantifying the potential impact of self-driving trucks*, EURO Journal on Transportation and Logistics 13, 100141；DOI 10.1016/j.ejtl.2024.100141 | 自动化中间运输与人工首末端结构 |

核实使用出版商记录与作者/机构来源：[Klapp作者机构页面](https://www2.isye.gatech.edu/faculty/Alan_Erera/research/2018/07/23/sameday-dispatching.html)、[Scherr出版商页面](https://www.sciencedirect.com/science/article/pii/S0191261520304021)、[He出版商页面](https://onlinelibrary.wiley.com/doi/abs/10.1002/net.22128)、[Wei出版商页面](https://www.sciencedirect.com/science/article/pii/S1366554523001230)、[Lee作者预印本](https://arxiv.org/abs/2305.03119)及[出版商页面](https://www.sciencedirect.com/science/article/pii/S2192437624000165)。部分出版商全文或 DOI 直连不可访问时，通过主来源摘要、元数据或作者记录交叉核对；没有声称全部全文均已访问。Wei 使用2023年正式期刊信息，不以预印本年份代替。原12条经典文献本轮保留，并检查引用和已有元数据；未声称重新逐篇阅读全文。

未引用这些工作的数值性能作为本稿结果，未将其理论结论直接移植成本稿保证。

## 5. 最终编译结果

修订前、修订后各调用一次内置 compile_latex_document，参数均为：

~~~text
path = D:/mycodelife/workshop/AI/multiagent-test/research/paper/final_manuscript/main.tex
root = D:/mycodelife/workshop/AI/multiagent-test/research/paper/final_manuscript
~~~

两次结果相同：

~~~text
kind: compile-failed
message: Could not prepare the LaTeX preview
diagnostics.log:
exec failed: unsupported operation: windows unelevated restricted-token sandbox cannot enforce split writable root sets directly; refusing to run unsandboxed
~~~

**最终状态：编译环境准备失败，TeX 编译器未正常启动；不是已定位的 TeX 语法错误，亦不是编译通过。** 这是执行环境的沙箱限制，不是用户拒绝授权，也不是自动审批审查拒绝。没有据此盲改源码。

只读检查未发现可用的 pdflatex、xelatex、lualatex 或 latexmk 终端程序。保持现有编辑器打开，未另建项目、替换主文件、导出另一份 PDF 或改开其他标签页。没有取得成功编译日志、参考文献排版结果或可供前后视觉比对的新 PDF；环境恢复后仍须在本项目完成实际编译与版面验收。

## 6. 独立审查和最终源稿指纹

Referee 完整读取任务书、新稿全部章节及17条文献记录，最终意见：

- **本轮保守修订交付：ACCEPT。**
- **正式投稿验收：尚未完成。** Direct 物理定义与实际编译仍为 P0 未决项；作者资料待补。没有要求本轮增加理论或实验。

审查后再次核对 mathematical_analysis.tex 的最终版本：仅把 F 与 F_k 的符号说明移到定理外，不影响审查结论。以下为最终冻结的论文源文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| main.tex | 80224ce7ef5e45d61767a30d569b2bea2dafee4f5213517937d05b986c63cf72 |
| references.bib | 13baee18dfa4cd8b432d8aefe817bd44c2869f7c6c4636deeb84b35909512a17 |
| sections/introduction.tex | 2b00e926d02015eb171f523f22b410c51e1921decccc8e90ac49b8130f0e5d9d |
| sections/related_work.tex | d0ec01fda63361ace29d1d32f62a661f9c23e7a1ce3d19ca2dee8281892c4192 |
| sections/optimization_model.tex | b30542b0ebe29fae56edf3aea66241aedbd8d9efe7ce38c2bb1a9467cf63b555 |
| sections/mathematical_analysis.tex | 2c8734b8624a50e6b6e0468cbe0880faadf80beff53a48a4423b73b5d2c479cd |
| sections/solution_method.tex | f4193ce07b7e037daff09983c6bdfb471244b63fdaa1f85e5f441c306a0761b6 |
| sections/discussion.tex | 2f244efb846b92919962f1c6fbba6f3094f6f92737eeff0159d480cff73db550 |
| sections/proofs.tex | 2222c1e2bbaa5654bb0be00d91429eaacda22ab6b1e00792e79e0813ac4253d7 |

修订前后额外比较的71个受保护源码文件（旧稿及相关 Python 实现、测试）哈希均一致。仓库在开始时已有其他修改及大量未跟踪研究文件，予以保留；不能把这些既有修改计为本轮修订。Git 状态读取会提示 .pytest_cache 访问受限，该缓存与本次论文无关，未改其权限。

## 7. 尚待解决的最小事项

| 优先级 | 事项 | 完成标准 |
|---|---|---|
| P0，作者决定 | 第3节 Direct 路线及计时定义 | 给出明确物理路线和车辆再次可用位置，确认是否需统一调整公式及实现 |
| P0，环境验证 | 实际编译及 PDF 排版 | 在现有主项目成功编译，核对未定义引用、参考文献、公式溢出和页面；当前无法声明通过 |
| P1，投稿资料 | 真实作者、机构及期刊要求的信息 | 作者提供资料，按目标期刊和盲审要求填写；不由 Agent 编造 |
| 本轮范围外 | 既定暂缓的数值部分 | 如实保留未开展状态；是否为目标期刊补充，由作者另行决定，本轮不新增实验计划或性能主张 |

本轮完成的是现有论证的澄清、收束和可核查修订，不把以上未决事项转化为新一轮模型扩展或理论研究。

