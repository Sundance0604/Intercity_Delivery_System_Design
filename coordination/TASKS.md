# TASKS — 有界任务与依赖

更新：2026-10-17。主协调者只负责协调文件；专业Agent各有唯一报告路径，不共享写入。未授权背景研究循环、代码迁移或求解。状态DONE仅表示指定审计交付，不表示科研命题已证。

## 本轮模型同步审计

| ID | Owner / 唯一写入范围 | 依赖 | 完成标准 | 状态 |
|---|---|---|---|---|
| MS-T | theorist：`research/theory/model_comparison.md` | 四TeX、本地代码/笔记、初始合同 | 逐公式新旧比较、代码映射、假设/证明/QVI缺口、行号 | DONE，Referee ACCEPT静态报告 |
| MS-E | experiment：`experiments/reports/model_code_gap.md` | 同上 | 文件/函数/行号映射，迁移依赖及测试验收；不运行求解 | DONE，Referee ACCEPT |
| MS-L | literature：`research/literature/model_positioning.md` | 同上及公开权威文献 | 问题/结构/贡献主张比较，引用核验及限制 | DONE，Referee ACCEPT有界核验 |
| MS-R | referee：`coordination/reviews/model_alignment_review.md` | MS-T/E/L全部交付后 | 独立原始来源复核，按严重度/证据/影响/补救/状态给verdict | DONE；源模型同步MAJOR REVISION |
| MS-PI | primary：MODEL_CONTRACT/CLAIMS/TASKS/PROGRESS、MODEL_DIFF/MIGRATION_PLAN | MS-R | 来源/主张/依赖/验收/作者问题分明；保护源文件 | DONE，文档交付；等待用户决定 |

## 待批准的实施队列

下表是拟议有限任务，不是已委派执行。完整Q1–Q12、阶段方法/验证/验收见[MIGRATION_PLAN](MIGRATION_PLAN.md)。

| ID | 拟Owner | 依赖 | 交付与完成标准 | 状态 |
|---|---|---|---|---|
| G0 | PI+作者 | 本轮报告 | wave/车数域/空返terminal/单位/信息/start-ext等关键规格书面确认 | WAITING_AUTHOR；非环境阻塞 |
| G1 | experiment | G0+批准 | 独立配置/容量/弧域模块，纯函数边界与代回测试 | NOT_STARTED |
| G2 | experiment | G1 | V1-M及公式追溯、独立schedule验证、批准后micro测试 | NOT_STARTED |
| G3 | experiment | G2及direct政策确认 | V1-F、空返、迁移、全日人工复位验证 | NOT_STARTED |
| G4 | experiment | G2/G3对应范围 | 新registry/schema、独立成本及scope、V0回归保护 | NOT_STARTED |
| G5 | experiment | G4 | V1-RH历史/库存/reveal/罚损/terminal账本 | NOT_STARTED |
| G6 | experiment+theorist后referee | G5 | 同母弧域、保守剪枝/安全性证明；full/reduced检查 | NOT_STARTED |
| G7 | experiment | G6 | wave增量容量启发式、返程、selectors/warm start与可行性 | NOT_STARTED |
| G8 | experiment | G7 | local branching邻域规格、恢复/预算/证书验证 | NOT_STARTED |
| G9 | experiment后referee | G8+单独预算批准 | 隔离公平实验计划、回归、日志；先smoke再正式 | NOT_STARTED，正式预算未申请 |

## 独立研究队列

| ID | 拟Owner | 依赖 | 完成标准 | 状态 |
|---|---|---|---|---|
| S1 | theorist后referee | Q10确认 | n≤W/active set/期限临界及可行域完整重审 | OPEN，未授权改稿 |
| S2 | theorist+experiment后referee | S1、Q1/Q9、V1基本验证 | S↔V1共同子模型/归一化/峰值车队/整数/首末期规格 | OPEN |
| L1 | literature | Q11材料及范围 | Bib、全文支持定位、校准/脚本来源；不认证未经检索的创新 | OPEN |
| QV1 | theorist | Q12确认 | 随机或确定性V2的状态/信息/控制/目标/边界合同 | OPEN |
| QV2 | theorist后referee | QV1 | 明确verification/threshold等命题与独立证明 | OPEN |

## 旧首次运行队列保留记录

R-001 landscape.md、T-001 qvi_assessment.md、E-001 inventory.md、V-001 initial.md、PI-001依赖V-001的初始综合是安装时预置队列。本轮用户改为上述模型同步审计，这些原指定文件未交付，不登记DONE；状态为DEFERRED/SCOPE_REPLACED_BY_MS，若未来恢复须重新明确范围。不得把本轮不同名称报告当作原任务已完成。
