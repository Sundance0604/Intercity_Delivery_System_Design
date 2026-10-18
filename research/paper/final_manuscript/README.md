# 论文新稿入口

主文件：[main.tex](main.tex)。标题：*Wave-Based Intercity Freight Scheduling with Fleet Circulation*。

正文按“容量机制 → 有限车队调度与滚动执行 → 独立稳态解释”组织。全部已有命题、假设和证明保留；数值实验章节与实验计划仍暂缓。

| 文件 | 内容 |
|---|---|
| main.tex | 标题、单段摘要、正文及附录装配 |
| sections/introduction.tex、related_work.tex | 两项主要贡献、三主题综述 |
| sections/service_capacities.tex | 系统、符号表、BHH几何解释、容量及pooling性质 |
| sections/optimization_model.tex | 完整MILP、车辆库存、货流、模式及真实终点 |
| sections/solution_method.tex | 接在MILP后的滚动步骤与时间线 |
| sections/mathematical_analysis.tex | 独立稳态基准、条件载荷和重复发车结论 |
| sections/discussion.tex | 三段讨论与结论 |
| sections/proofs.tex | 附录A：容量与载荷证明 |
| sections/dispatch_verification.tex | 附录B：日历最优性、QVI、整数Poisson比较 |
| sections/implementation_details.tex | 附录C：过滤及独立replay |
| references.bib | 18条实际引用，含Banerjee et al. (2022) |
| BHH_structure_revision_report.md | 本次修改、审查、未决项和编译记录 |
| BHH_source_and_claim_map.md | 结论依据与完整标签迁移表 |

[当前结构修订报告](BHH_structure_revision_report.md)和[结论映射](BHH_source_and_claim_map.md)为本轮入口。[BHH_revision_report.md](BHH_revision_report.md)及原final_manuscript_review是较早版本记录，其源文件指纹不代表当前稿。

作者已决定自行编译：以本目录为项目根、main.tex为主文件，沿用LaTeX/BibTeX工具链。内置编译此前在环境准备阶段失败，本轮没有新PDF或排版验收。Direct的目的接驳与车辆再次可用位置仍待作者确认；原式保持。作者信息为空，待填写真实资料。
