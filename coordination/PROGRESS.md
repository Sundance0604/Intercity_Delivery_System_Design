# PROGRESS — 已核实的模型同步进展

更新：2026-10-17（Asia/Shanghai）。本轮范围：模型分析、代码静态审计、文献定位、独立审查与条件化迁移计划。**实际代码迁移未开始，等待用户确认规格并批准。**

## 已完成的有界交付

1. 主协调者读取AGENTS、初始共享合同/账本/队列/进度、Git状态、四份TeX及相关本地源码；定位实际QVI笔记。
2. 并行Theorist/Experiment/Literature分别交付[数学比较](../research/theory/model_comparison.md)、[代码差距](../experiments/reports/model_code_gap.md)、[文献定位](../research/literature/model_positioning.md)。
3. 等待三稿后启动Referee，直接核对原始来源并交付[独立审查](reviews/model_alignment_review.md)。三稿ACCEPT限静态审计用途；源模型—算法对齐MAJOR REVISION。
4. PI更新[MODEL_CONTRACT](MODEL_CONTRACT.md)、[CLAIMS](CLAIMS.md)、[TASKS](TASKS.md)及本文件，新建[MODEL_DIFF](MODEL_DIFF.md)、[MIGRATION_PLAN](MIGRATION_PLAN.md)。合同登记源文件SHA-256，计划包含依赖/方法/验证/验收及Q1–Q12。

## 核实的知识与负面结果

- V0实际有M/F/RH不同模型/算法分支；旧论文与base的AV容量已不一致。F已有部分V1结构，不能一概当未实现。
- V1新增wave联合容量、stem、one-hot与direct终端/空返机制；正文、旧算法与代码未同步。已给旧启发式容量和history-only零车剪枝的限定手工反例，没有运行实验。
- V1与两份稳态稿尚不等价；sorting/δ/ω/timetable、车队/目标/连续近似及首末期需桥接。S的n≤W、deadline等号/可行边界需重审。
- CFS quantity是重量映射，不能当stop-count校准；partial horizon成本和window bound/gap不得解释为全日证书。
- 本地QVI笔记已找到，实际主要为确定性常率/时变率。限定周期代数可核；随机到达、完整verification、threshold与finite sorting同步最优仍未解决。
- Literature记录多项公开文献的存在性/主题核验与支持边界（清单含13项）；Referee另作有限抽查。未作穷尽novelty认证、未逐篇付费全文审定。Bib与稿称校准脚本缺失/未定位。
- Referee曾怀疑b=0破坏稳态2/3定律，复读as:bhh的b>0及cost-bound的ω>0后撤回；未把域外反例登记为稿内漏洞。V1与S参数域仍须分别保留。

## 未完成与明确限制

没有修改四份TeX、生产Python、tests或历史数据；没有运行Python/tests/solver、数值实验、TeX编译、依赖安装、Git提交/推送或网络发布。已有tests和solver许可证仅静态查看，运行状态UNTESTED。稿内表格、校准与“已检查”注释不是本轮复现证据。

已有大量untracked研究/实验文件，工作树不干净；`.pytest_cache`拒绝访问导致Git/文件扫描警告，不能断言全库所有角落均被遍历。没有修改权限或清理缓存。HEAD及输入散列见MODEL_CONTRACT，不能用HEAD替代untracked模型版本。

收尾检查：`intercity_delivery/`、`tests/`、`research/paper/`下的Python/TeX/Bib文件与`main.py`前后SHA-256快照一致；`git diff --check`通过，仅提示现有LF/CRLF转换策略。四份原协调文档更新、六份获准报告/计划新建；其余既有untracked状态不解释为本轮产物。Referee对六份协调文档作只读一致性复核，无阻断错误；文献清单数量已按复核结果更正。这不增加任何科学证明或数值验证状态。

## 待作者决定与下一步

最高优先级是Q1–Q9中影响批准范围的模型目标、wave分组/枚举域、空返/terminal、单位/揭示时间、start/ext与期限口径。S边界、缺失材料和V2方向可按Q10–Q12独立延期。

建议确认G0后先批准独立V1-M的G1–G2及对应G4验证；直送/RH/剪枝/启发式按依赖推进。微型测试预算只是建议，尚未获批；正式/大型实验须另申请。没有后台任务持续运行或自行推进实际迁移。

## 进展口径

审计发现、反例和明确的未决条件属于进展；文件数、页数、代码量、Agent数量不代表新定理或数值可靠性。CLAIMS保留原QVI-001/BHH-001为CONJECTURE。本轮没有新增NUMERIC_TOY、NUMERIC_FULL或全网络PROVED成果。
