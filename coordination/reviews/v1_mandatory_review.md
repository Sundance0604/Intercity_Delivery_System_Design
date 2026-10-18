# V1-M 独立审查

2026-10-18。Owner：Referee。范围：`intercity_delivery/v1/configuration.py`、`mandatory_optimizer.py`、`runner.py`、`validation.py`，两份新测试、profile、微型验收方案、实施报告、断言JSON和执行器。直接参照 `research/paper/main_v2_BHH.tex:67–137`、模型合同及已审纯内核。Reviewer不修改实现、不运行求解器。

**最终建议 ACCEPT，范围限作者采用的六项profile所定义的离线逐arc-wave强制换装基准。** 未发现M01–M15遗漏、联合容量偷换、零车selector或累计货物流的阻断错误。PI传达用户已回复“采用这组定义”，确认D1–D6-A；Reviewer已核实最终metadata、decision_id、确认文档和两份原始payload。用户确认这些定义不等于完整V1-F/RH、稳态或QVI已完成，也不意味着源TeX文本已同步。

## 按严重性列出的发现

| 严重性 | 证据 / 路径 | 影响 | 处理 | 状态 |
|---|---|---|---|---|
| 高：已明确的模型采用边界 | profile、TeX:71与82/106/110 | 同start文字与逐arc索引存在张力；不能把代码选择当成源文原先已唯一明确 | 用户采用D1–D6-A，当前结果author_confirmed_profile并带V1M-D1-D6-A-2026-10-18，历史conditional结果保留；源TeX未改，所选arc定义只用于此基准 | CLOSED_FOR_THIS_BENCHMARK；源文同步与F/RH另立事项 |
| 中：首轮原始证据不完整 | `v1_mandatory_micro_assertions.json`、实施报告第4节 | 首轮74项OK后JSON创建被拒，外围exit1，8份完整raw payload未保留。可引用通过断言，不能声称已归档该次完整解 | 记录明确raw未保存和外围exit1；执行器默认stdout，显式输出先exclusive创建。最终版本2run的真实payload已另存confirmed_micro_results，不冒充历史原解 | ACCEPTED_EVIDENCE_LIMITATION；新增两例原始证据已复核 |
| 数值/实验限制 | validator及基础kernel、8个T3微例 | 默认绝对容差1e-6、支持浮点范围有限；无真实数据、规模、统计或困难限时比较 | 将最优与可行限定该profile和数值容差，不作一般定理或性能优势声明；runner边界补纯mock | ACCEPTED_LIMITATION |

## 源式与实现对应

| 项目 | 判断 | 证据与范围 |
|---|---|---|
| M01–M02：f/h/K和母弧 | ACCEPT | 已审kernel对应TeX:74/82，模型调用min(vM,h(v[Δ−2rho]^+))；不使用旧V0线性逐车容量。local保留零容量母弧，AV由物理域生成两方向 |
| M03–M05：车辆 | ACCEPT | 对0…T核人工跨方向半开占用、AV总在途和两hub累计出发/到达；无AV终端复位。mandatory人工闭环，不凭空增加manual跨城/empty |
| M06–M07：wave | ACCEPT，逐arc定义 | 每(i,j,city,flow)至多一个binary u，x=Σv u，Σg≤ΣK(v)u。N=0时selector为空，x和cargo被等式/容量压为0，空域不报错 |
| M08：AV共享 | ACCEPT | 同弧全部订单共享M_auto×y；无该方向订单仍有y。反向单微例确实需要正向无货AV行动 |
| M09：窗 | ACCEPT，整数域profile | 只禁止origin起点<s和destination终点>e，匹配E；未把每腿套同一窗，未另加sorting延迟 |
| M10–M12：时序与需求 | ACCEPT | AV累计出发≤已完成pickup，destination累计出发≤已到AV，≤允许同刻交接；两城总g+z=d。T点D≤A≤P且P=D=d−z，推出AV总量相等；validator显式验三腿总量是完整闭合域下冗余校验，不可移用到有跨界存量的RH |
| M13：变量域 | ACCEPT，零值定义显式 | x/y非负整数，u binary，g/ĝ/z连续非负；z≤d由非负需求平衡蕴含。原Z+的解释由profile选择并披露 |
| M14：目标 | ACCEPT，分钟/每车小时定义 | local(j−i)t0/60与AVτt0/60各乘费率，另加逐单罚金；不加处理费或waiting项。只换算一次 |
| M15：导出与provenance | ACCEPT | 六组全量含零、小量和单元素tuple z；导出索引、输入/source散列、种子/容差、状态/incumbent/bound/gap与验证；HEAD不能代表未提交源码的限制明示 |
| 独立validator | ACCEPT，数值范围 | 自行从原f重建母弧，以load≤vM和f(load)≤v[Δ−2rho]^+核容量，不导入K表/network或Gurobi表达式；独立重放车辆、hub、逐单时序/需求与成本。共用validate_instance仅为schema，未共用被核验容量计算。六组齐全、域/重复键/NaN等有防线 |
| runner | ACCEPT，静态与已报执行范围 | 无incumbent成本/解为空；只有独立验证通过才schedule_valid；限时可行不标optimal。构造/provenance时间先扣，单线程Seed0。明确协作TimeLimit不保证验证/export结束前的硬实时截止 |
| 范围隔离 | ACCEPT | 旧registry和main没有接入本V1模型；旧V0与V0-F-EMPTY均保留。完整V1-F/RH、随机到达、稳态与QVI不在本交付 |

## 实际证据及来源

PI首次执行：配置8+微型MIP8+validator36+已有kernel22，共**74项OK，suite0.500秒**；8run均OPTIMAL、独立核验与目标断言通过，目标依次`.5,.5,12,100,.1,5,100,4`。Reviewer核读每个断言，没有运行这8次求解，也没有见到丢失raw解；8run结果属于PI证据，不是Reviewer亲测。

Reviewer实际运行：

```powershell
& 'C:/Users/32599/.conda/envs/pavane/python.exe' -B -m unittest tests.test_v1_mandatory.ConfigurationTests tests.test_v1_validation -v
```

**44项PASS、exit0，测试0.014秒、进程约0.247秒**，无MandatoryMicroSolveTests或额外solver run。正负例覆盖one-hot/缺selector、stem/原f、共享容量、车辆/hub、同刻交接、时序/窗/需求、连续z/小量、反向和empty AV、非法数值/键/域、objective篡改及溢出。Reviewer还只读核对断言JSON所列6份实现/测试SHA256，全部对应当时文件；metadata改版后的最终散列另记，不能假称历史执行对应新版指纹。

联合wave成本5仅证明该有限微例确受联合容量影响；没有执行旧优化器公平benchmark，不支持性能或普遍政策优势。没有随机重复、真实样本、统计比较或大规模gap证据。

## 最终变更复核

最终configuration常量、optimizer、runner、validator均使用 `author_confirmed_profile` 和决定ID `V1M-D1-D6-A-2026-10-18`；六个profile字段仍显式必填，确认没有扩大支持域或修改数学约束。实现profile与决策记录已登记作者采用，旧断言JSON保留历史条件语义。用户采用不自动修改TeX或完成F/RH/S/QVI。

`experiments/reports/v1_mandatory_confirmed_micro_results.json` 含两份真实完整payload。Reviewer读到两例分别OPTIMAL、schedule_valid、成本12和5、Threads1/Seed0、正确确认状态/决定ID；将各自保存的输入重建为V1MConfig/V1Order/V1MProfile，再把保存schedule与objective送入独立validator，**纯Python重放两例均通过，分别12.0与5.0，exit0**，无solver。首次尝试该只读重放的shell内联引号错误导致SyntaxError，切换here-string后才取得有效结果；未产生文件或求解。

PI报告两次最终run的进程exit0、约0.395秒，单runelapsed约0.156/0.047秒；加首次8run共10run，用完授权20秒理论上限。Reviewer没有增加solver次数。PI另报告配置/validator/kernel和4项runner纯mock合计70项PASS、0.043秒；这些是PI执行证据，不与历史74项误加为新增不同测试。

Reviewer已读 `tests/test_v1_runner_status.py`：建模耗尽预算不optimize、TIME_LIMIT无incumbent保留bound且无伪造解、依赖缺失、validator异常向上传播且dispose。Reviewer另执行 `python -B -m unittest tests.test_v1_runner_status -q`，**4项PASS、exit0、测试0.005秒、进程约0.174秒**；没有真实许可证异常或困难限时求解实测。证据保存异常与额度异常是两件事：上轮本review工具因 **workspace out of credits** 未执行，不是策略判定不安全或人工审批拒绝；首轮MIP执行器的 **PermissionError** 是Python写JSON权限问题。恢复额度后本review才实际落盘。

最终接受是对已采用定义的V1-M基准及有限数值证据，不包括源文完整语义等价、性能优势、RH或QVI。没有尚需追加solver才能修复的阻断项。
