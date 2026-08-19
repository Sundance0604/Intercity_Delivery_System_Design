# Simulation workspace

本目录保存按照 `docs/numerical_experiment_protocol.md` 执行的仿真过程和结果。

目录约定：

```text
simulation/
├── run_protocol.py                  # 可复现批次运行器
└── runs/
    └── <batch_id>/
        ├── README.md                # 批次说明和当前状态
        ├── manifest.json            # 环境、Git、数据和参数快照
        ├── preflight.json           # SQLite、城市对和订单预检
        ├── logs/                    # 完整终端日志
        └── results/                 # CSV、完整 JSON 和逐算例 JSON
```

正式结果应通过 `run_protocol.py` 生成，不手工修改。每个批次均使用绝对数据路径，并在
`manifest.json` 中保存文件大小、修改时间、Git commit、Python/Gurobi 版本、求解器、
随机种子和完整配置。

当前正式环境为 Conda `pavane`。


## 当前有效审计

- `fix_audit_20260819.md`：阶段 C 前的成本单位、Algorithm 2、Rolling Horizon、异常货流、路径和城市对校准修复；
- `runs/FIX3`：相同 LA–SF 订单、旧罚金 350 的最终有效对照；
- `runs/FIX4`：相同订单、默认罚金 10 的最终有效经济权衡对照。`FIX1/FIX2` 为无效中间审计批次。

`progress_20260819.md` 中阶段 B 的原始表格作为问题发现记录保留，但已经被修复审计取代。长批次标识会自动映射为最多 24 字符的“可读前缀 + 哈希”目录名，完整标识保存在 manifest。
## 阶段 C 先导实验（2026-08-19）

- [阶段 C 先导仿真报告](stage_c_pilot_20260819.md)：罚金 1–20 校准、三种运营机制、三档车队以及 200/500 单规模测试；
- `stage_c_20260819/stage_c_rows.csv`：统一的逐次求解结果；
- `stage_c_20260819/stage_c_summary.csv` 和 `stage_c_summary.json`：按规模、罚金、机制、车队和算法分组的统计量；
- `analyze_stage_c.py`：从一个或多个批次的 `run_summary.json` 重新生成统一汇总。

阶段 C 运行器支持 `--penalty-values`、`--mechanisms` 和 `--fleet-scales`。当前先导结果将 `penalty_lost=1` 作为 LA–SF、100 单基准；该数值需要随城市对和需求规模重新校准。