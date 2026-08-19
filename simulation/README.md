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