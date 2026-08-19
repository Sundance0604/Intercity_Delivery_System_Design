# Stage B paired comparison

- 批次：B1F, B1P, B2F, B2P
- 配对订单校验：通过（30 个算例）

| Orders | Solver | N | Mean cost | Mean unserved | Mean time (s) | Max window (s) | Mean arc reduction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | flexible_direct_mip | 10 | 32550.00 | 32.62% | 0.228 | 0.0000 |  |
| 10 | flexible_direct_rolling | 10 | 81200.00 | 100.00% | 2.437 | 0.0151 |  |
| 10 | paper_candidate_mip | 10 | 39085.00 | 50.16% | 0.137 | 0.0309 | 97.28% |
| 10 | paper_priority_heuristic | 10 | 103500.00 | 0.00% | 0.001 | 0.0026 | 97.43% |
| 20 | flexible_direct_mip | 10 | 47820.00 | 13.03% | 0.495 | 0.0000 |  |
| 20 | flexible_direct_rolling | 10 | 126490.00 | 100.00% | 4.782 | 0.0425 |  |
| 20 | paper_candidate_mip | 10 | 62405.00 | 21.96% | 0.251 | 0.0529 | 96.86% |
| 20 | paper_priority_heuristic | 10 | 135000.00 | 0.00% | 0.010 | 0.0048 | 97.26% |
| 40 | flexible_direct_mip | 10 | 68160.00 | 9.31% | 1.465 | 0.0000 |  |
| 40 | flexible_direct_rolling | 10 | 221270.00 | 100.00% | 11.014 | 0.1939 |  |
| 40 | paper_candidate_mip | 10 | 93290.00 | 7.27% | 0.409 | 0.0725 | 97.05% |
| 40 | paper_priority_heuristic | 10 | 145500.00 | 0.00% | 0.020 | 0.0089 | 97.20% |

逐算例目标差距与订单 SHA-256 见 `comparison.json`。
