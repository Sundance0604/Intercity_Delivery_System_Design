# Stage B paired comparison

- 批次：B1F, B1P
- 配对订单校验：通过（9 个算例）

| Orders | Solver | N | Mean cost | Mean unserved | Mean time (s) | Max window (s) | Mean arc reduction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | flexible_direct_mip | 3 | 33833.33 | 61.79% | 0.217 | 0.0000 |  |
| 10 | flexible_direct_rolling | 3 | 38733.33 | 100.00% | 2.343 | 0.0110 |  |
| 10 | paper_candidate_mip | 3 | 38000.00 | 73.59% | 0.127 | 0.0256 | 97.30% |
| 10 | paper_priority_heuristic | 3 | 115000.00 | 0.00% | 0.003 | 0.0015 | 97.30% |
| 20 | flexible_direct_mip | 3 | 50733.33 | 19.17% | 0.477 | 0.0000 |  |
| 20 | flexible_direct_rolling | 3 | 74200.00 | 100.00% | 4.517 | 0.0275 |  |
| 20 | paper_candidate_mip | 3 | 66850.00 | 43.16% | 0.257 | 0.0459 | 97.01% |
| 20 | paper_priority_heuristic | 3 | 135000.00 | 0.00% | 0.010 | 0.0030 | 97.23% |
| 40 | flexible_direct_mip | 3 | 69750.00 | 15.65% | 1.390 | 0.0000 |  |
| 40 | flexible_direct_rolling | 3 | 145016.67 | 100.00% | 9.740 | 0.0834 |  |
| 40 | paper_candidate_mip | 3 | 97516.67 | 15.57% | 0.400 | 0.0588 | 97.20% |
| 40 | paper_priority_heuristic | 3 | 145000.00 | 0.00% | 0.020 | 0.0060 | 97.20% |

逐算例目标差距与订单 SHA-256 见 `comparison.json`。
