# V1 N40 follow-up independent review

Date: 2026-10-19 (Asia/Shanghai). Reviewer: independent referee. Scope: delivered follow-up design, protocol, wrapper and pure tests; later result review will be added below. No Gurobi calls or edits to model, paper, experiment code or raw results by the reviewer.

## Pre-run decision: LIMITED ACCEPT

The delivered artifacts are `research/theory/v1_followup_design_20261019.md`, `experiments/specs/V1_FOLLOWUP_20261019_PROTOCOL.md`, `experiments/specs/run_v1_followup_20261019.py`, and `tests/test_followup_20261019.py`. I read these against MODEL_CONTRACT section 16 and the existing frozen harness/metrics interfaces. The coordinator reported 13 pure tests passing; this statement is attributed to the specialist, not a new reviewer execution.

Final inspected SHA-256 identities:

- Wrapper: `d03db69a1eeac075c622c9e36b7375005d8ca8b55d2a660c1837438fd2e8f5a0`.
- Tests: `de8739799f0a8b2e3fae67ceb63d60034e81b45209156af21c9435ae4ec17bfa`.
- Protocol: `a69ed13ee9e173e795bff6ab3876ff3fe44534406213156bb95bdc0a3dc94194`.

| Severity | Evidence / artifact | Impact | Remedy / condition | Status / recommendation |
|---|---|---|---|---|
| No blocking finding | Wrapper `prepare`, `load`, `worker`: original preflight and frozen-source hashes, original batch manifest, immutable input bytes/content, prior micro evidence, reference record/raw hashes, additional script/test/protocol hashes | Sources and reused observations remain distinguishable from new solves | Do not refresh original 63-source manifest; verify actual prepared manifest before interpreting results | Pre-run ACCEPT, actual batch identity pending |
| No blocking finding | Wrapper `run`, `worker`; fake-execution tests | Exclusive target creation, parent registration, exclusive worker claim and retained existing directories prevent automatic replacement/retry. Normal TIME_LIMIT without incumbent is kept without zero cost; replay/worker failures stop later solves | Preserve all 30 planned rows, including absent or interrupted attempts | Pre-run ACCEPT, actual records pending |
| No blocking finding | BUDGET and `run` | 30 new full attempts, logical 5 s, process hard 15 s, total 300 s including prepare and sessions. Timeout is rechecked after registration and shrinks with remaining budget | Report soft logical overruns separately from process/total budget; do not infer total compliance solely from solver runtime | Pre-run ACCEPT, actual elapsed evidence pending |
| Interpretation limitation | Design sections 1–3 and protocol sections 1, 4 | Five reused N40 seeds were chosen after observing prior Direct adoption. These are exploratory paired policy comparisons, not independent confirmatory evidence | No sample-size inflation from reuse; no seven-way significance selection or generalization | ACCEPT with stated limitation |
| Interpretation limitation | `cost_difference_interval`, seven COMPARISONS, inherited metrics field `global_optimal_oracle` | Bound interval formula for right minus left is correct; inherited field name can suggest exact mathematical optimality although status OPTIMAL respects solver tolerances | Retain actual LB/UB/gap, old 4103 TIME_LIMIT and old 4105 nonzero gap; call these solver-tolerance certificates, not exact oracles | ACCEPT with stated limitation |
| Verification outstanding | Future 35 records and 35 seed-pair rows | No new numerical result is established by this pre-run review | Independently hash, replay and recompute all delivered rows after execution; no additional solve | Results NOT YET REVIEWED |

The policy nesting argument concerns mathematical feasible sets and exact optimum direction; it does not constrain ordering of finite-time incumbents. The lower-bound difference intervals and across-seed Student-t descriptive intervals answer different questions and must remain separate. Partial/no-incumbent results must not enter complete-cost means. Direct remains the approved destination-city abstract-node assumption; these solves do not validate physical hub connectivity, QVI, or lossless RH pruning.

## Delivered batch review: LIMITED ACCEPT

Reviewed batch: `simulation/v1_runs/P1_FOLLOWUP_20261019_01/`. The coordinator delivered manifest, 30 immutable new job directories, five original reference identities, execution sessions, outer execution records, `independent_replay.json`, `tables/analysis.json`, and two CSV tables. I independently checked their contents after execution, using a readonly Python process and the original-f validator; I did not call or import Gurobi or launch any solve. The reviewer all-35 original-f replay took 15.765 s, separate from the coordinator's 22.625 s post-execution replay.

Reviewed derived identities:

- `tables/analysis.json`: `5b99ed1691dffc02f8e12d7b261446fba2d6315448f97259aa5b4444464b83b8`.
- `independent_replay.json`: `4207d257b1463186dc4c690a2532502c1edaf4588707302fa7c29f53cfe491d1`.
- Original 63-source content remains `5c8b233fbc50cda4d23383f404915027e2b8d63de5faa03a30f6703063b1c6c4`; no refresh was needed.

### Verified evidence

1. `load` passes frozen and supplemental identity gates. All five copied input identities, original-reference record/raw identities and prior micro identities match. For every one of 35 records I checked raw SHA, config SHA and saved config content against the prescribed configuration, input/orders identities, and record/raw solver status.
2. Independently replayed all 35 full schedules with `validate_flexible_schedule`, original profile, original orders, complete horizon and reported objective. All passed. No run or pair was excluded.
3. Exactly 30 new full attempts and 30 new optimize calls occurred: 26 OPTIMAL, four TIME_LIMIT. Including the five reused observations, 35 unique records comprise 30 OPTIMAL and five TIME_LIMIT, each with a complete independently feasible incumbent. The five reused optimize calls are historical, not new calls.
4. Recomputed all 35 seed-pair incumbent differences and all 35 lower/upper-bound difference intervals directly from raw costs and bounds. Recomputed seven groups' n, mean, sample sd, median, min/max and t4 descriptive intervals. Checked all 35 run CSV costs, both CSV row counts and all 35 phase-time sums. They agree with the delivered tables.
5. Internal prepare plus execution wall time is 92.578 s; independently reported outer process time is 92.813 s. Both are below 300 s. Maximum new worker parent time is 7.734 s, below 15 s. The four new TIME_LIMIT runs have `budget_exceeded=true` and runner elapsed about 5.360–5.391 s, so the logical 5 s budget is a soft limit and was exceeded. These overruns must be reported; the hard process and cumulative bounds were respected. No stopping error, replacement, retry, enlarged scale or additional sensitivity job is present.

### Findings and claim decisions

| Severity | Evidence / artifact | Impact | Remedy / required interpretation | Status / recommendation |
|---|---|---|---|---|
| No blocking finding | Batch manifest, source/supplement gates, all 35 record/raw identities and independent replay | Traceable model/profile/input comparison and feasible incumbent costs | Retain raw/status/bound/gap and all planned rows | Verified; ACCEPT for reproducibility and feasibility of this batch |
| Moderate limitation | Four new seed4103 TIME_LIMIT runs: T2/.1, T3/.1, T2/.05, T2/.2; old T3/0 seed4103 also TIME_LIMIT | Equal saved incumbent costs do not establish exact policy equivalence | Distinguish incumbent differences from optimum certificate ranges; preserve original 4105 nonzero gap as well | Disclosed; ACCEPT with limitation |
| Minor operational limitation | `run_execution.json`, sessions and raw timing | Four logical-limit overruns despite compliance with 15 s / 300 s hard limits | Report all three time notions and retain the runs, rather than relabeling them exactly five-second solves | Disclosed; ACCEPT with limitation |
| Supported finite-instance claim | T2/.1 minus T0: per-seed incumbent differences −75.8066615, −125, −25, −125, −25; all five certificate intervals strictly negative | Direct-enabled policy improves optimal objective relative to T0 for these five fixed synthetic instances, within solver certificate accuracy. The 4103 interval is [−49.6392297, −24.9010022], so this instance remains directionally supported despite a limited incumbent | Keep finite-instance and aggregate-node scope; do not use the descriptive CI as general-population confirmation | ACCEPT within stated scope |
| Supported descriptive claim | T2/.1 minus T0 mean −75.1613323, t4 CI [−137.2461481, −13.0765165]; T3/0 gives same saved incumbent differences | Observed mean cost decrease under the fixed five-seed design | Call CI descriptive, exploratory and one of seven comparisons; no general benefit probability or independent replication claim | ACCEPT within stated scope |
| Unresolved general effect | T1 versus T0, matched-m T3 versus T2, and minimum sensitivities have saved cost differences at floating precision zero | No observed additional incumbent-cost benefit from explicit empty or changed minimum. Seed4103 certificate intervals for these comparisons include zero | Do not assert exact universal equivalence or uselessness of empty/minimum; lack of effect is limited to saved runs | ACCEPT for observation; REJECT any general equivalence claim |
| Supported saved-schedule observation | All 35 raw schedules: explicit empty count zero, zero-cargo Direct count zero, minimum positive Direct wave-average load 3.0 | The tested .05/.1/.2 lower bounds are slack on the saved Direct schedules; different direct quantities can reflect alternative solutions | State this about the saved schedules, not every optimum or all instances | ACCEPT within stated scope |
| Scope limitation | Five old N40 samples, same synthetic parameters and prior-result-aware selection | This is useful verified follow-up knowledge but not operational calibration, independent confirmation, large-scale validation, pruning attribution or QVI evidence | Keep those wider claims deferred | ACCEPT for bounded follow-up; wider claims unsupported |

Group means independently recovered: T0 and T1 cost 1745 and served quantity 32.2; T2/.1, T3/.1, T2/.05, T2/.2 and reused T3/0 cost 1669.8386677 and served quantity 37.5016133, up to floating precision. The service increase is 5.3016133 continuous synthetic equivalents; it is not an integer count of fully delivered real shipments. All five paired Direct-versus-T0 improvements have the same intended instance, parameter and offline-information basis.

The matched empty and minimum contrasts remain especially uncertain at seed4103: T3/.1−T2/.1 [−16.0249738, 24.6392297]; T3/.1−T3/0 [−16.0249738, 16.0884982]; T2/.05−T2/.1 [−24.1237528, 24.6392297]; T2/.2−T2/.1 [−20.2327015, 24.6392297]. Near-zero incumbent differences cannot erase these uncertainty ranges.

**Final batch verdict: LIMITED ACCEPT.** The prepared design was executed without widening scope, the full planned denominator is present, the reported costs are independently feasible, and the paired statistics and optimization certificate intervals are correct. Acceptance concerns these delivered artifacts and five synthetic instances. A final prose report must preserve the above limitations and may be checked separately after the writer delivers it.

## Final prose consistency check

The completed `experiments/reports/v1_followup_20261019_results.md` was subsequently delivered. I read its seven sections and independently recalculated its additional certificate-mean intervals, busy fractions, direct shares, lead-time group means, maximum replay residual and all six listed artifact SHA values. These agree with the saved evidence. Its fixed-sample, offline, solver-tolerance and logical-budget limitations are explicit; it does not turn zero incumbent contrasts into a general equivalence result. The report may replace its administrative `INDEPENDENT_REVIEW_PENDING` state with `LIMITED_ACCEPT` and link this review. No substantive blocker remains. **Report verdict: ACCEPT with the same batch limitations.**
