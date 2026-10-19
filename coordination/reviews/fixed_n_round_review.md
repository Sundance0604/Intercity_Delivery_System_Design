# Fixed-N round independent review

2026-10-19, Asia/Shanghai. Reviewer owns this report only. No solver was
initialized or optimization run by the reviewer.

## Scope and evidence

The coordinator identified the delivered specialist artifacts:
`research/theory/fixed_n_round_contract.md`, `experiments/fixed_n/round.py`,
`experiments/fixed_n/pooling.py`, `experiments/fixed_n/default_design.json`,
`experiments/run_fixed_n.py`, their two test files, and
`experiments/specs/FIXED_N_ROUND_GUIDE.md`. The review also reads
`coordination/MODEL_CONTRACT.md` section 17 and the frozen optimizer's use of
local and Direct capacity tables. The final source identity and tiny artifacts
are reviewed below. Earlier pre-run findings retain their resolution status.

## Preliminary mathematical assessment

**ACCEPT, limited to the stated counterfactual.** For integer v>=1,
`K_separate(v)=v*K_pooled(1)` is the equal-share independent-vehicle local
surrogate, with `q<=v*M` and `a*q+b*sqrt(v*q)<=v*B`. Because
`sqrt(v*q)>=sqrt(q)`, separate-feasible waves are pooled-feasible under the
stated nonnegative coefficients. The exact full-horizon feasible-set
inclusion does not order time-limited incumbents. v=0 is explicitly zero.

The implementation changes the instance local capacity table before build;
Direct capacity and mother arcs are retained. Its additional replay uses the
original inequality, independently of its K-table constructor, after the full
pooled validator checks the rest of the schedule. Passing the pooled validator
alone is correctly not treated as a separate-model certificate. This is not a
claim about empirical routes, all BHH terms, or complete elimination of
spatial pooling from Direct.

## Findings and resolutions

| Severity | Evidence | Impact | Remedy | Status |
|---|---|---|---|---|
| Medium | `round.py::make_plan`, float levels formatted with `:g` | Distinct custom levels can collide in case IDs, silently skip attempts or overwrite comparison lookup. | Reject duplicate case IDs/labels before prepare; retain exact levels in identity. | RESOLVED: final code rejects colliding IDs; covered by a pure test. |
| Medium | `round.py::paired_rows` | Incumbent differences alone do not establish mechanism/pooling optimum differences when limits bind. | Export full/full certificate difference intervals, keep RH global bounds undefined. | RESOLVED: final intervals checked against independent arithmetic; RH intervals are null. |
| Low / boundary | `round.py::run_round` final-only session record and exclusive `run.lock` | Abrupt controller termination leaves the persistent lock; the next run fails closed before any worker can start. It does not offer arbitrary crash recovery. | Document that unresolved locks require inspection and are not silently deleted to resume or recharge budget. | RESOLVED by existing exclusive lock; normal-resume budget protection remains applicable. |
| Low | Guide initial N parity sentence | Odd N>=3 is supported by ceil/floor sampling but documented as disallowed. | Remove even-N restriction and disclose directional imbalance by one for odd N. | RESOLVED in final guide; plan N=3 pure test passes. |
| Low / scope | `round.py::report` and RH raw window hashes | Saving hashes is not itself an aligned dense/pruned domain comparison; equal domains also do not imply equal histories. | Independently align windows before asserting domain equality; retain trajectory caveat. | RESOLVED for this tiny batch by independent comparison of all 12 aligned windows; no generic same-history claim. |

## Pre-run decision

**LIMITED ACCEPT for the proposed N=4, seed4101, 14-top-level-task functional
smoke, cumulative 180 seconds, logical 3 seconds/run and hard 10 seconds/worker.**
Final code now rejects colliding IDs and resume parameter drift, exports
full/full reported-bound cost-difference intervals, and leaves RH global
intervals undefined. Odd-N and persistent-lock boundaries are corrected in the
guide. The reviewer independently ran all 23 new pure tests with
`sys.modules['gurobipy']=None`: 23 passed in 0.092 seconds. The initial restricted
attempt encountered temporary-directory PermissionError in three fixture-writing
tests; the permission-enabled rerun passed. This is an environment retry of
pure tests, not an optimization attempt or replacement of scientific results.

The mathematical counterfactual and predeclared default 14-job/one-seed design
provide no blocking issue for that bounded functional validation. This does not
accept new scientific efficacy claims or a full 70-job research round.

## Post-run evidence

Coordinator delivered
`simulation/fixed_n_runs/FIXED_N_SMOKE_20261019_01/`, including first execution
evidence, a completed-round resume, immutable manifest/plan/inputs, all 14
job raw/record/request files, two sessions and timestamped analyses. This
review uses the latest analysis `20261019T011701065665Z` and independently
checks source/input identity through `load_round` and each raw provenance.
Supplementary suite SHA256:
`9c1fa8f3d7acbef31159486d2be05a804e62f7bf87b18e96936ac7f3d7e3bb26`.
Core content SHA256 remains
`5c8b233fbc50cda4d23383f404915027e2b8d63de5faa03a30f6703063b1c6c4`.

The reviewer directly called the frozen original full schedule validator on
all 14 raw schedules with their requested information profiles and objective,
without trusting prior `PASSED` flags. All passed. For the separate model,
the reviewer separately reconstructed aggregate local cargo per wave from raw
keys and checked `q<=v*M` and `a*q+b*sqrt(v*q)<=v*B`; maximum positive residual
was zero. It has a distinct model ID and a comparison-only full bound scope.
No solver import or solve was needed (`gurobipy` was disabled).

Verified facts:

- 14 unique planned and attempted top-level runs: 10 full `OPTIMAL` and four
  RH `COMPLETED`. They used 58 actual optimize calls: ten full calls and 48
  RH window calls. No error, timeout, incomplete schedule or unattempted case
  occurred in this functional batch.
- Stage creation order is mechanism, algorithms, sensitivity, pooling. The
  four stage reference counts are 5, 3, 9, 2; these 19 references reuse the
  14 actual runs and are not 19 independent samples. The one frozen input is
  used throughout.
- All 15 pair edges are eligible. Cost differences and full/full bound
  intervals recompute correctly; RH has no global bound/gap or global
  difference interval. All one-seed confidence intervals are null.
- Dense/pruned mother-domain hashes and counts match at all 12 aligned
  windows, including the start/control/completion endpoints. Matching hashes
  do not certify matching inherited histories or general lossless pruning.
- CSV row counts agree with raw/plan: runs 14, pairs 15, direction tables
  5/3/9/2 and windows 48. Exclusive phase timings sum to each raw elapsed
  time within 1e-6 seconds.
- Initial prepare/controller time is 30.593 seconds within the 180-second
  cap. First postprocessing is separately 1.688 seconds, outer elapsed
  32.438 seconds. The second session attempts zero workers; cumulative
  execution becomes 30.624 seconds. All 14 current raw hashes equal both
  before/after maps in `resume_execution.json`, and the second session's
  attempt list is empty. No persistent lock or failure-stop file remains.

All 14 tiny outcomes cost 350 and serve all four equivalent units. These are
functional observations, not evidence that mechanisms, sensitivity levels,
pruning or pooling are generally equivalent. A single seed and tiny runtimes
cannot establish effect sizes, scaling or speedup. Real recovery from an
unfinished batch and actual solver/license failure were not deliberately
induced here; no-retry/error/budget paths have pure-test and code evidence,
while the real resume evidence is specifically for an already complete round.

## Final recommendations

| Deliverable or claim | Recommendation | Status and limits |
|---|---|---|
| Fixed-N mathematical comparison contract | ACCEPT | Algebra and model/information distinctions are explicit; local-separate is a comparison model. |
| Unified automation and guide | ACCEPT within stated scope | Immutable plans and sources, four ordered stages, shared baseline, one-attempt semantics, cumulative budget and report output verified. Persistent unresolved locks fail closed. |
| Tiny functional execution | ACCEPT | All 14 schedules replay independently, separate adds its own stricter check, 15 comparisons and zero-attempt resume verified. |
| New operational benefits, general pruning validity, scalability or pooling efficacy | REJECT as a claim from this batch | This batch supplies no such evidence and the delivered documents do not assert it. A separately declared research round is needed. |

Overall: **LIMITED ACCEPT of implementation and bounded functional
verification.** No production model, theory/manuscript, old raw result or
frozen source file was edited by the reviewer; this report is the reviewer's
only repository artifact.
