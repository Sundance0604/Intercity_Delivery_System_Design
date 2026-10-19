# Fixed-N four-direction experiment contract

2026-10-19, Asia/Shanghai. Status: **design audit; no new operational result**.

This contract follows the author's choice to run all four directions at one
fixed order count, then change that count only in a later round. It reads the
actual V1-F implementation, `coordination/MODEL_CONTRACT.md`, and Steps 7–10 of
`experiments/specs/source_snapshots/BHH_CFS_V1_Numerical_Experiment_Runbook.md`.
The old runbook's order-count recommendations do not override this choice.
No production solve, model edit, or change to the prior 63-source freeze is
part of this mathematical audit.

## 1. Common sample and identifiers

For each sample seed, materialize one immutable collection of exactly N orders.
Use those identical order records, quantities, windows, directions, penalty,
fleets, time unit, real horizon, terminal convention, and synthetic-unit contract
through all four directions. Pair by input hash and seed, not order count alone.
Record configuration, information profile, algorithm, minimum-load policy,
capacity variant, H/h/E, source identities, and solver controls for each run.

Within a round a reused result is one solver attempt, not another independent
sample. References must point to the actual run and its verified hashes. Prior
batches are reusable only after matching all sample/model/configuration/source
and certificate requirements, with reuse declared explicitly. A later N with
the same seed is not automatically a nested or paired sample: existing sampling
is not asserted to have the prefix property. Fixed N improves cross-direction
control but does not make different mathematical models or information sets
identical.

Define the following per-seed run IDs; suffixes denote policy and parameters,
not independent observations:

| ID | Direction first creating it | Model / algorithm | Changed setting |
|---|---|---|---|
| A0 | D1 mechanism | pooled, full offline, T0 | m=0 |
| A1 | D1 mechanism | pooled, full offline, T1 | m=0 |
| A2 | D1 mechanism | pooled, full offline, T2 | m=0.1 |
| A3 | D1 mechanism | pooled, full offline, T3 | m=0 |
| A4 | D1 matched minimum | pooled, full offline, T3 | m=0.1 |
| B0 | D2 algorithm | pooled, RH-dense, T3 m=0 | H=12, h=2, E=12 |
| B1 | D2 algorithm | pooled, RH-pruned, T3 m=0 | H=12, h=2, E=12 |
| C0 | D3 minimum-load | pooled, full offline, T2 | m=0.05 |
| C1 | D3 minimum-load | pooled, full offline, T2 | m=0.2 |
| C2 | D3 Direct price | pooled, full offline, T3 m=0 | c_direct=0.75 × baseline |
| C3 | D3 Direct price | pooled, full offline, T3 m=0 | c_direct=1.25 × baseline |
| C4 | D3 look-ahead | pooled, RH-pruned, T3 m=0 | H=8, h=2, E=12 |
| C5 | D3 look-ahead | pooled, RH-pruned, T3 m=0 | H=16, h=2, E=12 |
| D0 | D4 local pooling | localSeparate, full offline, T3 m=0 | local capacity only |

Thus D1 creates 5, D2 creates 2, D3 creates 6, D4 creates 1: **14 unique
top-level runs per seed**, or 70 for five seeds. RH window optimize calls and
micro-validation calls are additional counters, not additional sample seeds.

## 2. Reference/comparison mapping

All differences below mean comparison minus reference; speedup uses reference
wall time divided by comparison wall time. Positive cost difference is worse
under identical objective coefficients. Retain every failed/no-incumbent/partial
attempt in the denominator; eligible complete pairs alone receive cost deltas.

| Direction | Reference | Comparison | Interpretation |
|---|---|---|---|
| D1 | A0 | A1 | independent empty enabled with Direct disabled |
| D1 | A0 | A2 | loaded Direct policy added with m=0.1 |
| D1 | A0 | A3 | complete T3 policy versus T0 |
| D1 matched | A2 | A4 | independent empty enabled, same m=0.1 |
| D1 matched | A3 | A4 | impose m=0.1 within T3 |
| D2 | B0 | B1 | cargo pruning with the same release-revealed rule |
| D2 oracle | A3 | B0 and B1 separately | offline reference versus online execution |
| D3 load | A2 | C0 and C1 separately | one-factor minimum average load |
| D3 price | A3 | C2 and C3 separately | one-factor Direct arc price |
| D3 H | B1 | C4 and C5 separately | one-factor start look-ahead H |
| D4 | A3 | D0 | one-factor local spatial pooling comparison model |

These are 15 comparison edges per seed when the two oracle edges are retained,
but still 14 unique runs. Do not count an edge as a new observation.

The actual `flexible_rolling.py` reveals orders with release at or before the
current time; increasing H does not reveal future orders. It changes the start
domain and, because completion end is min(T, current+H+E), the completion domain.
Keeping E=12 therefore means a fixed extension length, not a fixed absolute
completion endpoint. Near real T, H values can yield identical clipped domains.
Online versus offline results combine information and control-policy effects;
they do not estimate a pure algorithmic gap under common information.

## 3. Mechanism and price interpretation

The minimum-load inequality is q >= m v for total Direct wave cargo q and
integer vehicles v. It excludes zero cargo and positive average loads below m;
it does not ensure each individual vehicle is loaded. T2 disables independent
empty. T3 with m=0 retains the source's zero-cargo Direct arcs, which can move
vehicles. Consequently A3 versus A2 is not a clean independent-empty ablation;
A4 versus A2 supplies that comparison. Report zero-cargo Direct movements and
explicit empty movements separately. If either is zero, retain that result
without retuning the sample or model to force usage.

For C2/C3 explicitly set c_empty_hour=25. In `flexible_configuration.py`, a
None empty rate falls back to c_direct_hour, so leaving it unset would silently
change two cost components. Even with explicit empty fixed, T3 m=0 permits
zero-cargo Direct: changing c_direct changes that channel's cost too. Call this
Direct arc-price sensitivity, not exclusively the price of freight carriage.
Different price settings have different objectives; their cost change is a
scenario response, not a quality ranking under one shared objective. Report
service and mode choices alongside their scenario-specific objective values.

## 4. LocalSeparate definition and independent feasibility

**Proved algebraic statements under a>0, b>=0, rho>=0, M>0, integer v>=0:**
let f(q)=a q+b sqrt(q), duration D>=0, and B=max(D-2rho,0). The adopted pooled
local capacity is min(v M, f_inverse(v B)); the controlled comparison is
min(v M, v f_inverse(B)). For v=0 both capacities are zero. For v>0 localSeparate
is independently checked by q<=v M and f(q/v)<=B, equivalently

    a q + b sqrt(v q) <= v B.

This directly checks the separate time inequality from f, without reusing an
optimizer capacity lookup. The existing pooled test f(q)<=v B is weaker for
v>=1 and cannot alone certify a localSeparate result. The normal independent
validator must still check all cargo, vehicle, selector, time, terminal, cost,
and mechanism constraints, followed by the separate original-f check. Its
arithmetic tolerance and units must be disclosed.

For q>=0, sqrt(v q)>=sqrt(q) when v>=1. Therefore any localSeparate-feasible
wave is pooled-feasible; with identical remaining constraints, the full-horizon
comparison model's feasible set is a subset of the pooled model's. Hence its
true minimum cost cannot be smaller. This is a model-level statement, not a
claim about real routing or the order of time-limited incumbent costs.

Only local capacity changes: retain Direct formula/capacities, explicit empty,
local/Direct/AV/selector mother domains, costs, and full offline information.
Different local optimum schedules are expected, not evidence of domain drift.
This comparison excludes cross-vehicle local spatial concentration within the
stated surrogate; it does not disable every BHH term everywhere in the model.

Capacity-curve diagnostics use v=1,2,3,5 without optimizer calls. Equality is
expected at v=1, b=0, zero B, or common physical-cap saturation, and can leave
operational cost unchanged. D>0 with D<2rho has zero cargo capacity under the
adopted clipped-budget convention; this does not certify physical zero-load
tour geometry. Nonzero capacity separation does not imply a strict system cost
or service improvement, since another resource or rejection can dominate.

## 5. Certificates, domains, and execution boundaries

Use full-horizon certified optimum only when solver certificate permits it.
Otherwise keep incumbent upper bound and valid lower bound separately. A
certified cost improvement of comparison B over reference A requires
LB_A > UB_B beyond tolerance, not merely two different incumbent values.
Never compare an RH window bound to a whole-day objective or rank a partial
bill against complete execution. Service is not the objective being maximized,
so a lower-cost solution can serve less.

For B0/B1, verify vehicle/selector mother keys per aligned window; matching
keys does not imply matching inherited commitments, bounds, cargo candidates,
or future trajectories. A final cost difference does not by itself identify a
single pruning deletion as its cause. This suite reports trajectory outcomes;
same-history single-window diagnosis is a separate task if needed.

Declare solver budgets and one-attempt/resume semantics before execution.
Zero served quantity gives an undefined delivered-flow mean/Direct share,
not a fabricated zero; denominator-zero ratios and speedups must be explicit.
An infeasible/no-incumbent comparison yields no complete cost difference.
Five seed CIs describe the chosen sample distribution, not timing repeats or
network-wide causal identification. Tiny runtimes require cautious claims.

Bind the additional suite sources separately, including localSeparate builder
and validator, while continuing to verify the old 63-file source manifest.
An old result's pooled replay cannot be relabeled as a separate-model
certificate. This contract makes no new operational-benefit, general lossless
pruning, empirical BHH-calibration, or QVI claim.
