# MODEL CONTRACT - source-of-truth mapping

Status: INITIAL / UNVERIFIED. The coordinator must fill this from the
actual local code, paper, and user's QVI notes before scientific analysis.

| Component | Production Intercity/BHH model (verify) | Toy QVI model (specify) | Evidence/source |
|---|---|---|---|
| System and objectives | TBD | TBD | TBD |
| State variables | TBD | TBD | TBD |
| Arrival process / information | TBD | TBD | TBD |
| Admissible actions and intervention times | TBD | TBD | TBD |
| Vehicle and routing constraints | TBD | Explicitly omitted or modeled | TBD |
| BHH scaling and range of validity | TBD | TBD | TBD |
| Cost function and normalization | TBD | TBD | TBD |
| Timing, horizon, and terminal condition | TBD | TBD | TBD |
| Data and initial conditions | TBD | TBD | TBD |
| Metrics / quantities compared | TBD | TBD | TBD |

## Model alignment rule

A proof for the toy model does not prove a property of the full network.
A simulation of the production MIP does not prove a QVI theorem. State
the mapping (or lack of mapping) before comparing outputs.

## Outstanding checks

- Locate and read the actual QVI development notes in the current worktree.
- Determine whether stopping, impulse control, or another DP framework
  reflects the admissible batching/dispatch decisions.
- Distinguish deterministic rolling-horizon information from dynamically
  revealed order arrivals.
