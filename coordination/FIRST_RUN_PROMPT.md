# First four-agent run - paste this into the primary Codex session

You are the primary research coordinator for this local BHH / Intercity
project, not the DRL reinforcement-learning project.

First read AGENTS.md, coordination/MODEL_CONTRACT.md, CLAIMS.md,
TASKS.md, and PROGRESS.md. Inspect git status and the actual local tree.
Locate my current BHH and QVI mathematical notes. If they are absent,
report that honestly and do not invent an updated QVI model.

PHASE 1: Spawn three distinct custom subagents concurrently:
- literature: Map prior work on BHH scaling, batching, stochastic
  arrivals, dynamic dispatch, optimal stopping, impulse control, and QVI.
  Verify citations and write research/literature/landscape.md.
- theorist: Based on the actual local notes, audit whether a proper QVI
  is justified. List state, intervention operator, assumptions, gaps,
  candidate statements, and tiny counterexamples. Write
  research/theory/qvi_assessment.md. Small toy numerical checks are OK.
- experiment: Only audit the runnable code and current experiments.
  Identify CLI/tests, runtime, data, algorithm baselines, and quick
  reproducibility checks. Prefer a dry-run; do not run Gurobi full solves
  or modify existing algorithms. Write experiments/reports/inventory.md.

Wait for all three. Do not let them write to the same files.

PHASE 2: Spawn the referee agent to independently assess the three
deliverables, especially whether the toy QVI and the full BHH optimizer
solve the same problem. Write coordination/reviews/initial.md.

PHASE 3: As primary coordinator, read the reports and update
MODEL_CONTRACT.md, CLAIMS.md, TASKS.md, and PROGRESS.md based only on
evidenced conclusions. Summarize results, open disputes, and the next
2-3 research tasks for my approval. No baseline code changes, no expensive
numerical runs, no pushes, and no fabricated references.
