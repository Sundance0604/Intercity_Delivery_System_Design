# BHH Codex multi-agent kit

Custom subagents: `literature`, `experiment`, `theorist`, `referee`.
Main Codex remains the research coordinator. The kit changes neither
research source code nor solver dependencies. This kit is for BHH and
Intercity delivery, not DRL_CO.

## Start on Windows PowerShell

```powershell
Set-Location 'D:\mycodelife\workshop\AI\multiagent-test'
codex --version
codex
```

Confirm Codex trusts this project when prompted so `.codex/config.toml`
can be loaded. Paste the contents of
`coordination/FIRST_RUN_PROMPT.md` into Codex.

In the interactive Codex CLI, run `/subagents` to inspect subagent
activity after they have been spawned. Subagents are not persistent OS
processes or independent API keys; the primary Codex starts them when
the prompt asks it to.

## Why this project has two kinds of computation

`experiment` handles production Gurobi, rolling-horizon, and full-model
benchmarks. `theorist` handles proofs and standalone, small toy-model
numerical calculations. Neither result substitutes for the other.

## Existing config and instructions

If an installer warning reports a pre-existing `.codex/config.toml`,
edit it instead of overwriting. Ensure its `[agents]` table contains:

```toml
[agents]
enabled = true
max_concurrent_threads_per_session = 4
```

To let the literature agent verify external papers using Codex search,
you may set top-level `web_search = "live"` (outside any TOML table),
subject to account, workspace, sandbox, and network availability.

The installer appends its `AGENTS.md` block when the repository already
has a root AGENTS.md; review and resolve any instruction conflicts.

## Operation / safety

Start with the initial audit. No long unattended solver jobs, changes
to existing baselines, or claims of a completed QVI proof. Keep outputs
in the role-specific directories; review `git diff` before committing.
All four custom agents share the root session's sandbox/approvals.
