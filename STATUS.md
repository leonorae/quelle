# Status — Quelle

> Quick briefing for agents and humans. Read this instead of traversing
> the full wiki to understand what is happening right now.
>
> **Updated**: 2026-03-15 (three-condition comparison complete on bootstrap)

---

## Active Experiments

| Experiment | Status | Owner | Next action |
|---|---|---|---|
| `behavioral-projections` | bootstrap-results | claude-opus-4-6 | Diagnose C3 collapses (rank sweep at L9,11,16,17); rerun on 7k corpus |
| `variable-bitrate-reasoning` | running | claude-sonnet-4-6 | Run full training, fill RESULTS.md |
| `VVVVVV` | planning | claude-sonnet-4-6 | Run Phase 0 training at d12, fill RESULTS.md Q0.1–Q0.3 |

## Backlog

| Experiment | Priority |
|---|---|
| `gumbel-compression-ablation` | Medium |
| `manifold-capability-probing` | Low |
| `multi-task-vbr` | Low |

## Repo Health

- `wiki/findings/`: empty (behavioral-projections bootstrap results in RESULTS.md, awaiting full-corpus confirmation before wiki writeup)
- `visualize.py`: still stub in `variable-bitrate-reasoning`
- Dashboard: not yet built

## Open Questions

- VVVVVV Phase 0: Do spike channels in nanochat (relu²) fall in first 32 indices? (→ Q0.1)
- VVVVVV Phase 0: Is BOS residual document-varying or near-constant? (→ Q0.2)
- These gate the learned projection gate and BOS-conditioned table extensions respectively.
