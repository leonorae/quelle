# Status — Quelle

> Quick briefing for agents and humans. Read this instead of traversing
> the full wiki to understand what is happening right now.
>
> **Updated**: 2026-03-09

---

## Active Experiments

| Experiment | Status | Owner | Next action |
|---|---|---|---|
| `variable-bitrate-reasoning` | running | claude-sonnet-4-6 | Run full training, fill RESULTS.md |
| `VVVVVV` | planning | claude-sonnet-4-6 | Run Phase 0 training at d12, fill RESULTS.md Q0.1–Q0.3 |

## Backlog

| Experiment | Priority |
|---|---|
| `gumbel-compression-ablation` | Medium |
| `manifold-capability-probing` | Low |
| `multi-task-vbr` | Low |

## Repo Health

- `wiki/findings/`: empty (no completed runs yet)
- `visualize.py`: still stub in `variable-bitrate-reasoning`
- Dashboard: not yet built

## Open Questions

- VVVVVV Phase 0: Do spike channels in nanochat (relu²) fall in first 32 indices? (→ Q0.1)
- VVVVVV Phase 0: Is BOS residual document-varying or near-constant? (→ Q0.2)
- These gate the learned projection gate and BOS-conditioned table extensions respectively.
