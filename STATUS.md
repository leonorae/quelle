# Status — Quelle

> Quick briefing for agents and humans. Read this instead of traversing
> the full wiki to understand what is happening right now.
>
> **Updated**: 2026-03-14

---

## Active Experiments

| Experiment | Status | Owner | Next action |
|---|---|---|---|
| `variable-bitrate-reasoning` | ready to run | — | Launch on Colab (train_colab.sh) |
| `VVVVVV` | ready to run | — | Launch on Colab (Phase 0: setup → train → probes) |
| `crystal-lattice` | code complete | — | Launch on Colab (curiosity_loop.py) |

## Archived

| Experiment | Reason |
|---|---|
| `geometric-self-awareness-reasoning` | Too simplistic; archived 2026-03-14 |

## Backlog

| Experiment | Priority |
|---|---|
| `gumbel-compression-ablation` | Medium |
| `manifold-capability-probing` | Low |
| `multi-task-vbr` | Low |

## Parallel Execution Setup

All three active experiments can run simultaneously on separate Colab runtimes.
The generic Colab notebook (`tools/training/colab_train.ipynb`) reads each
experiment's `colab.yaml` — just change `EXPERIMENT` in the config cell:

| Colab instance | Set `EXPERIMENT` to | GPU needed | Est. time |
|---|---|---|---|
| 1 | `VVVVVV` | T4+ | ~4-6h train + 30min probes |
| 2 | `variable-bitrate-reasoning` | Any GPU | ~2-4h |
| 3 | `crystal-lattice` | Any GPU | ~1-2h |

After each run, results sync to Google Drive under `quelle_artifacts/<slug>/`.
Update `RESULTS.md` per the chat-bridge protocol in CLAUDE.md.

## Repo Health

- `wiki/findings/`: empty (no completed runs yet)
- `visualize.py`: still stub in `variable-bitrate-reasoning`
- Dashboard: not yet built

## Open Questions

- VVVVVV Phase 0: Do spike channels in nanochat (relu²) fall in first 32 indices? (→ Q0.1)
- VVVVVV Phase 0: Is BOS residual document-varying or near-constant? (→ Q0.2)
- These gate the learned projection gate and BOS-conditioned table extensions respectively.
