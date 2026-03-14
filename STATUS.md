# Status — Quelle

> Quick briefing for agents and humans. Read this instead of traversing
> the full wiki to understand what is happening right now.
>
> **Updated**: 2026-03-14

---

## Active Experiments

| Experiment | Status | Owner | Next action |
|---|---|---|---|
| `VVVVVV` | ready to run | — | Launch on Colab (Phase 0: setup → train → probes) |
| `crystal-lattice` | **blocked** | — | Fix Wormhole operator (see vsa-encoding-fidelity results) |
| `probe-signal-comparison` | planned | — | Needs VVVVVV d12 checkpoint |
| `vsa-encoding-fidelity` | **done** | — | Wormhole operator ineffective; see RESULTS.md |
| `curiosity-vs-random` | planned | — | Needs crystal-lattice to run first |
| `cln-iteration-dynamics` | planned | — | Needs crystal-lattice training logs |

## Archived

| Experiment | Reason |
|---|---|
| `variable-bitrate-reasoning` | Concentration metric better as diagnostic than control signal |
| `geometric-self-awareness-reasoning` | Too simplistic |

## Backlog

| Experiment | Priority |
|---|---|
| `gumbel-compression-ablation` | Medium |
| `manifold-capability-probing` | Low |
| `multi-task-vbr` | Low |

## Key Finding: Wormhole Operator Broken

The VSA encoding fidelity test (2026-03-14) found that crystal-lattice's
Wormhole operator is ineffective:

- Ring vs chain classification: **57.5%** (near chance)
- Closure tag cosine similarity delta: **0.0003** (negligible)
- Chain-ring HV cosine similarity: **>0.92** (nearly identical)

**Root cause**: bundling (addition) of a single closure-tag HV into a
molecule HV with N atom-position terms dilutes the signal to ~1/(N+1).

**Fix options**: scale closure term by sqrt(N), use separate topology
channel, or multiplicative binding instead of bundling.

Crystal-lattice training should NOT proceed until this is fixed.

## Parallel Execution Setup

The generic Colab notebook (`tools/training/colab_train.ipynb`) reads each
experiment's `colab.yaml`. Set `EXPERIMENT` in the config cell.

## Shared Tools

- `tools/analysis/geometry/` — concentration, velocity, effective dimensionality
  diagnostics. Extracted from VBR for cross-experiment use.

## Open Questions

- VVVVVV Phase 0: Do spike channels in nanochat (relu²) fall in first 32 indices? (→ Q0.1)
- VVVVVV Phase 0: Is BOS residual document-varying or near-constant? (→ Q0.2)
- Crystal-lattice: Best fix for Wormhole operator dilution?
- Probe-signal-comparison: Do geometric signals add information beyond tuned lens?
