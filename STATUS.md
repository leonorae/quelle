# Status — Quelle

> Quick briefing for agents and humans. Read this instead of traversing
> the full wiki to understand what is happening right now.
>
> **Updated**: 2026-03-15 (VVVVVV reframe)

---

## Active Experiments

| Experiment | Status | Owner | Next action |
|---|---|---|---|
| `VVVVVV` | Phase 0 ready; Phases 1-4 conjectured dead | — | Train d12, run Phase 0 diagnostics (interpretability, not architecture) |
| `behavioral-projections` | active | — | Phase 0.6 (ve table analysis) + frame_ratio curve |
| `vsa-encoding-fidelity` | needs fix | — | Implement multiplicative binding, re-run probe |
| `curiosity-vs-random` | needs lit review | — | Survey active learning literature for low-budget regime |
| `cln-iteration-dynamics` | needs lit review | — | Survey looped transformer ablations |
| `probe-signal-comparison` | planned | — | Needs VVVVVV d12 checkpoint |

## Decomposed

| Experiment | Reason |
|---|---|
| `crystal-lattice` | Decoupled into independent threads (2026-03-15). See `wiki/findings/crystal-lattice-decomposition.md` |

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

---

## Recent Change: VVVVVV Reframe (2026-03-15)

The multi-ve architecture experiment (VVVVVV Phases 1-4) is `conjectured dead`.
Three reasons: gate-input may read spike channels (not content), ve_local is
redundant with attention, and the interesting question shifted from architecture
to interpretability. Phase 0 diagnostics still run — they now serve the
interpretability angle. Ve table content analysis moves to behavioral-projections
as Phase 0.6. See `wiki/findings/vvvvvv-reframe.md` and
`wiki/concepts/factorization-taxonomy.md`.

## Recent Change: Crystal-Lattice Decomposition (2026-03-15)

The crystal-lattice experiment stacked four unvalidated ideas in a linear
dependency chain: VSA encoding → CLN refinement → curiosity selection → LLM
mutation. The first layer failed validation. Rather than fix-and-rebuild
linearly, the experiment was decomposed into independent threads:

- **VSA encoding** (`vsa-encoding-fidelity`): Can VSA represent molecular
  topology? Fix the Wormhole operator and re-test. Standalone.
- **Iterative refinement** (`cln-iteration-dynamics`): Does looping help?
  Literature review first — recent looped-transformer papers may already answer
  this. Test on any representation, not necessarily VSA.
- **Active learning** (`curiosity-vs-random`): Does curiosity beat random at
  small budgets? Literature review first. Test with any model, not necessarily
  the full stack.
- **LLM molecular mutation**: Deferred. Low priority until other threads validate.

Speculative recombination paths noted in
`wiki/findings/crystal-lattice-decomposition.md`. Each thread succeeds or
fails on its own terms.

---

## Dependency Graph (Updated)

```
VVVVVV Phase 0 (ready to run — interpretability, not architecture)
    ├─→ probe-signal-comparison (needs d12 checkpoint)
    └─→ ve table content analysis (behavioral-projections Phase 0.6)
         └─→ [if multi-modal structure found] VVVVVV Phases 1-4 revival

behavioral-projections (active, independent)
    ├─→ frame_ratio curve (blocked on tuned lens)
    └─→ Phase 0.6 ve table analysis (needs nanochat checkpoint)

vsa-encoding-fidelity (needs fix, independent)
curiosity-vs-random (needs lit review, independent)
cln-iteration-dynamics (needs lit review, independent)
```

No linear chains. VVVVVV Phases 1-4 conjectured dead unless ve table analysis
motivates revival.

---

## Parallel Execution Setup

The generic Colab notebook (`tools/training/colab_train.ipynb`) reads each
experiment's `colab.yaml`. Set `EXPERIMENT` in the config cell.

## Shared Tools

- `tools/analysis/geometry/` — concentration, velocity, effective dimensionality
  diagnostics. Extracted from VBR for cross-experiment use.

## Open Questions

- VVVVVV Phase 0: Do spike channels in nanochat (relu²) fall in first 32 indices? (→ Q0.1)
- VVVVVV Phase 0: Is BOS residual document-varying or near-constant? (→ Q0.2)
- **Ve table**: Do ve rows cluster by syntactic function? Does ve cosine correlate with bisimulation distance? What's the effective rank?
- VSA: Does multiplicative binding fix the topology encoding problem?
- Active learning: Does the literature already resolve curiosity-vs-random at small budgets?
- Iterative refinement: Do looped-transformer ablations already characterize iteration dynamics?
