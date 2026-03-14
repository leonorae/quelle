---
title: Variable-Bitrate Reasoning with Geometric Self-Awareness
slug: variable-bitrate-reasoning
status: running
owner: claude-sonnet-4-6
created: 2026-02-28
implemented: 2026-03-07
depends_on: []
tags: [compression, geometry, transformer, arithmetic, dsd, gumbel-softmax]
---

# Variable-Bitrate Reasoning with Geometric Self-Awareness

## Hypothesis

A transformer that measures the geometric concentration of its own hidden states
can learn to allocate representational bandwidth adaptively—compressing more when
representations are geometrically tight (high certainty) and less when they are
diffuse (high uncertainty). This policy should emerge from task loss alone, with
no reinforcement learning.

## Specification

- **Implementation spec** (architecture, training loop, metrics): top-level `README.md`
- **Research context** (related work, novelty, feasibility, risks): `wiki/humans/latent-research-context.md`

## Status

`running` — code implemented 2026-03-07, awaiting first full training run.

Implementation checklist:

- [x] Tokenizer for arithmetic expressions
- [x] Dataset generator with difficulty labels
- [x] Transformer (4 layers, 4 heads, d=128)
- [x] Compression head with differentiable dimension selection (soft-sigmoid)
- [x] Concentration computation (per-example, returns `(B,)` tensor)
- [x] Learnable alpha/beta parameters
- [x] Training loop with all three losses (LM + future + curvature)
- [x] Logging infrastructure (CSV, written every `log_every` steps)
- [x] Evaluation script with metrics (accuracy by difficulty, λ–conc correlation)
- [x] Baseline implementations (fixed-high, fixed-low, random, no-compression)
- [ ] Visualization (PCA/UMAP, scatter plots) — `src/visualize.py` still stub

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Model ignores λ_t | Add small entropy bonus to encourage variation |
| Representation collapse | DSD stop-gradient prevents collapse; monitor latent variance |
| No correlation with concentration | Still informative — null result is a finding |
| λ_t saturates at extremes | Adjust α/β init; add temperature annealing |

## Success Criteria

1. λ_t negatively correlated with concentration (r < -0.3).
2. Adaptive compression outperforms fixed baselines on hard problems.
3. Future prediction loss decreases stably.
4. UMAP reveals distinct clusters for different step types.
5. Linear probes predict storage requirement from hidden states (>60% accuracy).

Even partial success (e.g. correlation exists but no accuracy gain) is informative
and worth documenting in `RESULTS.md`.

## Directory Layout

```
variable-bitrate-reasoning/
├── README.md          ← this file
├── RESULTS.md         ← fill in after run
├── src/
│   ├── model.py       ← VariableRateReasoner, CompressionHead, etc.
│   ├── data.py        ← ArithmeticDataset, tokenizer, generator
│   ├── train.py       ← training loop entry point
│   ├── evaluate.py    ← evaluation and baselines
│   └── visualize.py   ← PCA/UMAP, scatter plots
├── configs/
│   └── default.yaml   ← hyperparameters
├── data/              ← generated datasets (gitignored if large)
├── outputs/           ← logs, checkpoints, plots (gitignored if large)
├── notebooks/         ← exploratory analysis
└── .gitignore
```

## Related Work

See `wiki/concepts/related-work.md` for summaries of the six papers that
directly inform this experiment (latent vocabulary superposition, LT-Tuning,
TRAAC, GAIN-RL, Geometry of Thought, DCoLT).

## Open Questions

- ~~Two-pass vs. single-pass forward~~ **Resolved**: single-pass implemented
  (information leak noted; upgrade if results look suspicious — see
  `wiki/humans/decisions.md` 2026-02-28).
- ~~Hard-threshold fallback for Gumbel-Softmax~~ **Resolved**: not needed;
  soft-sigmoid mask was stable in smoke-testing.
