---
title: Variable-Bitrate Reasoning with Geometric Self-Awareness
slug: variable-bitrate-reasoning
status: planning
owner: unassigned
created: 2026-02-28
depends_on: []
tags: [compression, geometry, transformer, arithmetic]
---

# Variable-Bitrate Reasoning with Geometric Self-Awareness

## Hypothesis

A transformer that measures the geometric concentration of its own hidden states
can learn to allocate representational bandwidth adaptively—compressing more when
representations are geometrically tight (high certainty) and less when they are
diffuse (high uncertainty). This policy should emerge from task loss alone, with
no reinforcement learning.

## Specification

Full implementation spec lives in the top-level `README.md`.

## Status

`planning`

Implementation checklist (from spec):

- [ ] Tokenizer for arithmetic expressions
- [ ] Dataset generator with difficulty labels
- [ ] Transformer (4 layers, 4 heads, d=128)
- [ ] Compression head with differentiable dimension selection (Gumbel-Softmax)
- [ ] Concentration computation
- [ ] Learnable alpha/beta parameters
- [ ] Training loop with all three losses
- [ ] Logging infrastructure (CSV or TensorBoard)
- [ ] Evaluation script with metrics
- [ ] Baseline implementations (fixed-high, fixed-low, random, no-compression)
- [ ] Visualization (PCA/UMAP, scatter plots)

## Success Criteria

1. Training converges (loss decreases stably).
2. Lambda shows negative correlation with concentration (r < -0.3).
3. Adaptive compression outperforms fixed baselines on hard problems.
4. UMAP reveals distinct clusters for different step types.
5. Linear probes predict storage requirement from hidden states (>60% accuracy).

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

## Open Questions

- Two-pass vs. single-pass forward for future prediction target (see spec Note 2).
- Whether to start with hard threshold gradient bypass if Gumbel-Softmax is
  unstable, then switch.
