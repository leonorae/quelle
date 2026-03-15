---
title: MLP Context Steering — Ablation Study of MLP Contributions to Contextual Representations
slug: mlp-context-steering
status: scaffolded
owner: claude-opus-4-6
created: 2026-03-15
depends_on: []
tags: [mechanistic-interpretability, mlp, ablation, transformer-lens, effective-rank, attention-sinks]
---

# MLP Context Steering

## Hypothesis

MLP layers in transformer language models actively steer contextual
representations away from low-rank, attention-sink-dominated subspaces.
Ablating MLP outputs should cause measurable collapse in effective rank and
amplification of sink intensity, with the effect propagating through subsequent
layers rather than being locally contained.

## Phases

1. **Phase 1 — Baseline profiling**: Collect per-layer metrics (effective rank,
   sink intensity, max activation) on unmodified forward passes across diverse
   prompts.
2. **Phase 2 — Ablation**: Zero out MLP outputs at target layers and measure
   metric shifts. Two modes: "strong" (zero entire MLP) and "surgical" (zero
   specific neurons).
3. **Phase 3 — Controls**: Permutation controls to distinguish MLP-specific
   effects from generic perturbation effects.

## Status

`scaffolded` — directory skeleton and core metrics implemented. Awaiting tests
and remaining module implementation.

Implementation checklist:

- [x] Experiment skeleton (README, DECISIONS, configs)
- [x] `src/metrics.py` — erank, sink_intensity, max_activation, collect_layer_metrics
- [x] `src/ablation.py` — hook factories (strong, surgical, permutation)
- [ ] `src/baseline.py` — prompt loading and baseline collection
- [ ] `src/ablation.py` — collect_ablated loop
- [ ] `src/analysis.py` — Phase 2 computations
- [ ] `src/controls.py` — Phase 3 control collection
- [ ] `src/summarize.py` — result summarization
- [x] Tests for metrics.py

## Directory Layout

```
mlp-context-steering/
├── README.md
├── RESULTS.md
├── DECISIONS.md
├── src/
│   ├── __init__.py
│   ├── metrics.py        ← erank, sink_intensity, max_activation
│   ├── ablation.py       ← hook factories and ablation collection
│   ├── baseline.py       ← prompt loading and baseline collection
│   ├── analysis.py       ← Phase 2 metric comparison
│   ├── controls.py       ← Phase 3 permutation controls
│   └── summarize.py      ← result summarization
├── tests/
│   └── test_metrics.py   ← unit tests for pure tensor metrics
├── configs/
│   └── default.yaml
├── data/
├── outputs/
└── notebooks/
```

## Open Questions

- Do hook point names in TransformerLens match our assumptions? (flagged as TRAP in ablation.py)
- Is hook execution order deterministic for surgical ablation? (flagged as TRAP in ablation.py)
