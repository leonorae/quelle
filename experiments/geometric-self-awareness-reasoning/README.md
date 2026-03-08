---
title: Geometric Self-Awareness for Reasoning Integrity
slug: geometric-self-awareness-reasoning
status: planning
owner: unassigned
created: 2026-03-03
depends_on:
  - wiki/concepts/geometric-self-awareness.md
  - wiki/concepts/ris-scoring.md
tags: [geometry, reasoning, gsm8k, hidden-states, trajectory, interpretability]
---

# Geometric Self-Awareness for Reasoning Integrity

## Hypothesis

Angle concentration and trajectory geometry (velocity, curvature, manifold
dimensionality) in a language model's hidden states correlate with
human-judged reasoning quality (RIS) and can predict reasoning flaws **before
they manifest in final answers** — enabling training-free, interpretable error
detection.

## Motivation

Recent work establishes that:

- Correctness lives in a low-dimensional subspace of hidden states (Confidence
  Manifold, arXiv:2602.08159): centroid distance is a zero-shot predictor.
- Velocity, curvature, and flow similarity of reasoning trajectories correlate
  with answer quality (Geometry of Reasoning, arXiv:2510.09782).
- Models produce "right for wrong reasons" traces that fool surface-level
  evaluators (Advani et al. 2026 — RIS metric).

If geometry reflects *process* (not just *outcome*), we gain an interpretable
real-time signal — no gradient, no retraining required.

## Experimental Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **0B** | Generate ~1000 reasoning traces (GSM8K + Qwen2.5-7B-Instruct) | `planning` |
| **1** | Score traces with LLM judge ensemble (DeepSeek-V3, Qwen-72B, Llama-3.1-70B) using RIS rubric | `not started` |
| **2** | Extract geometric features from hidden states (angle, velocity, curvature, manifold dim) | `not started` |
| **3** | Correlation analysis, predictor training, temporal early-warning test | `not started` |

## Directory Layout

```
geometric-self-awareness-reasoning/
├── README.md                   ← this file
├── CONTEXT.md                  ← comprehensive context document for implementer agent
├── RESULTS.md                  ← fill in after Phase 3
├── src/
│   ├── phase0b_generate_traces.py   ← Phase 0B: trace generation
│   ├── phase1_ris_scoring.py        ← Phase 1: LLM judge ensemble (stub)
│   ├── phase2_extract_geometry.py   ← Phase 2: hidden state geometry (stub)
│   └── phase3_analysis.py           ← Phase 3: correlation + predictors (stub)
├── configs/
│   └── phase0b.yaml            ← generation hyperparameters
├── data/                       ← generated JSONL files (gitignored if large)
├── outputs/                    ← logs, plots, reports (gitignored if large)
├── notebooks/                  ← exploratory analysis
└── .gitignore
```

## Success Criteria

- **Phase 0B**: 1000 clean traces; baseline pass@1 accuracy > 70%.
- **Phase 3**: Pearson r between geometric metrics and RIS > 0.5; AUC for
  flaw prediction > 0.8; temporal early-warning AUC > 0.7.

## Open Questions

- GPU availability for Qwen2.5-7B-Instruct (24GB+ VRAM required). Cloud
  instance or quantized model if unavailable.
- Whether to use vLLM (fast) or Transformers (simple) for Phase 0B.
- Which judge models to use for Phase 1 (API cost vs. quality trade-off).
- Whether to also run ProcessBench/PRMBench alongside GSM8K for pre-annotated
  step-level error labels (could shortcut Phase 1).
