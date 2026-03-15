---
title: "Cross-Cutting Framing: Method Novelty, Not Phenomenon Novelty"
date: 2026-03-15
experiments: [behavioral-projections, translator-matrix-analysis, cln-iteration-dynamics, VVVVVV]
type: meta-finding
---

# Method Novelty, Not Phenomenon Novelty

All six novelty reviews (March 2026) converge on a single pattern: every proposed
contribution is **method novelty**, not phenomenon novelty. The phenomena under
study — layer regimes, rank dynamics, representation similarity gaps, iterative
refinement — are all well-established in the literature by multiple independent
groups.

## Standard Framing

For all publications from this research programme:

> We introduce a formal method that provides new analytical leverage on a
> recognized but under-formalized phenomenon.

Each paper must extensively cite prior work establishing the phenomenon it analyzes.

## Per-Review Summary

| Review | Phenomenon (known) | Method (novel) |
|--------|-------------------|----------------|
| [Tuned lens translator gap](tuned-lens-translator-gap.md) | Translator matrices exist and encode layer structure | SVD / difference / decomposition analysis of the matrices themselves |
| [Bisimulation probe](bisimulation-probe-prior-art.md) | Representational vs functional similarity gap | Learned probe targeting bisimulation distance |
| [Dherin-Dong-Sun triangle](dherin-dong-sun-triangle.md) | Implicit regularization, rank dynamics, layer regimes (each independently) | Three-way synthesis connecting them |
| [Cross-model behavioral geometry](cross-model-behavioral-geometry.md) | Models can be compared | Bisimulation probe metric space for cross-model RSA |
| [ODE interpretability](ode-interpretability-prior-art.md) | Transformers have dynamics, ODEs can model dynamics | Post-hoc ODE fitting as interpretability tool |
| [Frame/state decomposition](frame-state-decomposition-prior-art.md) | Layer regimes exist (8+ independent lines) | Formal decomposition of translator matrices into frame and state |

## Risk

The primary risk across all proposals is **overclaiming phenomenon discovery**.
Every novelty review found that the underlying phenomenon is documented by 3-8
prior works. Reviewers will catch this. Framing must be precise: the contribution
is a new lens, not a new observation.
