---
title: "Frame/State Decomposition Prior Art"
date: 2026-03-15
experiments: [behavioral-projections, translator-matrix-analysis]
type: novelty-review
---

# Frame/State Decomposition Prior Art

## Regime Boundaries: Thoroughly Established

At least eight independent lines of evidence establish that transformer layers
operate in distinct regimes:

1. **Tenney et al.** — "BERT rediscovers the classical NLP pipeline" (syntax →
   semantics progression)
2. **Voita et al.** — information-theoretic analysis of layer functions
3. **Sun et al. (Painters)** — layer similarity structure, early/middle/late regimes
4. **Phang et al.** — CKA clustering reveals layer groups
5. **ShortGPT** — Block Influence scores show dispensable middle layers
6. **Intrinsic dimensionality** — "hunchback" curve (rises then falls across layers)
7. **Alpay et al.** — critical depth γ_c ≈ 0.42, phase transition in representation
   structure
8. **Logit lens** — failures in early/middle layers reveal format incompatibility
   (the frame concept)

**Do not claim to discover regime boundaries.** They are consensus knowledge.

## Closest Decomposition Work

**Geva et al. (arXiv:2203.14680)** — decomposes transformer representations but
separates content types (entity vs attribute vs relation), not frame vs state.
This is the closest structural decomposition but with a different decomposition
axis.

## What Logit Lens Failures Motivate

The logit lens applies the unembedding matrix directly to intermediate layers. It
fails in early/middle layers because those layers haven't yet formatted their
representations for the unembedding matrix. This is exactly the frame concept:
early layers are still in a different representational format. The tuned lens fixes
this by learning a per-layer affine transform — the translator matrix — which is
literally a frame conversion.

The frame/state decomposition formalizes this: the translator matrix can be
decomposed into a component that handles format conversion (frame) and a component
that projects task-relevant content (state).

## Novel Contribution

The **formal decomposition method**, not regime discovery. Specifically:

- A mathematical decomposition of translator matrices into frame and state components
- Criteria for separating "format conversion" from "content projection"
- Quantitative metrics for frame stability and state information content across layers

## Framing

"Method that provides analytical leverage on a recognized but under-formalized
phenomenon." The regimes are known. The decomposition is new. Cite all eight
regime-discovery lines extensively.

Source: novelty review, March 2026.
