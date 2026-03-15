---
title: "Cross-Model Behavioral Geometry"
date: 2026-03-15
experiments: [behavioral-projections]
type: novelty-review
---

# Cross-Model Behavioral Geometry

Cross-model comparison via behavioral geometry is a real gap but narrower than
initially claimed.

## Closest Prior Art

| Work | What they do | Gap |
|------|-------------|-----|
| Oyama et al. (arXiv:2502.16173) | Model-to-model distances from log-likelihoods | Closest — uses outputs, but raw log-likelihoods, not a learned metric space |
| Huh et al. (arXiv:2405.07987, Platonic Representation Hypothesis) | Compares distance structures across models | From activations, not outputs; tests convergence hypothesis |
| Klabunde et al. (arXiv:2305.06329) | Survey of representation similarity | Documents the representational/functional divide this bridges |

## Novel Element

Using a **learned bisimulation probe's metric space** for cross-model RSA. No
precedent for:

1. Training bisimulation probes on two models independently
2. Comparing the resulting metric spaces (not the raw representations)
3. Asking whether behaviorally similar models produce similar behavioral geometries

This is strictly more informative than CKA/CCA (which compare raw representations)
and more structured than log-likelihood distances (which discard geometric
information).

## Key Framing

**Method novelty, not phenomenon novelty.** The question "how similar are two
models?" is well-studied. The contribution is a specific pipeline:
probe → metric space → RSA comparison. The novelty is in the pipeline, not in
the question it answers.

Source: novelty review, March 2026.
