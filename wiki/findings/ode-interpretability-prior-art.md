---
title: "ODE Interpretability Prior Art"
date: 2026-03-15
experiments: [cln-iteration-dynamics]
type: novelty-review
---

# ODE Interpretability Prior Art

The post-hoc ODE fitting idea is novel but the gap is closing. Three convergent
2025 papers are moving in the same direction from different angles.

## Convergent Work (2025)

| Paper | Approach | Key result | Gap from our idea |
|-------|----------|------------|-------------------|
| Tong et al. (arXiv:2503.01329, ICLR 2025) | ODE built into architecture | Spectral analysis + Lyapunov exponents for interpretability | ODE is designed in, not fit post-hoc |
| "Transformer Dynamics" (arXiv:2502.12131) | Low-dim projection + trajectory analysis | Applied to Llama 3.1 8B | Descriptive, not parametric — no ODE fit |
| PDE Perspective (arXiv:2408.09523) | Fit PDE to transformer activations | MSE 0.031, validates fit quality | Validates fit but doesn't interpret parameters |

## Conceptual Analog

**PHOENIX** (biological neural ODE) — fits ODE to biological neural recordings,
reads off regulatory structure from the learned dynamics. This is the closest
conceptual match: fit a dynamical system to observed trajectories, then interpret
the fitted parameters. Never applied to artificial neural networks as analysis
tools.

## Additional Context

**Liquid Neural Networks (LNNs)** — continuous-time neural networks with
interpretable dynamics. Used as models, never as analysis tools for other networks.
The idea of using an LNN-style ODE as a post-hoc probe on a transformer is
unexplored.

## Assessment

Frame as **natural synthesis** of these convergent lines, not as a wholly original
idea. The three 2025 papers each have one of the three ingredients (ODE,
interpretability, transformer analysis) but none combines all three in the post-hoc
fitting paradigm.

Cite all three prominently. The contribution is the specific combination: fit ODE
post-hoc to frozen transformer dynamics, then interpret the ODE parameters as a
description of the transformer's computational structure.

Source: novelty review, March 2026.
