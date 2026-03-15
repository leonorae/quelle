---
title: "Dherin-Dong-Sun Triangle Corrections"
date: 2026-03-15
experiments: [behavioral-projections, VVVVVV]
type: novelty-review
---

# Dherin-Dong-Sun Triangle Corrections

Two corrections to the initial framing, plus clarification of what is actually novel.

## Correction 1: Skip Connections, Not MLPs, Are Primary Rank Preservers

Replace "MLP's rank-restoring force" with: **skip connections are the primary
rank-preserving mechanism; MLPs are secondary amplifiers.**

Dong et al. is clear on this. The residual stream's skip connection maintains rank
by construction (adding a full-rank signal). MLPs amplify certain directions but are
not the primary force preventing rank collapse.

Supporting empirical data (arXiv:2508.16929): attention layers have ~60% effective
rank (erank), MLP layers have ~90% erank. This is consistent with MLPs maintaining
rank but attention compressing it.

## Correction 2: Dong-Sinks Edge Already Established

The connection between rank collapse dynamics (Dong et al.) and attention sinks is
**not novel** — Barbero et al. (arXiv:2504.02732, COLM 2025) already establishes
this edge explicitly.

## What Is Actually Novel

The **two Dherin edges** and the **three-way synthesis**:

1. **Dherin ↔ Dong**: Implicit regularization (path-length penalty from discrete
   gradient updates) as a mechanism driving the rank dynamics Dong describes
2. **Dherin ↔ Sun**: Implicit regularization as an explanation for the layer-regime
   boundaries Sun observes (why early layers settle into format-conversion mode)
3. **Three-way closure**: The triangle Dherin-Dong-Sun, where implicit regularization
   connects rank dynamics to layer regimes through a unified mechanism

## Related Mathematical Infrastructure

- **Geshkovski et al. (arXiv:2512.01868)** — potential mathematical framework for
  the saddle-point framing (transformer dynamics as navigation between saddle points
  in a loss landscape shaped by implicit regularization)

## Framing

The three individual phenomena (implicit regularization, rank dynamics, layer regimes)
are each well-established. The contribution is the synthesis — showing they form a
coherent triangle rather than three independent observations.

Source: novelty review, March 2026.
