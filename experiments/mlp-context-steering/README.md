---
status: scaffolded
owner: null
dependencies: []
---

# MLP Context-Steering and Rank Dynamics

**Hypothesis**: The MLP's context-dependent computation actively steers
representation rank across layers. When this steering is ablated, effective rank
decays faster and attention sinks intensify. These two effects correlate across
prompts — sinks are equilibrium markers for failed context-steering.

This is the empirical anchor for the Dherin-Dong-Sun triangle synthesis
(see `wiki/findings/` — novelty reviews, Review C).

## Core Prediction

Ablating the MLP's context-dependent component causes:
1. Faster effective rank decay across layers
2. Intensified attention sink formation
3. These two effects correlate across prompts

## Model

Pythia-410m via TransformerLens (`HookedTransformer.from_pretrained("pythia-410m")`).

- 24 layers, pre-norm (LayerNorm before attention and MLP)
- Standard GELU (NOT SwiGLU — Sun et al.'s "directional quadratic amplifier"
  mechanism does not directly apply; flag any discrepancies)
- TransformerLens is new to this repo; no existing infrastructure to reuse

## Compute

Light — forward passes only, no training. Single GPU, hours not days.

## Background (minimum context)

Three independent results predict a specific relationship:

1. **Dong et al. (ICML 2021)**: Pure self-attention loses rank doubly exponentially
   with depth. Skip connections are the primary defense. MLPs are secondary.

2. **Dherin et al. (arXiv:2507.16003)**: Self-attention + MLP implicitly produces a
   rank-1 weight update to the MLP's effective computation, derived from attended
   context. Exact for linear attention, approximate for softmax.

3. **Sun et al. (arXiv:2603.05498)**: Massive activations persist as near-constant
   implicit parameters. Attention sinks modulate attention locally. Pre-norm enables
   co-occurrence. Early SwiGLU blocks act as directional quadratic amplifiers.

**The untested connection**: The rank-1 context-dependent MLP update (Dherin) is the
mechanism by which context navigates the rank landscape (Dong), with sinks (Sun)
marking equilibrium points where navigation has saturated or failed.

## Protocol

### Phase 0: Baseline Measurements
Collect three curves per prompt across all 24 layers:
- **0a** — Effective rank (erank) of residual stream activations
- **0b** — Attention sink intensity (attention mass on BOS)
- **0c** — Massive activation magnitude (max abs value per layer)

### Phase 1: MLP Context Ablation
Two ablation variants at each layer, propagated through subsequent layers:
- **Strong**: Replace MLP input with mean across positions
- **Surgical**: Subtract attention output, re-add mean attention output

### Phase 2: Correlation Analysis
- **2a** — Rank decay acceleration (baseline vs ablated)
- **2b** — Sink intensification (baseline vs ablated)
- **2c** — Cross-prompt correlation between rank loss and sink gain
- **2d** — Per-layer regime detection (where both change sharply)

### Phase 3: Controls
- **3a** — Random permutation ablation (preserves marginal stats)
- **3b** — Layer-specific ablation (one layer at a time)
- **3c** — MLP-only vs attention-only rank contribution decomposition

## Success Criteria

| Outcome | What it means |
|---|---|
| **Strong confirmation** | All three predictions hold + early/middle layers most sensitive |
| **Partial confirmation** | Rank decay accelerates but sink intensification weak/uncorrelated |
| **Refutation** | Ablation does NOT accelerate rank decay |
| **Interesting null** | Rank decays but sinks don't intensify (sinks ≠ equilibrium markers) |

## Connections

- `behavioral-projections`: frame_ratio measures regime boundaries from a different
  angle; regime boundary layers should overlap with layers where Δerank and Δsink
  change sharply (Phase 2d)
- `tools/analysis/geometry/`: `effective_dimensionality` uses PCA participation
  ratio — this experiment uses entropy-based erank instead (different metric,
  complementary)
- VVVVVV Phase 0: spike channel measurements overlap with Phase 0c (massive
  activation magnitude)
