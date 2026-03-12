---
title: Behavioral Projections
status: draft
created: 2026-03-11
tags: [projections, interpretability, bisimulation, geometry]
related_experiments: [behavioral-projections]
---

# Behavioral Projections

## The gap

Existing activation analysis tools decompose (SAEs, transcoders) or test for
specific properties (linear probes, TCAV, activation patching). Neither learns
a general-purpose map from activation space to a behaviorally meaningful metric
space defined by the model's own computation.

## Four projections

### 1. Bisimulation probe (output-distribution distance)

Learn P such that ‖P(h₁) − P(h₂)‖ predicts KL(f(h₁) ‖ f(h₂)).

**Column space:** directions the model considers functionally distinct.
**Null space:** directions the model treats as interchangeable.

Prior work: bisimulation metrics in RL (Zhang et al., ICLR 2021); Tuned Lens
(Belrose et al., 2023) maps to absolute distributions but not pairwise distance;
DAS (Geiger et al., 2024) requires a predefined causal graph.

### 2. Contrastive prompt discrimination

InfoNCE loss on same-prompt vs different-prompt activation pairs.

**Column space:** what's specific to this input.
**Null space:** general-purpose computational machinery shared across inputs.

Augmentation uses register-token prepending (grounded in the model's attention
sink behavior) rather than arbitrary noise. SimCLR projector-head pattern.

Prior work: SimCSE (Gao et al., EMNLP 2021); Representation Engineering
(Zou et al., 2023).

### 3. Layer-contrastive (delta projection)

Map the residual stream update (h_{l+1} − h_l) to predict functional role or
output distribution change.

**Column space:** directions of maximum transformation between layers.
**Null space:** information carried through unchanged (residual backbone).

Prior work: Crosscoders (Anthropic, 2024); Voita et al. (2019) layer
difference analysis.

### 4. Perturbation-sensitivity map

Learn W such that ‖W·δ‖ predicts |Δoutput| for perturbation δ.
Linearized, amortized approximation of the model's Jacobian.

**Column space:** load-bearing directions.
**Null space:** slack (perturbation has no effect).

Prior work: Attribution patching (Nanda, 2023); AtP* (Kramár et al., 2024).
Both are per-input; the learned amortization is novel.

Key tension: Jacobian varies per input. Fixed W captures average sensitivity.
Gap between per-input and amortized sensitivity is itself measurable.

## Endogenous scaffolding

The intersection of all four null spaces isolates directions carrying no
content, no output sensitivity, no layer-delta, no perturbation response.
This is endogenous scaffolding — the geometric analog of attention sinks and
register tokens.

Synthetic tokens (sinks, registers, CLS) are prosthetic scaffolding using
sequence positions for zero content. The null-space intersection is endogenous
scaffolding encoded in representational geometry.

## Known failure modes

1. **Dimensional collapse** in contrastive approaches — mitigate with projector heads, whitening, orthogonality constraints (Jing et al., ICLR 2022)
2. **Nonlinear dark matter** — SAE reconstruction errors are pathological, hurting models more than random perturbations of same magnitude (Engels et al., 2024; Gurnee, 2024). Linear projections inherit this floor. Iterative peeling characterizes how much is high-rank-linear vs genuinely nonlinear.
3. **LayerNorm boundaries** — ~50% error in linear attribution for GPT-2. Perturbation-sensitivity probe directly affected. May need per-segment training.
4. **Probing selectivity** — strong probes learn structure not in the representation (Hewitt & Liang, 2019). Use MDL probes or control tasks.
5. **Interpretability illusion** — interventions can activate dormant pathways (Makelov et al., ICLR 2024). Validate with counterfactual tests.

## Connection to repo experiments

- **Slicer:** Ridge from LLM activations to CLIP space showed R²/nn_recall@5 tension — reconstruction fidelity and structural preservation are different objectives. Behavioral projections replace external target spaces with the model's own signals.
- **Variable-bitrate reasoning:** Training-time sensitivity feedback is VBR's adaptive compression applied to representational geometry rather than tokens.
- **Saddle detection:** Bisimulation probe gives the metric for geometric saddle definition — two prompts are at a saddle when bisimulation distance is small despite large activation distance.
