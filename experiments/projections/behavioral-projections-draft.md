---
title: "Behavioral Projections: Learned Linear Maps for LLM Activation Analysis"
status: draft
created: 2026-03-11
authors: [apophenia, opus-4.6]
depends_on: [slicer, variable-bitrate-reasoning, saddle-detection]
tags: [projections, interpretability, bisimulation, dark-matter, geometry]
---

# Behavioral Projections

## The gap

Existing tools for analyzing LLM activations fall into two categories: decomposition methods (SAEs, transcoders, crosscoders) that break activations into interpretable atoms, and diagnostic methods (linear probes, TCAV, activation patching) that test for specific properties. Neither learns a general-purpose map from activation space to a behaviorally meaningful metric space.

We propose four learned linear projections, each optimized for a different behavioral objective derived from the model's own computation. Together they decompose activation space into content, computation, sensitivity, and transformation — with the complement (the intersection of all four null spaces) isolating the model's endogenous scaffolding.

## The four projections

### 1. Output-distribution probe (bisimulation)

**Objective:** Learn P such that ‖P(h₁) − P(h₂)‖ predicts KL(f(h₁) ‖ f(h₂)), where f is the model's next-token distribution.

**What it captures:** The model's own notion of behavioral equivalence. Two activations are "close" if the model would do the same thing from either state. Column space = directions the model considers functionally distinct. Null space = directions the model treats as interchangeable.

**Closest prior work:** Bisimulation metrics in RL (Zhang et al., ICLR 2021). Tuned Lens (Belrose et al., 2023) maps activations to absolute output distributions but doesn't learn a pairwise behavioral distance. DAS (Geiger et al., 2024) learns behaviorally relevant subspaces but requires a predefined causal graph.

**Novelty:** Adapting bisimulation metric learning to LLM activation spaces. No causal model required — the metric is learned directly from the model's own output behavior.

### 2. Contrastive prompt discrimination

**Objective:** InfoNCE loss where positive pairs are activations from the same prompt under different conditions, negatives are different prompts.

**What it captures:** Input identity vs. generic computation. Column space = what's specific to this input. Null space = what's shared across all inputs (the model's general-purpose computational machinery).

**Augmentation strategy:** Rather than arbitrary noise, use the model's own attention sink behavior. Run same prompt with/without prepended register-style tokens. Content representation should be invariant; administrative representation shifts. The contrastive projection learns to ignore bookkeeping variation and preserve content identity — grounded in the architecture's actual failure mode rather than an arbitrary noise model.

**Closest prior work:** SimCSE (Gao et al., EMNLP 2021) applies contrastive learning to sentence embeddings. Representation Engineering (Zou et al., 2023) finds concept directions via mean-difference + PCA. Neither learns a projection on frozen intermediate activations with a contrastive objective.

### 3. Layer-contrastive (delta projection)

**Objective:** Learn a map on the residual stream update (h_{l+1} − h_l) that predicts either the change in output distribution or the functional role of each layer's contribution.

**What it captures:** What computation adds at each step. Column space = directions of maximum transformation between layers. Null space = information the model carries through unchanged (the residual stream backbone).

**Closest prior work:** Crosscoders (Anthropic, 2024) apply joint dictionary learning across layers. Tuned Lens gives indirect delta information. Voita et al. (2019) measured consecutive-layer differences analytically. A "delta tuned lens" — probing h_{l+1} − h_l directly — is a modest but genuine novelty.

### 4. Perturbation-sensitivity map

**Objective:** Learn W such that ‖W·δa‖ predicts ‖Δoutput‖ for arbitrary perturbation δa.

**What it captures:** A linearized, amortized approximation of the model's input-output Jacobian. Column space = load-bearing directions (perturbation causes output change). Null space = slack (perturbation has no effect).

**Closest prior work:** Attribution patching (Nanda, 2023) computes the exact local analog but requires a fresh backward pass per input. AtP* (Kramár et al., 2024) improves attribution patching but remains per-input. The learned amortization — a reusable linear map trained once and applied everywhere — is genuinely novel.

**Key tension:** The Jacobian varies per input. A fixed linear map captures average or dominant sensitivity directions. Works where the model's sensitivity structure is stable across inputs; fails on outliers. The gap between per-input and amortized sensitivity is itself a measurable quantity.

## The intersection: endogenous scaffolding

The intersection of all four null spaces — directions carrying no content, no output sensitivity, no layer-delta, no perturbation response — is the model's endogenous scaffolding. This maps directly onto the attention sink / register token phenomenon observed from outside: models hijack sequence positions for bookkeeping that carries zero task-relevant content but is structurally necessary.

Synthetic tokens (attention sinks, registers, CLS, task-conditioning tokens) are *prosthetic* scaffolding occupying explicit sequence positions. The null-space intersection is *endogenous* scaffolding encoded in representational geometry. The compression argument: synthetic tokens are uncompressed infrastructure using attention bandwidth for zero content. Geometric self-awareness (access to the bisimulation metric) could let the model compress infrastructure into geometry, freeing sequence positions.

## The line from observation to architecture

**Slicer** (external geometric probing via Ridge to CLIP) reveals the projection problem: CLIP's geometry imposes its own null space on the observation. The R² vs nn_recall tension demonstrates that reconstruction fidelity and structural preservation are fundamentally different objectives.

**The four projections** replace external target spaces with the model's own behavioral signals. Each defines "what matters" differently; together they cover the space. The iterative residual peeling (train projection on activations → measure residual → train next projection on residual) characterizes the rank structure of behavioral relevance: the number of iterations to saturation gives the effective dimensionality of the model's functionally relevant geometry.

**Training-time feedback.** If the perturbation-sensitivity map (or the bisimulation probe) is made available as an auxiliary signal during training, the model can learn to concentrate computation in load-bearing directions and minimize use of slack. This is the variable-bitrate paper's adaptive compression, but applied to representational geometry rather than the token bottleneck. The projections start as interpretability tools and become architectural interventions.

The model with access to its own bisimulation metric can self-allocate scaffolding vs. content capacity intrinsically, making input-side prosthetics unnecessary. This dissolves the synthetic token question: the model doesn't need register tokens if it can manage its own computational bookkeeping through learned geometric organization.

## Known failure modes

1. **Dimensional collapse** in contrastive approaches (Jing et al., ICLR 2022). Mitigate with projector heads, whitening, orthogonality constraints.

2. **Nonlinear dark matter.** Engels et al. (2024) show a constant nonlinear error persists regardless of SAE scale. Linear projections inherit this floor. The iterative peeling approach characterizes how much of the dark matter is high-rank-linear vs genuinely nonlinear.

3. **LayerNorm boundaries** introduce ~50% error in linear attribution for GPT-2. The perturbation-sensitivity projection is directly affected. May need to train per-segment (between LayerNorms) rather than per-layer.

4. **Probing selectivity** (Hewitt & Liang, 2019). Strong probes learn structure not present in the representation. Use minimum description length probes or control tasks to calibrate.

5. **Interpretability illusion** (Makelov et al., ICLR 2024). Learned subspace interventions can activate dormant pathways. Validate with counterfactual tests that check whether identified directions are used in normal operation, not just interventionally active.

## Experimental plan

**Phase 0: Cache activations.** Extract and store per-layer activations from GPT-2 and Pythia-410m for a diverse prompt set (~10k samples). One 3090 session. Everything downstream runs on cached data.

**Phase 1: Bisimulation probe.** Train Ridge/linear head mapping activation pairs to KL divergence in output distribution. Validate by checking that close points in the learned space produce similar model outputs on held-out data. Compare column space to SAE feature directions. Run on both models.

**Phase 2: Iterative peeling.** Train projection 1 on activations → output divergence. Measure residual. Train projection 2 on residual. Plot variance-captured curve. The saturation point gives effective dimensionality of behaviorally relevant geometry. The residual after saturation is measured dark matter.

**Phase 3: Contrastive with register augmentation.** Train on same-prompt ± register tokens. Extract the administrative subspace directly from difference vectors. Measure its dimensionality. Compare to projection-learned content space.

**Phase 4: Perturbation-sensitivity.** Train on (random perturbation direction, measured output change) pairs. Requires multiple forward passes per activation but amortizes into a reusable map. Compare to attribution patching results on known circuits.

**Phase 5: Intersection analysis.** Compute the intersection of all four null spaces. Measure its dimensionality. Characterize what lives there. Test whether removing these directions (projecting them out) affects model performance.

**Open question:** Does the iterative peeling run before or in parallel with the bisimulation probe? The greedy approach characterizes rank structure; the bisimulation probe gives the distance metric directly. They inform each other — the rank at which peeling saturates tells you the effective dimensionality of the bisimulation metric's column space. Recommendation: run bisimulation probe first (it's cheapest and gives the behavioral ground truth), then use its residual structure to guide the peeling analysis.

## Connection to prior work in this repo

- **Slicer:** The projections replace CLIP as the target space with the model's own behavior. The R²/nn_recall tension that motivated the alpha experiments is the symptom; the projections are the treatment.
- **Variable-bitrate reasoning:** The training-time feedback loop (sensitivity map → auxiliary loss) is the VBR paper's adaptive compression applied to geometry rather than tokens.
- **Saddle detection:** The bisimulation probe provides the metric for defining saddle points geometrically rather than behaviorally. Two prompts are at a saddle when their bisimulation distance is small (similar output) despite large activation distance (different representations).
- **Crystal lattice:** The iterative peeling characterizes what "crystallization" looks like in terms of behavioral projection rank — a representation has crystallized when the peeling curve saturates in few iterations (low effective rank, concentrated structure).
