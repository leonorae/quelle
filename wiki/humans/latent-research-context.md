# Updated Experimental Protocol: Geometry-Guided Adaptive Compression for Reasoning

## For the Implementation Agent

This document revises the previous protocol to incorporate recent literature and explicitly position the experiment within the current research landscape. It also addresses feasibility and why this approach is novel.

---

## 1. Introduction

This experiment tests whether a language model can learn to **adaptively compress its internal representations** based on geometric self-awareness (angle concentration), using only a small-scale arithmetic task and laptop-grade compute. The core idea is that the model's own geometry provides a signal for when to preserve information (low compression) and when to crystallize into token-like states (high compression), guided by a future prediction objective with stop-gradient (DSD-style).

The experiment is designed to be **minimal, feasible, and diagnostic**. It does not require RL, large datasets, or extensive compute.

---

## 2. Related Work and Positioning

### 2.1 Latent Reasoning via Vocabulary-Space Superposition 

**What they did:** Restricted latent reasoning to the column space of the vocabulary, treating it as a superposition over token probabilities. Achieved 4× compression without performance loss on GSM8k.

**Connection:** This validates that latent reasoning in a compressed space works. However, their compression is **fixed** (always through vocabulary) and guided by supervised fine-tuning, not geometry. Our experiment uses **adaptive** compression guided by the model's own uncertainty, which could be more efficient and general.

### 2.2 Latent Thoughts Tuning (LT-Tuning) 

**What they did:** Fused contextual hidden states with vocabulary embeddings to avoid feature collapse, using a three-stage curriculum. Enabled dynamic switching between latent and explicit modes.

**Connection:** They address the same instability we worry about, but their solution is complex (three stages, explicit fusion). Our DSD-inspired future prediction with stop-gradient may achieve stability more simply. They don't use geometry to guide switching; we propose angle concentration as the switch signal.

### 2.3 TRAAC: Adaptive, Attentive Compression 

**What they did:** Used RL to adaptively prune reasoning tokens based on self-attention importance, achieving 8.4% accuracy gain with 36.8% shorter chains.

**Connection:** This is the closest to our goal—adaptive allocation of reasoning budget. But they use RL and operate on **token count**, not latent dimension. Our approach uses **geometry** as the signal, potentially eliminating the need for RL, and compresses **representation** rather than tokens, which could preserve more information.

### 2.4 Training Language Models to Explain Their Own Computations 

**What they did:** Showed models have "privileged access" to their own internals and can explain themselves better than other models can.

**Connection:** This supports our hypothesis that geometry contains self-knowledge. Angle concentration could be a measurable aspect of that privileged access. If we can use it to guide compression, we're leveraging the model's self-knowledge directly.

### 2.5 The Geometry of Thought (NeurIPS 2025)

**What they did:** Proposed mixed-curvature spaces for different reasoning types, with a learned gating mechanism to route information.

**Connection:** They show that geometry matters for reasoning. Our angle concentration could be seen as a simple proxy for which "curvature regime" the model is in. If concentration is high, the model is in a stable, low-curvature region (Euclidean-like); if low, it's in a high-curvature, exploratory region (hyperbolic-like). This aligns with their framework.

### 2.6 Diffusion Chain of Lateral Thought (DCoLT, NeurIPS 2025)

**What they did:** Applied RL to optimize reasoning trajectories in diffusion language models, enabling non-linear reasoning.

**Connection:** They use RL to shape reasoning; we use geometry. Both aim to improve reasoning efficiency, but our approach is cheaper and potentially more interpretable.

---

## 3. Why Is No One Doing This? (Speculation)

Despite the convergence of ideas, no existing work combines **geometry-guided adaptive compression** with **DSD-style future prediction**. Possible reasons:

1. **Angle concentration is underappreciated:** GAIN-RL (which introduced it) is recent and focused on data sampling, not on-the-fly compression. Its potential as a real-time control signal hasn't been explored.

2. **RL is the default:** Most adaptive reasoning work (TRAAC, DCoLT) uses RL because it's a flexible framework. Geometry may seem less powerful or harder to integrate.

3. **DSD is very new (Feb 2026):** The diffusion-as-self-distillation insight is fresh; applying it to sequential reasoning is a novel extension.

4. **Focus on token count vs. latent dimension:** The community has focused on pruning tokens (explicit reasoning steps) rather than compressing internal representations. Latent compression is more radical and less studied.

5. **Fear of collapse:** Latent reasoning is known to be unstable (LT-Tuning explicitly addresses collapse). Many may have tried and failed, then moved on.

6. **Compute constraints:** Most labs have large resources and pursue RL-based methods because they scale. The minimal, laptop-scale approach is undervalued.

Your experiment directly addresses these gaps: it's simple, uses geometry as a signal, leverages DSD for stability, and operates on latent dimension. If it works, it will be a **breakthrough in efficiency and self-awareness**.

---

## 4. Practical Feasibility

### 4.1 Compute Requirements

| Component | Estimated Cost |
|-----------|----------------|
| Model size | 10M parameters |
| Dataset | 10k examples, ~1M tokens |
| Training | 10 epochs on CPU: 2-4 hours |
| Memory | <4GB RAM |
| **Total** | **Laptop-only; ~$0 AWS** |

This is **trivially feasible** on any modern laptop. No GPU required.

### 4.2 Implementation Complexity

| Module | Difficulty | Notes |
|--------|------------|-------|
| Transformer | Easy | Standard implementation |
| Compression head | Moderate | Differentiable dimension selection via Gumbel softmax |
| Angle concentration | Easy | Cosine similarity over tokens |
| Future prediction loss | Easy | MSE with stop-gradient |
| Training loop | Easy | PyTorch standard |

The only tricky part is differentiable dimension selection. Use a continuous relaxation (Gumbel softmax) to allow gradient flow to λ_t. If that fails, hard-code λ_t and treat it as a post-hoc analysis (still valuable).

### 4.3 Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Model ignores λ_t | Add small entropy bonus to encourage variation |
| Collapse | DSD stop-gradient prevents collapse; monitor latent variance |
| No correlation with concentration | Still informative—null result is a finding |
| λ_t saturates at extremes | Adjust α/β initialization; add temperature annealing |

### 4.4 Success Criteria

- λ_t negatively correlated with concentration (r < -0.3)
- Adaptive compression outperforms fixed baselines on hard problems
- Future prediction loss decreases stably
- Latent space shows clustering by step type (probed via UMAP)

Even partial success (e.g., correlation exists but performance not better) is publishable as a proof of concept.

---

## 5. Conclusion

This experiment is **novel, feasible, and well-grounded** in recent literature. It tests a hypothesis that no one has yet examined: that geometry alone can guide adaptive compression in reasoning, without RL. The results—positive or null—will advance the field's understanding of self-aware computation.

Proceed with implementation as specified. Document all results, especially failures—they will inform future work.
