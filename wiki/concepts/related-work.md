# Related Work

> Summaries of papers directly informing `variable-bitrate-reasoning`.
> See `LATENT.md` §2 for the full positioning discussion.

**Relevant experiments**: `variable-bitrate-reasoning`

---

## 1. Latent Reasoning via Vocabulary-Space Superposition

**What they did**: Restricted latent reasoning to the column space of the
vocabulary matrix, treating it as a superposition over token probabilities.
Achieved 4× compression without performance loss on GSM8k.

**Connection**: Validates that reasoning in a compressed latent space works.
Their compression is *fixed* (always through vocabulary) and driven by
supervised fine-tuning rather than geometry. We use *adaptive* compression
guided by the model's own angle concentration, which may be more efficient and
general.

---

## 2. Latent Thoughts Tuning (LT-Tuning)

**What they did**: Fused contextual hidden states with vocabulary embeddings
to avoid feature collapse, using a three-stage curriculum. Enabled dynamic
switching between latent and explicit reasoning modes.

**Connection**: Addresses the same instability risk we face, but via a complex
three-stage setup. Our DSD stop-gradient objective may achieve comparable
stability more simply. They do not use geometry to guide the latent/explicit
switch; we propose angle concentration as that signal.

---

## 3. TRAAC: Adaptive, Attentive Compression

**What they did**: Used RL to adaptively prune reasoning *tokens* based on
self-attention importance, achieving 8.4% accuracy gain with 36.8% shorter
chains.

**Connection**: Closest in spirit to our goal. Key differences:
- They use RL; we use geometry.
- They prune *token count*; we compress *representation dimension*.
- Compressing dimension may preserve more information than dropping tokens.

---

## 4. GAIN-RL (Angle Concentration for Data Sampling)

**What they did**: Introduced angle concentration as a signal for selecting
high-quality training data in RL-based reasoning pipelines.

**Connection**: Source of the angle concentration metric used in our
compression policy. They use it for offline data selection; we repurpose it as
an online, per-step control signal during inference/training.

---

## 5. The Geometry of Thought (NeurIPS 2025)

**What they did**: Proposed mixed-curvature spaces for different reasoning
types, with a learned gating mechanism to route information.

**Connection**: Shows geometry matters for reasoning. Our angle concentration
can be read as a proxy for "which curvature regime" the model is in:
- High concentration → stable, low-curvature (Euclidean-like) → safe to compress.
- Low concentration → exploratory, high-curvature (hyperbolic-like) → preserve information.

---

## 6. Diffusion Chain of Lateral Thought (DCoLT, NeurIPS 2025)

**What they did**: Applied RL to optimize reasoning trajectories in diffusion
language models, enabling non-linear reasoning paths.

**Connection**: Both DCoLT and this experiment aim to improve reasoning
efficiency, but through different mechanisms — RL vs. geometry. Our approach
is cheaper (no reward model needed) and potentially more interpretable.

---

## Why No One Has Combined These Yet

From `LATENT.md` §3:

1. Angle concentration is underappreciated as a real-time control signal (GAIN-RL is recent).
2. RL is the community default for adaptive reasoning.
3. DSD-style future prediction is very new (Feb 2026).
4. The field focuses on pruning *tokens* rather than compressing *dimensions*.
5. Latent reasoning is known to be unstable; many may have abandoned it.
6. Most labs have GPU resources and prefer RL methods that scale.
