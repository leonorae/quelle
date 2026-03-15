# Implementation Decisions — mlp-context-steering

> Non-obvious implementation decisions for this experiment. Repo-level
> decisions live in `wiki/humans/decisions.md`.

---

## D1 — 2026-03-15 — TransformerLens for model access

**Decision**: Use TransformerLens (neel-nanda-1/TransformerLens) for model
loading, hook registration, and cache access rather than raw HuggingFace +
manual hooks.

**Rationale**: TransformerLens provides standardised hook points and a cache
API that maps directly to the per-layer residual stream, MLP output, and
attention pattern tensors we need. Avoids reinventing hook infrastructure.

---

## D2 — 2026-03-15 — Effective rank via entropy (erank), NOT PCA participation ratio

**Decision**: Measure representation dimensionality using the entropy-based
effective rank (erank): compute SVD of the activation matrix, normalise
singular values into a probability distribution, and exponentiate the Shannon
entropy.

**Rationale**: The PCA participation ratio in `tools/analysis/geometry/` (if it
existed) would measure a different quantity. Erank is more sensitive to the
tail of the singular value distribution, which is where MLP ablation effects
are expected to appear. The participation ratio can miss redistribution among
small singular values.

**Reference**: Roy & Bhattacharya (2007), "Effective Rank of a Matrix".

---

## D3 — 2026-03-15 — Ablation propagates through all subsequent layers

**Decision**: When ablating MLP at layer L, the hook zeros the MLP output at
layer L and lets the modified residual stream propagate naturally through
layers L+1, L+2, ... This is NOT isolated per-layer ablation.

**Rationale**: The hypothesis is about MLP's role in steering downstream
representations. Isolating per-layer effects would measure local contribution
only, missing the propagation dynamics. Isolated per-layer ablation is
deferred to Phase 3b if needed.

---

## D4 — 2026-03-15 — Batch size 4, 500 prompts from Pile validation

**Decision**: Default config uses batch_size=4 and 500 prompts sampled from
The Pile validation split.

**Rationale**: Small batch size keeps memory manageable on single-GPU setups
(T4/A100). 500 prompts provides enough statistical power for detecting
moderate effect sizes while keeping wall time reasonable.
