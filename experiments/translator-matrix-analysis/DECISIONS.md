# Translator Matrix Analysis — Implementation Decisions

Decisions specific to this experiment. Repo-level decisions live in
`wiki/humans/decisions.md`.

---

## 2026-03-15

### D1 — CPU only, no GPU dependency

**Decision:** All computation runs on CPU. No CUDA dependency.

**Rationale:** SVD of 1024×1024 matrices takes seconds on CPU. Adding a GPU
requirement would increase setup friction for zero performance benefit. The
entire experiment is matrix algebra on pre-computed artifacts.

**Implication:** `torch.linalg.svd` on CPU tensors. No device management code.

---

### D2 — Pickle loading: accept the security tradeoff

**Decision:** Load matrices via `pickle.load()` from the HuggingFace release.

**Rationale:** Pickle is inherently unsafe (arbitrary code execution), but these
files come from a known academic source (Yom Din et al.) distributed via
HuggingFace. The alternative would be re-training the matrices, which defeats
the purpose. Accept the risk, but document it.

**Mitigation:** Download from HuggingFace only (not arbitrary URLs). Verify
tensor shapes after loading (should be 1024×1024). Do not load pickle files
from other sources without review.

---

### D3 — erank implementation local, not promoted to tools/

**Decision:** Implement erank in `src/metrics.py` within this experiment, even
though `mlp-context-steering` has an identical implementation.

**Rationale:** CLAUDE.md says "Do not move code to tools/ until it is needed
by ≥2 separate experiments." This IS the second experiment using erank, but
the implementations may diverge (this one operates on square matrices, the
other on rectangular activation matrices). Keep local for now. If the two
implementations stay identical after both experiments are complete, promote
to `tools/analysis/geometry/`.

---

### D4 — Adjacent pairs are primary; all-pairs is secondary

**Decision:** Phase 2 (adjacent differences) is the core analysis. Phase 3
(full atlas) is secondary — run it, but don't block on it.

**Rationale:** The frame-change story is about adjacent-layer transitions.
The all-pairs atlas is richer but noisier. Phase 0b composition validation
determines how much the all-pairs structure adds: if composition works well,
adjacent pairs are sufficient; if it fails, the all-pairs structure is
independently informative.

---

### D5 — Frame distance: Frobenius norm of (M − I)

**Decision:** Primary frame distance metric is ‖M_{i→j} − I‖_F. Report
spectral norm ‖M_{i→j} − I‖_2 as secondary.

**Rationale:** Frobenius norm captures total magnitude of frame change across
all dimensions. Spectral norm captures the single largest directional change.
Both are informative; Frobenius is the primary because it's sensitive to
distributed changes (which we predict for MLP-heavy frame shifts).

---

### D6 — Regime boundary detection: derivative peaks, not thresholds

**Decision:** Identify regime boundaries as layers where |d/dl ‖Delta_l‖_F|
is locally maximal, rather than layers where ‖Delta_l‖_F exceeds a threshold.

**Rationale:** Thresholds require arbitrary choices. Peaks in the derivative
of the frame-change profile are threshold-free — they mark where the rate of
frame change itself changes. A boundary is where the model transitions between
one regime of smooth evolution and another.

---

### D7 — Matrix inventory is known: 300 upper-triangular pairs

**Decision:** The HuggingFace release contains exactly 300 matrices: all pairs
(i, j) where 0 ≤ i < j ≤ 24. Layer indices 0 = post-embedding, 24 = final
layer output, giving 25 "layers" total (not 24). Only forward-direction maps
(i < j) are available — no backward maps.

**Verified:** Naming convention is `{i}_{j}.pickle`. Each file contains a
`torch.Tensor` of shape (1024, 1024), dtype float32. Left-multiplication
convention: `v_target = M @ v_source`.

**Implication:** All adjacent pairs (l, l+1) for l ∈ [0, 23] are available.
All layer-to-final pairs (l, 24) are available. Composition testing uses
triples i < k < j (all valid triples exist). No need for graceful degradation.

---

### D8 — Matrices are NOT near-identity; reframe Phase 1c

**Decision:** Phase 1c (near-identity analysis) must be reframed. The matrices
are far from identity — Frobenius norm of M_{0→1} is ~628 vs ~32 for I.

**Rationale:** No regularization was applied during training (plain OLS via
`sklearn.linear_model.LinearRegression(fit_intercept=False)`). No weight
decay, no orthogonality constraint, no initialization toward identity. The
matrices are unconstrained least-squares fits to the training data.

**Implication:** The "perturbative frame change" narrative from the protocol
may not hold. Delta_l = M_{l→l+1} − I may be dominated by the large deviation
from identity rather than revealing subtle frame-change structure. Phase 1c
should report ‖M − I‖ / ‖M‖ (relative deviation) alongside absolute, and
Phase 2 should consider whether raw Delta_l = M − I or normalized versions
are more informative. The fact that Tuned Lens initializes at identity but
these unconstrained matrices are far from identity is itself a finding.

---

### D9 — Final-token-only training: flag as limitation

**Decision:** Document prominently that the matrices were trained on
**sentence-final token representations only** from Wikipedia.

**Rationale:** This means the matrices capture the frame transformation
specific to that positional/contextual distribution. Frame structure may
differ for earlier positions, longer contexts, or non-Wikipedia text. All
claims about "general" regime structure should be qualified.

---

### D10 — LayerNorm handling: matrices map raw residual stream

**Decision:** The matrices operate in **unnormalized residual stream space**.
The final LayerNorm (`ln_f`) was explicitly excluded during data collection
(`model._no_ln_f = True`). Per-block LayerNorms are part of each block's
forward pass and thus baked into the target representations.

**Implication:** When interpreting the matrices, remember they map post-block
residual stream states (after the block's internal LN + attn + LN + MLP + skip
connections). To get logits from a matrix prediction at the final layer, apply
`ln_f` then the unembedding manually. For the frame-change analysis, this is
fine — we're studying the matrices themselves, not using them for prediction.

---

### D11 — Submodule-level matrices exist but are not released

**Decision:** The codebase (`sashayd/mat`) includes code for sub-block level
matrices (`p{layer}_{subpart_from}_{subpart_to}.pickle`) that decompose each
block into ln_1, attention, residual, ln_2, MLP, residual components. These
are **not uploaded to HuggingFace** and would require re-running the training
code to obtain.

**Implication:** If Phase 2 results are interesting enough to warrant
sub-block decomposition (e.g., isolating MLP vs attention contributions to
frame change within a single layer), this is possible but requires running
the training pipeline. Defer unless results demand it.
