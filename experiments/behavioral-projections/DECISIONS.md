# Behavioral Projections — Implementation Decisions

Decisions specific to this experiment. Repo-level decisions live in
`wiki/humans/decisions.md`.

---

## 2026-03-11

### D1 — safetensors over HDF5 for activation storage

**Decision:** Store cached activations as safetensors files with JSON metadata
sidecars, not HDF5.

**Rationale:** safetensors is already a transitive dependency (transformers uses
it), supports zero-copy mmap for large tensors, and has a simpler API than h5py.
HDF5's hierarchical key structure is unnecessary — we use one safetensors file
per batch of prompts plus a metadata JSON file for prompt IDs, categories, and
group linkage.

**Layout:**
```
data/activations/{model_slug}/
  batch_{n}.safetensors      # keys: layer_0..layer_L (last-token), logits_indices, logits_values
  batch_{n}_full.safetensors  # keys: layer_0..layer_L (full sequence)
  metadata.json               # [{prompt_id, category, group_id, token_count}, ...]
```

---

### D2 — Pythia-410m for pipeline development

**Decision:** Build and validate the full Phase 0→1 pipeline on Pythia-410m
before running on Qwen2.5-7B.

**Rationale:** Pythia-410m runs on CPU, caches fast (~480MB for 10k prompts),
and has 24 layers — enough structure for meaningful per-layer analysis. Pipeline
bugs are cheaper to find at this scale.

---

### D3 — Symmetric KL for pair targets

**Decision:** Use symmetric KL: `y_ij = (KL(p_i||p_j) + KL(p_j||p_i)) / 2`
as the pair target for the bisimulation probe.

**Rationale:** Asymmetric KL makes pair order matter, which complicates pair
sampling and doubles the effective dataset for no clear benefit. Symmetric KL
is a proper divergence and avoids artifacts from which distribution is the
reference.

**Note:** This is not JSD (which uses the mixture as reference). It's the
arithmetic mean of both KL directions.

---

### D4 — Ridge baseline first, gate MLP on Ridge results

**Decision:** Phase 1 trains Ridge regression before any learned projection.
MLP is gated on Ridge R² < 0.3.

**Rationale:** If Ridge R² > 0.3, the finding is stronger: a simple linear map
captures behavioral geometry. The Ridge coefficient vector IS a 1D projection
from activation differences to KL. For multi-dimensional projection, use
multi-output Ridge or SVD of the Gram matrix.

**Precision note:** Ridge predicts scalar KL from `h_i - h_j`. Its coefficient
vector `w ∈ R^d_hidden` defines a 1D projection. For d_proj > 1, train
multi-output Ridge mapping `h_i - h_j → [KL, ΔKL_perturbed, ...]` or use SVD
of the matrix `X_diff^T @ diag(y_kl) @ X_diff` to find the top-k directions
that explain KL variance.

---

### D5 — Top-100 logprobs, not full vocabulary

**Decision:** Store top-100 logprobs and their vocabulary indices per prompt.
Also store residual log-mass `log(1 - sum(top_100_probs))`.

**Rationale:** Pythia vocab = 50,304. Qwen vocab = 152,064. Full distributions
at float32 would be 200KB–600KB per prompt. Top-100 covers >99% probability
mass for most generations. KL approximation error is bounded by the tail
contribution, which we can estimate from the stored residual mass.

**KL computation:** For pairs where both distributions have the same token in
top-100, use exact logprob difference. For tokens in one top-100 but not the
other, approximate the missing logprob as `log(residual_mass / (vocab_size - 100))`.
This is conservative (uniform over tail).

---

### D6 — Bootstrap 500 prompts, scale later

**Decision:** Start with ~500 prompts for pipeline validation. Scale to 5–10k
after Phase 0→1 works end-to-end on Pythia-410m.

**Rationale:** The handoff specifies 5–10k across 6 categories. Curating that
set is a multi-day effort. A 500-prompt bootstrap set validates the full
pipeline in hours. Categories: 200 benchmark, 100 semantic diversity, 100
perturbation pairs (20 base × 5), 100 agent-relevant.
