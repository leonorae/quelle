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

**Smoke-test outcome (2026-03-12):** `make_bootstrap_set()` generated 221 prompts
(template coverage limits). Phase 0→1 pipeline validated end-to-end on CPU.
R²≈0 is expected — placeholder prompts lack behavioral diversity.

---

### D7 — sample_pairs caps at unique-pair count

**Decision:** `sample_pairs` caps `n_pairs` at `n_total*(n_total-1)//2` and
deduplicates using a set, rather than sampling with replacement.

**Rationale:** With small prompt sets (221 bootstrap prompts → 24310 unique pairs),
requesting 50k pairs produced duplicate rows in the Ridge design matrix, causing
ill-conditioned matrix warnings and inflated pair counts. With ≥5k real prompts
(≥12.5M unique pairs) this limit is never hit; the fix costs negligible overhead
and makes the code correct at all scales.

---

## 2026-03-12

### D8 — Full prompt set: diversity + LLM-informed selection (5–10k)

**Decision:** The 5–10k prompt set uses five components, combining surface diversity
with model-behavior-informed refinement:

| Component | Target size | Selection strategy |
|---|---|---|
| **Benchmark anchors** | ~2k | Curated from MMLU/GSM8K subsets — hand-pick subjects for domain breadth |
| **KL-spectrum filling** | ~2k | Run Pythia-410m on a ~20k candidate pool, compute pairwise KL on a sample, select prompts that maximize coverage of the KL range (greedy facility-location or similar) |
| **Perturbation families** | ~2k | ~400 base prompts × 5 variants; use LLM-generated rephrasings instead of templates |
| **Semantic diversity** | ~2k | Embed candidate prompts, cluster, sample from each cluster |
| **Sensitivity probes** | ~1k | After Phase 1, select prompts where the bisimulation probe has high residual — stress-test the geometry |

**Rationale:** Pure surface diversity is wasteful — many "diverse" prompts cluster
in similar KL ranges in activation space. Pure behavior-informed selection risks
overfitting to model-size quirks. The staged approach lets us start with curated
diversity (benchmarks + semantic + perturbations), run Phase 0–1, then use results
to fill gaps for Phases 2–4.

**Staging:** Components 1–4 can be built before any model runs. Component 5
(sensitivity probes) is gated on Phase 1 results — add it after the bisimulation
probe is trained.

**Format:** Same JSONL schema as D6. New categories: `kl_selected`, `sensitivity`.
`group_id` required for perturbation families, null for others.

---

### D9 — LLM-generated rephrasings over template perturbations

**Decision:** For the full prompt set, generate perturbation variants (rephrase,
context_added, authority_bias, negation) using an LLM rather than templates.
Store generated variants explicitly in JSONL.

**Rationale:** Template perturbations (e.g., "Please answer: {X}") are mechanical
and produce shallow variation in output distributions. LLM rephrasings produce
more naturalistic within-group variation, which is critical for Phase 3
(contrastive discrimination) to learn meaningful same-prompt clustering.
Templates remain acceptable for `register_shift` (padding tokens), which is
intentionally non-semantic.

---

### D10 — KL-spectrum filling via greedy selection

**Decision:** After running Phase 0 on a large candidate pool (~20k), select ~2k
prompts that maximize coverage of the pairwise KL range. Use a greedy
facility-location approach: iteratively add the prompt whose inclusion most
increases coverage of underrepresented KL bins.

**Rationale:** The bisimulation probe needs training pairs spanning the full KL
range. Random sampling over-represents the modal KL values and under-represents
the tails. Greedy selection is simple, deterministic, and doesn't require tuning
a clustering algorithm. This component is gated on Phase 0 completing on the
candidate pool.

---

### D11 — Shared corpus with slicer (CLIP projection)

**Decision:** Design the behavioral-projections corpus as the primary prompt set
and share it with the slicer experiment (Pythia → CLIP projection). Do not
design a separate slicer-specific corpus.

**Rationale:**

1. *Direct comparability.* Same activations, different projection targets
   (CLIP vs KL-behavioral). Sharing inputs makes null-space comparison
   well-posed: "what does CLIP consider irrelevant that the bisimulation
   probe considers critical?" requires identical input activations.

2. *Infrastructure alignment.* Both experiments read from the same
   `probe_cache/` slugs. Slicer's `probe_clip_vectors` and behavioral
   projection metrics sit on the same layer axis. No architectural changes
   needed.

3. *No cross-pollution.* CLIP projection and bisimulation projection have
   independent training targets. Sharing input prompts creates alignment,
   not leakage.

**Design constraint:** Ensure the semantic diversity component (D8, ~2k prompts)
includes a concrete/perceptual cluster (~300–500 prompts with strong visual
grounding — objects, scenes, spatial relationships) so slicer has adequate
signal in the shared set.

**If slicer needs more:** Add visual-semantic prompts as a sixth component
rather than distorting the existing five. Do not skew the corpus toward
perceptual content at the expense of abstract reasoning, math, or code coverage.

**Pythia training set (The Pile):** No filtering required. We measure activation
geometry (KL divergence, projection structure), not factual accuracy.
Memorized-vs-novel is real signal, not contamination. The five-component design
already prevents over-representation of Pile-adjacent text.

---

### D12 — Periphery probes: formally diverse inputs for activation-space coverage

**Decision:** Add a sixth corpus component (~600 prompts) of intentionally
unusual, malformed, or out-of-distribution inputs designed to activate
low-density regions of the model's representational geometry.

**Problem:** Components 1–4 are all well-formed English questions. This
produces dense coverage of the "question-answering" region of activation space
but systematically misses the periphery. The corpus has topical diversity but
no formal diversity — every prompt is grammatical, single-language, and
question-shaped.

**Why periphery matters:** If the bisimulation probe's effective rank is 50 on
routine text but jumps to 200 on unusual text, that's a finding: the model uses
more of its geometry for hard/unfamiliar inputs. We can't detect
context-dependent behavioral complexity without inputs that push into
low-density regions. The KL-spectrum filling component (D10) will partially
address this post-Phase-0, but we need some peripheral coverage in the initial
corpus to get meaningful Phase 0/1 results at all.

**Subcategories:**

| Subcategory | ~Count | Rationale |
|---|---|---|
| Malformed syntax | ~80 | Truncated sentences, garbled grammar, typos — tests robustness of representational structure |
| Mixed-language | ~80 | Code-switching, non-English, transliteration — different tokenization paths, different activation patterns |
| Self-contradictory | ~60 | "Explain why water is dry" — forces model into unusual output distributions |
| High-perplexity / nonsense | ~80 | Random token sequences, adversarial strings — extreme periphery of activation space |
| Unusual registers | ~100 | Poetry, legalese, IRC logs, commit messages, recipes, stage directions — same semantics, different form |
| Raw naturalistic text | ~100 | Plain prose excerpts (not questions) — fills the actual center of Pythia's training distribution, which our question-heavy corpus undersells |
| Domain outliers | ~100 | Highly specialized jargon, archaic text, technical notation — things Pythia would find genuinely strange |

**Design note:** These prompts are not expected to produce "good" model outputs.
The point is the activations, not the generations. A prompt that produces
confused or degenerate output is exactly the kind of input where the probe needs
to characterize geometry.

**Interaction with D11 (slicer):** Periphery probes are less useful for slicer
(CLIP has no meaningful embedding for nonsense or garbled text). This is fine —
they serve the behavioral-projections experiment specifically. Slicer can ignore
this component or use it as a negative control.

---

### D13 — Phase 0.5: Tuned Lens baseline before bisimulation probe

**Decision:** Before investing in the learned bisimulation probe, test whether
a simple Tuned Lens (per-layer affine map h → logits, trained to minimize KL
with the model's final output distribution) already captures pairwise behavioral
distance.

**Protocol:**

1. Train one `nn.Linear(1024, 50304)` per layer, targeting the model's true
   output distribution (reconstructed exactly from cached `layer_24` via
   `ln_f` + `embed_out`).
2. For each pair (i, j) at each layer: decode both hidden states through the
   lens independently, compute symmetric KL between the decoded distributions.
   Call this `tuned_lens_kl`.
3. Compute `true_kl` from cached output logprobs (existing infrastructure).
4. Regress `tuned_lens_kl` on `true_kl`. Report R² and Spearman ρ, **per layer**.

**Primary output:** Per-layer R² curve, not a single number. The shape of the
curve matters more than its magnitude:

- Layer 24 (final): near-perfect reconstruction expected (it IS the output pathway)
- Early layers: both Tuned Lens and bisimulation probe should be poor
- **Middle layers are the decision point.** If Tuned Lens already has decent R²
  at layers 10–18, independent decoding captures most pairwise structure and
  the bisimulation probe adds little. If Tuned Lens R² is poor at those layers
  while the bisimulation probe (when built) is better, the pairwise formulation
  captures structure that independent decoding misses.

**Gate:** The bisimulation probe (Phase 1) is only worth building if:
- Tuned Lens R² at middle layers is below ~0.3, OR
- The bisimulation probe can significantly exceed Tuned Lens R² at middle layers

**Pair sampling for evaluation:** Must be **stratified by KL range**, not
uniform random. Without stratification, most pairs have very high KL
(different prompts → very different outputs), making regression trivially good
("far-apart things are far apart"). Stratified sampling draws roughly equal
proportions from:
- Low-KL pairs (perturbation groups, same base prompt)
- Medium-KL pairs (same domain, different prompts)
- High-KL pairs (cross-domain)

This ensures R² and Spearman reflect ranking quality across the full KL range,
not just the easy separations.

**Implementation:** Self-contained `src/tuned_lens_baseline.py`, not the
`tuned-lens` PyPI package. ~250 lines, reuses existing activation loading and
KL computation infrastructure. No new dependencies.

---

### D14 — Three-condition pairwise comparison design

**Decision:** Run three conditions on the same pair set to determine whether
pairwise structure exists in activation space and which tool captures it:

| Condition | Architecture | Objective | File |
|---|---|---|---|
| C1: Standard Tuned Lens | Per-layer Linear(d,V) | KL(lens ‖ true) per prompt | `src/tuned_lens_baseline.py` |
| C2: Pairwise Tuned Lens | Per-layer Linear(d,V) | MSE(lens_KL_pair, true_KL_pair) | `src/pairwise_lens.py` |
| C3: Bisimulation Probe | P ∈ R^{d_proj × d} | MSE(‖P(h_i - h_j)‖₂, true_KL) | `src/bisimulation_probe.py` |

**Evaluation:** Same for all three — R² and Spearman ρ on held-out stratified
pairs, per layer. `src/compare_conditions.py` runs all three and produces
comparison plots.

**Interpretation:**
- C1 ≈ C2 ≈ C3 → no pairwise structure, standard lens suffices
- C2 > C1 → pairwise optimization helps at those layers
- C3 > C2 → direct pairwise metric captures structure that decode-then-compare misses

**Effective rank sweep (C3 only):** After main comparison, sweep
d_proj ∈ {1024, 512, 256, 128, 64, 32, 16}. The knee in R² vs d_proj is the
effective behavioral dimensionality per layer.

**Rationale:** The tuned lens baseline (D13) showed ρ ≈ 0.88 but R² ≈ 0.5-0.6
at middle layers, with an R² dip at layers 17–19 while Spearman stays high.
This means ranking is good but magnitudes are wrong — exactly the gap a pairwise
objective might close. The three-condition comparison determines whether pairwise
structure exists and which downstream tool to use.

---

### D15 — Learned projection for multi-dimensional bisimulation

**Decision:** Condition 3 uses two approaches: Ridge regression (1D baseline,
already existed) and a learned linear projection via Adam (multi-dimensional).

**Ridge:** `w^T (h_i - h_j) → scalar KL`. The coefficient vector is a 1D
projection. Already validated in smoketest.

**Learned:** `BisimulationProjection(d_hidden, d_proj)`: a single `nn.Linear`
(no bias) mapping h-differences to d_proj dimensions. Predicted KL =
L2 norm of the projection. Trained with MSE loss via Adam.

**Why both:** Ridge is the closed-form baseline — fast, no hyperparameters
beyond alpha, guaranteed global optimum for the linear case. The learned
projection handles the non-linearity of the L2 norm and supports d_proj > 1
for the rank sweep.

**Note:** The L2 norm makes the loss non-convex in P even though the
architecture is linear. Ridge avoids this by predicting scalar KL directly.

---

## 2026-03-15

### D16 — Targeted rank sweep for C3 collapse diagnosis

**Decision:** Run rank sweep at collapse layers (9, 11, 16, 17) plus
control layers (6, 12, 20), with multiple seeds per (layer, d_proj) and
participation ratio measurement.

**Motivation:** C3 has catastrophic R² collapses at layers 9, 11, 16, 17
while Spearman ρ stays reasonable. Ridge (1D) captures nothing at any layer.
Three competing (non-exclusive) hypotheses:

**(2a) Intrinsic dimensionality mismatch:** Behavioral structure at collapse
layers lives in ~30–50 dims; the 1024-dim projection has ~1000 noise dims
dominating the L2 norm. **Signature:** sharp knee in R² vs d_proj at intrinsic
dim, then plateau. Low seed variance above the knee. Higher participation
ratio at collapse layers than controls.

**(2b) Sample complexity failure:** 221 prompts → ~24k pairs is insufficient
to estimate a 1024-dim projection regardless of intrinsic dim. **Signature:**
monotonic R² improvement as d_proj shrinks (fewer params = better
generalization), no knee. PR uninformative.

**(2c) Optimization failure:** L2 norm makes loss non-convex; at high d_proj
the landscape has more saddle points. Learned projection finds bad local
minimum that fits a few large-distance pairs (preserving rank ≈ Spearman)
while failing on scale (collapsing R²). **Signature:** high R² variance
across seeds at the same d_proj, especially at collapse layers and large
d_proj.

**Diagnostics (all run in single script):**
1. Rank sweep: d_proj ∈ {8, 16, 32, 64, 128, 256, 512, 1024} — enough
   resolution to detect a knee in the 32–256 range
2. Multiple seeds (default 3, flag for 5) per (layer, d_proj) — measures
   optimization variance to test (2c)
3. Participation ratio of learned projection's singular values at each
   (layer, d_proj) — tests whether intrinsic dim differs at collapse layers

**Implementation:** `src/diagnose_c3_collapses.py` with config override
`configs/collapse_sweep.yaml`.

**Outputs:**
- `collapse_rank_sweep.png`: R², ρ, PR vs d_proj with error bands
- `seed_variance.png`: R² std across seeds per d_proj (collapse vs control)
- `diagnosis_summary.txt`: automated hypothesis evaluation

