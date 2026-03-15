---
status: scaffolded
owner: null
dependencies: []
---

# Translator Matrix Structural Analysis

**Hypothesis**: The structure of inter-layer translator matrices encodes how the
model's interpretive frame shifts across depth. Adjacent-matrix differences
isolate frame change from state change. Layers where frame change dominates are
regime boundaries.

Addresses two novelty gaps: translator matrices as unstudied objects (Review A)
and frame/state decomposition for regime detection (Review F).

## Model & Data

GPT-2 Medium (24 layers, d_model = 1024). Pre-trained translator matrices from
Yom Din et al. (arXiv:2303.09435, LREC-COLING 2024), released at
`huggingface.co/sashay/linear-shortcut` under `gpt2-medium/wikipedia/`.

These are linear (no bias) maps M_{i→j} between arbitrary layer pairs.
Trained on Wikipedia. Code at `github.com/sashayd/mat`.

## Compute

Negligible — SVD of 1024×1024 matrices, CPU only, minutes total.

## Background

- **Yom Din et al.**: Trained all-pairs linear shortcuts. Published analysis is
  about output quality (how well shortcuts predict final-layer logits). Nobody
  has examined the matrices themselves.
- **Tuned Lens** (Belrose et al.): Trains affine maps layer→final. Shows
  hidden-state covariance drifts across layers but never examines what the
  translators encode about that drift. Layer-to-final only; Yom Din is all-pairs
  (strictly richer).
- **Key distinction**: Yom Din matrices are linear (no bias) and map between
  arbitrary layer pairs — a complete "atlas" of inter-layer frame transformations.

## Protocol

### Phase 0: Acquisition and Validation
- **0a** — Download matrices from HuggingFace
- **0b** — Validate composition: M_{i→j} ≈ M_{k→j} @ M_{i→k} for sampled triples

### Phase 1: SVD of Individual Matrices
- **1a** — Full SVD of each matrix
- **1b** — Singular value spectra: erank, condition number, spectral entropy
- **1c** — Near-identity structure: ‖M_{l→l+1} − I‖ for adjacent pairs
- **1d** — Cross-pair spectral comparison, regime boundary candidates

### Phase 2: Adjacent Differences as Frame Change (core novel analysis)
- **2a** — Frame deltas: Delta_l = M_{l→l+1} − I, SVD of each Delta
- **2b** — Frame change magnitude profile: ‖Delta_l‖_F and ‖Delta_l‖_2 across layers
- **2c** — Frame change dimensionality: erank(Delta_l) across layers
- **2d** — Principal frame-change directions (top-k right singular vectors)
- **2e** — Alignment with known GPT-2 structural features (if available)

### Phase 3: Full Atlas Analysis
- **3a** — Frame distance matrix + hierarchical clustering → regime structure
- **3b** — Compare to CKA layer similarity (Sun et al. 2407.09298, Phang et al.)
- **3c** — Composition residuals as nonlinearity markers

## Success Criteria

| Outcome | What it means |
|---|---|
| **Strong** | Clear regime boundaries from frame-change magnitude, matching CKA. Interpretable principal directions. |
| **Moderate** | Smooth profiles but clustering still reveals block structure (gradual transitions) |
| **Null** | All near-identity with similar spectra, no structure beyond slow drift |
| **Surprising** | Large composition errors in specific layer ranges (nonlinearity pockets) |

## Connections

- **mlp-context-steering**: Identifies regime boundaries via MLP ablation sensitivity.
  If both methods agree on boundary layers, that's convergent evidence from
  independent methods.
- **behavioral-projections**: Principal frame-change directions (Phase 2d) can be
  compared to bisimulation probe's learned projection. Prediction: bisimulation
  weights state-change directions more than frame-change directions.
- **Tuned Lens baseline** (behavioral-projections Phase 0.5c on Pythia-410m):
  If regime boundaries match across GPT-2 Medium and Pythia-410m, validates
  cross-model generality. Differences may track architecture (pre-norm vs
  post-norm).

## Data Details (verified)

- **300 matrices**: all (i, j) pairs where 0 ≤ i < j ≤ 24. Layer 0 =
  post-embedding, layer 24 = final output (25 layer indices total).
- **Not near-identity**: Frobenius norm of M_{0→1} is ~628 vs ~32 for I.
  Unconstrained OLS fit (no regularization, no identity initialization).
- **Final-token only**: trained on sentence-final token representations from
  Wikipedia. Frame structure may differ for other positions/corpora.
- **Raw residual stream**: final LayerNorm (`ln_f`) excluded during data
  collection. Per-block LayerNorms are baked into the block forward pass.
- **Left-multiplication**: v_target = M @ v_source.
- **Sub-block matrices** exist in the codebase (attention, MLP, etc.) but
  are NOT released on HuggingFace. Available if needed (requires re-running
  training).

## Limitations

- Trained on Wikipedia sentence-final tokens only — frame structure may be
  position-dependent and dataset-dependent.
- Linear maps (no bias) — the bias term in Tuned Lens captures mean shift;
  these matrices fold it into the linear transformation or ignore it.
- GPT-2 Medium uses post-norm (LayerNorm after attention and MLP), unlike
  Pythia which uses pre-norm. This affects what the matrices are mapping between.
- Matrices are far from identity — the "perturbative frame change" narrative
  may not hold. Phase 1c must characterize the actual deviation structure.
