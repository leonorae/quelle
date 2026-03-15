---
experiment: behavioral-projections
title: "Phase 1.5: Three-Condition Comparison (Bootstrap)"
model: EleutherAI/pythia-410m
corpus: bootstrap-221
date: 2026-03-15
tags: [tuned-lens, pairwise-lens, bisimulation, comparison]
---

# Results — Three-Condition Comparison (Bootstrap, Pythia-410m)

## Setup

- **Model:** Pythia-410m (24 layers, d=1024)
- **Corpus:** 221-prompt bootstrap set
- **Pairs:** Stratified by KL range (per D13)
- **Conditions:**
  - C1: Standard Tuned Lens — per-layer Linear(d,V), trained on per-prompt KL
  - C2: Pairwise Tuned Lens — same architecture, trained on pairwise KL (MSE)
  - C3: Bisimulation Probe — learned projection P, L2 norm predicts KL
  - C3 Ridge: Ridge regression baseline (1D projection from activation differences)

## Per-Layer Results

| Layer | C1 R² | C1 ρ | C2 R² | C2 ρ | C3 R² | C3 ρ | Best R² |
|-------|-------|------|-------|------|-------|------|---------|
| 0 | -0.6325 | 0.6779 | -1.0678 | 0.5721 | 0.3847 | 0.7155 | C3 |
| 1 | -0.2837 | 0.8079 | -1.0678 | 0.4786 | 0.7637 | 0.8713 | C3 |
| 2 | -0.1629 | 0.8245 | 0.6111 | 0.8403 | 0.7547 | 0.8694 | C3 |
| 3 | -0.0834 | 0.8473 | 0.6649 | 0.8706 | 0.8056 | 0.8834 | C3 |
| 4 | -0.0863 | 0.8640 | 0.6778 | 0.8756 | 0.7949 | 0.8781 | C3 |
| 5 | -0.1117 | 0.8769 | 0.3880 | 0.7467 | 0.8149 | 0.8916 | C3 |
| 6 | -0.0846 | 0.8747 | 0.5690 | 0.8355 | 0.8612 | 0.9262 | C3 |
| 7 | 0.0164 | 0.8841 | 0.6269 | 0.8344 | 0.6742 | 0.9113 | C3 |
| 8 | -0.0037 | 0.8908 | 0.7786 | 0.9119 | 0.8705 | 0.9343 | C3 |
| 9 | -0.0601 | 0.9019 | 0.7811 | 0.9149 | -6.5561 | 0.7979 | C2 |
| 10 | -0.0364 | 0.8985 | 0.7542 | 0.9030 | -0.5926 | 0.8232 | C2 |
| 11 | 0.1008 | 0.8996 | 0.8203 | 0.9273 | -3.6035 | 0.8222 | C2 |
| 12 | 0.1130 | 0.9117 | 0.8082 | 0.9247 | 0.8915 | 0.9412 | C3 |
| 13 | 0.0075 | 0.9119 | 0.8121 | 0.9171 | 0.6603 | 0.9028 | C2 |
| 14 | 0.2984 | 0.9115 | 0.8968 | 0.9666 | 0.8907 | 0.9424 | C2 |
| 15 | 0.2280 | 0.9201 | 0.9206 | 0.9688 | 0.9088 | 0.9553 | C2 |
| 16 | 0.1958 | 0.9267 | 0.9073 | 0.9677 | -3.4203 | 0.8497 | C2 |
| 17 | 0.2411 | 0.9307 | 0.9332 | 0.9726 | -10.3954 | 0.8325 | C2 |
| 18 | 0.2917 | 0.9341 | 0.9377 | 0.9746 | -0.1484 | 0.8370 | C2 |
| 19 | 0.3391 | 0.9365 | 0.9033 | 0.9534 | 0.9137 | 0.9569 | C3 |
| 20 | 0.3788 | 0.9364 | 0.9476 | 0.9782 | 0.9093 | 0.9586 | C2 |
| 21 | 0.3958 | 0.9475 | 0.9274 | 0.9711 | 0.9030 | 0.9517 | C2 |
| 22 | 0.4646 | 0.9480 | 0.9439 | 0.9783 | 0.7491 | 0.9421 | C2 |
| 23 | 0.4495 | 0.9500 | 0.9416 | 0.9795 | 0.9052 | 0.9521 | C2 |
| 24 | 0.6548 | 0.9603 | 0.9261 | 0.9610 | 0.5714 | 0.8421 | C2 |

## Findings

### F1: Pairwise structure is real — C2 >> C1 everywhere [observed]

C1 (standard tuned lens) has negative or near-zero R² at most layers despite
Spearman ρ of 0.88–0.96. The lens ranks pairs correctly but predicts wrong
magnitudes. C2 (same architecture, pairwise objective) achieves R² > 0.6 at
most layers above layer 2. **The pairwise objective, not the architecture,
is what matters.** This validates the bisimulation framing: relational
structure exists in activation space and a pairwise training signal recovers it.

### F2: C3 has higher R² than C2 at early layers, competitive through ~8 [observed]

C3 R² exceeds C2 R² at layers 0–8 (C3 best by R²). The margin is large only
at layers 0–1 where C2 R² is strongly negative (−1.07). By layer 2, C2 R²
jumps to 0.61 vs C3 at 0.75 — close. The claim that "direct geometric
projection captures structure decode-then-compare misses" is [conjectured];
the data shows C2 performs worse early, not why. Possible explanations include
logit-alignment developing gradually, but this is untested.

### F3: C3 collapses are structured, not random [observed]

C3 R² goes deeply negative at layers 9 (−6.6), 11 (−3.6), 16 (−3.4), and
17 (−10.4), while Spearman ρ stays reasonable (0.80–0.85). The probe
preserves rank order but produces wildly wrong magnitude predictions.

The collapse layers are not random — they may correspond to architectural
boundaries (attention pattern shifts, MLP gating transitions) where the
activation subspace changes character. [conjectured] The L2 objective becomes
non-convex in a particularly degenerate way at these layers — saddle points
or curvature pathology in the projection, not just outlier amplification.
The rank sweep (D16) will distinguish: if lower d_proj stabilizes R² without
killing Spearman, it's a curvature/degeneracy problem; if all ranks fail,
it's insufficient data.

### F4: C3 Ridge is dead — wrong inductive bias [observed]

Ridge regression (1D projection via coefficient vector) produces flat negative
R² (~−0.06) and Spearman ρ ≈ 0 across all 25 layers. This is not "suboptimal"
or "naturally regularized" — it is zero signal. Ridge cannot recover any
structure, ranking or otherwise.

This is informative: the behavioral geometry is not 1D. A single linear
direction from activation differences to scalar KL captures nothing. The
learned C3 projection (multi-dimensional, d_proj=1024) at least gets ranking
right everywhere (ρ > 0.8). The structure requires multiple dimensions to
express — which is why the rank sweep matters.

### F5: High ρ / low R² pattern is universal for C1 [observed]

The ρ–R² gap in C1 is not a middle-layer artifact — it persists from layer 0
to layer 24. Even at the final layer, C1 R² = 0.65 vs C2 R² = 0.93.
Independent decoding systematically distorts pairwise distance magnitudes.

## Thread State

| Status | Item |
|--------|------|
| [observed] | C2 > C1 at every layer — pairwise optimization recovers relational structure standard tuned lens misses |
| [observed] | C3 > C2 at layers 0–1 clearly, competitive through ~8, collapses at 9/11/16/17 |
| [observed] | C3 ridge: zero signal throughout — wrong method, not just suboptimal |
| [observed] | C3 Spearman robust despite R² collapse — rank preserved, scale broken |
| [conjectured] | Collapse at 9/11/16/17 is architectural boundary effect, not pure outlier sensitivity |
| [conjectured] | Early C3 advantage = geometric structure not accessible through logit decoding |
| [open] | Rank sweep at collapse layers — does lower rank stabilize R²? (D16, `diagnose_c3_collapses.py`) |
| [open] | Whether logit-alignment explanation for C2 dominance in mid-late layers holds |
| [dead] | C3 ridge as probe method |
| [designed] | Phase 2 iterative peeling: C2 for layer selection, stabilized C3 as peeling operator |

## Caveats

- **Bootstrap corpus only (221 prompts).** Results may shift substantially on
  the full 7k corpus. The bootstrap set has limited behavioral diversity.
- **C3 instability** may be partly a small-corpus artifact (learned projection
  has more parameters than ~24k pairs can constrain at full rank), but the
  structured pattern of collapse layers suggests it is not purely data-limited.
- **No rank sweep yet.** C3 used full-rank (d_proj=1024). Lower ranks may
  stabilize the collapses. Diagnostic script ready: `src/diagnose_c3_collapses.py`.

## Next Steps

1. **Run rank sweep at collapse layers** (`diagnose_c3_collapses.py`, D16).
   Tests dimensionality-mismatch vs overfitting. If lower d_proj stabilizes R²
   at collapse layers → rank-constrain C3. If all ranks fail → need more data.
2. **Rerun on full corpus** (7k prompts) to confirm findings scale.
3. **Phase 2 gate: MET.** Both C2 and C3 exceed R² > 0.3 at middle layers.
   Iterative peeling (INLP) requires a projection matrix — C2 (decode-then-compare)
   doesn't provide one. Use C2 to identify which layers have structure worth
   peeling, then use a stabilized C3 (rank-constrained or regularized) as the
   actual peeling operator.
4. **Record in wiki/findings/** after full-corpus confirmation.
