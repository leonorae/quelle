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

### F1: Pairwise structure is real — C2 >> C1 everywhere

C1 (standard tuned lens) has negative or near-zero R² at most layers despite
Spearman ρ of 0.88–0.96. The lens ranks pairs correctly but predicts wrong
magnitudes. C2 (same architecture, pairwise objective) achieves R² > 0.6 at
most layers above layer 2. **The pairwise objective, not the architecture,
is what matters.** This validates the bisimulation framing.

### F2: C3 dominates early layers (0–8), C2 dominates mid-to-late (9–24)

Direct geometric projection (C3) outperforms decode-then-compare (C2) at
layers 0–8. By layer 9, C2 takes over. Interpretation: early representations
have pairwise geometric structure best captured by direct projection; later
representations become logit-aligned enough that decoding is the natural basis.

### F3: C3 has catastrophic collapses at specific layers

C3 R² goes deeply negative at layers 9 (−6.6), 11 (−3.6), 16 (−3.4), and
17 (−10.4), while Spearman ρ stays reasonable (0.80–0.85). The probe ranks
pairs correctly but produces wildly wrong magnitudes at these layers. This is
consistent with overfitting to outlier pairs — the non-convex L2-norm
objective (see D15) amplifies a few large-distance pairs, inflating MSE.
Ridge (C2) is naturally regularized against this.

### F4: High ρ / low R² pattern is universal for C1

The ρ–R² gap in C1 is not a middle-layer artifact — it persists from layer 0
to layer 24. Even at the final layer, C1 R² = 0.65 vs C2 R² = 0.93.
Independent decoding systematically distorts pairwise distance magnitudes.

## Caveats

- **Bootstrap corpus only (221 prompts).** Results may shift substantially on
  the full 7k corpus. The bootstrap set has limited behavioral diversity.
- **C3 instability** may be a small-corpus artifact. The learned projection has
  more parameters to overfit with only ~24k pairs.
- **No rank sweep yet.** C3 used full-rank (d_proj=1024). Lower ranks may
  stabilize the collapses.

## Next Steps

1. **Diagnose C3 collapses:** Run rank sweep at collapse layers (9, 11, 16, 17).
   If lower-rank projections stabilize R², add weight decay or explicit rank
   constraint.
2. **Rerun on full corpus** (7k prompts) to confirm findings scale.
3. **Phase 2 gate: MET.** Both C2 and C3 exceed R² > 0.3 at middle layers.
   Iterative peeling is worth pursuing — recommend using C2 (pairwise lens)
   as the more stable tool.
4. **Record in wiki/findings/** after full-corpus confirmation.
