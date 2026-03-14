---
title: Results — Variable-Bitrate Reasoning
slug: variable-bitrate-reasoning
status: completed
completed: 2026-03-14
---

# Results

## Soft-Mask Dimension Ordering Experiment

**Hypothesis:** VBR's soft dimension mask (sigmoid gate over dimension indices)
causes the model to sort information by importance across dimensions, so that
early dimensions carry more task-relevant signal than later ones.

**Setup:** 2000 training examples, 3 epochs, d_model=128, 4 layers, CPU.
Analysis on 500 test examples.

## Summary

**The hypothesis is not supported.** After 3 epochs of training, the model shows
no significant dimension ordering effect. Information (as measured by per-dimension
correlation with answer value and difficulty) is distributed roughly uniformly
across all 128 dimensions, with no systematic front-loading of important dimensions.

The compression parameters (alpha, beta) barely moved from their initial values,
and lambda does not differentiate across difficulty levels. The soft mask mechanism
has not yet induced the expected information-sorting pressure.

## Dimension Ordering Analysis

### Per-dimension correlation with answer value

Correlation (absolute) between each dimension's activation and the answer value,
averaged over 8 bins of 16 dimensions each:

| Layer | Bin 1 (d0-15) | Bin 2 | Bin 3 | Bin 4 | Bin 5 | Bin 6 | Bin 7 | Bin 8 (d112-127) |
|-------|---------------|-------|-------|-------|-------|-------|-------|------------------|
| 0     | 0.060 | 0.056 | 0.036 | 0.069 | 0.054 | 0.060 | 0.051 | 0.058 |
| 1     | 0.068 | 0.079 | 0.053 | 0.069 | 0.063 | 0.058 | 0.056 | 0.066 |
| 2     | 0.063 | 0.067 | 0.050 | 0.077 | 0.062 | 0.056 | 0.066 | 0.071 |
| 3     | 0.068 | 0.068 | 0.056 | 0.082 | 0.075 | 0.067 | 0.074 | 0.061 |

All bins show similar correlation magnitudes (~0.05-0.08). No front-loading pattern.

### First-k vs last-k informativeness (answer correlation)

| Layer | k=32 first | k=32 last | Ratio |
|-------|-----------|----------|-------|
| 0     | 0.058 | 0.054 | 1.07 |
| 1     | 0.073 | 0.061 | 1.20 |
| 2     | 0.065 | 0.068 | 0.95 |
| 3     | 0.068 | 0.067 | 1.02 |

Ratios hover near 1.0 across all layers. Layer 1 shows a slight trend (1.20x)
but this is not statistically significant.

### Spearman rank correlation (dim index vs informativeness)

| Layer | rho | p-value | Significant? |
|-------|------|---------|-------------|
| 0 | -0.014 | 0.878 | No |
| 1 | -0.095 | 0.287 | No |
| 2 | +0.037 | 0.678 | No |
| 3 | +0.013 | 0.889 | No |

No layer shows a statistically significant monotonic relationship between
dimension index and informativeness.

## Compression Policy (Learned alpha and beta)

| Layer | alpha (init=2.0) | beta (init=0.5) |
|-------|-----------------|----------------|
| 0     | 1.997 | 0.514 |
| 1     | 1.989 | 0.514 |
| 2     | 1.988 | 0.513 |
| 3     | 1.986 | 0.515 |

Alpha and beta have barely moved from initialisation. The compression policy
has not been significantly shaped by training. This suggests 3 epochs on 2000
examples is insufficient for the compression head to learn a meaningful policy.

## Lambda by Difficulty Level

| Difficulty | Mean lambda | Std |
|-----------|------------|-----|
| Easy      | 0.626 | 0.072 |
| Medium    | 0.618 | 0.062 |
| Hard      | 0.623 | 0.055 |

Lambda is nearly identical across difficulty levels (~0.62), with overlapping
standard deviations. The model does **not** compress differently for different
difficulties at this training stage.

## Interpretation

1. **Insufficient training:** 3 epochs on 2000 examples is likely far too little
   for the ordering effect to emerge. The compression parameters barely moved
   from init, suggesting the gradient signal through the soft mask has not had
   time to reshape the representation geometry.

2. **Weak gradient signal:** The soft mask's gradient with respect to the hidden
   dimensions may be too diffuse. The sigmoid cutoff creates a smooth gate, but
   the model needs sustained training pressure to learn that early dimensions
   should carry critical information.

3. **Lambda plateau:** With mean lambda around 0.62, the model is keeping roughly
   the first 49 dimensions at full weight and fading the rest. But this cutoff
   is driven by concentration (geometric alignment), not by learned importance
   ordering.

## Next Steps / Extensions

- **Longer training:** Run with full 10000 examples and 10+ epochs to test
  whether ordering emerges with more gradient steps.
- **Stronger compression pressure:** Increase the future-prediction loss weight
  to force the model to pack information into the kept dimensions.
- **Direct dimension penalty:** Add an L1 penalty on activations in high-index
  dimensions to provide explicit pressure for ordering.
- **Alternative proxy:** Use mutual information estimation (e.g., MINE) instead
  of linear correlation, which may miss nonlinear relationships.
- **Ablation:** Train without the soft mask (no_compression baseline) and compare
  dimension correlation profiles to isolate the mask's causal effect.
