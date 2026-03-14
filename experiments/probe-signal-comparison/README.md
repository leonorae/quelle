---
status: planned
owner: null
dependencies: [VVVVVV]
---

# Probe Signal Comparison

**Hypothesis**: Geometric signals (concentration, velocity, effective dimensionality)
carry information about layer importance that is complementary to — not redundant
with — probe-based signals (tuned lens entropy, layer-skip delta).

## Background

Geometric concentration was originally proposed as a control signal for adaptive
compression (variable-bitrate-reasoning, now archived). Tuned lens and projection
probes are the established tools for understanding what a model "knows" at each
layer. This experiment tests whether geometric signals add anything.

## Method

Use a frozen nanochat d12 checkpoint (from VVVVVV Phase 0). For each layer, compute:

| Signal | What it measures | Cost |
|---|---|---|
| Concentration | Mean pairwise cosine sim across tokens | O(S²d) |
| Representation velocity | ‖centroid_L - centroid_{L-1}‖ | O(d) |
| Effective dimensionality | PCA participation ratio | O(d²) |
| Tuned lens entropy | Uncertainty of decoded logits | O(dV) + trained probe |
| Layer-skip delta | Change in prediction when layer is ablated | O(dV) |

Evaluate each against:
1. Whether the layer changes the final prediction (binary)
2. Loss contribution via leave-one-out ablation
3. Correlation with input difficulty (if measurable)

## Success criteria

- At least one geometric signal has Pearson r > 0.3 with layer importance
  (as measured by ablation loss delta) after controlling for tuned lens entropy.
- If no geometric signal adds information beyond probes, that's a clean null
  result that closes the "geometry as control signal" line of inquiry.

## Dependencies

- Trained nanochat d12 checkpoint (from VVVVVV experiment)
- `tools/analysis/geometry/` for geometric metrics
- Tuned lens library or custom implementation
