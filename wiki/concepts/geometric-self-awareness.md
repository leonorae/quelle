# Geometric Self-Awareness

> Background concept for understanding what the angle concentration metric
> measures and why it might proxy for representational certainty.

**Relevant experiments**: `variable-bitrate-reasoning`, `geometric-self-awareness-reasoning`

---

## Core Idea

When a transformer processes tokens, its hidden states live in a high-dimensional
space ℝᵈ. At any layer, we can look at the *geometry* of the set of token
representations:

- **Concentrated** (low diversity): token vectors point in similar directions.
  This suggests the model has collapsed distinct tokens into a shared
  representation — often a sign that those tokens are playing similar
  functional roles, or that the model has "committed" to an interpretation.

- **Diffuse** (high diversity): token vectors are spread across many directions.
  This suggests the model is still tracking multiple distinct features or has
  not yet resolved ambiguity.

## Angle Concentration Metric

```python
def concentration(h):
    # h: (batch, seq_len, d_model)
    h_norm = F.normalize(h, dim=-1)
    sim = torch.bmm(h_norm, h_norm.transpose(1, 2))  # pairwise cosine sim
    mask = ~torch.eye(sim.shape[1], dtype=bool, device=sim.device)
    return sim[:, mask].mean()  # mean off-diagonal cosine similarity
```

Range: [-1, 1]. Higher = more concentrated (vectors more aligned).

## Relationship to Uncertainty

The hypothesis (to be tested in `variable-bitrate-reasoning`) is:

> **High concentration → high certainty → more compression is safe.**
> **Low concentration → high uncertainty → preserve more information.**

This is analogous to entropy in information theory: when the model's
representations are geometrically uniform (low entropy in direction space),
less bandwidth is needed to transmit the state forward in time.

## Trajectory Geometry (used in `geometric-self-awareness-reasoning`)

Beyond per-step concentration, we measure how the *centroid* of a step's token
representations moves through hidden-state space across reasoning steps:

| Feature | Definition |
|---------|-----------|
| **Velocity** | `‖ c_t - c_{t-1} ‖₂` — how far the centroid moved between steps |
| **Curvature** | Menger curvature of three consecutive centroids `c_{t-1}, c_t, c_{t+1}` — detects sharp direction changes |
| **Manifold dim** | Intrinsic dimensionality of token vectors within a step, estimated via PCA or MLE — proxy for effective degrees of freedom |
| **Centroid distance** | Distance from `c_t` to the centroid of the "correct-answer" manifold vs. the "incorrect-answer" manifold |

The hypothesis (Geometry of Reasoning, arXiv:2510.09782) is that high-quality
reasoning steps exhibit smooth, low-curvature trajectories in a
low-dimensional manifold, while errors produce abrupt curvature spikes or
high-velocity jumps.

## References

- Cosine similarity as a proxy for semantic overlap: standard NLP practice.
- Von Mises–Fisher distribution on the hypersphere: formal treatment of
  directional statistics.
- Related to "representation collapse" concerns in contrastive learning (SimCLR,
  BYOL), but here collapse is seen as a *signal* rather than a failure mode.
- Geometry of Reasoning (arXiv:2510.09782) — defines velocity, curvature, and
  flow similarity for reasoning trajectories.
- Confidence Manifold (arXiv:2602.08159) — correctness lives in a
  low-dimensional subspace; centroid distance is a zero-shot predictor.
