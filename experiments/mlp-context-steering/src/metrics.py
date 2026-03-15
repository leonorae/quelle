"""Core metrics for the MLP context-steering experiment.

All functions take plain tensors. No model dependency.

Three metrics, each computed per-layer:
  - erank: entropy-based effective rank of the residual stream
  - sink_intensity: fraction of attention mass on position 0 (BOS)
  - max_activation: max absolute activation value across positions/channels
"""

from __future__ import annotations

import torch


def erank(H: torch.Tensor) -> float:
    """Entropy-based effective rank of a matrix.

    erank(H) = exp(entropy(σ / sum(σ)))

    where σ are the singular values of H.

    This is NOT the same as tools/analysis/geometry/effective_dimensionality,
    which uses PCA participation ratio (threshold-dependent). Erank is continuous
    and matches Dong et al.'s definition. See DECISIONS.md D2.

    Parameters
    ----------
    H : (seq_len, d_model) or (batch, seq_len, d_model)
        If 3D, batch and seq dimensions are flattened (treating all positions
        across all batch elements as samples).

    Returns
    -------
    float — effective rank, in [1, min(n_samples, d_model)]
    """
    if H.dim() == 3:
        H = H.reshape(-1, H.shape[-1])

    # SVD — we only need singular values
    sigma = torch.linalg.svdvals(H)

    # Normalize to a distribution
    sigma = sigma[sigma > 0]  # drop exact zeros
    p = sigma / sigma.sum()

    # Shannon entropy → effective rank
    entropy = -(p * p.log()).sum()
    return float(entropy.exp().item())


def sink_intensity(
    attention_pattern: torch.Tensor,
    sink_position: int = 0,
) -> float:
    """Fraction of attention mass directed at the sink position.

    Parameters
    ----------
    attention_pattern : (batch, n_heads, seq_len, seq_len)
        Standard attention weight tensor from one layer.
        attention_pattern[b, h, q, k] = how much query q attends to key k.
    sink_position : int
        Which key position is the sink (default: 0 = BOS).

    Returns
    -------
    float — mean attention mass on sink_position, averaged across all
            batch elements, heads, and query positions (excluding the sink
            position itself as a query, since its self-attention is trivially
            high in causal models).
    """
    B, H, S, _ = attention_pattern.shape

    # Attention mass on sink from all query positions except the sink itself
    # Shape: (B, H, S-1) — one value per non-sink query position
    sink_mass = attention_pattern[:, :, 1:, sink_position]  # skip q=0

    return float(sink_mass.mean().item())


def max_activation(H: torch.Tensor) -> dict:
    """Max absolute activation value across positions and channels.

    Parameters
    ----------
    H : (batch, seq_len, d_model)

    Returns
    -------
    dict with:
        - 'max_value': float — the maximum absolute activation
        - 'channel_idx': int — which channel (d_model dimension) has the max
        - 'position_idx': int — which position has the max
        - 'per_channel_max': Tensor (d_model,) — max abs per channel, for
          identifying persistent spike channels across prompts
    """
    abs_H = H.abs()

    # Global max
    max_val, flat_idx = abs_H.reshape(-1).max(dim=0)

    # Decode flat index → (batch, pos, channel)
    d = H.shape[-1]
    s = H.shape[-2]
    channel_idx = int(flat_idx % d)
    position_idx = int((flat_idx // d) % s)

    # Per-channel max (across batch and positions)
    per_channel_max = abs_H.reshape(-1, d).max(dim=0).values

    return {
        'max_value': float(max_val.item()),
        'channel_idx': channel_idx,
        'position_idx': position_idx,
        'per_channel_max': per_channel_max,
    }


def collect_layer_metrics(
    residual: torch.Tensor,
    attention_pattern: torch.Tensor | None = None,
) -> dict:
    """Compute all three metrics for one layer.

    Parameters
    ----------
    residual : (batch, seq_len, d_model) — residual stream at this layer
    attention_pattern : (batch, n_heads, seq_len, seq_len) or None
        If None, sink_intensity is not computed (e.g., for embedding layer).

    Returns
    -------
    dict with keys 'erank', 'sink_intensity', 'max_activation'
    """
    result = {
        'erank': erank(residual),
        'max_activation': max_activation(residual),
    }

    if attention_pattern is not None:
        result['sink_intensity'] = sink_intensity(attention_pattern)
    else:
        result['sink_intensity'] = None

    return result
