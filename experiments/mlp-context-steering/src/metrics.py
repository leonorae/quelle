"""Per-layer metrics for MLP context steering analysis.

All functions are pure tensor operations — no model dependency.
Designed to work on cached activations from TransformerLens.
"""

import torch
from torch import Tensor


def erank(X: Tensor) -> Tensor:
    """Entropy-based effective rank of an activation matrix.

    Given a matrix X of shape (batch, seq_len, d_model) or (seq_len, d_model),
    computes the effective rank as exp(H(p)) where p is the normalised
    singular value distribution and H is Shannon entropy.

    Uses D2 decision: entropy-based erank, NOT PCA participation ratio.

    Args:
        X: Activation matrix. Shape (seq_len, d_model) or (batch, seq_len, d_model).
           If 3D, activations are flattened across batch and seq_len dimensions
           before SVD (i.e., we treat all token positions as samples).

    Returns:
        Scalar tensor: effective rank (float).
    """
    if X.dim() == 3:
        # Flatten batch and seq dims: (B, T, D) -> (B*T, D)
        X = X.reshape(-1, X.shape[-1])
    elif X.dim() != 2:
        raise ValueError(f"Expected 2D or 3D tensor, got {X.dim()}D")

    # SVD — we only need singular values
    s = torch.linalg.svdvals(X)

    # Remove near-zero singular values to avoid log(0)
    s = s[s > 1e-10]

    if s.numel() == 0:
        return torch.tensor(1.0, device=X.device, dtype=X.dtype)

    # Normalise to probability distribution
    p = s / s.sum()

    # Shannon entropy
    H = -(p * torch.log(p)).sum()

    # Effective rank
    return torch.exp(H)


def sink_intensity(attn_patterns: Tensor, sink_token_index: int = 0) -> Tensor:
    """Fraction of total attention mass directed to the sink token.

    Measures how much attention is concentrated on a single token (typically
    BOS/token 0), which indicates attention sink behaviour.

    Args:
        attn_patterns: Attention weights. Shape (batch, n_heads, seq_len, seq_len)
                       or (n_heads, seq_len, seq_len).
        sink_token_index: Which token position is the sink (default 0 = BOS).

    Returns:
        Scalar tensor: mean fraction of attention to sink token, averaged
        over all heads, query positions, and batch elements.
    """
    if attn_patterns.dim() == 3:
        # Add batch dim: (H, T, T) -> (1, H, T, T)
        attn_patterns = attn_patterns.unsqueeze(0)
    elif attn_patterns.dim() != 4:
        raise ValueError(
            f"Expected 3D or 4D attention tensor, got {attn_patterns.dim()}D"
        )

    # attn_patterns[:, :, :, sink_token_index] gives attention TO sink
    # from every query position, for every head and batch element
    sink_attn = attn_patterns[:, :, :, sink_token_index]  # (B, H, T)

    # Mean across all dimensions
    return sink_attn.mean()


def max_activation(X: Tensor) -> Tensor:
    """Maximum absolute activation value across all positions and dimensions.

    A simple health metric: if this spikes after ablation, something is
    numerically unstable.

    Args:
        X: Activation tensor of any shape.

    Returns:
        Scalar tensor: max |x| across entire tensor.
    """
    return X.abs().max()


def collect_layer_metrics(
    resid_post: Tensor,
    attn_patterns: Tensor,
    sink_token_index: int = 0,
) -> dict[str, Tensor]:
    """Collect all metrics for a single layer.

    Convenience function that bundles erank, sink_intensity, and max_activation
    into a single dict.

    Args:
        resid_post: Residual stream after this layer. Shape (batch, seq_len, d_model).
        attn_patterns: Attention patterns for this layer. Shape (batch, n_heads, seq_len, seq_len).
        sink_token_index: Token index for sink intensity computation.

    Returns:
        Dict with keys "erank", "sink_intensity", "max_activation", each
        mapping to a scalar tensor.
    """
    return {
        "erank": erank(resid_post),
        "sink_intensity": sink_intensity(attn_patterns, sink_token_index),
        "max_activation": max_activation(resid_post),
    }
