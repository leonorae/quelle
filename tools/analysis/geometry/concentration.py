"""Concentration and geometric diagnostics for hidden state analysis.

All functions take plain tensors. No model or framework dependency beyond torch.

Usage as a diagnostic (not a control signal):
    Concentration tells you how aligned token representations are at a given layer.
    High = tokens collapsed into shared representation (commitment/certainty).
    Low = tokens spread across directions (ambiguity/exploration).

    Use alongside probes (tuned lens, projection probes) — concentration measures
    geometry, probes measure output alignment. They answer different questions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def concentration(h: torch.Tensor) -> torch.Tensor:
    """Mean pairwise cosine similarity across token positions.

    Parameters
    ----------
    h : (batch, seq_len, d_model)

    Returns
    -------
    (batch,) — values in [-1, 1]. Higher = more concentrated.
    """
    B, S, _ = h.shape
    if S <= 1:
        return torch.zeros(B, device=h.device)

    h_norm = F.normalize(h, dim=-1)
    sim = torch.bmm(h_norm, h_norm.transpose(1, 2))  # (B, S, S)

    mask = ~torch.eye(S, dtype=torch.bool, device=h.device)
    sim_off = sim.reshape(B, S * S)[:, mask.reshape(S * S)]
    return sim_off.mean(dim=1)


def concentration_per_layer(
    hidden_states: list[torch.Tensor],
) -> list[float]:
    """Compute mean concentration at each layer.

    Parameters
    ----------
    hidden_states : list of (batch, seq_len, d_model) tensors, one per layer

    Returns
    -------
    list of floats, one per layer
    """
    return [float(concentration(h).mean().item()) for h in hidden_states]


def representation_velocity(
    hidden_states: list[torch.Tensor],
) -> list[float]:
    """L2 distance between consecutive layer centroids.

    Parameters
    ----------
    hidden_states : list of (batch, seq_len, d_model) tensors

    Returns
    -------
    list of floats (length = len(hidden_states) - 1)
    """
    centroids = [h.mean(dim=1) for h in hidden_states]  # each (B, D)
    velocities = []
    for i in range(1, len(centroids)):
        delta = (centroids[i] - centroids[i - 1]).norm(dim=-1)  # (B,)
        velocities.append(float(delta.mean().item()))
    return velocities


def effective_dimensionality(h: torch.Tensor, threshold: float = 0.95) -> float:
    """Effective dimensionality via PCA participation ratio.

    Computes how many principal components explain `threshold` fraction
    of the variance. Uses the mean-centered, batch-flattened representation.

    Parameters
    ----------
    h : (batch, seq_len, d_model) or (N, d_model)
    threshold : fraction of variance to explain

    Returns
    -------
    float — effective number of dimensions
    """
    if h.dim() == 3:
        h = h.reshape(-1, h.shape[-1])  # flatten batch and seq

    h_centered = h - h.mean(dim=0, keepdim=True)
    # SVD on centered data
    _, s, _ = torch.linalg.svd(h_centered, full_matrices=False)
    variance = s ** 2
    variance_ratio = variance / variance.sum()
    cumulative = variance_ratio.cumsum(dim=0)
    n_dims = int((cumulative < threshold).sum().item()) + 1
    return float(n_dims)


@dataclass
class GeometricSummary:
    """Per-layer geometric diagnostics."""
    layer: int
    concentration: float
    velocity: float | None  # None for first layer
    effective_dim: float


def geometric_summary(
    hidden_states: list[torch.Tensor],
    dim_threshold: float = 0.95,
) -> list[GeometricSummary]:
    """Compute full geometric summary across layers.

    Parameters
    ----------
    hidden_states : list of (batch, seq_len, d_model) tensors
    dim_threshold : PCA variance threshold for effective dimensionality

    Returns
    -------
    list of GeometricSummary, one per layer
    """
    concs = concentration_per_layer(hidden_states)
    vels = representation_velocity(hidden_states)
    dims = [effective_dimensionality(h, dim_threshold) for h in hidden_states]

    summaries = []
    for i in range(len(hidden_states)):
        summaries.append(GeometricSummary(
            layer=i,
            concentration=concs[i],
            velocity=vels[i - 1] if i > 0 else None,
            effective_dim=dims[i],
        ))
    return summaries
