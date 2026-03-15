"""Shared metric functions for translator matrix analysis.

All functions take plain tensors (matrices). No model dependency, no GPU needed.
"""

from __future__ import annotations

import torch


def erank(M: torch.Tensor) -> float:
    """Entropy-based effective rank.

    erank(M) = exp(entropy(σ / sum(σ)))

    Same definition as mlp-context-steering/src/metrics.py (see D3 in
    DECISIONS.md for why it's duplicated rather than promoted to tools/).

    Parameters
    ----------
    M : (m, n) tensor

    Returns
    -------
    float — effective rank, in [1, min(m, n)]
    """
    sigma = torch.linalg.svdvals(M)
    sigma = sigma[sigma > 0]
    p = sigma / sigma.sum()
    entropy = -(p * p.log()).sum()
    return float(entropy.exp().item())


def spectral_entropy(M: torch.Tensor) -> float:
    """Normalized spectral entropy.

    Like erank but normalized to [0, 1] by dividing by log(min(m, n)).
    Value of 1 = perfectly uniform singular values (full rank).
    Value near 0 = dominated by a single singular value.

    Parameters
    ----------
    M : (m, n) tensor
    """
    sigma = torch.linalg.svdvals(M)
    sigma = sigma[sigma > 0]
    p = sigma / sigma.sum()
    entropy = -(p * p.log()).sum()
    max_entropy = torch.tensor(len(sigma), dtype=torch.float).log()
    if max_entropy == 0:
        return 0.0
    return float((entropy / max_entropy).item())


def condition_number(M: torch.Tensor) -> float:
    """Condition number: σ_max / σ_min.

    Parameters
    ----------
    M : (m, n) tensor

    Returns
    -------
    float — condition number. Inf if M is rank-deficient.
    """
    sigma = torch.linalg.svdvals(M)
    if sigma[-1] == 0:
        return float('inf')
    return float((sigma[0] / sigma[-1]).item())


def matrix_summary(M: torch.Tensor) -> dict:
    """Compute all scalar metrics for a single matrix.

    Returns dict with: erank, spectral_entropy, condition_number,
    frobenius_norm, spectral_norm, singular_values (full vector).
    """
    sigma = torch.linalg.svdvals(M)

    # Reuse sigma for erank/spectral_entropy to avoid redundant SVD
    sigma_pos = sigma[sigma > 0]
    p = sigma_pos / sigma_pos.sum()
    ent = -(p * p.log()).sum()
    max_ent = torch.tensor(len(sigma_pos), dtype=torch.float).log()

    return {
        'erank': float(ent.exp().item()),
        'spectral_entropy': float((ent / max_ent).item()) if max_ent > 0 else 0.0,
        'condition_number': float((sigma[0] / sigma[-1]).item()) if sigma[-1] > 0 else float('inf'),
        'frobenius_norm': float(sigma.square().sum().sqrt().item()),
        'spectral_norm': float(sigma[0].item()),
        'singular_values': sigma,
    }
