"""Phase 4: Perturbation-sensitivity map.

Learn W such that ‖W·δ‖ predicts |Δoutput| for perturbation δ.

This is a linearized, amortized approximation of the model's Jacobian.
Column space = load-bearing directions. Null space = slack.

Algorithm:
    For each cached activation h at layer l:
        Sample ~100 random unit perturbation directions δ
        Run perturbed activation h + ε·δ through remaining layers
        Measure ΔKL = KL(f(h) || f(h + ε·δ))
        Train W: ‖W·δ‖₂ ≈ ΔKL via least squares

NOTE: This requires LIVE MODEL forward passes (not just cached data).
      Budget GPU time for perturbation passes.

Alternative: Exact Jacobian via torch.autograd.functional.jacobian.
    Then W = J^T J is the exact sensitivity matrix.
    Expensive for large models — stochastic approximation scales better.

Known issue: LayerNorm boundaries introduce ~50% error in linear
    attribution (draft §Known failure modes #3). May need per-segment
    training (between LayerNorms). Address when implementing, not before.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import torch


def sample_perturbations(
    model: torch.nn.Module,
    hidden_state: torch.Tensor,
    layer_idx: int,
    n_dirs: int = 100,
    epsilon: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random unit perturbations and measure output change.

    Args:
        model: the full model (needed for forward passes through remaining layers)
        hidden_state: (d_hidden,) activation at target layer
        layer_idx: which layer this activation is from
        n_dirs: number of random perturbation directions
        epsilon: perturbation magnitude
    Returns:
        perturbation_dirs: (n_dirs, d_hidden) unit vectors
        delta_kl: (n_dirs,) KL divergence caused by each perturbation
    """
    raise NotImplementedError("Phase 4 — requires live model, implement after Phase 1")


def train_sensitivity_map(
    perturbation_dirs: torch.Tensor,
    delta_kl: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Learn sensitivity map W via least squares.

    Minimize: Σ_k (‖W @ δ_k‖₂ - ΔKL_k)²

    W^T W gives the sensitivity matrix (analogous to J^T J).

    Args:
        perturbation_dirs: (M, d_hidden)
        delta_kl: (M,)
    Returns:
        W: (d_proj, d_hidden)
        metrics: {r2, effective_rank, ...}
    """
    raise NotImplementedError("Phase 4 — requires live model, implement after Phase 1")
