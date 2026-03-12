"""Phase 2: Iterative residual peeling.

Characterize whether "dark matter" (what bisimulation can't see) is
high-rank linear or genuinely nonlinear.

Algorithm:
    1. Train P_1 on activations → KL divergence (Phase 1 output)
    2. Orthonormalize P_1 via QR decomposition
    3. Compute residual: r_i = h_i - Q Q^T h_i  (project out P_1's column space)
    4. Train P_2 on residuals → KL divergence
    5. Repeat, plot cumulative KL variance explained

Key output: saturation curve. Fast saturation = dark matter is linear.
Slow saturation = genuinely nonlinear structure.

IMPORTANT: P must be orthonormalized (QR) before computing residuals.
Ridge coefficients are NOT orthonormal. See DECISIONS.md D4 precision note.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
import numpy as np


def orthonormalize_projection(P: np.ndarray) -> np.ndarray:
    """QR-orthonormalize projection rows.

    Args:
        P: (d_proj, d_hidden) or (d_hidden,) projection matrix/vector.
    Returns:
        Q: orthonormalized projection with same row space as P.
    """
    raise NotImplementedError("Phase 2 — implement after Phase 1 results")


def compute_residual(
    activations: torch.Tensor,
    Q: np.ndarray,
) -> torch.Tensor:
    """Project out Q's column space from activations.

    r_i = h_i - Q^T Q h_i

    Args:
        activations: (N, d_hidden) float
        Q: (d_proj, d_hidden) orthonormalized projection
    Returns:
        residuals: (N, d_hidden) float
    """
    raise NotImplementedError("Phase 2 — implement after Phase 1 results")


def run_peeling(
    cache_dir: Path,
    layer: int,
    max_iterations: int = 10,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Iterative peeling loop.

    Returns:
        List of dicts per iteration: {
            projection: np.ndarray,
            r2: float,
            cumulative_variance_explained: float,
        }
    """
    raise NotImplementedError("Phase 2 — implement after Phase 1 results")
