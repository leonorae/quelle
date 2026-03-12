"""Cross-projection intersection analysis.

Compute the intersection of null spaces from multiple trained projections.
This intersection isolates the model's endogenous scaffolding — directions
that carry no content, no output sensitivity, no perturbation response.

Requires: ≥2 trained projection matrices from Phases 1-4.

Key analysis:
    - Dimensionality of null space intersection
    - What lives in the intersection (characterize via nearest SAE features)
    - Ablation: project out intersection directions, measure performance impact
    - Compare intersection across layers
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np


def compute_null_space(P: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Compute null space of projection matrix via SVD.

    Args:
        P: (d_proj, d_hidden)
        threshold: singular value cutoff for null space
    Returns:
        null_basis: (d_null, d_hidden) orthonormal basis for null space
    """
    raise NotImplementedError("Intersection analysis — implement after ≥2 projections trained")


def intersect_null_spaces(
    projections: list[np.ndarray],
    threshold: float = 1e-6,
) -> np.ndarray:
    """Compute intersection of multiple null spaces.

    Stack projections vertically, compute null space of the stack.

    Args:
        projections: list of (d_proj_i, d_hidden) projection matrices
    Returns:
        intersection_basis: (d_int, d_hidden) orthonormal basis
    """
    raise NotImplementedError("Intersection analysis — implement after ≥2 projections trained")


def analyze_intersections(
    projection_dir: Path,
    layers: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Full intersection analysis across layers.

    Returns:
        {layer: {intersection_dim, fraction_of_total, ...}, ...}
    """
    raise NotImplementedError("Intersection analysis — implement after ≥2 projections trained")
