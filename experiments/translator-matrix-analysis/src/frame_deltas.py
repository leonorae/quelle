"""Phase 2: Adjacent differences as frame change.

This is the CORE NOVEL ANALYSIS. Everything else supports this.

For each layer transition l → l+1:
    Delta_l = M_{l→l+1} - I
    SVD(Delta_l) → principal frame-change directions and magnitudes

Usage:
    python -m src.frame_deltas --cache-dir data/matrices --output-dir outputs/frame_deltas

Outputs:
    outputs/frame_deltas/
        magnitude_profile.json   — ‖Delta_l‖_F and ‖Delta_l‖_2 per layer
        dimensionality_profile.json — erank(Delta_l) per layer
        principal_directions.pt  — top-k singular vectors per layer
        regime_boundaries.json   — identified boundary layers with evidence
        figures/
            magnitude_profile.png
            dimensionality_profile.png
            direction_similarity.png  — cosine sim between adjacent layers' principal directions
"""

from __future__ import annotations

from pathlib import Path

import torch

from .metrics import matrix_summary


def compute_frame_deltas(
    matrices: dict[tuple[int, int], torch.Tensor],
    n_layer_indices: int = 25,
) -> dict[int, dict]:
    """Compute Delta_l = M_{l→l+1} - I and its SVD for each layer.

    Returns dict mapping layer l → {
        'delta': Tensor — the Delta matrix itself
        'U': Tensor — left singular vectors (directions in l+1 space most affected)
        'S': Tensor — singular values (magnitudes of frame change)
        'Vh': Tensor — right singular vectors (directions in l space that drive change)
        'summary': dict — scalar metrics from matrix_summary(Delta)
    }
    """
    I = torch.eye(matrices[next(iter(matrices))].shape[0])
    deltas = {}

    for l in range(n_layer_indices - 1):
        key = (l, l + 1)
        if key not in matrices:
            continue

        Delta = matrices[key] - I
        U, S, Vh = torch.linalg.svd(Delta, full_matrices=False)

        deltas[l] = {
            'delta': Delta,
            'U': U,
            'S': S,
            'Vh': Vh,
            'summary': matrix_summary(Delta),
        }

    return deltas


def magnitude_profile(deltas: dict[int, dict]) -> dict:
    """Phase 2b: Frame change magnitude across layers.

    Returns per-layer ‖Delta_l‖_F and ‖Delta_l‖_2.
    Peaks indicate regime boundary candidates.
    """
    layers = sorted(deltas.keys())
    return {
        'layers': layers,
        'frobenius': [deltas[l]['summary']['frobenius_norm'] for l in layers],
        'spectral': [deltas[l]['summary']['spectral_norm'] for l in layers],
    }


def dimensionality_profile(deltas: dict[int, dict]) -> dict:
    """Phase 2c: Frame change dimensionality across layers.

    erank(Delta_l) across layers.
    - Low erank: frame change is low-rank (few directions rotate/scale)
    - High erank: frame change is distributed (general coordinate transform)

    The triangle experiment predicts MLP-heavy layers should show higher-rank
    frame changes. This is a structural prediction testable without forward
    passes.
    """
    layers = sorted(deltas.keys())
    return {
        'layers': layers,
        'erank': [deltas[l]['summary']['erank'] for l in layers],
        'spectral_entropy': [deltas[l]['summary']['spectral_entropy'] for l in layers],
    }


def principal_direction_consistency(
    deltas: dict[int, dict],
    top_k: int = 10,
) -> dict:
    """Phase 2d: Track principal frame-change directions across layers.

    For each layer transition, extract the top-k right singular vectors of
    Delta_l (the directions in residual stream space that rotate most).

    Compute cosine similarity between adjacent layers' top-k subspaces.
    - High similarity → persistent rotational drift (smooth evolution)
    - Low similarity → abrupt direction change (discrete regime transition)

    Returns per-layer-pair subspace similarity scores.
    """
    layers = sorted(deltas.keys())
    similarities = []

    for idx in range(len(layers) - 1):
        l1, l2 = layers[idx], layers[idx + 1]

        # Top-k right singular vectors (rows of Vh)
        Vh1 = deltas[l1]['Vh'][:top_k]  # (k, d_model)
        Vh2 = deltas[l2]['Vh'][:top_k]

        # Subspace similarity: mean of max cosine similarities
        # For each direction in Vh1, find the most aligned direction in Vh2
        cos_sim = torch.mm(Vh1, Vh2.T).abs()  # (k, k)
        # Grassmann-like similarity: average of column maxima
        sim = float(cos_sim.max(dim=1).values.mean().item())

        similarities.append({
            'layers': (l1, l2),
            'subspace_similarity': sim,
        })

    return {
        'top_k': top_k,
        'similarities': similarities,
    }


def detect_regime_boundaries(
    deltas: dict[int, dict],
    direction_consistency: dict,
) -> list[dict]:
    """Phase 2 synthesis: Identify regime boundaries.

    A regime boundary is a layer where:
    1. Frame change magnitude peaks or changes slope (Phase 2b)
    2. Frame change dimensionality shifts (Phase 2c)
    3. Principal directions change abruptly (Phase 2d)

    Returns list of boundary candidates with evidence scores.

    See DECISIONS.md D6: use derivative peaks, not thresholds.
    """
    # TODO: Implement
    # Compute d/dl of magnitude profile → find local maxima
    # Compute d/dl of dimensionality profile → find shift points
    # Find dips in direction consistency → abrupt transitions
    # Combine evidence: layers that appear in multiple criteria are strong candidates
    raise NotImplementedError


def generate_figures(
    deltas: dict[int, dict],
    mag_profile: dict,
    dim_profile: dict,
    dir_consistency: dict,
    boundaries: list[dict],
    output_dir: Path,
) -> None:
    """Generate Phase 2 figures.

    Required:
    1. Frame change magnitude profile: ‖Delta_l‖_F and ‖Delta_l‖_2 vs layer
       with regime boundaries marked
    2. Frame change dimensionality: erank(Delta_l) vs layer
    3. Direction consistency: subspace similarity between adjacent layers
    4. Combined regime boundary evidence plot

    Use matplotlib. Save as PNG and PDF.
    """
    # TODO: Implement
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: argparse, load matrices, compute everything, save, plot
    raise NotImplementedError
