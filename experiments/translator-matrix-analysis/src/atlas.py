"""Phase 3: Full atlas analysis.

Uses all available M_{i→j} matrices (not just adjacent pairs) to build
a frame distance matrix, cluster layers into regimes, and identify where
the linear shortcut approximation breaks down.

Usage:
    python -m src.atlas --cache-dir data/matrices --output-dir outputs/atlas

Outputs:
    outputs/atlas/
        frame_distance_matrix.pt   — (n_layers, n_layers) distance matrix
        clustering.json            — hierarchical clustering results
        composition_residuals.json — composition error for sampled triples
        figures/
            frame_distance_heatmap.png
            clustering_dendrogram.png
            composition_residual_heatmap.png
"""

from __future__ import annotations

from pathlib import Path

import torch


def build_frame_distance_matrix(
    matrices: dict[tuple[int, int], torch.Tensor],
    n_layer_indices: int = 25,
) -> torch.Tensor:
    """Phase 3a: Compute frame distance between all available layer pairs.

    D[i, j] = ‖M_{i→j} − I‖_F

    For pairs where M_{i→j} is not directly available, leave as NaN.
    For i == j, D[i, i] = 0.

    Returns (n_layers, n_layers) tensor.
    """
    I = torch.eye(matrices[next(iter(matrices))].shape[0])
    D = torch.full((n_layers, n_layers), float('nan'))

    for i in range(n_layers):
        D[i, i] = 0.0

    for (i, j), M in matrices.items():
        if i < n_layers and j < n_layers:
            D[i, j] = float((M - I).norm('fro').item())

    return D


def hierarchical_clustering(D: torch.Tensor) -> dict:
    """Cluster layers by frame distance.

    Use scipy's hierarchical clustering on the frame distance matrix.
    The cluster structure IS the regime structure — layers within a cluster
    share an approximate interpretive frame.

    Implementation notes for the sonnet:
    ------------------------------------
    Handle NaN entries: for clustering, use only the subset of pairs where
    D[i,j] is available. If the distance matrix is very sparse, fall back to
    adjacent-only distances (which should always be available).

    Use scipy.cluster.hierarchy.linkage with method='ward' or 'average'.
    Use scipy.cluster.hierarchy.dendrogram for visualization.
    Use scipy.cluster.hierarchy.fcluster with a range of thresholds to find
    stable cluster assignments.
    """
    # TODO: Implement
    # from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    # from scipy.spatial.distance import squareform
    #
    # 1. Convert D to condensed distance vector (squareform)
    # 2. Handle NaN: impute or use only available pairs
    # 3. Compute linkage
    # 4. Find stable number of clusters (try 2, 3, 4)
    # 5. Return cluster assignments and linkage matrix
    raise NotImplementedError


def compare_to_cka(
    clustering_result: dict,
) -> dict:
    """Phase 3b: Compare frame-distance clustering to published CKA results.

    Sun et al. (2407.09298, "Transformer Layers as Painters") found three
    layer classes (early, middle, final). Phang et al. (2109.08406) found
    block-diagonal CKA structure.

    This is a qualitative comparison. Report:
    - How many clusters emerge from frame distance?
    - Do the cluster boundaries match the published CKA boundaries?
    - If 3 clusters: are they early/middle/final?

    The CKA results are from the literature, not computed here. The sonnet
    should look up the specific boundary layers from the papers and compare.

    GPT-2 Medium (24 layers) CKA clustering from literature:
    - Early: layers 0-7 (approximate, varies by paper)
    - Middle: layers 8-17
    - Final: layers 18-23
    These numbers are approximate — check the actual papers for GPT-2 Medium.
    """
    # TODO: Implement comparison
    raise NotImplementedError


def composition_residual_analysis(
    matrices: dict[tuple[int, int], torch.Tensor],
) -> dict:
    """Phase 3c: Composition residuals as nonlinearity markers.

    For triples (i, k, j) where all three matrices exist:
        error = ‖M_{i→j} − M_{k→j} @ M_{i→k}‖_F / ‖M_{i→j}‖_F

    High error = the computation between layers i and j is essentially
    nonlinear (not capturable by composing linear shortcuts). These are
    layers where the frame/state decomposition breaks down.

    Returns per-triple errors, organized for heatmap visualization.
    """
    # TODO: Implement (partially done in acquire.validate_composition,
    # but this version should be more systematic — compute ALL valid
    # triples, not just a sample, and organize for visualization).
    raise NotImplementedError


def generate_figures(
    D: torch.Tensor,
    clustering_result: dict,
    composition_result: dict,
    output_dir: Path,
) -> None:
    """Generate Phase 3 figures.

    Required:
    1. Frame distance heatmap — D[i,j] with layer indices on both axes.
       NaN entries shown as white/gray.
    2. Clustering dendrogram — from hierarchical clustering on D.
    3. Composition residual heatmap — for the intermediate layer k,
       show error(i, k, j) as a function of i and j, with k as a
       slider or separate subplot per k.

    Use matplotlib. Save as PNG and PDF.
    """
    # TODO: Implement
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: argparse, load matrices, run analyses, save, plot
    raise NotImplementedError
