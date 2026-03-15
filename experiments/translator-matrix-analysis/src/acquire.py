"""Phase 0: Acquisition and validation of translator matrices.

Downloads pre-trained linear shortcut matrices from HuggingFace and validates
composition consistency.

Usage:
    python -m src.acquire [--cache-dir data/matrices]

Data source: huggingface.co/sashay/linear-shortcut (gpt2-medium/wikipedia/)
Paper: Yom Din et al., arXiv:2303.09435
Code: github.com/sashayd/mat
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch


# GPT-2 Medium: 25 layer indices (0 = post-embedding, 24 = final output)
# 300 matrices available: all (i, j) pairs where 0 <= i < j <= 24
N_LAYER_INDICES = 25  # 0 through 24
D_MODEL = 1024


def download_matrices(cache_dir: Path) -> dict[tuple[int, int], torch.Tensor]:
    """Download all available matrices from HuggingFace.

    Returns dict mapping (i, j) → M_{i→j} tensor of shape (D_MODEL, D_MODEL).

    Implementation notes for the sonnet:
    ------------------------------------
    Use huggingface_hub to download:
        from huggingface_hub import hf_hub_download, list_repo_tree

    1. List all files under gpt2-medium/wikipedia/ in the repo
       "sashay/linear-shortcut"
    2. Download each .pickle file
    3. Load with pickle.load() — yields a tensor or numpy array
    4. Convert to torch.Tensor if needed
    5. Verify shape is (1024, 1024)

    TRAP: pickle.load() can execute arbitrary code. These files are from
    a known academic source on HuggingFace. Still, verify tensor shapes
    after loading and don't load from untrusted sources.

    Naming convention: "{i}_{j}.pickle" where 0 <= i < j <= 24.
    300 files total (upper-triangular only). Layer 0 = post-embedding,
    layer 24 = final output. Left-multiplication: v_target = M @ v_source.

    TRAP: Matrices are NOT near-identity. M_{0→1} has Frobenius norm ~628
    vs ~32 for I. They're unconstrained OLS fits (no regularization).
    See DECISIONS.md D8.

    TRAP: Trained on final-token representations only (sentence-final
    tokens from Wikipedia). See DECISIONS.md D9.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Implement download
    # from huggingface_hub import hf_hub_download, list_repo_tree
    #
    # repo_id = "sashay/linear-shortcut"
    # prefix = "gpt2-medium/wikipedia/"
    #
    # Step 1: List available files
    # Step 2: Download each to cache_dir
    # Step 3: Load and validate
    raise NotImplementedError


def load_matrices(cache_dir: Path) -> dict[tuple[int, int], torch.Tensor]:
    """Load previously downloaded matrices from local cache.

    Returns dict mapping (i, j) → M_{i→j} tensor.
    """
    matrices = {}
    for pkl_path in sorted(cache_dir.glob("*.pickle")):
        # Parse (i, j) from filename
        stem = pkl_path.stem  # e.g., "6_9"
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        try:
            i, j = int(parts[0]), int(parts[1])
        except ValueError:
            continue

        with open(pkl_path, "rb") as f:
            M = pickle.load(f)

        if isinstance(M, torch.Tensor):
            pass
        else:
            M = torch.tensor(M, dtype=torch.float32)

        assert M.shape == (D_MODEL, D_MODEL), f"Unexpected shape {M.shape} for {pkl_path}"
        matrices[(i, j)] = M

    return matrices


def inventory(matrices: dict[tuple[int, int], torch.Tensor]) -> dict:
    """Report what's available.

    Returns dict with:
        'n_total': int — number of matrices available
        'has_adjacent': list[int] — which adjacent pairs (l, l+1) are present
        'has_to_final': list[int] — which layer-to-final pairs are present
        'missing_adjacent': list[int] — which adjacent pairs are missing
        'coverage': float — fraction of 576 possible pairs present
    """
    final_layer = N_LAYER_INDICES - 1  # 24
    adjacent = []
    missing_adjacent = []
    to_final = []

    for l in range(N_LAYER_INDICES - 1):
        if (l, l + 1) in matrices:
            adjacent.append(l)
        else:
            missing_adjacent.append(l)

    for l in range(N_LAYER_INDICES - 1):
        if (l, final_layer) in matrices:
            to_final.append(l)

    # Expected: 300 matrices = C(25, 2) = 25*24/2
    n_possible = N_LAYER_INDICES * (N_LAYER_INDICES - 1) // 2
    return {
        'n_total': len(matrices),
        'n_possible': n_possible,
        'has_adjacent': adjacent,
        'has_to_final': to_final,
        'missing_adjacent': missing_adjacent,
        'coverage': len(matrices) / n_possible,
    }


def validate_composition(
    matrices: dict[tuple[int, int], torch.Tensor],
    n_triples: int = 50,
) -> dict:
    """Check M_{i→j} ≈ M_{k→j} @ M_{i→k} for sampled triples.

    If composition works well, the all-pairs atlas is redundant with adjacent
    pairs and the frame-change story is clean. If composition fails badly,
    the all-pairs structure contains information that adjacent differences miss.

    Parameters
    ----------
    n_triples : int
        Number of random (i, k, j) triples to test.

    Returns
    -------
    dict with:
        'mean_error': float — mean ‖M_{i→j} − M_{k→j} @ M_{i→k}‖_F / ‖M_{i→j}‖_F
        'max_error': float — worst-case relative error
        'errors': list[dict] — per-triple details (i, k, j, absolute_error, relative_error)
        'composition_holds': bool — True if mean relative error < 0.1
    """
    # TODO: Implement
    # Find all valid triples where M_{i→k}, M_{k→j}, and M_{i→j} all exist.
    # Sample n_triples of them (or all, if fewer exist).
    # For each: compute M_{k→j} @ M_{i→k}, compare to M_{i→j}.
    # Report errors.
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: argparse for cache_dir, then:
    #   matrices = download_matrices(cache_dir)
    #   inv = inventory(matrices)
    #   print(json.dumps(inv, indent=2))
    #   comp = validate_composition(matrices)
    #   print(json.dumps({k: v for k, v in comp.items() if k != 'errors'}, indent=2))
    raise NotImplementedError
