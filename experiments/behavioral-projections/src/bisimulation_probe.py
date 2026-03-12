"""Phase 1: Bisimulation probe.

Learn a linear projection P where ‖P(h₁) - P(h₂)‖₂ predicts the symmetric
KL divergence between the model's output distributions at h₁ and h₂.

Math:
    Given hidden states {h_i} at layer l with output distributions {p_i}:
    Target: y_ij = (KL(p_i||p_j) + KL(p_j||p_i)) / 2   (symmetric KL, D3)
    Ridge: w = argmin ||X_diff @ w - y||² + α||w||²
           where X_diff[k] = h_i - h_j for pair k
           w ∈ R^d_hidden defines a 1D projection (scalar prediction)
    Learned: P ∈ R^{d_proj × d_hidden}, loss = MSE(‖P(h_i) - P(h_j)‖₂, y_ij)

See DECISIONS.md D3 (symmetric KL), D4 (Ridge first), D5 (top-k KL approx).

Usage:
    python src/bisimulation_probe.py --config configs/default.yaml --cache data/activations/pythia-410m/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from safetensors.torch import load_file
from sklearn.linear_model import Ridge
from tqdm import tqdm


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# --- KL computation from cached top-k logprobs ---

def compute_pairwise_kl(
    logprobs_i: torch.Tensor,     # (top_k,) float32
    indices_i: torch.Tensor,      # (top_k,) int64
    residual_i: float,            # log(1 - sum(top_k_probs_i))
    logprobs_j: torch.Tensor,
    indices_j: torch.Tensor,
    residual_j: float,
    vocab_size: int = 50304,      # Pythia default; override for Qwen
) -> float:
    """Compute symmetric KL from stored top-k logprobs.

    For tokens in both top-k sets: exact logprob difference.
    For tokens in one set but not the other: approximate the missing logprob
    as log(residual_mass / (vocab_size - top_k)).  (See DECISIONS.md D5.)

    Returns: (KL(p_i||p_j) + KL(p_j||p_i)) / 2
    """
    top_k = len(logprobs_i)

    # Build lookup: token_idx -> logprob for each distribution
    lp_i = dict(zip(indices_i.tolist(), logprobs_i.tolist()))
    lp_j = dict(zip(indices_j.tolist(), logprobs_j.tolist()))

    # Uniform approximation for tail tokens
    tail_count = vocab_size - top_k
    tail_lp_i = residual_i - np.log(max(tail_count, 1))
    tail_lp_j = residual_j - np.log(max(tail_count, 1))

    # KL(p_i || p_j) = sum_x p_i(x) * (log p_i(x) - log p_j(x))
    all_tokens = set(lp_i.keys()) | set(lp_j.keys())

    kl_ij = 0.0
    kl_ji = 0.0
    for tok in all_tokens:
        li = lp_i.get(tok, tail_lp_i)
        lj = lp_j.get(tok, tail_lp_j)
        pi = np.exp(li)
        pj = np.exp(lj)
        kl_ij += pi * (li - lj)
        kl_ji += pj * (lj - li)

    return (kl_ij + kl_ji) / 2.0


def compute_pairwise_kl_batched(
    logprobs: torch.Tensor,       # (N, top_k) float32
    indices: torch.Tensor,        # (N, top_k) int64
    residuals: torch.Tensor,      # (N,) float32
    pairs: torch.Tensor,          # (M, 2) int64 — pair indices
    vocab_size: int = 50304,
) -> torch.Tensor:
    """Compute symmetric KL for a batch of pairs. Returns (M,) float32."""
    kl_values = []
    for idx in range(len(pairs)):
        i, j = pairs[idx, 0].item(), pairs[idx, 1].item()
        kl = compute_pairwise_kl(
            logprobs[i], indices[i], residuals[i].item(),
            logprobs[j], indices[j], residuals[j].item(),
            vocab_size=vocab_size,
        )
        kl_values.append(kl)
    return torch.tensor(kl_values, dtype=torch.float32)


# --- Data loading ---

def load_cached_layer(cache_dir: Path, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load all cached activations for one layer.

    Returns:
        hidden_states: (N, d_hidden) float16
        logprobs: (N, top_k) float32
        logits_indices: (N, top_k) int64
        logits_residual: (N,) float32
    """
    batch_files = sorted(cache_dir.glob("batch_[0-9]*.safetensors"))
    # Exclude full-sequence files
    batch_files = [f for f in batch_files if "_full" not in f.name]

    hidden_states = []
    logprobs_list = []
    indices_list = []
    residual_list = []

    for bf in batch_files:
        data = load_file(bf)
        hidden_states.append(data[f"layer_{layer}"])
        logprobs_list.append(data["logits_values"])
        indices_list.append(data["logits_indices"])
        residual_list.append(data["logits_residual"])

    return (
        torch.cat(hidden_states, dim=0),
        torch.cat(logprobs_list, dim=0),
        torch.cat(indices_list, dim=0),
        torch.cat(residual_list, dim=0),
    )


def sample_pairs(
    n_total: int,
    n_pairs: int,
    metadata: list[dict] | None = None,
    same_group_ratio: float = 0.3,
    seed: int = 42,
) -> torch.Tensor:
    """Sample pairs of prompt indices.

    If metadata is provided, same_group_ratio fraction of pairs come from
    prompts sharing a group_id (perturbation pairs). Rest are random.

    Returns: (n_pairs, 2) int64
    """
    rng = np.random.RandomState(seed)
    pairs = []

    if metadata is not None:
        # Group prompts by group_id
        groups: dict[str, list[int]] = {}
        for idx, m in enumerate(metadata):
            gid = m.get("group_id")
            if gid is not None:
                groups.setdefault(gid, []).append(idx)

        # Sample within-group pairs
        n_group = int(n_pairs * same_group_ratio)
        group_keys = [k for k, v in groups.items() if len(v) >= 2]
        for _ in range(n_group):
            if not group_keys:
                break
            gk = rng.choice(group_keys)
            members = groups[gk]
            i, j = rng.choice(len(members), size=2, replace=False)
            pairs.append((members[i], members[j]))

    # Fill remaining with random pairs
    n_remaining = n_pairs - len(pairs)
    for _ in range(n_remaining):
        i, j = rng.choice(n_total, size=2, replace=False)
        pairs.append((i, j))

    return torch.tensor(pairs, dtype=torch.int64)


# --- Ridge baseline ---

def train_ridge(
    hidden_states: torch.Tensor,  # (N, d_hidden)
    logprobs: torch.Tensor,
    indices: torch.Tensor,
    residuals: torch.Tensor,
    pairs: torch.Tensor,          # (M, 2)
    alpha: float = 1.0,
    vocab_size: int = 50304,
) -> tuple[Ridge, dict[str, float]]:
    """Train Ridge regression: X_diff -> KL.

    The coefficient vector w defines a 1D projection from activation space
    to behavioral distance. See DECISIONS.md D4.

    Returns:
        ridge: trained sklearn Ridge model
        metrics: {r2_train, mean_kl, std_kl}
    """
    print("Computing pairwise KL targets...")
    y = compute_pairwise_kl_batched(logprobs, indices, residuals, pairs, vocab_size)

    # X_diff = h_i - h_j
    h = hidden_states.float()
    X_diff = h[pairs[:, 0]] - h[pairs[:, 1]]  # (M, d_hidden)
    X_np = X_diff.numpy()
    y_np = y.numpy()

    print(f"Training Ridge (alpha={alpha}, {X_np.shape[0]} pairs, d={X_np.shape[1]})...")
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_np, y_np)

    r2 = ridge.score(X_np, y_np)
    metrics = {
        "r2_train": float(r2),
        "mean_kl": float(y_np.mean()),
        "std_kl": float(y_np.std()),
        "n_pairs": int(len(y_np)),
    }
    print(f"Ridge R² (train): {r2:.4f}")
    return ridge, metrics


def evaluate_ridge(
    ridge: Ridge,
    hidden_states: torch.Tensor,
    logprobs: torch.Tensor,
    indices: torch.Tensor,
    residuals: torch.Tensor,
    pairs: torch.Tensor,
    vocab_size: int = 50304,
) -> dict[str, float]:
    """Evaluate Ridge on held-out pairs."""
    from scipy.stats import spearmanr

    y = compute_pairwise_kl_batched(logprobs, indices, residuals, pairs, vocab_size)
    h = hidden_states.float()
    X_diff = (h[pairs[:, 0]] - h[pairs[:, 1]]).numpy()
    y_np = y.numpy()

    y_pred = ridge.predict(X_diff)
    r2 = ridge.score(X_diff, y_np)

    rho, pval = spearmanr(y_np, y_pred)

    metrics = {
        "r2_val": float(r2),
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "n_pairs": int(len(y_np)),
    }
    print(f"Ridge R² (val): {r2:.4f}, Spearman ρ: {rho:.4f}")
    return metrics


def analyze_projection(
    ridge: Ridge,
    layer: int,
    output_dir: Path,
) -> dict[str, Any]:
    """SVD analysis of the Ridge coefficient vector.

    For a scalar-output Ridge, coef_ is (d_hidden,). This is a 1D projection.
    SVD of the reshaped weight gives the effective rank structure.

    For multi-output Ridge, coef_ is (d_out, d_hidden) and SVD gives the
    full rank structure.
    """
    w = ridge.coef_  # (d_hidden,) for scalar output
    if w.ndim == 1:
        w = w.reshape(1, -1)  # (1, d_hidden)

    U, S, Vh = np.linalg.svd(w, full_matrices=False)

    # Effective rank at thresholds
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    thresholds = [0.90, 0.95, 0.99]
    effective_ranks = {}
    for t in thresholds:
        rank = int(np.searchsorted(cumvar, t)) + 1
        effective_ranks[f"rank_{int(t*100)}"] = rank

    result = {
        "layer": layer,
        "singular_values": S.tolist(),
        "effective_ranks": effective_ranks,
        "total_dims": int(w.shape[1]),
    }

    # Save projection matrix
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"ridge_coef_layer_{layer}.npy", ridge.coef_)
    with open(output_dir / f"ridge_analysis_layer_{layer}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Layer {layer}: effective rank (90/95/99%) = "
          f"{effective_ranks.get('rank_90', 'N/A')}/"
          f"{effective_ranks.get('rank_95', 'N/A')}/"
          f"{effective_ranks.get('rank_99', 'N/A')}")
    return result


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Bisimulation probe (Phase 1)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--cache", type=str, required=True, help="Path to cached activations dir")
    parser.add_argument("--layer", type=int, default=None, help="Single layer to probe (default: all)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    cache_dir = Path(args.cache)
    bisim_cfg = config["bisimulation"]

    output_dir = Path(args.output_dir) if args.output_dir else Path(config["output"]["projection_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(cache_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Determine layers to probe
    # Peek at first batch file to count layers
    first_batch = sorted(f for f in cache_dir.glob("batch_[0-9]*.safetensors") if "_full" not in f.name)[0]
    sample_data = load_file(first_batch)
    n_layers = sum(1 for k in sample_data.keys() if k.startswith("layer_"))
    del sample_data

    if args.layer is not None:
        layers = [args.layer]
    elif bisim_cfg.get("layers"):
        layers = bisim_cfg["layers"]
    else:
        layers = list(range(n_layers))

    print(f"Probing {len(layers)} layers: {layers}")

    # Sample pairs (shared across layers for comparability)
    n_total = len(metadata)
    n_train = bisim_cfg["num_pairs_train"]
    n_val = bisim_cfg["num_pairs_val"]
    same_group_ratio = bisim_cfg.get("pair_sampling", {}).get("same_group_ratio", 0.3)

    train_pairs = sample_pairs(n_total, n_train, metadata, same_group_ratio, seed=config.get("seed", 42))
    val_pairs = sample_pairs(n_total, n_val, metadata, same_group_ratio, seed=config.get("seed", 42) + 1)

    all_results = []

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        hidden_states, logprobs, indices, residuals = load_cached_layer(cache_dir, layer)
        print(f"Loaded {hidden_states.shape[0]} activations, d={hidden_states.shape[1]}")

        # Ridge baseline
        ridge, train_metrics = train_ridge(
            hidden_states, logprobs, indices, residuals,
            train_pairs, alpha=bisim_cfg["ridge_alpha"],
        )

        val_metrics = evaluate_ridge(
            ridge, hidden_states, logprobs, indices, residuals, val_pairs,
        )

        analysis = analyze_projection(ridge, layer, output_dir)

        result = {
            "layer": layer,
            "train": train_metrics,
            "val": val_metrics,
            "analysis": analysis,
        }
        all_results.append(result)

    # Save summary
    with open(output_dir / "bisimulation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # CSV log for quick inspection
    with open(output_dir / "bisimulation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "r2_train", "r2_val", "spearman_rho", "rank_90", "rank_95"])
        for r in all_results:
            writer.writerow([
                r["layer"],
                f"{r['train']['r2_train']:.4f}",
                f"{r['val']['r2_val']:.4f}",
                f"{r['val']['spearman_rho']:.4f}",
                r["analysis"]["effective_ranks"].get("rank_90", ""),
                r["analysis"]["effective_ranks"].get("rank_95", ""),
            ])

    print(f"\nResults saved to {output_dir}")
    print("\nSummary:")
    for r in all_results:
        print(f"  Layer {r['layer']:2d}: R²={r['val']['r2_val']:.4f}, "
              f"ρ={r['val']['spearman_rho']:.4f}")


if __name__ == "__main__":
    main()
