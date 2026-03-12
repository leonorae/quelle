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
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from safetensors.torch import load_file
from scipy.stats import spearmanr
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

    Caps n_pairs at the number of unique pairs available (n_total choose 2)
    to avoid duplicates when the prompt set is small.

    Returns: (n_pairs, 2) int64
    """
    max_unique = n_total * (n_total - 1) // 2
    if n_pairs > max_unique:
        print(f"Warning: requested {n_pairs} pairs but only {max_unique} unique pairs "
              f"exist for {n_total} prompts. Capping at {max_unique}.")
        n_pairs = max_unique

    rng = np.random.RandomState(seed)
    pair_set: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []

    def add_pair(a: int, b: int) -> bool:
        key = (min(a, b), max(a, b))
        if key not in pair_set:
            pair_set.add(key)
            pairs.append((a, b))
            return True
        return False

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
        attempts = 0
        while len(pairs) < n_group and group_keys and attempts < n_group * 10:
            attempts += 1
            gk = rng.choice(group_keys)
            members = groups[gk]
            i, j = rng.choice(len(members), size=2, replace=False)
            add_pair(members[i], members[j])

    # Fill remaining with random pairs
    attempts = 0
    max_attempts = (n_pairs - len(pairs)) * 20
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(n_total, size=2, replace=False)
        add_pair(int(i), int(j))

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


# --- Learned multi-dimensional projection (Condition 3) ---

class BisimulationProjection(nn.Module):
    """Learned linear projection P: R^d_hidden → R^d_proj.

    Predicts pairwise behavioral distance as ||P(h_i - h_j)||_2.
    """

    def __init__(self, d_hidden: int, d_proj: int):
        super().__init__()
        self.proj = nn.Linear(d_hidden, d_proj, bias=False)

    def forward(self, delta_h: torch.Tensor) -> torch.Tensor:
        """(B, d_hidden) → (B,) predicted KL."""
        projected = self.proj(delta_h)  # (B, d_proj)
        return projected.norm(dim=-1)   # (B,)


def train_learned_projection(
    hidden_states: torch.Tensor,  # (N, d_hidden)
    true_kl: torch.Tensor,        # (M,) precomputed pair targets
    pairs: torch.Tensor,          # (M, 2)
    d_proj: int,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    epochs: int = 50,
    batch_size: int = 256,
    patience: int = 10,
) -> tuple[BisimulationProjection, dict[str, Any]]:
    """Train a learned linear projection for bisimulation distance.

    Returns the trained projection and training metrics.
    """
    d_hidden = hidden_states.shape[1]
    model = BisimulationProjection(d_hidden, d_proj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    h = hidden_states.float()
    n_pairs = len(pairs)
    n_batches = math.ceil(n_pairs / batch_size)
    best_loss = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        perm = torch.randperm(n_pairs)
        epoch_loss = 0.0

        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            pair_batch = pairs[idx]
            target = true_kl[idx]

            delta_h = h[pair_batch[:, 0]] - h[pair_batch[:, 1]]
            pred = model(delta_h)

            loss = nn.functional.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    metrics = {
        "best_mse": float(best_loss),
        "epochs_trained": epoch + 1,
        "d_proj": d_proj,
    }
    return model, metrics


def evaluate_learned_projection(
    model: BisimulationProjection,
    hidden_states: torch.Tensor,
    true_kl: torch.Tensor,
    pairs: torch.Tensor,
) -> dict[str, float]:
    """Evaluate learned projection on held-out pairs."""
    h = hidden_states.float()

    with torch.no_grad():
        delta_h = h[pairs[:, 0]] - h[pairs[:, 1]]
        pred = model(delta_h).numpy()

    y_np = true_kl.numpy()

    ss_res = np.sum((y_np - pred) ** 2)
    ss_tot = np.sum((y_np - y_np.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    rho, pval = spearmanr(y_np, pred)

    return {
        "r2": float(r2),
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "n_pairs": int(len(y_np)),
    }


def run_rank_sweep(
    cache_dir: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Sweep d_proj for the bisimulation probe across all layers.

    For each layer and each d_proj, train a learned projection and evaluate.
    Returns per-layer, per-d_proj results.

    The effective rank is the d_proj where performance starts dropping.
    """
    bisim_cfg = config["bisimulation"]
    proj_cfg = config.get("projection", {})
    seed = config.get("seed", 42)

    d_proj_values = bisim_cfg.get("rank_sweep", [1024, 512, 256, 128, 64, 32, 16])
    lr = proj_cfg.get("lr", 0.001)
    wd = proj_cfg.get("weight_decay", 0.01)
    epochs = proj_cfg.get("epochs", 50)
    batch_size = proj_cfg.get("batch_size", 256)
    patience_val = proj_cfg.get("patience", 10)

    with open(cache_dir / "metadata.json") as f:
        metadata = json.load(f)
    n_total = len(metadata)

    same_group_ratio = bisim_cfg.get("pair_sampling", {}).get("same_group_ratio", 0.3)
    n_train = bisim_cfg.get("num_pairs_train", 50000)
    n_val = bisim_cfg.get("num_pairs_val", 10000)

    train_pairs = sample_pairs(n_total, n_train, metadata, same_group_ratio, seed=seed + 600)
    val_pairs = sample_pairs(n_total, n_val, metadata, same_group_ratio, seed=seed + 601)

    # Determine layers
    first_batch = sorted(
        f for f in cache_dir.glob("batch_[0-9]*.safetensors")
        if "_full" not in f.name
    )[0]
    sample_data = load_file(first_batch)
    n_layers = sum(1 for k in sample_data if k.startswith("layer_"))
    del sample_data

    layers = bisim_cfg.get("layers") or list(range(n_layers))

    all_results = []

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Rank sweep — Layer {layer}")
        print(f"{'='*60}")

        hidden_states, logprobs, indices, residuals = load_cached_layer(cache_dir, layer)
        d_hidden = hidden_states.shape[1]

        # Compute KL targets for this layer's pairs
        train_kl = compute_pairwise_kl_batched(
            logprobs, indices, residuals, train_pairs,
        )
        val_kl = compute_pairwise_kl_batched(
            logprobs, indices, residuals, val_pairs,
        )

        for d_proj in d_proj_values:
            if d_proj > d_hidden:
                continue

            print(f"\n  d_proj = {d_proj}")
            model, train_metrics = train_learned_projection(
                hidden_states, train_kl, train_pairs, d_proj,
                lr=lr, weight_decay=wd, epochs=epochs,
                batch_size=batch_size, patience=patience_val,
            )

            val_metrics = evaluate_learned_projection(
                model, hidden_states, val_kl, val_pairs,
            )

            result = {
                "layer": layer,
                "d_proj": d_proj,
                "train_mse": train_metrics["best_mse"],
                "epochs_trained": train_metrics["epochs_trained"],
                "r2": val_metrics["r2"],
                "spearman_rho": val_metrics["spearman_rho"],
                "n_pairs": val_metrics["n_pairs"],
            }
            all_results.append(result)
            print(f"    R² = {val_metrics['r2']:.4f}, ρ = {val_metrics['spearman_rho']:.4f}")

    # Save
    sweep_dir = output_dir / "rank_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    with open(sweep_dir / "rank_sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(sweep_dir / "rank_sweep_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "d_proj", "r2", "spearman_rho", "train_mse"])
        for r in all_results:
            writer.writerow([
                r["layer"], r["d_proj"],
                f"{r['r2']:.4f}", f"{r['spearman_rho']:.4f}",
                f"{r['train_mse']:.4f}",
            ])

    print(f"\nRank sweep results saved to {sweep_dir}")
    return all_results


def run_condition3(
    cache_dir: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run Condition 3: Ridge baseline per layer + learned projection at d_proj=d_hidden.

    This is the main per-layer comparison entry point (not the rank sweep).
    Returns per-layer results with R² and Spearman.
    """
    bisim_cfg = config["bisimulation"]
    proj_cfg = config.get("projection", {})
    seed = config.get("seed", 42)

    with open(cache_dir / "metadata.json") as f:
        metadata = json.load(f)
    n_total = len(metadata)

    same_group_ratio = bisim_cfg.get("pair_sampling", {}).get("same_group_ratio", 0.3)
    n_train = bisim_cfg.get("num_pairs_train", 50000)
    n_val = bisim_cfg.get("num_pairs_val", 10000)

    train_pairs = sample_pairs(n_total, n_train, metadata, same_group_ratio, seed=seed)
    val_pairs = sample_pairs(n_total, n_val, metadata, same_group_ratio, seed=seed + 1)

    first_batch = sorted(
        f for f in cache_dir.glob("batch_[0-9]*.safetensors")
        if "_full" not in f.name
    )[0]
    sample_data = load_file(first_batch)
    n_layers = sum(1 for k in sample_data if k.startswith("layer_"))
    del sample_data

    layers = bisim_cfg.get("layers") or list(range(n_layers))

    results = []
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Condition 3 — Layer {layer}")
        print(f"{'='*60}")

        hidden_states, logprobs, indices, residuals = load_cached_layer(cache_dir, layer)
        d_hidden = hidden_states.shape[1]

        # Ridge baseline
        ridge, ridge_train = train_ridge(
            hidden_states, logprobs, indices, residuals,
            train_pairs, alpha=bisim_cfg.get("ridge_alpha", 1.0),
        )
        ridge_val = evaluate_ridge(
            ridge, hidden_states, logprobs, indices, residuals, val_pairs,
        )

        # Learned projection at d_proj = d_hidden
        train_kl = compute_pairwise_kl_batched(
            logprobs, indices, residuals, train_pairs,
        )
        val_kl = compute_pairwise_kl_batched(
            logprobs, indices, residuals, val_pairs,
        )

        proj_model, proj_train = train_learned_projection(
            hidden_states, train_kl, train_pairs, d_proj=d_hidden,
            lr=proj_cfg.get("lr", 0.001),
            weight_decay=proj_cfg.get("weight_decay", 0.01),
            epochs=proj_cfg.get("epochs", 50),
            batch_size=proj_cfg.get("batch_size", 256),
            patience=proj_cfg.get("patience", 10),
        )
        proj_val = evaluate_learned_projection(
            proj_model, hidden_states, val_kl, val_pairs,
        )

        result = {
            "layer": layer,
            "ridge_r2": ridge_val["r2_val"],
            "ridge_spearman": ridge_val["spearman_rho"],
            "learned_r2": proj_val["r2"],
            "learned_spearman": proj_val["spearman_rho"],
            "n_pairs": ridge_val["n_pairs"],
        }
        results.append(result)
        print(f"  Ridge:   R² = {ridge_val['r2_val']:.4f}, ρ = {ridge_val['spearman_rho']:.4f}")
        print(f"  Learned: R² = {proj_val['r2']:.4f}, ρ = {proj_val['spearman_rho']:.4f}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "condition3_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "condition3_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "ridge_r2", "ridge_spearman", "learned_r2", "learned_spearman"])
        for r in results:
            writer.writerow([
                r["layer"],
                f"{r['ridge_r2']:.4f}", f"{r['ridge_spearman']:.4f}",
                f"{r['learned_r2']:.4f}", f"{r['learned_spearman']:.4f}",
            ])

    return results


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Bisimulation probe (Phase 1)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--cache", type=str, required=True, help="Path to cached activations dir")
    parser.add_argument("--layer", type=int, default=None, help="Single layer to probe (default: all)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--mode", choices=["ridge", "condition3", "rank-sweep"], default="ridge",
        help="ridge: original Ridge-only run. "
             "condition3: Ridge + learned projection per layer. "
             "rank-sweep: d_proj sweep for effective rank analysis.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cache_dir = Path(args.cache)
    bisim_cfg = config["bisimulation"]

    output_dir = Path(args.output_dir) if args.output_dir else Path(config["output"]["projection_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "condition3":
        run_condition3(cache_dir, config, output_dir)
        return

    if args.mode == "rank-sweep":
        run_rank_sweep(cache_dir, config, output_dir)
        return

    # --- Original Ridge-only mode ---
    with open(cache_dir / "metadata.json") as f:
        metadata = json.load(f)

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

    with open(output_dir / "bisimulation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

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
