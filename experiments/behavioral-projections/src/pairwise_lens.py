"""Condition 2: Pairwise-optimized Tuned Lens.

Same architecture as the standard Tuned Lens (per-layer affine map h → logits),
but trained with a pairwise objective:

    Loss = MSE(
        symKL(softmax(W_l h_i + b_l), softmax(W_l h_j + b_l)),
        true_KL(prompt_i, prompt_j)
    )

This trains the lens to decode in a way that *preserves pairwise behavioral
distances*, even if individual decoded distributions are less accurate.

See the three-condition comparison design in DECISIONS.md D14.

Usage:
    python -m src.pairwise_lens --config configs/default.yaml \
        --cache data/activations/pythia-410m/
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
from scipy.stats import spearmanr
from tqdm import tqdm

try:
    from .bisimulation_probe import (
        compute_pairwise_kl_batched,
        load_cached_layer,
        load_config,
        sample_pairs,
    )
    from .tuned_lens_baseline import (
        TunedLens,
        pairwise_kl_from_logits,
        sample_stratified_pairs,
    )
except ImportError:
    from bisimulation_probe import (
        compute_pairwise_kl_batched,
        load_cached_layer,
        load_config,
        sample_pairs,
    )
    from tuned_lens_baseline import (
        TunedLens,
        pairwise_kl_from_logits,
        sample_stratified_pairs,
    )


# ---------------------------------------------------------------------------
# Pairwise training
# ---------------------------------------------------------------------------

def _sym_kl_from_logits_pair(
    logits_i: torch.Tensor,  # (B, V)
    logits_j: torch.Tensor,  # (B, V)
) -> torch.Tensor:
    """Differentiable symmetric KL between two batches of logit vectors.

    Returns (B,) — one scalar per pair.
    """
    lp_i = torch.log_softmax(logits_i, dim=-1)
    lp_j = torch.log_softmax(logits_j, dim=-1)
    p_i = lp_i.exp()
    p_j = lp_j.exp()
    kl_ij = (p_i * (lp_i - lp_j)).sum(dim=-1)
    kl_ji = (p_j * (lp_j - lp_i)).sum(dim=-1)
    return (kl_ij + kl_ji) / 2


def train_pairwise_lens(
    cache_dir: Path,
    config: dict[str, Any],
) -> tuple[TunedLens, dict[int, dict[str, Any]]]:
    """Train per-layer lenses with the pairwise objective.

    For each layer:
      1. Sample pairs using group-aware sampling
      2. Compute true KL targets (from cached top-k logprobs)
      3. Optimize lens weights so decoded-KL ≈ true-KL
    """
    pl_cfg = config["pairwise_lens"]
    bisim_cfg = config["bisimulation"]
    seed = config.get("seed", 42)

    lr = pl_cfg.get("lr", 0.001)
    wd = pl_cfg.get("weight_decay", 0.01)
    epochs = pl_cfg.get("epochs", 30)
    batch_size = pl_cfg.get("pair_batch_size", 256)
    n_train_pairs = pl_cfg.get("num_pairs_train", 20000)

    # Load metadata for group-aware pair sampling
    with open(cache_dir / "metadata.json") as f:
        metadata = json.load(f)
    n_total = len(metadata)

    same_group_ratio = bisim_cfg.get("pair_sampling", {}).get("same_group_ratio", 0.3)

    # Count layers and dimensions
    from safetensors.torch import load_file
    first_batch = sorted(
        f for f in cache_dir.glob("batch_[0-9]*.safetensors")
        if "_full" not in f.name
    )[0]
    sample_data = load_file(first_batch)
    n_layers = sum(1 for k in sample_data if k.startswith("layer_"))
    d_hidden = sample_data["layer_0"].shape[1]
    vocab_size = pl_cfg.get("vocab_size", 50304)
    del sample_data

    layers = pl_cfg.get("layers") or list(range(n_layers))

    # Sample training pairs once (shared across layers)
    train_pairs = sample_pairs(
        n_total, n_train_pairs, metadata, same_group_ratio, seed=seed + 300,
    )

    # Compute true KL targets once
    print("Computing true KL targets for pairwise lens training...")
    _, logprobs, indices, residuals = load_cached_layer(cache_dir, 0)
    true_kl = compute_pairwise_kl_batched(
        logprobs, indices, residuals, train_pairs, vocab_size=vocab_size,
    )
    print(f"True KL: mean={true_kl.mean():.3f}, std={true_kl.std():.3f}, "
          f"range=[{true_kl.min():.3f}, {true_kl.max():.3f}]")

    lens = TunedLens(n_layers, d_hidden, vocab_size)
    weights_dir = cache_dir / "_pairwise_lens_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[int, dict[str, Any]] = {}

    for layer in layers:
        print(f"\n--- Training pairwise lens for layer {layer} ---")
        hidden_states = load_cached_layer(cache_dir, layer)[0].float()  # (N, d)

        linear = lens.ensure_layer(layer)
        optimizer = torch.optim.Adam(linear.parameters(), lr=lr, weight_decay=wd)

        n_pairs = len(train_pairs)
        n_batches = math.ceil(n_pairs / batch_size)
        best_loss = float("inf")
        patience = pl_cfg.get("patience", 10)
        no_improve = 0

        for epoch in range(epochs):
            perm = torch.randperm(n_pairs)
            epoch_loss = 0.0

            for b in range(n_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                pair_batch = train_pairs[idx]  # (B, 2)
                target_kl = true_kl[idx]       # (B,)

                h_i = hidden_states[pair_batch[:, 0]]  # (B, d)
                h_j = hidden_states[pair_batch[:, 1]]  # (B, d)

                logits_i = lens(h_i, layer)  # (B, V)
                logits_j = lens(h_j, layer)  # (B, V)

                pred_kl = _sym_kl_from_logits_pair(logits_i, logits_j)

                loss = torch.nn.functional.mse_loss(pred_kl, target_kl)

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
                print(f"  Early stop at epoch {epoch+1}")
                break

        all_metrics[layer] = {
            "best_mse": best_loss,
            "epochs_trained": epoch + 1,
        }
        print(f"  Layer {layer}: best MSE = {best_loss:.4f}")

        # Persist layer weights and free memory
        torch.save(linear.state_dict(), weights_dir / f"layer_{layer}.pt")
        lens.drop_layer(layer)
        del hidden_states, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return lens, all_metrics


# ---------------------------------------------------------------------------
# Evaluation (same protocol as Condition 1)
# ---------------------------------------------------------------------------

def evaluate_pairwise_lens(
    cache_dir: Path,
    lens: TunedLens,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate pairwise lens: does lens_kl predict true_kl on held-out pairs?

    Same metrics as Condition 1: R² and Spearman ρ per layer.
    """
    pl_cfg = config["pairwise_lens"]
    bisim_cfg = config["bisimulation"]
    n_pairs_eval = pl_cfg.get("num_pairs_eval", 10000)
    seed = config.get("seed", 42)

    with open(cache_dir / "metadata.json") as f:
        metadata = json.load(f)
    n_total = len(metadata)

    from safetensors.torch import load_file
    first_batch = sorted(
        f for f in cache_dir.glob("batch_[0-9]*.safetensors")
        if "_full" not in f.name
    )[0]
    sample_data = load_file(first_batch)
    n_layers = sum(1 for k in sample_data if k.startswith("layer_"))
    del sample_data

    layers = pl_cfg.get("layers") or list(range(n_layers))
    same_group_ratio = bisim_cfg.get("pair_sampling", {}).get("same_group_ratio", 0.3)

    # Sample eval pairs (different seed from training)
    candidate_pairs = sample_pairs(
        n_total, min(n_pairs_eval * 3, n_total * (n_total - 1) // 2),
        metadata, same_group_ratio, seed=seed + 400,
    )

    # True KL for stratification
    print("Computing true KL for evaluation pair stratification...")
    _, logprobs, indices, residuals = load_cached_layer(cache_dir, 0)
    true_kl_all = compute_pairwise_kl_batched(
        logprobs, indices, residuals, candidate_pairs,
    )

    eval_pairs = sample_stratified_pairs(
        metadata, true_kl_all, candidate_pairs, n_pairs_eval, seed=seed + 500,
    )

    true_kl = compute_pairwise_kl_batched(
        logprobs, indices, residuals, eval_pairs,
    )
    true_kl_np = true_kl.numpy()

    print(f"Evaluating on {len(eval_pairs)} stratified pairs "
          f"(KL range: {true_kl_np.min():.3f} – {true_kl_np.max():.3f})")

    results = []
    weights_dir = cache_dir / "_pairwise_lens_weights"

    for layer in tqdm(layers, desc="Evaluating pairwise lens"):
        hidden_states = load_cached_layer(cache_dir, layer)[0].float()

        # Load layer weights on demand
        layer_weights = weights_dir / f"layer_{layer}.pt"
        if layer_weights.exists() and str(layer) not in lens.lenses:
            linear = lens.ensure_layer(layer)
            linear.load_state_dict(torch.load(layer_weights, weights_only=True))

        with torch.no_grad():
            lens_logits = lens(hidden_states, layer)

        lens_kl = pairwise_kl_from_logits(lens_logits, eval_pairs)
        lens_kl_np = np.clip(np.nan_to_num(lens_kl.numpy(), nan=0.0, posinf=1e6), 0, None)

        del hidden_states, lens_logits
        lens.drop_layer(layer)

        ss_res = np.sum((true_kl_np - lens_kl_np) ** 2)
        ss_tot = np.sum((true_kl_np - true_kl_np.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        rho, pval = spearmanr(true_kl_np, lens_kl_np)

        results.append({
            "layer": layer,
            "r2": float(r2),
            "spearman_rho": float(rho),
            "spearman_pval": float(pval),
            "mean_lens_kl": float(lens_kl_np.mean()),
            "mean_true_kl": float(true_kl_np.mean()),
            "n_pairs": int(len(eval_pairs)),
        })
        print(f"  Layer {layer:2d}: R² = {r2:.4f}, ρ = {rho:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Condition 2: Pairwise-optimized Tuned Lens",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache", required=True, help="Path to cached activations")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    cache_dir = Path(args.cache)
    pl_cfg = config["pairwise_lens"]
    output_dir = Path(args.output_dir or pl_cfg.get("output_dir", "outputs/pairwise_lens"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    lens, train_metrics = train_pairwise_lens(cache_dir, config)

    # Consolidate per-layer weights into a single file
    weights_dir = cache_dir / "_pairwise_lens_weights"
    full_state = {}
    for wf in sorted(weights_dir.glob("layer_*.pt")):
        layer_idx = wf.stem.split("_")[1]
        layer_sd = torch.load(wf, weights_only=True)
        for k, v in layer_sd.items():
            full_state[f"lenses.{layer_idx}.{k}"] = v
    torch.save(full_state, output_dir / "pairwise_lens_weights.pt")

    # Evaluate
    results = evaluate_pairwise_lens(cache_dir, lens, config)

    # Merge training metrics
    for r in results:
        layer = r["layer"]
        if layer in train_metrics:
            r["train_mse"] = train_metrics[layer]["best_mse"]
            r["epochs_trained"] = train_metrics[layer]["epochs_trained"]

    # Save
    with open(output_dir / "pairwise_lens_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "pairwise_lens_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "train_mse", "r2", "spearman_rho"])
        for r in results:
            writer.writerow([
                r["layer"],
                f"{r.get('train_mse', float('nan')):.4f}",
                f"{r['r2']:.4f}",
                f"{r['spearman_rho']:.4f}",
            ])

    print(f"\n{'='*60}")
    print("Pairwise Lens — Per-Layer Results")
    print(f"{'='*60}")
    print(f"{'Layer':>6} {'Train MSE':>10} {'R²':>8} {'Spearman ρ':>10}")
    print(f"{'-'*6:>6} {'-'*10:>10} {'-'*8:>8} {'-'*10:>10}")
    for r in results:
        mse = r.get("train_mse", float("nan"))
        print(f"{r['layer']:6d} {mse:10.4f} {r['r2']:8.4f} {r['spearman_rho']:10.4f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
