"""Phase 0.5: Tuned Lens baseline.

Does a simple per-layer affine decoder already capture pairwise behavioral
distance?  If yes, the bisimulation probe adds nothing.  If no, pairwise
structure exists that independent decoding misses.

For each layer l, train a linear map  h_l → logits  to minimize KL with the
model's true output distribution.  Then for each pair (i, j):
    tuned_lens_kl = symKL( softmax(lens(h_i)), softmax(lens(h_j)) )
Compare to true_kl.  Report R² and Spearman ρ per layer.

Primary output: per-layer R² curve.  Shape matters more than magnitude.
See DECISIONS.md D13.

Usage:
    python -m src.tuned_lens_baseline --config configs/default.yaml \\
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
from safetensors.torch import load_file
from scipy.stats import spearmanr
from tqdm import tqdm

try:
    from .bisimulation_probe import (
        compute_pairwise_kl_batched,
        load_cached_layer,
        load_config,
        sample_pairs,
    )
except ImportError:
    from bisimulation_probe import (
        compute_pairwise_kl_batched,
        load_cached_layer,
        load_config,
        sample_pairs,
    )


# ---------------------------------------------------------------------------
# Target logit reconstruction
# ---------------------------------------------------------------------------

def reconstruct_target_logits(
    cache_dir: Path,
    model_name: str,
    dtype: str = "float16",
) -> torch.Tensor:
    """Reconstruct exact full-vocabulary logits from cached final-layer hidden states.

    Loads only the model's final LayerNorm + output embedding (embed_out),
    applies them to cached layer_{n-1}, and returns (N, vocab_size) float32.
    """
    from transformers import AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype, torch.float16)
    print(f"Loading model {model_name} to extract output head...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype,
    )

    ln_f = model.gpt_neox.final_layer_norm
    embed_out = model.embed_out

    # Find the final layer index from cached files
    first_batch = sorted(
        f for f in cache_dir.glob("batch_[0-9]*.safetensors")
        if "_full" not in f.name
    )[0]
    sample_data = load_file(first_batch)
    n_layers = sum(1 for k in sample_data if k.startswith("layer_"))
    final_layer = n_layers - 1
    del sample_data

    # Load cached final-layer hidden states
    h_final = load_cached_layer(cache_dir, final_layer)[0].float()  # (N, d_hidden)
    print(f"Reconstructing logits from layer_{final_layer} ({h_final.shape[0]} prompts)...")

    # Apply output head
    with torch.no_grad():
        ln_f = ln_f.float()
        embed_out = embed_out.float()
        target_logits = embed_out(ln_f(h_final))  # (N, vocab_size)

    # Free model
    del model, ln_f, embed_out
    print(f"Target logits: {target_logits.shape}, {target_logits.dtype}")
    return target_logits


# ---------------------------------------------------------------------------
# Tuned Lens module
# ---------------------------------------------------------------------------

class TunedLens(nn.Module):
    """Per-layer affine map from hidden state to vocabulary logits.

    Layers are allocated lazily to avoid OOM when vocab_size is large.
    Call ``ensure_layer(layer)`` before forward if the layer may not exist yet.
    """

    def __init__(self, n_layers: int, d_hidden: int, vocab_size: int):
        super().__init__()
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.vocab_size = vocab_size
        # Sparse ModuleDict keyed by str(layer) — only materialised layers live in memory.
        self.lenses = nn.ModuleDict()

    def ensure_layer(self, layer: int) -> nn.Linear:
        """Allocate a single layer if it doesn't exist yet. Return the Linear."""
        key = str(layer)
        if key not in self.lenses:
            self.lenses[key] = nn.Linear(self.d_hidden, self.vocab_size)
        return self.lenses[key]

    def drop_layer(self, layer: int) -> None:
        """Free a layer's parameters to reclaim memory."""
        key = str(layer)
        if key in self.lenses:
            del self.lenses[key]

    def forward(self, hidden_state: torch.Tensor, layer: int) -> torch.Tensor:
        """(batch, d_hidden) → (batch, vocab_size)"""
        return self.lenses[str(layer)](hidden_state)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tuned_lens(
    cache_dir: Path,
    model_name: str,
    config: dict[str, Any],
) -> tuple[TunedLens, dict[str, Any]]:
    """Train per-layer lenses to reconstruct the model's output distribution.

    Returns the trained lens and per-layer reconstruction metrics.
    """
    tl_cfg = config["tuned_lens"]
    model_dtype = config["model"].get("dtype", "float16")

    # Reconstruct target logits (exact, from cached final layer)
    target_logits = reconstruct_target_logits(cache_dir, model_name, model_dtype)
    target_log_probs = torch.log_softmax(target_logits, dim=-1)  # (N, V)
    n_prompts, vocab_size = target_logits.shape

    # Count layers
    first_batch = sorted(
        f for f in cache_dir.glob("batch_[0-9]*.safetensors")
        if "_full" not in f.name
    )[0]
    sample_data = load_file(first_batch)
    n_layers = sum(1 for k in sample_data if k.startswith("layer_"))
    d_hidden = sample_data["layer_0"].shape[1]
    del sample_data

    # Determine which layers to train
    layers = tl_cfg.get("layers") or list(range(n_layers))

    lens = TunedLens(n_layers, d_hidden, vocab_size)
    lr = tl_cfg.get("lr", 0.001)
    wd = tl_cfg.get("weight_decay", 0.01)
    epochs = tl_cfg.get("epochs", 30)
    batch_size = tl_cfg.get("batch_size", 64)

    # Train + save one layer at a time to stay within memory.
    # Layer weights are saved to disk; the returned lens is empty and can be
    # reloaded selectively during evaluation.
    weights_dir = cache_dir / "_lens_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    for layer in layers:
        print(f"\n--- Training lens for layer {layer} ---")
        hidden_states = load_cached_layer(cache_dir, layer)[0].float()  # (N, d)

        linear = lens.ensure_layer(layer)
        optimizer = torch.optim.Adam(linear.parameters(), lr=lr, weight_decay=wd)

        n_batches = math.ceil(n_prompts / batch_size)
        best_loss = float("inf")

        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(n_prompts)
            epoch_loss = 0.0

            for b in range(n_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                h = hidden_states[idx]          # (B, d)
                target = target_log_probs[idx]  # (B, V)

                pred_logits = lens(h, layer)    # (B, V)
                pred_log_probs = torch.log_softmax(pred_logits, dim=-1)

                # KL(target || pred) = sum target_p * (log target_p - log pred_p)
                target_p = target.exp()
                kl = (target_p * (target - pred_log_probs)).sum(dim=-1).mean()

                optimizer.zero_grad()
                kl.backward()
                optimizer.step()

                epoch_loss += kl.item()

            avg_loss = epoch_loss / n_batches
            if avg_loss < best_loss:
                best_loss = avg_loss

        all_metrics[layer] = {
            "reconstruction_kl": best_loss,
            "epochs_trained": epochs,
        }
        print(f"  Layer {layer}: best KL = {best_loss:.4f}")

        # Persist this layer's weights and free memory before next layer
        torch.save(linear.state_dict(), weights_dir / f"layer_{layer}.pt")
        lens.drop_layer(layer)
        del hidden_states, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return lens, all_metrics


# ---------------------------------------------------------------------------
# Stratified pair sampling (D13)
# ---------------------------------------------------------------------------

def sample_stratified_pairs(
    metadata: list[dict],
    true_kl_values: torch.Tensor | None,
    all_pairs: torch.Tensor,
    n_target: int,
    seed: int = 42,
) -> torch.Tensor:
    """Sample pairs stratified by KL range.

    Draws roughly equal counts from low/medium/high KL terciles.
    Falls back to uniform sampling if true_kl_values is None.
    """
    if true_kl_values is None or len(true_kl_values) == 0:
        # Fallback: just use provided pairs
        return all_pairs[:n_target]

    rng = np.random.RandomState(seed)

    kl_np = true_kl_values.numpy()
    terciles = np.percentile(kl_np, [33.3, 66.7])

    low_mask = kl_np <= terciles[0]
    mid_mask = (kl_np > terciles[0]) & (kl_np <= terciles[1])
    high_mask = kl_np > terciles[1]

    n_per = n_target // 3

    def _sample_from_mask(mask: np.ndarray, n: int) -> np.ndarray:
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return np.array([], dtype=np.int64)
        chosen = rng.choice(indices, size=min(n, len(indices)), replace=False)
        return chosen

    low_idx = _sample_from_mask(low_mask, n_per)
    mid_idx = _sample_from_mask(mid_mask, n_per)
    high_idx = _sample_from_mask(high_mask, n_target - len(low_idx) - len(mid_idx))

    selected = np.concatenate([low_idx, mid_idx, high_idx])
    rng.shuffle(selected)
    return all_pairs[selected]


# ---------------------------------------------------------------------------
# Pairwise KL from full logits (exact, no top-k approximation)
# ---------------------------------------------------------------------------

def pairwise_kl_from_logits(
    logits: torch.Tensor,      # (N, V)
    pairs: torch.Tensor,       # (M, 2)
    batch_size: int = 512,
) -> torch.Tensor:
    """Exact symmetric KL between lens-decoded distributions.

    Returns (M,) float32.
    """
    log_probs = torch.log_softmax(logits, dim=-1)  # (N, V)
    kl_values = []

    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        lp_i = log_probs[batch_pairs[:, 0]]  # (B, V)
        lp_j = log_probs[batch_pairs[:, 1]]  # (B, V)
        p_i = lp_i.exp()
        p_j = lp_j.exp()
        kl_ij = (p_i * (lp_i - lp_j)).sum(dim=-1)
        kl_ji = (p_j * (lp_j - lp_i)).sum(dim=-1)
        kl_values.append(((kl_ij + kl_ji) / 2).detach())

    return torch.cat(kl_values, dim=0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline(
    cache_dir: Path,
    lens: TunedLens,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate: does tuned_lens_kl predict true_kl?

    Returns per-layer results with R² and Spearman ρ.
    """
    tl_cfg = config["tuned_lens"]
    n_pairs_eval = tl_cfg.get("num_pairs_eval", 10000)

    with open(cache_dir / "metadata.json") as f:
        metadata = json.load(f)

    n_total = len(metadata)

    # Determine layers
    first_batch = sorted(
        f for f in cache_dir.glob("batch_[0-9]*.safetensors")
        if "_full" not in f.name
    )[0]
    sample_data = load_file(first_batch)
    n_layers = sum(1 for k in sample_data if k.startswith("layer_"))
    del sample_data

    layers = tl_cfg.get("layers") or list(range(n_layers))
    seed = config.get("seed", 42)

    # Sample candidate pairs (oversample, then stratify)
    candidate_pairs = sample_pairs(
        n_total, min(n_pairs_eval * 3, n_total * (n_total - 1) // 2),
        metadata, same_group_ratio=0.3, seed=seed + 100,
    )

    # Compute true KL once (shared across layers)
    print("Computing true KL for pair stratification...")
    _, logprobs, indices, residuals = load_cached_layer(cache_dir, 0)
    true_kl_all = compute_pairwise_kl_batched(
        logprobs, indices, residuals, candidate_pairs,
    )

    # Stratified selection
    eval_pairs = sample_stratified_pairs(
        metadata, true_kl_all, candidate_pairs, n_pairs_eval, seed=seed + 200,
    )

    # True KL for selected pairs
    true_kl = compute_pairwise_kl_batched(
        logprobs, indices, residuals, eval_pairs,
    )
    true_kl_np = true_kl.numpy()

    print(f"Evaluating on {len(eval_pairs)} stratified pairs "
          f"(KL range: {true_kl_np.min():.3f} – {true_kl_np.max():.3f}, "
          f"median: {np.median(true_kl_np):.3f})")

    results = []
    weights_dir = cache_dir / "_lens_weights"

    for layer in tqdm(layers, desc="Evaluating layers"):
        hidden_states = load_cached_layer(cache_dir, layer)[0].float()

        # Load this layer's weights on demand
        layer_weights = weights_dir / f"layer_{layer}.pt"
        if layer_weights.exists() and str(layer) not in lens.lenses:
            linear = lens.ensure_layer(layer)
            linear.load_state_dict(torch.load(layer_weights, weights_only=True))

        with torch.no_grad():
            lens_logits = lens(hidden_states, layer)  # (N, V)

        lens_kl = pairwise_kl_from_logits(lens_logits, eval_pairs)
        lens_kl_np = lens_kl.numpy()

        # Free layer memory before next iteration
        del hidden_states, lens_logits
        lens.drop_layer(layer)

        # Clamp negative/nan KL values (numerical edge cases)
        lens_kl_np = np.clip(lens_kl_np, 0, None)
        lens_kl_np = np.nan_to_num(lens_kl_np, nan=0.0, posinf=1e6)

        # R² via sklearn-style computation
        ss_res = np.sum((true_kl_np - lens_kl_np) ** 2)
        ss_tot = np.sum((true_kl_np - true_kl_np.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        rho, pval = spearmanr(true_kl_np, lens_kl_np)

        result = {
            "layer": layer,
            "r2": float(r2),
            "spearman_rho": float(rho),
            "spearman_pval": float(pval),
            "mean_lens_kl": float(lens_kl_np.mean()),
            "mean_true_kl": float(true_kl_np.mean()),
            "n_pairs": int(len(eval_pairs)),
        }
        results.append(result)
        print(f"  Layer {layer:2d}: R² = {r2:.4f}, ρ = {rho:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 0.5: Tuned Lens baseline for behavioral distance",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache", required=True, help="Path to cached activations")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    cache_dir = Path(args.cache)
    model_name = args.model or config["model"]["name"]
    tl_cfg = config["tuned_lens"]
    output_dir = Path(args.output_dir or tl_cfg.get("output_dir", "outputs/tuned_lens"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    lens, train_metrics = train_tuned_lens(cache_dir, model_name, config)

    # Consolidate per-layer weights into a single file
    weights_dir = cache_dir / "_lens_weights"
    full_state = {}
    for wf in sorted(weights_dir.glob("layer_*.pt")):
        layer_idx = wf.stem.split("_")[1]
        layer_sd = torch.load(wf, weights_only=True)
        for k, v in layer_sd.items():
            full_state[f"lenses.{layer_idx}.{k}"] = v
    torch.save(full_state, output_dir / "tuned_lens_weights.pt")

    # Evaluate
    results = evaluate_baseline(cache_dir, lens, config)

    # Merge training + eval metrics
    for r in results:
        layer = r["layer"]
        if layer in train_metrics:
            r["reconstruction_kl"] = train_metrics[layer]["reconstruction_kl"]

    # Save
    with open(output_dir / "tuned_lens_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "tuned_lens_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "recon_kl", "r2", "spearman_rho"])
        for r in results:
            writer.writerow([
                r["layer"],
                f"{r.get('reconstruction_kl', float('nan')):.4f}",
                f"{r['r2']:.4f}",
                f"{r['spearman_rho']:.4f}",
            ])

    # Print summary curve
    print(f"\n{'='*60}")
    print("Tuned Lens Baseline — Per-Layer R² Curve")
    print(f"{'='*60}")
    print(f"{'Layer':>6} {'Recon KL':>10} {'R²':>8} {'Spearman ρ':>10}")
    print(f"{'-'*6:>6} {'-'*10:>10} {'-'*8:>8} {'-'*10:>10}")
    for r in results:
        recon = r.get("reconstruction_kl", float("nan"))
        print(f"{r['layer']:6d} {recon:10.4f} {r['r2']:8.4f} {r['spearman_rho']:10.4f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
