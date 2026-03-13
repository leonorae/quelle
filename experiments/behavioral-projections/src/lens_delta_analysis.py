"""Phase 0.5c: Lens delta decomposition.

Decomposes the prediction change between adjacent layers into frame vs state
contributions:

    total_delta[i]  = P_{l+1}(h_{l+1}[i]) - P_l(h_l[i])     # total change
    frame_delta[i]  = P_{l+1}(h_l[i])     - P_l(h_l[i])      # same state, different lens
    state_delta[i]  = P_{l+1}(h_{l+1}[i]) - P_{l+1}(h_l[i])  # same lens, different state

These sum exactly: total_delta = frame_delta + state_delta.  All computed in
logit space (pre-softmax) for linear decomposability and numerical stability.

Additionally reports SVD of W_{l+1} - W_l (the lens weight difference matrix)
to characterise the "map gap" as a matrix object.

Gate condition: Phase 0.5 lens weights exist in _lens_weights/.

Usage:
    python -m src.lens_delta_analysis --config configs/default.yaml \\
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
from tqdm import tqdm

try:
    from .bisimulation_probe import load_cached_layer, load_config
    from .tuned_lens_baseline import TunedLens
except ImportError:
    from bisimulation_probe import load_cached_layer, load_config
    from tuned_lens_baseline import TunedLens


# ---------------------------------------------------------------------------
# Lens loading
# ---------------------------------------------------------------------------

def load_lens_pair(
    weights_dir: Path,
    layer_l: int,
    layer_l1: int,
    d_hidden: int,
    vocab_size: int,
) -> tuple[nn.Linear, nn.Linear]:
    """Load trained lens weights for two adjacent layers."""
    def _load_one(layer: int) -> nn.Linear:
        linear = nn.Linear(d_hidden, vocab_size)
        path = weights_dir / f"layer_{layer}.pt"
        linear.load_state_dict(torch.load(path, weights_only=True))
        linear.eval()
        return linear

    return _load_one(layer_l), _load_one(layer_l1)


# ---------------------------------------------------------------------------
# Delta decomposition
# ---------------------------------------------------------------------------

def decompose_layer_pair(
    cache_dir: Path,
    weights_dir: Path,
    layer_l: int,
    layer_l1: int,
    d_hidden: int,
    vocab_size: int,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Compute frame/state delta decomposition for one adjacent layer pair.

    All deltas computed in logit space.  Returns norms and summary stats.
    """
    lens_l, lens_l1 = load_lens_pair(
        weights_dir, layer_l, layer_l1, d_hidden, vocab_size,
    )

    # Load activations for both layers
    h_l = load_cached_layer(cache_dir, layer_l)[0].float()    # (N, d)
    h_l1 = load_cached_layer(cache_dir, layer_l1)[0].float()  # (N, d)
    n_prompts = h_l.shape[0]

    # Accumulate norms in streaming fashion to stay memory-safe
    total_norms = []
    frame_norms = []
    state_norms = []

    with torch.no_grad():
        for start in range(0, n_prompts, batch_size):
            end = min(start + batch_size, n_prompts)
            h_l_b = h_l[start:end]
            h_l1_b = h_l1[start:end]

            # Four logit vectors needed
            p_l_hl   = lens_l(h_l_b)    # P_l(h_l)
            p_l1_hl  = lens_l1(h_l_b)   # P_{l+1}(h_l)   — same state, next lens
            p_l1_hl1 = lens_l1(h_l1_b)  # P_{l+1}(h_{l+1})

            total_d = p_l1_hl1 - p_l_hl       # total prediction change
            frame_d = p_l1_hl  - p_l_hl        # pure map gap
            state_d = p_l1_hl1 - p_l1_hl       # pure content change

            total_norms.append(total_d.norm(dim=-1))   # (B,)
            frame_norms.append(frame_d.norm(dim=-1))
            state_norms.append(state_d.norm(dim=-1))

    total_norms = torch.cat(total_norms)  # (N,)
    frame_norms = torch.cat(frame_norms)
    state_norms = torch.cat(state_norms)

    # Frame ratio: ||frame|| / ||total|| per prompt, averaged
    # Guard against zero total (shouldn't happen, but be safe)
    ratio = frame_norms / total_norms.clamp(min=1e-8)

    # SVD of lens weight difference: W_{l+1} - W_l
    W_diff = lens_l1.weight.data.float() - lens_l.weight.data.float()  # (V, d)
    # Only need singular values, not full U/V
    singular_values = torch.linalg.svdvals(W_diff)
    top_10_sv = singular_values[:10].tolist()

    # Clean up
    del lens_l, lens_l1, h_l, h_l1, W_diff

    return {
        "layer_l": layer_l,
        "layer_l1": layer_l1,
        "total_norm_mean": total_norms.mean().item(),
        "total_norm_std": total_norms.std().item(),
        "frame_norm_mean": frame_norms.mean().item(),
        "frame_norm_std": frame_norms.std().item(),
        "state_norm_mean": state_norms.mean().item(),
        "state_norm_std": state_norms.std().item(),
        "frame_ratio_mean": ratio.mean().item(),
        "frame_ratio_std": ratio.std().item(),
        "top_10_singular_values": top_10_sv,
        "n_prompts": int(total_norms.shape[0]),
    }


# ---------------------------------------------------------------------------
# KL-based comparison (optional, for cross-referencing with C1/C2/C3)
# ---------------------------------------------------------------------------

def compute_kl_deltas(
    cache_dir: Path,
    weights_dir: Path,
    layer_l: int,
    layer_l1: int,
    d_hidden: int,
    vocab_size: int,
    batch_size: int = 32,
) -> dict[str, float]:
    """Compute KL-divergence-based frame/state decomposition.

    Uses symmetric KL between softmax distributions rather than logit-space
    L2 norms.  Not linearly decomposable, but directly comparable with
    C1/C2/C3 metrics.
    """
    lens_l, lens_l1 = load_lens_pair(
        weights_dir, layer_l, layer_l1, d_hidden, vocab_size,
    )

    h_l = load_cached_layer(cache_dir, layer_l)[0].float()
    h_l1 = load_cached_layer(cache_dir, layer_l1)[0].float()
    n_prompts = h_l.shape[0]

    total_kls = []
    frame_kls = []
    state_kls = []

    def _sym_kl(lp_a: torch.Tensor, lp_b: torch.Tensor) -> torch.Tensor:
        """Symmetric KL between two log-prob tensors, per-row."""
        p_a = lp_a.exp()
        p_b = lp_b.exp()
        kl_ab = (p_a * (lp_a - lp_b)).sum(dim=-1)
        kl_ba = (p_b * (lp_b - lp_a)).sum(dim=-1)
        return 0.5 * (kl_ab + kl_ba)

    with torch.no_grad():
        for start in range(0, n_prompts, batch_size):
            end = min(start + batch_size, n_prompts)
            h_l_b = h_l[start:end]
            h_l1_b = h_l1[start:end]

            lp_l_hl   = torch.log_softmax(lens_l(h_l_b), dim=-1)
            lp_l1_hl  = torch.log_softmax(lens_l1(h_l_b), dim=-1)
            lp_l1_hl1 = torch.log_softmax(lens_l1(h_l1_b), dim=-1)

            total_kls.append(_sym_kl(lp_l_hl, lp_l1_hl1))
            frame_kls.append(_sym_kl(lp_l_hl, lp_l1_hl))
            state_kls.append(_sym_kl(lp_l1_hl, lp_l1_hl1))

    del lens_l, lens_l1, h_l, h_l1

    return {
        "total_kl_mean": torch.cat(total_kls).mean().item(),
        "frame_kl_mean": torch.cat(frame_kls).mean().item(),
        "state_kl_mean": torch.cat(state_kls).mean().item(),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_lens_delta_analysis(
    cache_dir: Path,
    config: dict[str, Any],
    output_dir: Path,
    include_kl: bool = False,
) -> list[dict[str, Any]]:
    """Run delta decomposition for all adjacent layer pairs."""
    from safetensors.torch import load_file

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover layers from cache
    first_batch = sorted(
        f for f in cache_dir.glob("batch_[0-9]*.safetensors")
        if "_full" not in f.name
    )[0]
    sample_data = load_file(first_batch)
    n_layers = sum(1 for k in sample_data if k.startswith("layer_"))
    d_hidden = sample_data["layer_0"].shape[1]
    del sample_data

    vocab_size = config.get("pairwise_lens", {}).get("vocab_size", 50304)

    # Check gate: lens weights must exist
    weights_dir = cache_dir / "_lens_weights"
    if not weights_dir.exists():
        raise FileNotFoundError(
            f"Lens weights not found at {weights_dir}. Run Phase 0.5 first."
        )

    # Determine which layers have trained weights
    available = sorted(
        int(f.stem.split("_")[1])
        for f in weights_dir.glob("layer_*.pt")
    )
    if len(available) < 2:
        raise ValueError(f"Need ≥2 trained layers, found {len(available)}")

    # Build adjacent pairs from available layers
    pairs = [(available[i], available[i + 1]) for i in range(len(available) - 1)]
    print(f"Decomposing {len(pairs)} adjacent layer pairs "
          f"(d={d_hidden}, V={vocab_size})...")

    results = []
    for layer_l, layer_l1 in tqdm(pairs, desc="Layer pairs"):
        entry = decompose_layer_pair(
            cache_dir, weights_dir, layer_l, layer_l1,
            d_hidden, vocab_size,
        )
        if include_kl:
            kl_entry = compute_kl_deltas(
                cache_dir, weights_dir, layer_l, layer_l1,
                d_hidden, vocab_size,
            )
            entry.update(kl_entry)
        results.append(entry)

        # Progress
        print(f"  L{layer_l}→L{layer_l1}: "
              f"frame={entry['frame_norm_mean']:.2f} "
              f"state={entry['state_norm_mean']:.2f} "
              f"ratio={entry['frame_ratio_mean']:.3f} "
              f"sv1={entry['top_10_singular_values'][0]:.2f}")

    # ---- Save outputs ----

    # Per-pair CSV
    csv_path = output_dir / "lens_delta_results.csv"
    fieldnames = [
        "layer_l", "layer_l1",
        "total_norm_mean", "total_norm_std",
        "frame_norm_mean", "frame_norm_std",
        "state_norm_mean", "state_norm_std",
        "frame_ratio_mean", "frame_ratio_std",
        "sv_1", "sv_2", "sv_3",
    ]
    if include_kl:
        fieldnames += ["total_kl_mean", "frame_kl_mean", "state_kl_mean"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in fieldnames if k in r}
            # Flatten top-3 SVs into columns
            svs = r["top_10_singular_values"]
            row["sv_1"] = svs[0] if len(svs) > 0 else ""
            row["sv_2"] = svs[1] if len(svs) > 1 else ""
            row["sv_3"] = svs[2] if len(svs) > 2 else ""
            writer.writerow(row)

    # Summary JSON (full detail including all 10 SVs)
    json_path = output_dir / "lens_delta_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  CSV:  {csv_path.name}")
    print(f"  JSON: {json_path.name}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 0.5c: Lens delta decomposition (frame vs state)")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to experiment config")
    parser.add_argument("--cache", required=True,
                        help="Path to cached activations")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: outputs/lens_delta/)")
    parser.add_argument("--include-kl", action="store_true",
                        help="Also compute KL-based deltas for C1/C2/C3 comparison")
    args = parser.parse_args()

    config = load_config(args.config)
    cache_dir = Path(args.cache)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/lens_delta")

    run_lens_delta_analysis(cache_dir, config, output_dir, include_kl=args.include_kl)


if __name__ == "__main__":
    main()
