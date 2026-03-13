"""Three-condition comparison: Standard Lens vs Pairwise Lens vs Bisimulation Probe.

Runs all three conditions on the same pair set, produces comparison plots and
summary tables. See DECISIONS.md D14 for the experimental design.

Conditions:
  1. Standard Tuned Lens (loads existing results or trains)
  2. Pairwise-optimized Tuned Lens (trains with pairwise objective)
  3. Direct bisimulation probe (Ridge + learned projection)

Primary output: per-layer R² and Spearman curves, all three on the same axes.

Usage:
    python -m src.compare_conditions --config configs/default.yaml \
        --cache data/activations/pythia-410m/

Optional: --rank-sweep to also run the d_proj effective rank analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .bisimulation_probe import (
        load_cached_layer,
        load_config,
        run_condition3,
        run_rank_sweep,
        sample_pairs,
    )
    from .pairwise_lens import (
        evaluate_pairwise_lens,
        train_pairwise_lens,
    )
    from .tuned_lens_baseline import (
        evaluate_baseline,
        reconstruct_target_log_probs,
        train_tuned_lens,
    )
except ImportError:
    from bisimulation_probe import (
        load_cached_layer,
        load_config,
        run_condition3,
        run_rank_sweep,
        sample_pairs,
    )
    from pairwise_lens import (
        evaluate_pairwise_lens,
        train_pairwise_lens,
    )
    from tuned_lens_baseline import (
        evaluate_baseline,
        reconstruct_target_log_probs,
        train_tuned_lens,
    )


def load_or_train_condition1(
    cache_dir: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Load existing Condition 1 results or train from scratch."""
    import torch

    tl_cfg = config["tuned_lens"]
    tl_dir = output_dir / "condition1_tuned_lens"
    existing_dir = Path(tl_cfg.get("output_dir", "outputs/tuned_lens"))

    # Try loading existing results
    summary_path = existing_dir / "tuned_lens_summary.json"
    weights_path = existing_dir / "tuned_lens_weights.pt"

    if summary_path.exists() and weights_path.exists():
        print("Loading existing Condition 1 (Standard Tuned Lens) results...")
        with open(summary_path) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} layers from {summary_path}")
        return results

    # Train from scratch
    print("Training Condition 1 (Standard Tuned Lens)...")
    model_name = config["model"]["name"]
    lens, train_metrics = train_tuned_lens(cache_dir, model_name, config)

    tl_dir.mkdir(parents=True, exist_ok=True)
    # Consolidate per-layer weights saved during training
    weights_dir = cache_dir / "_lens_weights"
    full_state = {}
    for wf in sorted(weights_dir.glob("layer_*.pt")):
        layer_idx = wf.stem.split("_")[1]
        layer_sd = torch.load(wf, weights_only=True)
        for k, v in layer_sd.items():
            full_state[f"lenses.{layer_idx}.{k}"] = v
    torch.save(full_state, tl_dir / "tuned_lens_weights.pt")

    results = evaluate_baseline(cache_dir, lens, config)
    for r in results:
        layer = r["layer"]
        if layer in train_metrics:
            r["reconstruction_kl"] = train_metrics[layer]["reconstruction_kl"]

    with open(tl_dir / "tuned_lens_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_condition2(
    cache_dir: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run Condition 2: Pairwise-optimized Tuned Lens."""
    import torch

    print("\nTraining Condition 2 (Pairwise-optimized Tuned Lens)...")
    c2_dir = output_dir / "condition2_pairwise_lens"
    c2_dir.mkdir(parents=True, exist_ok=True)

    lens, train_metrics = train_pairwise_lens(cache_dir, config)
    # Consolidate per-layer weights saved during training
    weights_dir = cache_dir / "_pairwise_lens_weights"
    full_state = {}
    for wf in sorted(weights_dir.glob("layer_*.pt")):
        layer_idx = wf.stem.split("_")[1]
        layer_sd = torch.load(wf, weights_only=True)
        for k, v in layer_sd.items():
            full_state[f"lenses.{layer_idx}.{k}"] = v
    torch.save(full_state, c2_dir / "pairwise_lens_weights.pt")

    results = evaluate_pairwise_lens(cache_dir, lens, config)
    for r in results:
        layer = r["layer"]
        if layer in train_metrics:
            r["train_mse"] = train_metrics[layer]["best_mse"]

    with open(c2_dir / "pairwise_lens_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def format_comparison_table(
    c1: list[dict], c2: list[dict], c3: list[dict],
) -> str:
    """Format a comparison table for terminal output."""
    lines = []
    lines.append(f"{'':=<90}")
    lines.append("Three-Condition Comparison — Per-Layer Results")
    lines.append(f"{'':=<90}")
    lines.append(
        f"{'Layer':>6}  "
        f"{'C1 R²':>7} {'C1 ρ':>7}  "
        f"{'C2 R²':>7} {'C2 ρ':>7}  "
        f"{'C3 R²':>7} {'C3 ρ':>7}  "
        f"{'Best R²':>8}"
    )
    lines.append(
        f"{'-'*6:>6}  "
        f"{'-'*7:>7} {'-'*7:>7}  "
        f"{'-'*7:>7} {'-'*7:>7}  "
        f"{'-'*7:>7} {'-'*7:>7}  "
        f"{'-'*8:>8}"
    )

    # Index by layer
    c1_by_layer = {r["layer"]: r for r in c1}
    c2_by_layer = {r["layer"]: r for r in c2}
    c3_by_layer = {r["layer"]: r for r in c3}

    all_layers = sorted(set(c1_by_layer) | set(c2_by_layer) | set(c3_by_layer))

    for layer in all_layers:
        r1 = c1_by_layer.get(layer, {})
        r2 = c2_by_layer.get(layer, {})
        r3 = c3_by_layer.get(layer, {})

        r2_1 = r1.get("r2", float("nan"))
        rho_1 = r1.get("spearman_rho", float("nan"))
        r2_2 = r2.get("r2", float("nan"))
        rho_2 = r2.get("spearman_rho", float("nan"))
        # Condition 3: use learned projection if available, else Ridge
        r2_3 = r3.get("learned_r2", r3.get("ridge_r2", float("nan")))
        rho_3 = r3.get("learned_spearman", r3.get("ridge_spearman", float("nan")))

        r2_values = [("C1", r2_1), ("C2", r2_2), ("C3", r2_3)]
        best = max(r2_values, key=lambda x: x[1] if not np.isnan(x[1]) else -999)

        lines.append(
            f"{layer:6d}  "
            f"{r2_1:7.4f} {rho_1:7.4f}  "
            f"{r2_2:7.4f} {rho_2:7.4f}  "
            f"{r2_3:7.4f} {rho_3:7.4f}  "
            f"{best[0]:>8}"
        )

    return "\n".join(lines)


def save_comparison(
    c1: list[dict], c2: list[dict], c3: list[dict],
    output_dir: Path,
):
    """Save merged comparison CSV and JSON."""
    c1_by_layer = {r["layer"]: r for r in c1}
    c2_by_layer = {r["layer"]: r for r in c2}
    c3_by_layer = {r["layer"]: r for r in c3}
    all_layers = sorted(set(c1_by_layer) | set(c2_by_layer) | set(c3_by_layer))

    rows = []
    for layer in all_layers:
        r1 = c1_by_layer.get(layer, {})
        r2 = c2_by_layer.get(layer, {})
        r3 = c3_by_layer.get(layer, {})
        rows.append({
            "layer": layer,
            "c1_r2": r1.get("r2", None),
            "c1_spearman": r1.get("spearman_rho", None),
            "c2_r2": r2.get("r2", None),
            "c2_spearman": r2.get("spearman_rho", None),
            "c3_ridge_r2": r3.get("ridge_r2", None),
            "c3_ridge_spearman": r3.get("ridge_spearman", None),
            "c3_learned_r2": r3.get("learned_r2", None),
            "c3_learned_spearman": r3.get("learned_spearman", None),
        })

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(rows, f, indent=2)

    with open(output_dir / "comparison_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer",
            "c1_r2", "c1_spearman",
            "c2_r2", "c2_spearman",
            "c3_ridge_r2", "c3_ridge_spearman",
            "c3_learned_r2", "c3_learned_spearman",
        ])
        for r in rows:
            writer.writerow([
                r["layer"],
                f"{r['c1_r2']:.4f}" if r["c1_r2"] is not None else "",
                f"{r['c1_spearman']:.4f}" if r["c1_spearman"] is not None else "",
                f"{r['c2_r2']:.4f}" if r["c2_r2"] is not None else "",
                f"{r['c2_spearman']:.4f}" if r["c2_spearman"] is not None else "",
                f"{r['c3_ridge_r2']:.4f}" if r["c3_ridge_r2"] is not None else "",
                f"{r['c3_ridge_spearman']:.4f}" if r["c3_ridge_spearman"] is not None else "",
                f"{r['c3_learned_r2']:.4f}" if r["c3_learned_r2"] is not None else "",
                f"{r['c3_learned_spearman']:.4f}" if r["c3_learned_spearman"] is not None else "",
            ])

    print(f"\nComparison saved to {output_dir / 'comparison_results.csv'}")


def try_plot_comparison(
    c1: list[dict], c2: list[dict], c3: list[dict],
    output_dir: Path,
    rank_sweep: list[dict] | None = None,
):
    """Attempt to plot comparison curves. Falls back gracefully if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots. Install with: pip install matplotlib")
        return

    c1_by_layer = {r["layer"]: r for r in c1}
    c2_by_layer = {r["layer"]: r for r in c2}
    c3_by_layer = {r["layer"]: r for r in c3}
    all_layers = sorted(set(c1_by_layer) | set(c2_by_layer) | set(c3_by_layer))

    # --- R² comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    layers_arr = np.array(all_layers)
    c1_r2 = np.array([c1_by_layer.get(l, {}).get("r2", np.nan) for l in all_layers])
    c2_r2 = np.array([c2_by_layer.get(l, {}).get("r2", np.nan) for l in all_layers])
    c3_r2 = np.array([c3_by_layer.get(l, {}).get("learned_r2", np.nan) for l in all_layers])

    ax1.plot(layers_arr, c1_r2, "o-", label="C1: Standard Lens", alpha=0.8)
    ax1.plot(layers_arr, c2_r2, "s-", label="C2: Pairwise Lens", alpha=0.8)
    ax1.plot(layers_arr, c3_r2, "^-", label="C3: Bisim Probe", alpha=0.8)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("R²")
    ax1.set_title("Pairwise KL Prediction — R²")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # --- Spearman comparison ---
    c1_rho = np.array([c1_by_layer.get(l, {}).get("spearman_rho", np.nan) for l in all_layers])
    c2_rho = np.array([c2_by_layer.get(l, {}).get("spearman_rho", np.nan) for l in all_layers])
    c3_rho = np.array([c3_by_layer.get(l, {}).get("learned_spearman", np.nan) for l in all_layers])

    ax2.plot(layers_arr, c1_rho, "o-", label="C1: Standard Lens", alpha=0.8)
    ax2.plot(layers_arr, c2_rho, "s-", label="C2: Pairwise Lens", alpha=0.8)
    ax2.plot(layers_arr, c3_rho, "^-", label="C3: Bisim Probe", alpha=0.8)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Spearman ρ")
    ax2.set_title("Pairwise KL Prediction — Spearman ρ")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "comparison_r2_spearman.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_dir / 'comparison_r2_spearman.png'}")

    # --- Rank sweep plot (if available) ---
    if rank_sweep:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Group by layer
        by_layer: dict[int, list] = {}
        for r in rank_sweep:
            by_layer.setdefault(r["layer"], []).append(r)

        for layer, results in sorted(by_layer.items()):
            results.sort(key=lambda x: x["d_proj"])
            d_projs = [r["d_proj"] for r in results]
            r2s = [r["r2"] for r in results]
            rhos = [r["spearman_rho"] for r in results]

            ax1.plot(d_projs, r2s, "o-", label=f"Layer {layer}", alpha=0.7)
            ax2.plot(d_projs, rhos, "o-", label=f"Layer {layer}", alpha=0.7)

        ax1.set_xlabel("d_proj")
        ax1.set_ylabel("R²")
        ax1.set_title("Effective Rank Sweep — R²")
        ax1.set_xscale("log", base=2)
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("d_proj")
        ax2.set_ylabel("Spearman ρ")
        ax2.set_title("Effective Rank Sweep — Spearman ρ")
        ax2.set_xscale("log", base=2)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / "rank_sweep_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {output_dir / 'rank_sweep_curves.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Three-condition comparison for behavioral distance prediction",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache", required=True, help="Path to cached activations")
    parser.add_argument("--output-dir", default="outputs/comparison")
    parser.add_argument("--rank-sweep", action="store_true",
                        help="Also run d_proj rank sweep for Condition 3")
    parser.add_argument("--skip-c1", action="store_true",
                        help="Skip Condition 1 (use existing results)")
    parser.add_argument("--skip-c2", action="store_true",
                        help="Skip Condition 2")
    parser.add_argument("--skip-c3", action="store_true",
                        help="Skip Condition 3")
    args = parser.parse_args()

    config = load_config(args.config)
    cache_dir = Path(args.cache)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Condition 1: Standard Tuned Lens
    c1_results = []
    if not args.skip_c1:
        c1_results = load_or_train_condition1(cache_dir, config, output_dir)

    # Condition 2: Pairwise-optimized Tuned Lens
    c2_results = []
    if not args.skip_c2:
        c2_results = run_condition2(cache_dir, config, output_dir)

    # Condition 3: Direct bisimulation probe
    c3_results = []
    if not args.skip_c3:
        c3_dir = output_dir / "condition3_bisim"
        c3_results = run_condition3(cache_dir, config, c3_dir)

    # Print comparison
    if c1_results and c2_results and c3_results:
        table = format_comparison_table(c1_results, c2_results, c3_results)
        print(f"\n{table}")

    # Save
    save_comparison(c1_results, c2_results, c3_results, output_dir)

    # Rank sweep
    rank_sweep_results = None
    if args.rank_sweep and not args.skip_c3:
        sweep_dir = output_dir / "condition3_bisim"
        rank_sweep_results = run_rank_sweep(cache_dir, config, sweep_dir)

    # Plots
    try_plot_comparison(c1_results, c2_results, c3_results, output_dir, rank_sweep_results)

    # Interpretation guide
    print(f"\n{'='*60}")
    print("Interpretation Guide")
    print(f"{'='*60}")
    print("If C1 ≈ C2 ≈ C3 everywhere:")
    print("  → No pairwise structure. Standard lens suffices.")
    print("If C2 > C1 at some layers:")
    print("  → Pairwise optimization helps. Those layers have structure")
    print("    that absolute decoding misses.")
    print("If C3 > C2 at some layers:")
    print("  → Direct pairwise metric captures something the")
    print("    decode-then-compare pipeline can't, even optimized for pairs.")
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
