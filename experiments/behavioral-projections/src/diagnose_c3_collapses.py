"""Diagnose C3 bisimulation probe collapses via targeted rank sweep.

Runs the rank sweep only at collapse layers (9, 11, 16, 17) plus control
layers (6, 12, 20), with a finer d_proj grid going down to 4.

Tests the dimensionality-mismatch hypothesis: if lower d_proj improves R²
at collapse layers, the full-rank projection is fitting noise in unused
dimensions and the L2 norm is dominated by those dimensions.

Usage:
    python -m src.diagnose_c3_collapses \
        --config configs/default.yaml \
        --override configs/collapse_sweep.yaml \
        --cache data/activations/pythia-410m/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from .bisimulation_probe import (
        BisimulationProjection,
        compute_pairwise_kl_batched,
        evaluate_learned_projection,
        load_cached_layer,
        load_config,
        sample_pairs,
        train_learned_projection,
    )
except ImportError:
    from bisimulation_probe import (
        BisimulationProjection,
        compute_pairwise_kl_batched,
        evaluate_learned_projection,
        load_cached_layer,
        load_config,
        sample_pairs,
        train_learned_projection,
    )

from safetensors import safe_open
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def run_targeted_sweep(
    cache_dir: Path,
    config: dict[str, Any],
    layers: list[int],
    d_proj_values: list[int],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run rank sweep on specified layers only."""
    bisim_cfg = config["bisimulation"]
    proj_cfg = config.get("projection", {})
    seed = config.get("seed", 42)

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

    all_results = []

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Collapse diagnosis — Layer {layer}")
        print(f"{'='*60}")

        hidden_states, logprobs, indices, residuals = load_cached_layer(cache_dir, layer)
        d_hidden = hidden_states.shape[1]

        train_kl = compute_pairwise_kl_batched(logprobs, indices, residuals, train_pairs)
        val_kl = compute_pairwise_kl_batched(logprobs, indices, residuals, val_pairs)

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

            tag = " *** COLLAPSE" if val_metrics["r2"] < -0.5 else ""
            print(f"    R² = {val_metrics['r2']:.4f}, ρ = {val_metrics['spearman_rho']:.4f}{tag}")

    return all_results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

COLLAPSE_LAYERS = {9, 11, 16, 17}


def analyse_results(results: list[dict]) -> str:
    """Print diagnostic summary testing dimensionality-mismatch vs overfitting."""
    by_layer: dict[int, list[dict]] = {}
    for r in results:
        by_layer.setdefault(r["layer"], []).append(r)

    lines = []
    lines.append("=" * 70)
    lines.append("Collapse Diagnosis Summary")
    lines.append("=" * 70)

    for layer in sorted(by_layer):
        entries = sorted(by_layer[layer], key=lambda x: x["d_proj"])
        is_collapse = layer in COLLAPSE_LAYERS
        label = "COLLAPSE" if is_collapse else "CONTROL"

        lines.append(f"\n  Layer {layer} [{label}]")
        lines.append(f"  {'d_proj':>8}  {'R²':>8}  {'ρ':>8}  {'train_mse':>10}")

        best_r2 = max(e["r2"] for e in entries)
        for e in entries:
            marker = " <-- best" if e["r2"] == best_r2 else ""
            lines.append(
                f"  {e['d_proj']:>8}  {e['r2']:>8.4f}  {e['spearman_rho']:>8.4f}"
                f"  {e['train_mse']:>10.4f}{marker}"
            )

        # Find optimal d_proj
        best_entry = max(entries, key=lambda e: e["r2"])
        full_rank = next((e for e in entries if e["d_proj"] == max(e["d_proj"] for e in entries)), None)

        if is_collapse and full_rank:
            delta = best_entry["r2"] - full_rank["r2"]
            lines.append(f"  → Best d_proj = {best_entry['d_proj']}, "
                         f"R² improvement over full rank: {delta:+.4f}")
            if delta > 0.5:
                lines.append(f"  → SUPPORTS dimensionality-mismatch hypothesis")
            elif best_entry["r2"] < 0:
                lines.append(f"  → SUPPORTS overfitting hypothesis (no rank helps)")

    lines.append("")
    lines.append("=" * 70)
    lines.append("Verdict:")
    lines.append("  If collapse layers improve with lower d_proj → dimensionality mismatch")
    lines.append("  If collapse layers stay negative at all ranks → overfitting / need more data")
    lines.append("=" * 70)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sweep(results: list[dict], output_dir: Path):
    """Plot R² vs d_proj per layer, highlighting collapse layers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    by_layer: dict[int, list[dict]] = {}
    for r in results:
        by_layer.setdefault(r["layer"], []).append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for layer in sorted(by_layer):
        entries = sorted(by_layer[layer], key=lambda x: x["d_proj"])
        d_projs = [e["d_proj"] for e in entries]
        r2s = [e["r2"] for e in entries]
        rhos = [e["spearman_rho"] for e in entries]

        is_collapse = layer in COLLAPSE_LAYERS
        style = "o--" if is_collapse else "s-"
        lw = 2.0 if is_collapse else 1.0
        label = f"L{layer}" + (" *" if is_collapse else "")

        ax1.plot(d_projs, r2s, style, label=label, alpha=0.8, linewidth=lw)
        ax2.plot(d_projs, rhos, style, label=label, alpha=0.8, linewidth=lw)

    ax1.set_xlabel("d_proj")
    ax1.set_ylabel("R²")
    ax1.set_title("C3 Rank Sweep — R² (collapse layers dashed)")
    ax1.set_xscale("log", base=2)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)

    ax2.set_xlabel("d_proj")
    ax2.set_ylabel("Spearman ρ")
    ax2.set_title("C3 Rank Sweep — Spearman ρ")
    ax2.set_xscale("log", base=2)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "collapse_rank_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose C3 bisimulation probe collapses via targeted rank sweep",
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Base config")
    parser.add_argument("--override", default="configs/collapse_sweep.yaml",
                        help="Override config with layer/sweep settings")
    parser.add_argument("--cache", required=True,
                        help="Path to cached activations")
    parser.add_argument("--output-dir", default="outputs/collapse_diagnosis",
                        help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply overrides
    override_path = Path(args.override)
    if override_path.exists():
        with open(override_path) as f:
            overrides = yaml.safe_load(f)

        layers = overrides.get("layers", [6, 9, 11, 12, 16, 17, 20])
        d_proj_values = overrides.get("rank_sweep", [1024, 512, 256, 128, 64, 32, 16, 8, 4])

        # Merge projection overrides
        if "projection" in overrides:
            config.setdefault("projection", {}).update(overrides["projection"])
    else:
        layers = [6, 9, 11, 12, 16, 17, 20]
        d_proj_values = [1024, 512, 256, 128, 64, 32, 16, 8, 4]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)

    # Run sweep
    results = run_targeted_sweep(cache_dir, config, layers, d_proj_values, output_dir)

    # Save raw results
    with open(output_dir / "collapse_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "collapse_sweep_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "d_proj", "r2", "spearman_rho", "train_mse", "epochs_trained"])
        for r in results:
            writer.writerow([
                r["layer"], r["d_proj"],
                f"{r['r2']:.4f}", f"{r['spearman_rho']:.4f}",
                f"{r['train_mse']:.4f}", r["epochs_trained"],
            ])

    # Analysis
    summary = analyse_results(results)
    print(summary)

    with open(output_dir / "diagnosis_summary.txt", "w") as f:
        f.write(summary)

    # Plot
    plot_sweep(results, output_dir)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
