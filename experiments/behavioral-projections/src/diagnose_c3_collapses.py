"""Diagnose C3 bisimulation probe collapses via targeted rank sweep.

Three competing (non-exclusive) hypotheses for collapses at layers 9, 11, 16, 17:

  (2a) Intrinsic dimensionality mismatch — behavioral structure at collapse layers
       lives in ~30-50 dims; a 1024-dim projection has ~1000 noise dims dominating
       the L2 norm. Predicts: sharp knee in R² vs d_proj at intrinsic dim, then plateau.

  (2b) Sample complexity failure — 221 prompts yield ~24k pairs, insufficient to
       estimate 1024-dim projection regardless of intrinsic dim. Predicts: monotonic
       R² improvement as d_proj shrinks (fewer params = better generalization), no knee.

  (2c) Optimization failure — L2 norm makes loss non-convex; at high d_proj the
       landscape has more saddle points. Bad local minima fit a few large-distance
       pairs while failing on the rest. Predicts: high variance across seeds at
       same d_proj, especially at collapse layers.

Diagnostics:
  1. Rank sweep with resolution to detect a knee
  2. Multiple seeds per (layer, d_proj) to measure variance
  3. Participation ratio of learned projection's singular values

Usage:
    python -m src.diagnose_c3_collapses \\
        --config configs/default.yaml \\
        --override configs/collapse_sweep.yaml \\
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

from safetensors.torch import load_file


COLLAPSE_LAYERS = {9, 11, 16, 17}


# ---------------------------------------------------------------------------
# Participation ratio
# ---------------------------------------------------------------------------

def participation_ratio(weight: torch.Tensor) -> dict[str, float]:
    """Compute participation ratio from the projection weight matrix.

    The participation ratio PR = (sum(s_i))^2 / sum(s_i^2) where s_i are
    singular values. PR=1 means one dominant direction; PR=d_proj means
    all directions equally used. Normalized PR = PR / d_proj.

    Also reports effective ranks at 90/95/99% explained variance.
    """
    # weight is (d_proj, d_hidden)
    S = torch.linalg.svdvals(weight.float())
    S2 = S ** 2

    total_var = S2.sum().item()
    if total_var < 1e-12:
        return {"participation_ratio": 0.0, "normalized_pr": 0.0,
                "rank_90": 0, "rank_95": 0, "rank_99": 0,
                "top_sv": 0.0, "sv_entropy": 0.0}

    pr = (S.sum().item() ** 2) / total_var
    n_pr = pr / len(S)

    # Effective ranks at variance thresholds
    cumvar = torch.cumsum(S2, dim=0) / total_var
    rank_90 = int((cumvar < 0.90).sum().item()) + 1
    rank_95 = int((cumvar < 0.95).sum().item()) + 1
    rank_99 = int((cumvar < 0.99).sum().item()) + 1

    # Entropy of normalized singular value distribution
    p = S2 / total_var
    p_nz = p[p > 0]
    sv_entropy = -(p_nz * p_nz.log()).sum().item()

    return {
        "participation_ratio": float(pr),
        "normalized_pr": float(n_pr),
        "rank_90": rank_90,
        "rank_95": rank_95,
        "rank_99": rank_99,
        "top_sv": float(S[0].item()),
        "sv_entropy": float(sv_entropy),
    }


# ---------------------------------------------------------------------------
# Core sweep with multi-seed
# ---------------------------------------------------------------------------

def run_targeted_sweep(
    cache_dir: Path,
    config: dict[str, Any],
    layers: list[int],
    d_proj_values: list[int],
    n_seeds: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run rank sweep on specified layers with multiple seeds per (layer, d_proj)."""
    bisim_cfg = config["bisimulation"]
    proj_cfg = config.get("projection", {})
    base_seed = config.get("seed", 42)

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

    # Fixed pair sets (same across seeds — we're testing optimization variance)
    train_pairs = sample_pairs(n_total, n_train, metadata, same_group_ratio, seed=base_seed + 600)
    val_pairs = sample_pairs(n_total, n_val, metadata, same_group_ratio, seed=base_seed + 601)

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

            for seed_idx in range(n_seeds):
                seed = base_seed + seed_idx * 1000
                torch.manual_seed(seed)

                seed_label = f" (seed {seed_idx+1}/{n_seeds})" if n_seeds > 1 else ""
                print(f"\n  d_proj = {d_proj}{seed_label}")

                model, train_metrics = train_learned_projection(
                    hidden_states, train_kl, train_pairs, d_proj,
                    lr=lr, weight_decay=wd, epochs=epochs,
                    batch_size=batch_size, patience=patience_val,
                )

                val_metrics = evaluate_learned_projection(
                    model, hidden_states, val_kl, val_pairs,
                )

                # Participation ratio of learned projection
                with torch.no_grad():
                    pr_info = participation_ratio(model.proj.weight)

                result = {
                    "layer": layer,
                    "d_proj": d_proj,
                    "seed": seed_idx,
                    "train_mse": train_metrics["best_mse"],
                    "epochs_trained": train_metrics["epochs_trained"],
                    "r2": val_metrics["r2"],
                    "spearman_rho": val_metrics["spearman_rho"],
                    "n_pairs": val_metrics["n_pairs"],
                    **{f"pr_{k}": v for k, v in pr_info.items()},
                }
                all_results.append(result)

                tag = " *** COLLAPSE" if val_metrics["r2"] < -0.5 else ""
                print(f"    R² = {val_metrics['r2']:.4f}, "
                      f"ρ = {val_metrics['spearman_rho']:.4f}, "
                      f"PR = {pr_info['participation_ratio']:.1f}{tag}")

    return all_results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_results(results: list[dict]) -> str:
    """Diagnostic summary testing (2a) vs (2b) vs (2c)."""

    # Group by (layer, d_proj) → list of per-seed results
    grouped: dict[tuple[int, int], list[dict]] = {}
    for r in results:
        key = (r["layer"], r["d_proj"])
        grouped.setdefault(key, []).append(r)

    # Group by layer for per-layer summaries
    by_layer: dict[int, dict[int, list[dict]]] = {}
    for (layer, d_proj), seeds in grouped.items():
        by_layer.setdefault(layer, {})[d_proj] = seeds

    lines = []
    lines.append("=" * 78)
    lines.append("Collapse Diagnosis Summary")
    lines.append("=" * 78)

    for layer in sorted(by_layer):
        is_collapse = layer in COLLAPSE_LAYERS
        label = "COLLAPSE" if is_collapse else "CONTROL"
        d_proj_results = by_layer[layer]

        lines.append(f"\n  Layer {layer} [{label}]")
        lines.append(f"  {'d_proj':>8}  {'R²_mean':>8}  {'R²_std':>8}  "
                     f"{'ρ_mean':>8}  {'PR_mean':>8}  {'rank95':>8}")

        best_r2_mean = -float("inf")
        best_d_proj = None

        for d_proj in sorted(d_proj_results):
            seeds = d_proj_results[d_proj]
            r2s = [s["r2"] for s in seeds]
            rhos = [s["spearman_rho"] for s in seeds]
            prs = [s["pr_participation_ratio"] for s in seeds]
            rank95s = [s["pr_rank_95"] for s in seeds]

            r2_mean = np.mean(r2s)
            r2_std = np.std(r2s) if len(r2s) > 1 else 0.0
            rho_mean = np.mean(rhos)
            pr_mean = np.mean(prs)
            rank95_mean = np.mean(rank95s)

            if r2_mean > best_r2_mean:
                best_r2_mean = r2_mean
                best_d_proj = d_proj

            marker = ""
            if r2_std > 0.3:
                marker = " ← HIGH VARIANCE"

            lines.append(
                f"  {d_proj:>8}  {r2_mean:>8.4f}  {r2_std:>8.4f}  "
                f"{rho_mean:>8.4f}  {pr_mean:>8.1f}  {rank95_mean:>8.1f}{marker}"
            )

        lines.append(f"  → Best mean R² at d_proj = {best_d_proj} ({best_r2_mean:.4f})")

        if is_collapse:
            # Check for knee: is there a d_proj where R² plateaus?
            d_projs_sorted = sorted(d_proj_results.keys())
            r2_means = [np.mean([s["r2"] for s in d_proj_results[d]]) for d in d_projs_sorted]

            # Detect knee: largest second-derivative in R² vs log(d_proj)
            if len(r2_means) >= 3:
                diffs = [r2_means[i+1] - r2_means[i] for i in range(len(r2_means)-1)]
                diffs2 = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
                if diffs2:
                    knee_idx = int(np.argmax(np.abs(diffs2))) + 1
                    knee_d = d_projs_sorted[knee_idx]
                    lines.append(f"  → Knee candidate at d_proj = {knee_d}")

            # Check seed variance pattern
            max_d = max(d_proj_results.keys())
            min_d = min(d_proj_results.keys())
            if len(d_proj_results[max_d]) > 1 and len(d_proj_results[min_d]) > 1:
                var_high = np.std([s["r2"] for s in d_proj_results[max_d]])
                var_low = np.std([s["r2"] for s in d_proj_results[min_d]])
                if var_high > 3 * var_low and var_high > 0.1:
                    lines.append(f"  → Seed variance much higher at d_proj={max_d} "
                                 f"({var_high:.3f}) vs d_proj={min_d} ({var_low:.3f}) "
                                 f"→ supports optimization failure (2c)")

    # Cross-layer participation ratio comparison
    lines.append(f"\n{'='*78}")
    lines.append("Participation Ratio Comparison (full-rank projections)")
    lines.append(f"{'='*78}")

    max_d_proj = max(r["d_proj"] for r in results)
    for layer in sorted(by_layer):
        if max_d_proj in by_layer[layer]:
            seeds = by_layer[layer][max_d_proj]
            prs = [s["pr_participation_ratio"] for s in seeds]
            nprs = [s["pr_normalized_pr"] for s in seeds]
            is_collapse = layer in COLLAPSE_LAYERS
            label = " ***" if is_collapse else ""
            lines.append(f"  Layer {layer:>2}: PR = {np.mean(prs):>6.1f} "
                         f"(norm = {np.mean(nprs):.4f}){label}")

    lines.append("")
    lines.append("=" * 78)
    lines.append("Hypothesis signatures:")
    lines.append("  (2a) Intrinsic dim mismatch: sharp knee, low seed variance above knee,")
    lines.append("       higher PR at collapse layers than controls")
    lines.append("  (2b) Sample complexity:      monotonic improvement, moderate variance,")
    lines.append("       PR uninformative")
    lines.append("  (2c) Optimization failure:   high seed variance at large d_proj,")
    lines.append("       especially at collapse layers")
    lines.append("  These are not mutually exclusive.")
    lines.append("=" * 78)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sweep(results: list[dict], output_dir: Path):
    """Plot R² vs d_proj with error bands, plus participation ratio comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    # Group by (layer, d_proj)
    grouped: dict[tuple[int, int], list[dict]] = {}
    for r in results:
        grouped.setdefault((r["layer"], r["d_proj"]), []).append(r)

    by_layer: dict[int, list[int]] = {}
    for (layer, d_proj) in grouped:
        by_layer.setdefault(layer, set()).add(d_proj)
    for layer in by_layer:
        by_layer[layer] = sorted(by_layer[layer])

    # --- Plot 1: R² with error bands + Spearman ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_r2, ax_rho, ax_pr = axes

    for layer in sorted(by_layer):
        d_projs = by_layer[layer]
        r2_means, r2_stds = [], []
        rho_means = []
        pr_means = []

        for d in d_projs:
            seeds = grouped[(layer, d)]
            r2s = [s["r2"] for s in seeds]
            r2_means.append(np.mean(r2s))
            r2_stds.append(np.std(r2s) if len(r2s) > 1 else 0.0)
            rho_means.append(np.mean([s["spearman_rho"] for s in seeds]))
            pr_means.append(np.mean([s["pr_participation_ratio"] for s in seeds]))

        r2_means = np.array(r2_means)
        r2_stds = np.array(r2_stds)

        is_collapse = layer in COLLAPSE_LAYERS
        style = "o--" if is_collapse else "s-"
        lw = 2.0 if is_collapse else 1.0
        label = f"L{layer}" + (" *" if is_collapse else "")

        ax_r2.plot(d_projs, r2_means, style, label=label, alpha=0.8, linewidth=lw)
        if r2_stds.max() > 0:
            ax_r2.fill_between(d_projs, r2_means - r2_stds, r2_means + r2_stds,
                               alpha=0.15)

        ax_rho.plot(d_projs, rho_means, style, label=label, alpha=0.8, linewidth=lw)
        ax_pr.plot(d_projs, pr_means, style, label=label, alpha=0.8, linewidth=lw)

    ax_r2.set_xlabel("d_proj")
    ax_r2.set_ylabel("R² (mean ± std)")
    ax_r2.set_title("Rank Sweep — R²")
    ax_r2.set_xscale("log", base=2)
    ax_r2.legend(fontsize=7)
    ax_r2.grid(True, alpha=0.3)
    ax_r2.axhline(0, color="gray", linestyle="--", alpha=0.5)

    ax_rho.set_xlabel("d_proj")
    ax_rho.set_ylabel("Spearman ρ")
    ax_rho.set_title("Rank Sweep — Spearman ρ")
    ax_rho.set_xscale("log", base=2)
    ax_rho.legend(fontsize=7)
    ax_rho.grid(True, alpha=0.3)

    ax_pr.set_xlabel("d_proj")
    ax_pr.set_ylabel("Participation Ratio")
    ax_pr.set_title("Rank Sweep — Participation Ratio")
    ax_pr.set_xscale("log", base=2)
    ax_pr.legend(fontsize=7)
    ax_pr.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "collapse_rank_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")

    # --- Plot 2: Seed variance at each d_proj (collapse vs control) ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for layer in sorted(by_layer):
        d_projs = by_layer[layer]
        stds = []
        for d in d_projs:
            seeds = grouped[(layer, d)]
            r2s = [s["r2"] for s in seeds]
            stds.append(np.std(r2s) if len(r2s) > 1 else 0.0)

        is_collapse = layer in COLLAPSE_LAYERS
        style = "o--" if is_collapse else "s-"
        lw = 2.0 if is_collapse else 1.0
        label = f"L{layer}" + (" *" if is_collapse else "")
        ax.plot(d_projs, stds, style, label=label, alpha=0.8, linewidth=lw)

    ax.set_xlabel("d_proj")
    ax.set_ylabel("R² std across seeds")
    ax.set_title("Optimization Variance — R² std (collapse layers dashed)")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "seed_variance.png"
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
    parser.add_argument("--seeds", type=int, default=None,
                        help="Number of seeds per (layer, d_proj). "
                             "Overrides config value.")
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply overrides
    override_path = Path(args.override)
    if override_path.exists():
        with open(override_path) as f:
            overrides = yaml.safe_load(f)

        layers = overrides.get("layers", [6, 9, 11, 12, 16, 17, 20])
        d_proj_values = overrides.get("rank_sweep",
                                      [1024, 512, 256, 128, 64, 32, 16, 8])
        n_seeds = overrides.get("n_seeds", 3)

        if "projection" in overrides:
            config.setdefault("projection", {}).update(overrides["projection"])
    else:
        layers = [6, 9, 11, 12, 16, 17, 20]
        d_proj_values = [1024, 512, 256, 128, 64, 32, 16, 8]
        n_seeds = 3

    if args.seeds is not None:
        n_seeds = args.seeds

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)

    print(f"Layers: {layers}")
    print(f"d_proj values: {d_proj_values}")
    print(f"Seeds per (layer, d_proj): {n_seeds}")
    print(f"Total runs: {len(layers) * len(d_proj_values) * n_seeds}")

    # Run sweep
    results = run_targeted_sweep(
        cache_dir, config, layers, d_proj_values, n_seeds, output_dir,
    )

    # Save raw results
    with open(output_dir / "collapse_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "collapse_sweep_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "layer", "d_proj", "seed", "r2", "spearman_rho", "train_mse",
            "epochs_trained", "pr_participation_ratio", "pr_normalized_pr",
            "pr_rank_90", "pr_rank_95", "pr_rank_99", "pr_top_sv", "pr_sv_entropy",
        ]
        writer.writerow(header)
        for r in results:
            writer.writerow([
                r["layer"], r["d_proj"], r["seed"],
                f"{r['r2']:.4f}", f"{r['spearman_rho']:.4f}",
                f"{r['train_mse']:.6f}", r["epochs_trained"],
                f"{r['pr_participation_ratio']:.2f}",
                f"{r['pr_normalized_pr']:.6f}",
                r["pr_rank_90"], r["pr_rank_95"], r["pr_rank_99"],
                f"{r['pr_top_sv']:.4f}", f"{r['pr_sv_entropy']:.4f}",
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
