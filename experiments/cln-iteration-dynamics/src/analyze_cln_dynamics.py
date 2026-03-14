"""Analyze CLN iteration dynamics from crystal-lattice training logs.

Reads the per-iteration diagnostics logged during crystal-lattice training
and produces a summary of whether the CLN loop is doing useful work.

Usage:
    python -m src.analyze_cln_dynamics --metrics-file ../crystal-lattice/outputs/metrics.json
    python -m src.analyze_cln_dynamics --checkpoint ../crystal-lattice/outputs/checkpoint.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_metrics(path: Path) -> list[dict]:
    """Load metrics JSON from crystal-lattice training."""
    with open(path) as f:
        return json.load(f)


def analyze_iteration_profiles(metrics: list[dict]) -> dict:
    """Analyze per-iteration diagnostic profiles across training.

    Looks for the CLN diagnostics logged at each training step.
    Expected structure: each entry has a "cln_diagnostics" key with a list
    of per-iteration dicts (entropy, integrity, alpha, latent_norm).
    """
    # Collect per-iteration statistics across all training steps
    all_profiles = []
    for entry in metrics:
        diags = entry.get("cln_diagnostics") or entry.get("diagnostics")
        if diags:
            all_profiles.append(diags)

    if not all_profiles:
        return {"error": "No CLN diagnostics found in metrics"}

    n_iters = len(all_profiles[0])
    n_steps = len(all_profiles)

    # Aggregate per-iteration means
    keys = ["entropy", "integrity", "alpha", "latent_norm"]
    means = {k: [] for k in keys}
    stds = {k: [] for k in keys}

    for i in range(n_iters):
        for k in keys:
            values = [p[i].get(k, 0) for p in all_profiles if i < len(p)]
            means[k].append(float(np.mean(values)))
            stds[k].append(float(np.std(values)))

    # Detect "useful iteration" count: where does the profile plateau?
    # Use integrity as the primary signal — when does it stop increasing?
    integrity_deltas = np.diff(means["integrity"])
    plateau_iter = n_iters
    for i, delta in enumerate(integrity_deltas):
        if abs(delta) < 0.01 * abs(means["integrity"][0] - means["integrity"][-1] + 1e-8):
            plateau_iter = i + 1
            break

    # Alpha trajectory: does it decay?
    alpha_trend = "decaying" if means["alpha"][-1] < means["alpha"][0] * 0.7 else \
                  "increasing" if means["alpha"][-1] > means["alpha"][0] * 1.3 else "flat"

    return {
        "n_iterations": n_iters,
        "n_training_steps": n_steps,
        "means": means,
        "stds": stds,
        "plateau_iteration": plateau_iter,
        "alpha_trend": alpha_trend,
        "summary": {
            "entropy_range": [means["entropy"][0], means["entropy"][-1]],
            "integrity_range": [means["integrity"][0], means["integrity"][-1]],
            "alpha_range": [means["alpha"][0], means["alpha"][-1]],
        },
    }


def write_results(analysis: dict, output_path: Path) -> None:
    """Write RESULTS.md from analysis."""
    means = analysis["means"]
    n = analysis["n_iterations"]

    lines = [
        "# CLN Iteration Dynamics — Results\n",
        f"Analyzed {analysis['n_training_steps']} training steps, "
        f"{n} CLN iterations per step.\n",
        "## Per-Iteration Means\n",
        "| Iter | Entropy | Integrity | Alpha | Latent Norm |",
        "|------|---------|-----------|-------|-------------|",
    ]

    for i in range(n):
        lines.append(
            f"| {i} | {means['entropy'][i]:.4f} | {means['integrity'][i]:.4f} | "
            f"{means['alpha'][i]:.4f} | {means['latent_norm'][i]:.4f} |"
        )

    lines.extend([
        "",
        "## Summary\n",
        f"- **Plateau iteration**: {analysis['plateau_iteration']} "
        f"(integrity stops increasing meaningfully)",
        f"- **Alpha trend**: {analysis['alpha_trend']}",
        f"- **Entropy**: {analysis['summary']['entropy_range'][0]:.4f} → "
        f"{analysis['summary']['entropy_range'][-1]:.4f}",
        f"- **Integrity**: {analysis['summary']['integrity_range'][0]:.4f} → "
        f"{analysis['summary']['integrity_range'][-1]:.4f}",
        "",
        "## Interpretation\n",
    ])

    if analysis["plateau_iteration"] < n * 0.5:
        lines.append(
            f"The CLN plateaus at iteration {analysis['plateau_iteration']}/{n}. "
            "Most computation happens early — consider reducing the loop count."
        )
    elif analysis["alpha_trend"] == "decaying":
        lines.append(
            "Alpha decays across iterations (resonance pattern). The CLN "
            "relies on the VSA anchor early and trusts its own refinement later."
        )
    else:
        lines.append(
            "The CLN uses iterations relatively uniformly. The loop appears "
            "justified — each iteration contributes."
        )

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("../crystal-lattice/outputs/metrics.json"),
        help="Path to crystal-lattice metrics JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("RESULTS.md"),
        help="Output results file",
    )
    args = parser.parse_args()

    if not args.metrics_file.exists():
        print(f"Metrics file not found: {args.metrics_file}")
        print("Run crystal-lattice training first, then re-run this analysis.")
        sys.exit(1)

    metrics = load_metrics(args.metrics_file)
    analysis = analyze_iteration_profiles(metrics)

    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        sys.exit(1)

    write_results(analysis, args.output)

    # Also dump raw analysis as JSON
    json_out = args.output.with_suffix(".json")
    with open(json_out, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Raw analysis: {json_out}")


if __name__ == "__main__":
    main()
