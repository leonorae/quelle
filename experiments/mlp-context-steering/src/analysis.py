"""Phase 2: Correlation analysis.

Computes deltas between baseline and ablated measurements, tests the three
core predictions, and identifies regime boundary layers.

Usage:
    python -m src.analysis --config configs/default.yaml

Inputs:
    outputs/baseline/*.pt
    outputs/ablated_strong/*.pt
    outputs/ablated_surgical/*.pt

Outputs:
    outputs/analysis/
        rank_decay.json         — Δerank per layer, per prompt
        sink_intensification.json — Δsink per layer, per prompt
        cross_prompt_correlation.json — correlation between total rank loss
                                        and total sink gain
        regime_boundaries.json  — layers where both Δerank and Δsink peak
        figures/                — plots (erank curves, Δ plots, scatter)
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_measurements(directory: Path, n_prompts: int) -> dict:
    """Load per-prompt .pt files into stacked arrays.

    Returns dict with:
        'erank': Tensor (n_prompts, n_layers+1) — erank at each layer
        'sink_intensity': Tensor (n_prompts, n_layers) — sink at each layer
    """
    # TODO: Implement loading and stacking
    raise NotImplementedError


def compute_rank_decay(
    baseline_erank: torch.Tensor,
    ablated_erank: torch.Tensor,
) -> dict:
    """Phase 2a: Rank decay acceleration.

    Δerank(l) = erank_baseline(l) - erank_ablated(l)

    Predictions:
        - Δerank > 0 at most layers (ablation reduces rank)
        - Δerank increases with depth (effect accumulates)

    Returns dict with per-layer means, stds, and significance tests.
    """
    # TODO: Implement
    # Compute Δerank per prompt per layer
    # Test: is Δerank significantly > 0? (one-sample t-test or bootstrap)
    # Test: does Δerank increase with layer index? (linear regression slope)
    raise NotImplementedError


def compute_sink_intensification(
    baseline_sink: torch.Tensor,
    ablated_sink: torch.Tensor,
) -> dict:
    """Phase 2b: Sink intensification.

    Δsink(l) = sink_ablated(l) - sink_baseline(l)

    Prediction: Δsink > 0 (ablation increases sink formation)
    """
    # TODO: Implement
    raise NotImplementedError


def compute_cross_prompt_correlation(
    baseline_erank: torch.Tensor,
    ablated_erank: torch.Tensor,
    baseline_sink: torch.Tensor,
    ablated_sink: torch.Tensor,
) -> dict:
    """Phase 2c: Cross-prompt correlation.

    For each prompt, compute:
        total_rank_loss = sum over layers of Δerank
        total_sink_gain = sum over layers of Δsink

    Then compute Pearson correlation between total_rank_loss and
    total_sink_gain across prompts.

    Prediction: positive correlation (prompts where ablation most damages
    rank are the same prompts where sinks intensify most).
    """
    # TODO: Implement
    # Use scipy.stats.pearsonr or torch-native correlation
    # Report r, p-value, 95% CI (bootstrap)
    raise NotImplementedError


def detect_regime_boundaries(
    delta_erank: torch.Tensor,
    delta_sink: torch.Tensor,
) -> list[int]:
    """Phase 2d: Find layers where both Δerank and Δsink change sharply.

    These are candidate "regime boundary" layers where MLP context-steering
    is most load-bearing.

    Approach: identify layers where both |d/dl Δerank| and |d/dl Δsink|
    exceed their respective medians. These are layers where the ablation
    effect is changing fastest — transition points.
    """
    # TODO: Implement
    raise NotImplementedError


def generate_figures(output_dir: Path) -> None:
    """Generate all Phase 2 figures.

    Required figures:
    1. erank(l) curves: baseline vs strong-ablated vs surgical-ablated,
       with error bands (±1 std across prompts)
    2. sink_intensity(l) curves: same three conditions
    3. Δerank(l) and Δsink(l) on same axes (dual y-axis), per variant
    4. Cross-prompt scatter: total_rank_loss vs total_sink_gain
    5. Regime boundary layers highlighted on the Δ plots

    Use matplotlib. Save as PNG and PDF.
    """
    # TODO: Implement
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: Wire up: load measurements → compute all analyses → save → plot
    raise NotImplementedError
