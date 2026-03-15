"""Phase 2 analysis — comparing baseline and ablated metrics.

TODO: All functions below are stubs. Implement after baseline and ablation
collection are working.

Each function states its prediction in the docstring.
"""

from torch import Tensor


def erank_shift(baseline_metrics: dict, ablated_metrics: dict) -> dict:
    """Compute per-layer change in effective rank after ablation.

    TODO: Implement.

    Prediction: erank should DECREASE after MLP ablation. MLPs expand the
    effective dimensionality of representations; removing them should cause
    collapse toward a lower-rank subspace.

    Args:
        baseline_metrics: Layer -> metric dict from collect_baseline().
        ablated_metrics: Layer -> metric dict from collect_ablated().

    Returns:
        Dict mapping layer index to erank delta (ablated - baseline).
        Negative values confirm the prediction.
    """
    raise NotImplementedError


def sink_amplification(baseline_metrics: dict, ablated_metrics: dict) -> dict:
    """Compute per-layer change in sink intensity after ablation.

    TODO: Implement.

    Prediction: sink intensity should INCREASE after MLP ablation. Without
    MLP steering, attention patterns degrade toward the default sink
    pattern (all mass on token 0).

    Args:
        baseline_metrics: Layer -> metric dict from collect_baseline().
        ablated_metrics: Layer -> metric dict from collect_ablated().

    Returns:
        Dict mapping layer index to sink_intensity delta (ablated - baseline).
        Positive values confirm the prediction.
    """
    raise NotImplementedError


def propagation_profile(ablated_metrics: dict, ablated_layer: int) -> dict:
    """Characterise how ablation effects propagate to downstream layers.

    TODO: Implement.

    Prediction: ablation at layer L should have increasing effect on erank
    at layers L+1, L+2, ... (cumulative degradation), not just a local dip.

    Per D3: ablation propagates through all subsequent layers.

    Args:
        ablated_metrics: Layer -> metric dict from single-layer ablation.
        ablated_layer: Which layer was ablated.

    Returns:
        Dict with propagation statistics (downstream erank trend, etc.)
    """
    raise NotImplementedError


def activation_stability(baseline_metrics: dict, ablated_metrics: dict) -> dict:
    """Check for numerical instability after ablation.

    TODO: Implement.

    Uses max_activation metric. If ablation causes activation explosion
    (max_activation >> baseline), the experiment may need gradient clipping
    or reduced sequence length.

    Args:
        baseline_metrics: Layer -> metric dict from collect_baseline().
        ablated_metrics: Layer -> metric dict from collect_ablated().

    Returns:
        Dict with stability diagnostics per layer.
    """
    raise NotImplementedError
