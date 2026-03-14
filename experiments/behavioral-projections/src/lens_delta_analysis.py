"""Lens Delta Decomposition: frame_ratio curve across layers.

Decomposes prediction change between adjacent tuned lens translators into
state_delta (activation changed, same lens) and frame_delta (same activation,
lens changed).

Requires:
    - Trained tuned lens weights (one affine transform per layer)
    - Cached hidden state activations from a frozen model

Usage:
    python -m src.lens_delta_analysis \
        --lens-weights outputs/lens_weights.pt \
        --activations outputs/activations.pt \
        --output-dir outputs/lens_delta/

Outputs:
    - per_layer_pair.csv: layer_from, layer_to, mean_total_delta,
      mean_state_delta, mean_frame_delta, frame_ratio, state_ratio
    - summary.json: peaks, regime_boundaries, stability_metrics
    - weight_delta_svd.csv: SVD of W_{l+1} - W_l per layer pair
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class LayerPairDecomposition:
    """Decomposition results for one pair of adjacent layers."""
    layer_from: int
    layer_to: int
    mean_total_delta: float
    mean_state_delta: float
    mean_frame_delta: float
    frame_ratio: float
    state_ratio: float
    # SVD of weight matrix delta
    weight_delta_rank_90: int  # rank at 90% variance explained
    weight_delta_top_singular: float


def load_lens_weights(path: Path) -> list[dict[str, torch.Tensor]]:
    """Load tuned lens affine transforms.

    Expected format: list of dicts with 'weight' (d_model, d_model) and
    'bias' (d_model,), one per layer.
    """
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and "translators" in data:
        return data["translators"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected lens weights format: {type(data)}")


def apply_translator(
    translator: dict[str, torch.Tensor], h: torch.Tensor
) -> torch.Tensor:
    """Apply affine translator: h @ W.T + b.

    Parameters
    ----------
    translator : dict with 'weight' (d_out, d_in) and 'bias' (d_out,)
    h : (..., d_in)

    Returns
    -------
    (..., d_out)
    """
    return F.linear(h, translator["weight"], translator["bias"])


def decompose_layer_pair(
    h_l: torch.Tensor,
    h_l1: torch.Tensor,
    P_l: dict[str, torch.Tensor],
    P_l1: dict[str, torch.Tensor],
    layer_from: int,
    layer_to: int,
) -> LayerPairDecomposition:
    """Compute the three-way delta decomposition for one layer pair.

    total_delta = P_{l+1}(h_{l+1}) - P_l(h_l)
    state_delta = P_{l+1}(h_{l+1}) - P_{l+1}(h_l)
    frame_delta = P_{l+1}(h_l) - P_l(h_l)

    Parameters
    ----------
    h_l : (N, d_model) activations at layer l
    h_l1 : (N, d_model) activations at layer l+1
    P_l : translator dict for layer l
    P_l1 : translator dict for layer l+1
    """
    with torch.no_grad():
        p_l_h_l = apply_translator(P_l, h_l)       # P_l(h_l)
        p_l1_h_l = apply_translator(P_l1, h_l)     # P_{l+1}(h_l)
        p_l1_h_l1 = apply_translator(P_l1, h_l1)   # P_{l+1}(h_{l+1})

        total_delta = p_l1_h_l1 - p_l_h_l
        state_delta = p_l1_h_l1 - p_l1_h_l
        frame_delta = p_l1_h_l - p_l_h_l

        total_norm = total_delta.norm(dim=-1)       # (N,)
        state_norm = state_delta.norm(dim=-1)
        frame_norm = frame_delta.norm(dim=-1)

        # Avoid division by zero
        safe_total = total_norm.clamp(min=1e-8)
        frame_ratio = (frame_norm / safe_total).mean().item()
        state_ratio = (state_norm / safe_total).mean().item()

    # SVD of weight matrix delta
    W_l = P_l["weight"]    # (d_out, d_in)
    W_l1 = P_l1["weight"]
    W_delta = W_l1 - W_l

    _, s, _ = torch.linalg.svd(W_delta, full_matrices=False)
    var = s ** 2
    var_ratio = var / var.sum()
    cumvar = var_ratio.cumsum(dim=0)
    rank_90 = int((cumvar < 0.9).sum().item()) + 1

    return LayerPairDecomposition(
        layer_from=layer_from,
        layer_to=layer_to,
        mean_total_delta=float(total_norm.mean().item()),
        mean_state_delta=float(state_norm.mean().item()),
        mean_frame_delta=float(frame_norm.mean().item()),
        frame_ratio=frame_ratio,
        state_ratio=state_ratio,
        weight_delta_rank_90=rank_90,
        weight_delta_top_singular=float(s[0].item()),
    )


def find_regime_boundaries(
    decompositions: list[LayerPairDecomposition],
    threshold_std: float = 1.5,
) -> list[int]:
    """Identify regime boundaries as frame_ratio peaks.

    A layer pair is a regime boundary if its frame_ratio exceeds
    mean + threshold_std * std of the frame_ratio distribution.
    """
    ratios = [d.frame_ratio for d in decompositions]
    mean_r = sum(ratios) / len(ratios)
    std_r = (sum((r - mean_r) ** 2 for r in ratios) / len(ratios)) ** 0.5
    threshold = mean_r + threshold_std * std_r

    boundaries = []
    for d in decompositions:
        if d.frame_ratio > threshold:
            boundaries.append(d.layer_to)

    return boundaries


def run_analysis(
    lens_weights_path: Path,
    activations_path: Path,
    output_dir: Path,
) -> None:
    """Run full lens delta analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    translators = load_lens_weights(lens_weights_path)
    activations = torch.load(activations_path, map_location="cpu", weights_only=True)

    # activations: dict mapping layer index -> (N, d_model) tensor
    # or list of (N, d_model) tensors
    if isinstance(activations, list):
        act_dict = {i: a for i, a in enumerate(activations)}
    else:
        act_dict = activations

    n_layers = len(translators)
    decompositions: list[LayerPairDecomposition] = []

    for l in range(n_layers - 1):
        h_l = act_dict[l]
        h_l1 = act_dict[l + 1]
        P_l = translators[l]
        P_l1 = translators[l + 1]

        decomp = decompose_layer_pair(h_l, h_l1, P_l, P_l1, l, l + 1)
        decompositions.append(decomp)
        print(
            f"L{l}→L{l+1}: frame_ratio={decomp.frame_ratio:.4f} "
            f"state_ratio={decomp.state_ratio:.4f} "
            f"W_delta rank@90%={decomp.weight_delta_rank_90}"
        )

    # Write CSV
    csv_path = output_dir / "per_layer_pair.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(decompositions[0]).keys()))
        writer.writeheader()
        for d in decompositions:
            writer.writerow(asdict(d))

    # Find regime boundaries
    boundaries = find_regime_boundaries(decompositions)
    ratios = [d.frame_ratio for d in decompositions]

    # Write summary
    summary = {
        "n_layers": n_layers,
        "n_layer_pairs": len(decompositions),
        "frame_ratio_mean": sum(ratios) / len(ratios),
        "frame_ratio_std": (sum((r - sum(ratios)/len(ratios))**2 for r in ratios) / len(ratios)) ** 0.5,
        "frame_ratio_min": min(ratios),
        "frame_ratio_max": max(ratios),
        "regime_boundaries": boundaries,
        "n_regime_boundaries": len(boundaries),
        "per_layer_pair": [asdict(d) for d in decompositions],
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults written to {output_dir}")
    print(f"Regime boundaries at layers: {boundaries}")


def main():
    parser = argparse.ArgumentParser(
        description="Lens delta decomposition: frame_ratio curve across layers"
    )
    parser.add_argument(
        "--lens-weights", type=Path, required=True,
        help="Path to trained tuned lens weights (.pt)"
    )
    parser.add_argument(
        "--activations", type=Path, required=True,
        help="Path to cached activations (.pt)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/lens_delta"),
        help="Output directory for results"
    )
    args = parser.parse_args()

    run_analysis(args.lens_weights, args.activations, args.output_dir)


if __name__ == "__main__":
    main()
