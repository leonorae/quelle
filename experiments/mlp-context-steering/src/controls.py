"""Phase 3: Control conditions and sanity checks.

Three controls that test whether the main findings (if any) are specific
to context-steering or just artifacts of disrupting the forward pass.

Run AFTER Phase 2 analysis. If Phase 2 shows no signal, some controls
become unnecessary (see DECISIONS.md D9).

Usage:
    python -m src.controls --config configs/default.yaml --control {permutation,layer_specific,decomposition}
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from .metrics import collect_layer_metrics


# --- 3a: Random permutation control ---

def make_permutation_hook(layer: int):
    """Replace MLP input with a random permutation of positions.

    Preserves marginal statistics (same set of vectors, different ordering).
    If this produces the same effect as mean ablation, the result is about
    removing per-position variation, not specifically about context steering.
    """
    def hook_fn(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        B, S, D = activation.shape
        # Permute positions independently for each batch element
        for b in range(B):
            perm = torch.randperm(S, device=activation.device)
            activation[b] = activation[b, perm]
        return activation
    return hook_fn


# --- 3b: Layer-specific ablation ---

def collect_layer_specific_ablation(
    model: HookedTransformer,
    prompts: list[torch.Tensor],
    output_dir: Path,
) -> None:
    """Ablate one layer at a time, measure downstream effects.

    For each layer l in 0..n_layers-1:
        - Apply strong ablation at ONLY layer l
        - Measure erank and sink_intensity at all subsequent layers
        - Save results

    This identifies which layers are most sensitive. The triangle predicts
    early-to-middle layers (where Sun et al. find spike formation) should
    be most affected.
    """
    # TODO: Implement
    # For each target_layer:
    #   hooks = [(f'blocks.{target_layer}.hook_mlp_in', make_strong_ablation_hook(target_layer))]
    #   Run forward pass with hooks, collect metrics at all layers
    raise NotImplementedError


# --- 3c: MLP-only vs attention-only rank contribution ---

def decompose_rank_contributions(
    model: HookedTransformer,
    prompts: list[torch.Tensor],
    output_dir: Path,
) -> None:
    """Measure rank contribution of attention vs MLP at each layer.

    At each layer l, compute:
        erank(resid_pre_l + attn_out_l)   — rank after attention, before MLP
        erank(resid_post_l)                — rank after attention AND MLP

    The difference shows whether MLP increases or decreases rank.
    Dong predicts attention decreases rank; the triangle predicts
    MLP (via context-dependent updates) increases it.

    TransformerLens cache keys:
        cache['resid_pre', l]   — input to layer l
        cache['attn_out', l]    — attention output at layer l
        cache['resid_post', l]  — output of layer l (after MLP)
        cache['resid_mid', l]   — resid_pre + attn_out (before MLP)
    """
    # TODO: Implement
    # For each prompt:
    #   logits, cache = model.run_with_cache(tokens)
    #   for l in range(n_layers):
    #       erank_after_attn = erank(cache['resid_mid', l])
    #       erank_after_mlp = erank(cache['resid_post', l])
    #       mlp_rank_delta = erank_after_mlp - erank_after_attn
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: Add argparse for control type, wire up
    raise NotImplementedError
