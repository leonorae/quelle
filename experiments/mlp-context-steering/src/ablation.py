"""Phase 1: MLP context ablation.

Two ablation variants, both applied at every layer and propagated forward:

  Strong:   x_ablated[pos] = mean_over_positions(x)        for all pos
  Surgical: x_ablated[pos] = (x[pos] - attn_out[pos]) + mean(attn_out)

Collects the same three metrics (erank, sink_intensity, max_activation) as
baseline.py, under ablated conditions.

Usage:
    python -m src.ablation --config configs/default.yaml --variant strong
    python -m src.ablation --config configs/default.yaml --variant surgical
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from .metrics import collect_layer_metrics


def make_strong_ablation_hook(layer: int):
    """Hook that replaces MLP input with its mean across positions.

    Replaces every position's MLP input with the sequence mean,
    removing ALL per-position variation.

    TransformerLens hook point: 'blocks.{layer}.hook_mlp_in'
    Hook signature: hook_fn(activation, hook) -> modified_activation

    Parameters
    ----------
    layer : int — which layer this hook is for (for logging/debugging)
    """
    def hook_fn(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        # activation shape: (batch, seq_len, d_model)
        mean = activation.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        return mean.expand_as(activation)
    return hook_fn


def make_surgical_ablation_hook(layer: int):
    """Hook that removes per-position attention variation from MLP input.

    Subtracts the attention output and re-adds its mean, preserving
    each token's own residual contribution while removing position-specific
    context from attention.

    This requires access to the attention output at the same layer.
    Implementation approach: use two hooks per layer.
      1. Cache attn_out at 'blocks.{layer}.hook_attn_out'
      2. Modify MLP input at 'blocks.{layer}.hook_mlp_in'

    Parameters
    ----------
    layer : int — which layer this hook is for
    """
    # Shared state between the two hooks for this layer
    attn_out_cache = {}

    def cache_attn_out(activation: torch.Tensor, hook: HookPoint) -> None:
        # Just cache, don't modify
        attn_out_cache['value'] = activation  # (batch, seq_len, d_model)

    def modify_mlp_in(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        attn_out = attn_out_cache['value']
        # Remove per-position attention contribution, add back the mean
        mean_attn = attn_out.mean(dim=1, keepdim=True)
        modified = (activation - attn_out) + mean_attn.expand_as(attn_out)
        return modified

    return cache_attn_out, modify_mlp_in


def build_hook_list(
    model: HookedTransformer,
    variant: Literal["strong", "surgical"],
) -> list[tuple[str, callable]]:
    """Build the full list of (hook_point_name, hook_fn) pairs.

    For strong ablation: one hook per layer at hook_mlp_in.
    For surgical ablation: two hooks per layer (hook_attn_out + hook_mlp_in).

    Returns list suitable for model.run_with_hooks(fwd_hooks=...).
    """
    hooks = []
    for layer in range(model.cfg.n_layers):
        if variant == "strong":
            hooks.append((
                f'blocks.{layer}.hook_mlp_in',
                make_strong_ablation_hook(layer),
            ))
        elif variant == "surgical":
            cache_fn, modify_fn = make_surgical_ablation_hook(layer)
            hooks.append((f'blocks.{layer}.hook_attn_out', cache_fn))
            hooks.append((f'blocks.{layer}.hook_mlp_in', modify_fn))
    return hooks


def collect_ablated(
    model: HookedTransformer,
    prompts: list[torch.Tensor],
    variant: Literal["strong", "surgical"],
    output_dir: Path,
    batch_size: int = 4,
) -> None:
    """Run all prompts with ablation hooks and collect metrics.

    Same output format as baseline.collect_baseline, saved to
    outputs/ablated_{variant}/.

    Implementation notes for the sonnet:
    ------------------------------------
    Use model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="both")
    to get both logits and cache. The hooks modify activations in-place
    during the forward pass, and the cache captures the MODIFIED activations.

    TRAP: TransformerLens hook point names vary slightly between model
    families. For Pythia, verify the exact names by checking:
        model.hook_dict.keys()
    The expected pattern is 'blocks.{l}.hook_mlp_in' but some models
    use 'blocks.{l}.ln2.hook_normalized' for the pre-MLP layernorm output.
    If hook_mlp_in doesn't exist, the ablation must target the layernorm
    output instead (which is what actually feeds into the MLP).

    TRAP: For surgical ablation, hook execution order matters. The
    cache_attn_out hook MUST fire before modify_mlp_in within the same
    layer. TransformerLens processes hooks in the order they appear in
    the forward pass (attn before MLP), so this should work naturally,
    but verify.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    hooks = build_hook_list(model, variant)

    # TODO: Implement the collection loop.
    # Same structure as baseline.collect_baseline, but using
    # model.run_with_hooks(tokens, fwd_hooks=hooks) instead of
    # model.run_with_cache(tokens).
    #
    # Note: run_with_hooks returns logits by default. To also get the cache
    # (for computing metrics on modified activations), you need:
    #   model.run_with_cache(tokens, fwd_hooks=hooks)
    # which applies hooks AND caches the (modified) activations.
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: Add argparse for config path + variant, wire up
    raise NotImplementedError
