"""Ablation hooks and collection for MLP context steering.

Hook factories are fully implemented. The collection loop (collect_ablated)
is a TODO stub with guidance in its docstring.

TRAP 1: TransformerLens hook point names may differ between model families.
        For GPT-2: "blocks.{layer}.hook_mlp_out" is correct.
        For other models, verify with `model.hook_dict.keys()`.

TRAP 2: Hook execution order matters for surgical ablation. TransformerLens
        executes hooks in registration order. If you register an MLP-zeroing
        hook and a metric-collection hook, the collection hook must come AFTER
        the ablation hook to see the ablated activations.
"""

from typing import Callable

import torch
from torch import Tensor


# Type alias for TransformerLens hook functions
HookFn = Callable[[Tensor, object], Tensor]


def make_strong_ablation_hook() -> HookFn:
    """Create a hook that zeros the entire MLP output.

    Returns a hook function compatible with TransformerLens's
    `model.run_with_hooks()`. When attached to a
    "blocks.{L}.hook_mlp_out" hook point, it replaces the MLP output
    with zeros, effectively removing the MLP's contribution to the
    residual stream at that layer.

    Per D3: the modified residual stream propagates naturally through
    all subsequent layers.
    """

    def hook_fn(mlp_output: Tensor, hook: object) -> Tensor:
        return torch.zeros_like(mlp_output)

    return hook_fn


def make_surgical_ablation_hook(neuron_indices: list[int]) -> HookFn:
    """Create a hook that zeros specific neurons in the MLP output.

    Args:
        neuron_indices: List of neuron indices to zero out. These index
            into the last dimension of the MLP output tensor.

    Returns:
        Hook function that zeros only the specified neurons, leaving
        all others intact.
    """

    def hook_fn(mlp_output: Tensor, hook: object) -> Tensor:
        result = mlp_output.clone()
        result[..., neuron_indices] = 0.0
        return result

    return hook_fn


def make_permutation_hook(seed: int = 0) -> HookFn:
    """Create a hook that randomly permutes the MLP output across the token dimension.

    Used as a control in Phase 3: permutation preserves activation norms
    and statistics but destroys positional structure. If ablation effects
    are due to MLP-specific computation (not just magnitude), permutation
    should produce different metric shifts than zeroing.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Hook function that permutes token positions in the MLP output.
    """

    def hook_fn(mlp_output: Tensor, hook: object) -> Tensor:
        gen = torch.Generator(device=mlp_output.device)
        gen.manual_seed(seed)
        # Permute along the sequence (token) dimension
        # mlp_output shape: (batch, seq_len, d_model)
        seq_len = mlp_output.shape[1]
        perm = torch.randperm(seq_len, generator=gen, device=mlp_output.device)
        return mlp_output[:, perm, :]

    return hook_fn


def build_hook_list(
    n_layers: int,
    target_layers: list[int] | None = None,
    mode: str = "strong",
    neuron_indices: list[int] | None = None,
) -> list[tuple[str, HookFn]]:
    """Build a list of (hook_point_name, hook_fn) pairs for TransformerLens.

    Args:
        n_layers: Total number of layers in the model.
        target_layers: Which layers to ablate. None = all layers.
        mode: "strong" (zero entire MLP) or "surgical" (zero specific neurons).
        neuron_indices: Required if mode="surgical". Neuron indices to zero.

    Returns:
        List of (hook_point_name, hook_fn) tuples suitable for passing to
        `model.run_with_hooks(fwd_hooks=...)`.
    """
    if target_layers is None:
        target_layers = list(range(n_layers))

    hooks = []
    for layer in target_layers:
        hook_point = f"blocks.{layer}.hook_mlp_out"
        if mode == "strong":
            hook_fn = make_strong_ablation_hook()
        elif mode == "surgical":
            if neuron_indices is None:
                raise ValueError("neuron_indices required for surgical mode")
            hook_fn = make_surgical_ablation_hook(neuron_indices)
        else:
            raise ValueError(f"Unknown ablation mode: {mode}")
        hooks.append((hook_point, hook_fn))

    return hooks


def collect_ablated(model, prompts, config):
    """Collect per-layer metrics on ablated forward passes.

    TODO: Implement this function.

    Guidance:
    - Structure mirrors collect_baseline() in baseline.py.
    - Use build_hook_list() to construct hooks from config.
    - Run model.run_with_hooks(tokens, fwd_hooks=hooks) to get ablated cache.
    - Extract resid_post and attention patterns from cache using same keys
      as baseline: cache["resid_post", layer] and cache["pattern", layer].
    - Compute metrics per layer using collect_layer_metrics().
    - Return dict mapping layer index to metric dicts.

    TRAP: Verify hook point names match your model. Print
    `list(model.hook_dict.keys())` to check.

    Args:
        model: TransformerLens HookedTransformer.
        prompts: List of token tensors.
        config: Experiment config dict.

    Returns:
        Dict[int, Dict[str, Tensor]]: layer -> metrics mapping.
    """
    raise NotImplementedError("collect_ablated not yet implemented")
