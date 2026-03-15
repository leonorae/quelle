"""Baseline collection for MLP context steering.

TODO: Both functions below need implementation.
"""

from torch import Tensor


def load_prompts(config: dict) -> list[Tensor]:
    """Load and tokenise prompts from the configured data source.

    TODO: Implement this function.

    Guidance:
    - Use config["data"]["source"] to determine dataset. Default: "pile-validation".
    - Load n_prompts samples, truncate/pad to max_seq_len tokens.
    - Tokenise using model.tokenizer (TransformerLens provides this).
    - Return list of token tensors, each shape (seq_len,).
    - Set random seed from config["data"]["seed"] for reproducible sampling.

    Args:
        config: Experiment config dict.

    Returns:
        List of token tensors.
    """
    raise NotImplementedError("load_prompts not yet implemented")


def collect_baseline(model, prompts: list[Tensor], config: dict) -> dict:
    """Collect per-layer metrics on unmodified forward passes.

    TODO: Implement this function.

    Guidance:
    - Batch prompts according to config["collection"]["batch_size"].
    - Run model forward with caching: `logits, cache = model.run_with_cache(batch)`.
    - For each layer L, extract:
        - cache["resid_post", L]  → residual stream after layer L, shape (B, T, D)
        - cache["pattern", L]     → attention patterns, shape (B, H, T, T)
    - Pass to collect_layer_metrics() from metrics.py.
    - Accumulate metrics across batches (mean or concat, depending on metric).
    - Return dict mapping layer index to metric dicts.

    TransformerLens cache key format:
        cache["resid_post", layer_idx] — residual stream post-layer
        cache["pattern", layer_idx]    — attention weights (softmaxed)
        cache["mlp_out", layer_idx]    — MLP output (for reference)

    Args:
        model: TransformerLens HookedTransformer.
        prompts: List of token tensors from load_prompts().
        config: Experiment config dict.

    Returns:
        Dict[int, Dict[str, Tensor]]: layer -> {"erank": ..., "sink_intensity": ..., "max_activation": ...}
    """
    raise NotImplementedError("collect_baseline not yet implemented")
