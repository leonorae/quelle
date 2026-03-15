"""Phase 0: Baseline measurements.

Collects erank, sink_intensity, and max_activation across all layers
for a corpus of prompts under normal (unablated) forward passes.

Usage:
    python -m src.baseline --config configs/default.yaml

Outputs:
    outputs/baseline/{prompt_idx:05d}.pt  — per-prompt measurements
    outputs/baseline/summary.json         — aggregate statistics
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from .metrics import collect_layer_metrics


def load_model(model_name: str = "pythia-410m", device: str = "cuda") -> HookedTransformer:
    """Load Pythia-410m via TransformerLens."""
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    return model


def load_prompts(source: str, n_prompts: int = 500, seq_len: int = 2048) -> list[torch.Tensor]:
    """Load and tokenize prompts.

    Parameters
    ----------
    source : str
        Path to a text corpus or HuggingFace dataset name.
        Implementation should handle both.
    n_prompts : int
        Number of prompts to sample.
    seq_len : int
        Truncate/pad to this length.

    Returns
    -------
    list of (seq_len,) int tensors — tokenized prompts
    """
    # TODO: Implement prompt loading.
    # Options (in preference order):
    #   1. If a cached 7304-prompt corpus exists from another experiment, load it
    #   2. Load from The Pile validation set via HuggingFace datasets
    #   3. Load from OpenWebText via HuggingFace datasets
    #
    # Use model.tokenizer to tokenize. Truncate to seq_len, skip prompts
    # shorter than seq_len (don't pad — padding would contaminate erank).
    raise NotImplementedError


def collect_baseline(
    model: HookedTransformer,
    prompts: list[torch.Tensor],
    output_dir: Path,
    batch_size: int = 4,
) -> None:
    """Run all prompts through the model and collect per-layer metrics.

    For each prompt, saves a dict with:
        - 'erank': list[float] — erank at each layer (0 through n_layers)
        - 'sink_intensity': list[float|None] — sink intensity at each layer
        - 'max_activation': list[dict] — max activation info at each layer

    The layer-0 entry is the embedding output (before any transformer block).
    Layers 1..n_layers correspond to transformer blocks.

    Implementation notes for the sonnet:
    ------------------------------------
    TransformerLens caches all intermediate activations when you run:
        logits, cache = model.run_with_cache(tokens)

    Relevant cache keys:
        cache['resid_post', layer]     — residual stream after layer (B, S, D)
        cache['pattern', layer]        — attention pattern (B, H, S, S)
        cache['resid_pre', 0]          — embedding output (B, S, D)

    Process prompts in batches of batch_size to manage GPU memory.
    For each batch, run_with_cache, compute metrics, save, clear cache.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_layers = model.cfg.n_layers

    # TODO: Implement the collection loop.
    #
    # Pseudocode:
    #   for i, tokens in enumerate(prompts):
    #       logits, cache = model.run_with_cache(tokens.unsqueeze(0))
    #       layer_metrics = []
    #       # Layer 0: embedding output
    #       layer_metrics.append(collect_layer_metrics(
    #           cache['resid_pre', 0], attention_pattern=None
    #       ))
    #       # Layers 1..n_layers
    #       for l in range(n_layers):
    #           layer_metrics.append(collect_layer_metrics(
    #               cache['resid_post', l],
    #               cache['pattern', l],
    #           ))
    #       torch.save(layer_metrics, output_dir / f'{i:05d}.pt')
    #       del cache  # free GPU memory
    #
    # After all prompts: compute aggregate summary and save as JSON.
    raise NotImplementedError


def compute_summary(output_dir: Path, n_prompts: int) -> dict:
    """Aggregate per-prompt measurements into summary statistics.

    Loads all per-prompt .pt files, computes mean and std of each metric
    across prompts at each layer.

    Returns dict suitable for JSON serialization.
    """
    # TODO: Implement aggregation.
    # Load each {i:05d}.pt, stack erank values into (n_prompts, n_layers+1),
    # compute mean/std. Same for sink_intensity. For max_activation, track
    # which channels appear most frequently as the max channel (spike channel
    # consistency check — Sun et al. predict consistent channels for pre-norm).
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: Add argparse for config path, wire up load_model → load_prompts → collect_baseline
    raise NotImplementedError
