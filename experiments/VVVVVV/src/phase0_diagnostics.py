"""
Phase 0 — Diagnostic Baseline for Multi-Timescale Value Embedding (VVVVVV)

Three probes to characterise the existing single-table ve before touching k:
    Q0.1  probe_spike_channels()  — where do spike channels land?
    Q0.2  probe_bos_stability()   — is the BOS residual document-level?
    Q0.3  eval_with_ve_ablated()  — functional load of existing ve

All probes are non-destructive (weights are never permanently modified).
They are implemented as standalone functions that accept a nanochat GPT model
and a DataLoader. Import from nanochat's gpt.py before calling.

Usage
-----
    from nanochat.gpt import GPT, GPTConfig, has_ve
    from phase0_diagnostics import run_phase0

    model = GPT.from_pretrained(checkpoint_path)
    model.eval()
    results = run_phase0(model, val_dataloader)
    # results is a dict; log or save as needed.

Dependencies: torch, itertools (stdlib), nanochat's gpt.py on PYTHONPATH.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Q0.1 — Spike channel probe
# ---------------------------------------------------------------------------

def probe_spike_channels(
    model: Any,
    dataloader,
    has_ve_fn,
    n_batches: int = 50,
    gate_window: int = 32,
) -> dict[str, Any]:
    """
    Measure which residual channels carry the highest mean absolute activation
    at each ve-layer. Compare these against the gate's read window [:gate_window].

    Parameters
    ----------
    model       : nanochat GPT model (eval mode)
    dataloader  : yields (x, y) token-index batches
    has_ve_fn   : nanochat's has_ve(layer_idx, n_layer) function
    n_batches   : number of batches to accumulate over
    gate_window : width of the current gate's fixed read window (default 32)

    Returns
    -------
    dict with keys:
        "channel_magnitudes"  : {layer_idx: tensor(d_model)} mean |act| per channel
        "top_indices"         : {layer_idx: list of int} top-gate_window channel indices
        "overlap_fraction"    : {layer_idx: float} fraction of top channels in [:gate_window]
        "gate_window"         : int
        "interpretation"      : {layer_idx: str} "informative" / "near-noise" / "ambiguous"
    """
    n_layer = model.config.n_layer
    ve_layer_indices = [i for i in range(n_layer) if has_ve_fn(i, n_layer)]

    # Accumulate sum of |activations| per channel, per ve-layer
    accum: dict[int, torch.Tensor] = {}
    count: dict[int, int] = {}
    hooks = []

    def make_hook(idx: int):
        def hook(module, inp, out):
            # inp[0] is x entering the block (post-residual hidden state)
            x = inp[0].detach()  # (B, T, d_model)
            mag = x.abs().mean(dim=(0, 1))  # (d_model,)
            if idx not in accum:
                accum[idx] = torch.zeros_like(mag)
                count[idx] = 0
            accum[idx] += mag
            count[idx] += 1
        return hook

    for i, block in enumerate(model.transformer.h):
        if has_ve_fn(i, n_layer):
            hooks.append(block.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= n_batches:
                break
            x = x.to(next(model.parameters()).device)
            model(x)

    for h in hooks:
        h.remove()

    channel_magnitudes = {i: accum[i] / count[i] for i in accum}

    results: dict[str, Any] = {
        "channel_magnitudes": {},
        "top_indices": {},
        "overlap_fraction": {},
        "gate_window": gate_window,
        "interpretation": {},
    }

    for layer_idx, mag in channel_magnitudes.items():
        top_k = torch.topk(mag, gate_window).indices.tolist()
        overlap = sum(1 for ch in top_k if ch < gate_window) / gate_window
        if overlap > 0.5:
            interp = "informative"
        elif overlap < 0.2:
            interp = "near-noise"
        else:
            interp = "ambiguous"

        results["channel_magnitudes"][layer_idx] = mag.cpu().tolist()
        results["top_indices"][layer_idx] = top_k
        results["overlap_fraction"][layer_idx] = overlap
        results["interpretation"][layer_idx] = interp

    return results


# ---------------------------------------------------------------------------
# Q0.2 — BOS residual stability probe
# ---------------------------------------------------------------------------

def probe_bos_stability(
    model: Any,
    dataloader,
    has_ve_fn,
    n_batches: int = 20,
) -> dict[str, Any]:
    """
    Measure whether the BOS token residual carries document-level information
    or is a near-constant sink.

    For each ve-layer:
      - Collect BOS residuals (position 0 of each sequence in the batch)
      - Within-batch cross-doc cosine: mean cosine between all pairs of BOS
        residuals in the same batch (different documents)
      - Cross-batch cosine: mean cosine between BOS residuals from different
        batches (used to estimate "pure sink" baseline)

    Parameters
    ----------
    model       : nanochat GPT model (eval mode)
    dataloader  : yields (x, y) token-index batches
    has_ve_fn   : nanochat's has_ve(layer_idx, n_layer) function
    n_batches   : number of batches to collect from

    Returns
    -------
    dict with keys:
        "within_batch_cosine"  : {layer_idx: float} mean pairwise cosine within batch
        "cross_batch_cosine"   : {layer_idx: float} mean cosine across batches
        "classification"       : {layer_idx: str} "document-signal" / "pure-sink" / "no-stable-signal"
        "interpretation_note"  : str
    """
    n_layer = model.config.n_layer
    ve_layer_indices = [i for i in range(n_layer) if has_ve_fn(i, n_layer)]

    # Collect BOS residuals at each ve-layer, across batches
    # Shape per layer: list of tensors (B, d_model)
    bos_by_layer: dict[int, list[torch.Tensor]] = {i: [] for i in ve_layer_indices}
    hooks = []

    def make_hook(idx: int):
        def hook(module, inp, out):
            x = inp[0].detach()  # (B, T, d_model)
            bos = x[:, 0, :].cpu()  # BOS is always position 0
            bos_by_layer[idx].append(bos)
        return hook

    for i, block in enumerate(model.transformer.h):
        if has_ve_fn(i, n_layer):
            hooks.append(block.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= n_batches:
                break
            x = x.to(next(model.parameters()).device)
            model(x)

    for h in hooks:
        h.remove()

    results: dict[str, Any] = {
        "within_batch_cosine": {},
        "cross_batch_cosine": {},
        "classification": {},
        "interpretation_note": (
            "within >> cross: BOS carries document signal (conditioning viable). "
            "Both high: pure sink (BOS conditioning = fixed bias only). "
            "Both low: no stable BOS signal."
        ),
    }

    for layer_idx in ve_layer_indices:
        batches = bos_by_layer[layer_idx]  # list of (B, d) tensors
        if not batches:
            continue

        # Within-batch pairwise cosine: for each batch, all pairs of sequences
        within_cosines = []
        for batch_bos in batches:  # (B, d)
            normed = F.normalize(batch_bos, dim=-1)  # (B, d)
            sim_matrix = normed @ normed.T  # (B, B)
            B = normed.shape[0]
            if B > 1:
                # off-diagonal upper triangle
                indices = torch.triu_indices(B, B, offset=1)
                within_cosines.extend(
                    sim_matrix[indices[0], indices[1]].tolist()
                )

        # Cross-batch cosine: BOS from batch i vs batch i+1
        cross_cosines = []
        for a, b in zip(batches[:-1], batches[1:]):
            normed_a = F.normalize(a, dim=-1)
            normed_b = F.normalize(b, dim=-1)
            # mean of all cross-batch pairs
            sim = (normed_a @ normed_b.T).mean().item()
            cross_cosines.append(sim)

        within_mean = float(sum(within_cosines) / len(within_cosines)) if within_cosines else float("nan")
        cross_mean = float(sum(cross_cosines) / len(cross_cosines)) if cross_cosines else float("nan")

        # Classification
        if within_mean > 0.7 and cross_mean > 0.7:
            cls = "pure-sink"
        elif within_mean > 0.7 and cross_mean < 0.5:
            cls = "document-signal"
        elif within_mean < 0.3 and cross_mean < 0.3:
            cls = "no-stable-signal"
        else:
            cls = "ambiguous"

        results["within_batch_cosine"][layer_idx] = within_mean
        results["cross_batch_cosine"][layer_idx] = cross_mean
        results["classification"][layer_idx] = cls

    return results


# ---------------------------------------------------------------------------
# Q0.3 — ve ablation: functional load
# ---------------------------------------------------------------------------

def eval_with_ve_ablated(
    model: Any,
    dataloader,
    has_ve_fn,
    eval_fn,
    n_batches: int | None = None,
) -> dict[str, Any]:
    """
    Zero out all value_embeds weights, run eval, restore, and report
    val_bpb delta. This is the causal ground truth for "how much work is
    the single ve table doing" before touching k.

    Parameters
    ----------
    model       : nanochat GPT model (eval mode)
    dataloader  : validation dataloader
    has_ve_fn   : nanochat's has_ve(layer_idx, n_layer) function
    eval_fn     : callable(model, dataloader, n_batches) -> float (val_bpb or val_loss)
    n_batches   : passed to eval_fn; if None, eval_fn uses its own default

    Returns
    -------
    dict with keys:
        "baseline_bpb"    : float — val_bpb with ve enabled
        "ablated_bpb"     : float — val_bpb with ve zeroed
        "delta_bpb"       : float — ablated - baseline (positive = ve was helping)
        "relative_delta"  : float — delta / baseline
        "n_tables_zeroed" : int   — number of ve tables zeroed
    """
    n_layer = model.config.n_layer
    ve_layers = [i for i in range(n_layer) if has_ve_fn(i, n_layer)]

    # Baseline eval
    eval_kwargs = {} if n_batches is None else {"n_batches": n_batches}
    baseline_bpb = eval_fn(model, dataloader, **eval_kwargs)

    # Save and zero out all value_embeds
    saved_weights: dict[str, torch.Tensor] = {}
    n_zeroed = 0
    for key, emb in model.transformer.value_embeds.items():
        saved_weights[key] = emb.weight.data.clone()
        emb.weight.data.zero_()
        n_zeroed += 1

    # Ablated eval
    ablated_bpb = eval_fn(model, dataloader, **eval_kwargs)

    # Restore
    for key, weight in saved_weights.items():
        model.transformer.value_embeds[key].weight.data = weight

    delta = ablated_bpb - baseline_bpb
    relative = delta / baseline_bpb if baseline_bpb != 0 else float("nan")

    return {
        "baseline_bpb": baseline_bpb,
        "ablated_bpb": ablated_bpb,
        "delta_bpb": delta,
        "relative_delta": relative,
        "n_tables_zeroed": n_zeroed,
    }


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_phase0(
    model: Any,
    dataloader,
    has_ve_fn,
    eval_fn,
    output_path: str | Path | None = None,
    spike_batches: int = 50,
    bos_batches: int = 20,
    ablation_batches: int | None = None,
    gate_window: int = 32,
) -> dict[str, Any]:
    """
    Run all three Phase 0 probes and return a combined results dict.
    Optionally write results to a JSON file at output_path.

    Parameters
    ----------
    model           : nanochat GPT model (eval mode, correct device)
    dataloader      : validation DataLoader (yields (x, y) batches)
    has_ve_fn       : nanochat's has_ve function
    eval_fn         : callable(model, dataloader, **kwargs) -> float
    output_path     : if given, write JSON to this path
    spike_batches   : n_batches for Q0.1
    bos_batches     : n_batches for Q0.2
    ablation_batches: n_batches for Q0.3 (None → eval_fn default)
    gate_window     : current gate read window width (default 32)
    """
    print("=== Phase 0: Diagnostic Baseline ===")

    print("[Q0.1] Probing spike channels...")
    q01 = probe_spike_channels(
        model, dataloader, has_ve_fn,
        n_batches=spike_batches,
        gate_window=gate_window,
    )
    print(f"  Overlap fractions by layer: {q01['overlap_fraction']}")
    print(f"  Interpretations: {q01['interpretation']}")

    print("[Q0.2] Probing BOS residual stability...")
    q02 = probe_bos_stability(
        model, dataloader, has_ve_fn,
        n_batches=bos_batches,
    )
    print(f"  Within-batch cosine by layer: {q02['within_batch_cosine']}")
    print(f"  Cross-batch cosine by layer:  {q02['cross_batch_cosine']}")
    print(f"  Classifications: {q02['classification']}")

    print("[Q0.3] Evaluating ve functional load...")
    q03 = eval_with_ve_ablated(
        model, dataloader, has_ve_fn, eval_fn,
        n_batches=ablation_batches,
    )
    print(f"  Baseline val_bpb:  {q03['baseline_bpb']:.4f}")
    print(f"  Ablated val_bpb:   {q03['ablated_bpb']:.4f}")
    print(f"  Delta:             {q03['delta_bpb']:+.4f} ({q03['relative_delta']:+.2%})")

    combined = {"Q0.1_spike_channels": q01, "Q0.2_bos_stability": q02, "Q0.3_ve_ablation": q03}

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # channel_magnitudes tensors are already lists
        with open(path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"Results written to {path}")

    return combined
