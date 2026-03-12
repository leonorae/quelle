"""Phase 0: Activation caching.

Extract per-layer hidden states and top-k output logprobs from a pretrained
model for a set of prompts. Store as safetensors with JSON metadata sidecars.

Usage:
    python src/cache_activations.py --config configs/default.yaml --prompts prompts/benchmarks.jsonl

Output layout (see DECISIONS.md D1):
    data/activations/{model_slug}/
        batch_000.safetensors     # layer_0..layer_L (last-token), logits_indices, logits_values
        batch_000_full.safetensors  # (optional) full-sequence hidden states
        metadata.json             # [{prompt_id, category, group_id, token_count}, ...]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompts(path: str) -> list[dict[str, Any]]:
    """Load JSONL prompt file. See prompts/README.md for format."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def slugify_model_name(name: str) -> str:
    """EleutherAI/pythia-410m -> pythia-410m"""
    return name.split("/")[-1].lower()


def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "float16",
    device: str = "auto",
) -> tuple:
    """Load model with output_hidden_states=True."""
    torch_dtype = getattr(torch, dtype, torch.float16)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        output_hidden_states=True,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def extract_activations(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    top_k: int = 100,
    precision_logits: str = "float32",
) -> dict[str, torch.Tensor]:
    """Run forward pass, extract last-token hidden states and top-k logprobs.

    Returns dict with keys:
        layer_{i}: (batch, d_hidden) float16 — last-token hidden state per layer
        logits_values: (batch, top_k) float32 — top-k log-probabilities
        logits_indices: (batch, top_k) int64 — vocabulary indices for top-k
        logits_residual: (batch,) float32 — log(1 - sum(top_k_probs))
        token_counts: (batch,) int64 — number of tokens per prompt
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # tuple of (batch, seq, d_hidden)
    logits = outputs.logits                # (batch, seq, vocab)

    # Find last non-padding token position per sequence
    attention_mask = inputs["attention_mask"]  # (batch, seq)
    # last valid index = sum of mask - 1
    last_positions = attention_mask.sum(dim=1) - 1  # (batch,)
    batch_indices = torch.arange(len(texts), device=device)

    result = {}

    # Extract last-token hidden state per layer
    for layer_idx, hs in enumerate(hidden_states):
        # hs: (batch, seq, d_hidden)
        last_token_hs = hs[batch_indices, last_positions]  # (batch, d_hidden)
        result[f"layer_{layer_idx}"] = last_token_hs.half().cpu()

    # Extract top-k logprobs from last token position
    last_logits = logits[batch_indices, last_positions]  # (batch, vocab)
    logits_dtype = getattr(torch, precision_logits, torch.float32)
    last_logits = last_logits.to(logits_dtype)
    log_probs = torch.log_softmax(last_logits, dim=-1)

    top_values, top_indices = torch.topk(log_probs, k=top_k, dim=-1)
    result["logits_values"] = top_values.cpu()    # (batch, top_k) float32
    result["logits_indices"] = top_indices.cpu()   # (batch, top_k) int64

    # Residual mass: log(1 - sum(exp(top_k_logprobs)))
    top_probs_sum = top_values.exp().sum(dim=-1)   # (batch,)
    # Clamp to avoid log(0) when top-k covers full mass
    residual = torch.log(torch.clamp(1.0 - top_probs_sum, min=1e-10))
    result["logits_residual"] = residual.cpu()     # (batch,)

    result["token_counts"] = last_positions.cpu() + 1  # (batch,)

    return result


def extract_full_sequence(
    model,
    tokenizer,
    texts: list[str],
    device: str,
) -> dict[str, torch.Tensor]:
    """Extract full-sequence hidden states (all token positions).

    Returns dict with keys:
        layer_{i}: (batch, max_seq, d_hidden) float16
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    result = {}
    for layer_idx, hs in enumerate(outputs.hidden_states):
        result[f"layer_{layer_idx}"] = hs.half().cpu()

    return result


def cache_prompts(
    model,
    tokenizer,
    prompts: list[dict[str, Any]],
    output_dir: Path,
    device: str,
    config: dict[str, Any],
) -> None:
    """Cache activations for all prompts, writing batched safetensors files."""
    cache_cfg = config["cache"]
    batch_size = cache_cfg["batch_size"]
    prompts_per_file = cache_cfg["prompts_per_file"]
    top_k = cache_cfg["top_k_logits"]
    store_full = cache_cfg.get("store_full_sequence", False)
    precision_logits = cache_cfg.get("precision_logits", "float32")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = []
    file_idx = 0
    file_tensors: dict[str, list[torch.Tensor]] = {}
    file_full_tensors: dict[str, list[torch.Tensor]] = {}
    file_metadata: list[dict] = []

    num_batches = math.ceil(len(prompts) / batch_size)

    for batch_start in tqdm(range(0, len(prompts), batch_size), total=num_batches, desc="Caching"):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        texts = [p["text"] for p in batch_prompts]

        activations = extract_activations(
            model, tokenizer, texts, device,
            top_k=top_k, precision_logits=precision_logits,
        )

        # Accumulate into current file batch
        for key, tensor in activations.items():
            file_tensors.setdefault(key, []).append(tensor)

        if store_full:
            full_acts = extract_full_sequence(model, tokenizer, texts, device)
            for key, tensor in full_acts.items():
                file_full_tensors.setdefault(key, []).append(tensor)

        for i, p in enumerate(batch_prompts):
            file_metadata.append({
                "prompt_id": p["prompt_id"],
                "category": p.get("category", "unknown"),
                "group_id": p.get("group_id"),
                "token_count": int(activations["token_counts"][i].item()),
            })

        # Write file when we've accumulated enough prompts
        prompts_in_file = sum(t.shape[0] for t in file_tensors.get("layer_0", []))
        if prompts_in_file >= prompts_per_file or batch_start + batch_size >= len(prompts):
            # Concatenate accumulated tensors
            merged = {k: torch.cat(v, dim=0) for k, v in file_tensors.items()}
            save_file(merged, output_dir / f"batch_{file_idx:03d}.safetensors")

            if store_full and file_full_tensors:
                merged_full = {k: torch.cat(v, dim=0) for k, v in file_full_tensors.items()}
                save_file(merged_full, output_dir / f"batch_{file_idx:03d}_full.safetensors")

            all_metadata.extend(file_metadata)
            file_tensors = {}
            file_full_tensors = {}
            file_metadata = []
            file_idx += 1

    # Write metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Cached {len(all_metadata)} prompts in {file_idx} files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Cache LLM activations for projection analysis")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--prompts", type=str, required=True, help="Path to JSONL prompt file")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    args = parser.parse_args()

    config = load_config(args.config)

    model_name = args.model or config["model"]["name"]
    device = args.device or config["model"].get("device", "auto")
    dtype = config["model"].get("dtype", "float16")

    model_slug = slugify_model_name(model_name)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["cache"]["output_dir"]) / model_slug

    print(f"Model: {model_name} ({dtype})")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts")

    model, tokenizer, device = load_model_and_tokenizer(model_name, dtype, device)
    cache_prompts(model, tokenizer, prompts, output_dir, device, config)


if __name__ == "__main__":
    main()
