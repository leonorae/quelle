"""
Phase 0 runner script — Multi-Timescale Value Embedding (VVVVVV)

Usage
-----
Run from the repo root via the wrapper script:

    bash experiments/VVVVVV/src/run_phase0.sh

Or manually (NANOCHAT_BASE_DIR and PYTHONPATH must be set):

    export NANOCHAT_BASE_DIR=experiments/VVVVVV/outputs/nanochat_base
    export PYTHONPATH=/path/to/quelle/nanochat:/path/to/quelle/experiments/VVVVVV/src
    python experiments/VVVVVV/src/run_phase0.py \\
        --checkpoint-dir $NANOCHAT_BASE_DIR/base_checkpoints/d12

Checkpoint format
-----------------
nanochat saves checkpoints as:
    <checkpoint-dir>/model_<step:06d>.pt   -- model state dict (flat)
    <checkpoint-dir>/meta_<step:06d>.json  -- metadata including model_config

This script uses nanochat.checkpoint_manager.build_model() which handles both.
If --step is omitted, the latest available step is used.

Outputs a JSON file to experiments/VVVVVV/outputs/phase0_results.json
and prints a summary to stdout.
"""

import argparse
import math
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 diagnostics for VVVVVV")
    p.add_argument(
        "--checkpoint-dir", required=True,
        help="Path to nanochat checkpoint directory (e.g. $NANOCHAT_BASE_DIR/base_checkpoints/d12)",
    )
    p.add_argument(
        "--step", type=int, default=None,
        help="Checkpoint step to load (default: latest)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--spike-batches", type=int, default=50)
    p.add_argument("--bos-batches", type=int, default=20)
    p.add_argument("--ablation-batches", type=int, default=None)
    p.add_argument("--output", default=str(
        Path(__file__).parent.parent / "outputs" / "phase0_results.json"
    ))
    return p.parse_args()


def make_eval_fn(device, token_bytes, max_batches=100):
    """
    Returns an eval_fn compatible with eval_with_ve_ablated.
    Computes bits-per-byte using nanochat's byte-weighted bpb formula.

    token_bytes: 1-D tensor of shape (vocab_size,), bytes per token id
                 (0 for special tokens that should not be counted).
    """
    token_bytes = token_bytes.to(device)

    def eval_fn(model, dataloader, n_batches=None):
        model.eval()
        total_nats = 0.0
        total_bytes = 0
        limit = n_batches if n_batches is not None else max_batches
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                if i >= limit:
                    break
                x, y = x.to(device), y.to(device)
                loss2d = model(x, y, loss_reduction="none")  # (B, T)
                loss2d = loss2d.view(-1)
                y_flat = y.view(-1)
                nb = token_bytes[y_flat]
                total_nats += (loss2d * (nb > 0)).sum().item()
                total_bytes += nb.sum().item()
        if total_bytes == 0:
            return float("nan")
        return total_nats / (math.log(2) * total_bytes)

    return eval_fn


def main():
    args = parse_args()

    # --- Import nanochat ---
    try:
        from nanochat.gpt import has_ve  # type: ignore[import]
        from nanochat.checkpoint_manager import build_model, find_last_step  # type: ignore[import]
        from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit  # type: ignore[import]
        from nanochat.tokenizer import get_token_bytes  # type: ignore[import]
    except ImportError as e:
        print(
            f"ERROR: Could not import nanochat ({e}).\n"
            "Use the run_phase0.sh wrapper, which sets PYTHONPATH automatically.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load model ---
    device = torch.device(args.device)
    checkpoint_dir = args.checkpoint_dir
    step = args.step if args.step is not None else find_last_step(checkpoint_dir)
    print(f"Loading checkpoint: {checkpoint_dir}  step={step}")
    model, tokenizer, meta = build_model(checkpoint_dir, step, device, phase="eval")
    config = model.config
    print(f"Model loaded. n_layer={config.n_layer}, n_embd={config.n_embd}, step={step}")

    # --- Build validation dataloader ---
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, B=args.batch_size, T=config.sequence_len, split="val",
        device=args.device,
    )

    # --- Run Phase 0 ---
    sys.path.insert(0, str(Path(__file__).parent))
    from phase0_diagnostics import run_phase0  # type: ignore[import]

    token_bytes = get_token_bytes().to(device)
    eval_fn = make_eval_fn(device, token_bytes)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = run_phase0(
        model=model,
        dataloader=val_loader,
        has_ve_fn=has_ve,
        eval_fn=eval_fn,
        output_path=args.output,
        spike_batches=args.spike_batches,
        bos_batches=args.bos_batches,
        ablation_batches=args.ablation_batches,
    )

    print("\n=== Summary ===")
    q01 = results["Q0.1_spike_channels"]
    q02 = results["Q0.2_bos_stability"]
    q03 = results["Q0.3_ve_ablation"]

    print("\nQ0.1 — Spike channel overlap:")
    for layer, overlap in q01["overlap_fraction"].items():
        interp = q01["interpretation"][layer]
        print(f"  layer {layer}: {overlap:.1%} overlap → {interp}")

    print("\nQ0.2 — BOS residual stability:")
    for layer, cls in q02["classification"].items():
        w = q02["within_batch_cosine"][layer]
        c = q02["cross_batch_cosine"][layer]
        print(f"  layer {layer}: within={w:.3f}, cross={c:.3f} → {cls}")

    print(f"\nQ0.3 — ve functional load:")
    print(f"  baseline bpb:  {q03['baseline_bpb']:.4f}")
    print(f"  ablated bpb:   {q03['ablated_bpb']:.4f}")
    print(f"  delta:         {q03['delta_bpb']:+.4f} ({q03['relative_delta']:+.2%})")

    print(f"\nFull results saved to: {args.output}")

    # --- Guidance on next steps ---
    print("\n=== Implications for Phase 1 / Architecture ===")
    for layer, interp in q01["interpretation"].items():
        if interp == "near-noise":
            print(f"  [Q0.1 layer {layer}] Gate reads near-noise channels. "
                  "Consider learned projection gate (§6.1) — flag in DECISIONS.md.")
        elif interp == "informative":
            print(f"  [Q0.1 layer {layer}] Gate reads informative channels. "
                  "Fixed [:32] slice may be sufficient.")

    any_sink = any(v == "pure-sink" for v in q02["classification"].values())
    any_doc = any(v == "document-signal" for v in q02["classification"].values())
    if any_sink:
        print("  [Q0.2] BOS is a pure sink at some layers. "
              "BOS-conditioned table (§6.2) will not carry document signal there.")
    if any_doc:
        print("  [Q0.2] BOS carries document signal at some layers. "
              "BOS-conditioned table (§6.2) is viable — proceed after Phase 1.")

    if q03["delta_bpb"] < 0.001:
        print("  [Q0.3] WARNING: ve ablation delta is very small. "
              "Either ve is not load-bearing at this scale, or the eval window is too short.")


if __name__ == "__main__":
    main()
