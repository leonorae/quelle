"""
Phase 0 runner script — Multi-Timescale Value Embedding (VVVVVV)

Usage
-----
    python run_phase0.py --checkpoint path/to/ckpt.pt --data_dir path/to/data

The nanochat package must be importable (add it to PYTHONPATH or run from a
directory where nanochat/ is a sibling). Example:

    PYTHONPATH=/path/to/nanochat python run_phase0.py --checkpoint ...

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
    p.add_argument("--checkpoint", required=True, help="Path to nanochat checkpoint (.pt)")
    p.add_argument("--data_dir", required=True, help="Path to tokenised data directory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--spike_batches", type=int, default=50)
    p.add_argument("--bos_batches", type=int, default=20)
    p.add_argument("--ablation_batches", type=int, default=None)
    p.add_argument("--output", default=str(
        Path(__file__).parent.parent / "outputs" / "phase0_results.json"
    ))
    return p.parse_args()


def make_eval_fn(device, max_batches=100):
    """
    Returns an eval_fn compatible with eval_with_ve_ablated.
    Computes mean cross-entropy loss and converts to bpb (bits per byte).
    Assumes GPT-2 tokeniser (byte-pair; bpb ≈ loss / log(2)).
    """
    def eval_fn(model, dataloader, n_batches=None):
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        limit = n_batches if n_batches is not None else max_batches
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                if i >= limit:
                    break
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                B, T = x.shape
                total_loss += loss.item() * B * T
                total_tokens += B * T
        mean_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
        bpb = mean_loss / math.log(2)
        return bpb
    return eval_fn


def main():
    args = parse_args()

    # --- Import nanochat ---
    try:
        from nanochat.gpt import GPT, GPTConfig, has_ve  # type: ignore[import]
    except ImportError:
        print(
            "ERROR: Could not import nanochat. "
            "Set PYTHONPATH to the nanochat repo root and retry.\n"
            "  e.g.  PYTHONPATH=/path/to/nanochat python run_phase0.py ...",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    model.eval()
    print(f"Model loaded. n_layer={config.n_layer}, n_embd={config.n_embd}")

    # --- Build dataloader ---
    # nanochat uses a simple binary-memmapped DataLoader.
    # Adjust the import path if nanochat's dataloader lives elsewhere.
    try:
        from nanochat.data import get_dataloader  # type: ignore[import]
        val_loader = get_dataloader(
            data_dir=args.data_dir,
            split="val",
            batch_size=ckpt.get("config", {}).get("batch_size", 4),
            seq_len=config.block_size,
        )
    except ImportError:
        # Fallback: minimal DataLoader using numpy memmap (common nanochat pattern)
        import numpy as np

        class _MinimalLoader:
            def __init__(self, data_dir, seq_len, batch_size=4):
                import os
                val_bin = Path(data_dir) / "val.bin"
                if not val_bin.exists():
                    raise FileNotFoundError(f"val.bin not found at {val_bin}")
                self.data = np.memmap(val_bin, dtype=np.uint16, mode="r")
                self.seq_len = seq_len
                self.batch_size = batch_size

            def __iter__(self):
                B, T = self.batch_size, self.seq_len
                n = len(self.data)
                pos = 0
                while pos + B * T + 1 <= n:
                    x = torch.tensor(
                        self.data[pos: pos + B * T].reshape(B, T).astype("int64"),
                        dtype=torch.long,
                    )
                    y = torch.tensor(
                        self.data[pos + 1: pos + B * T + 1].reshape(B, T).astype("int64"),
                        dtype=torch.long,
                    )
                    yield x, y
                    pos += B * T

        val_loader = _MinimalLoader(args.data_dir, config.block_size)
        print("Warning: using minimal fallback DataLoader (nanochat.data not found).")

    # --- Run Phase 0 ---
    from phase0_diagnostics import run_phase0

    eval_fn = make_eval_fn(args.device)
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
