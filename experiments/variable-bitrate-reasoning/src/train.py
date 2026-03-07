"""train.py — Training entry point for Variable-Bitrate Reasoning.

Usage (from the experiment root directory):
    python -m src.train --config configs/default.yaml
"""

import argparse
import csv
import pathlib
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .data import ArithmeticDataset, VOCAB_SIZE, PAD_ID, collate_fn
from .model import VariableRateReasoner


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loader(cfg: dict, split: str):
    from torch.utils.data import DataLoader

    n = cfg["data"]["num_train"] if split == "train" else cfg["data"]["num_test"]
    # Top-level seed for training, seed+1 for test (distinct data)
    seed = cfg["seed"] if split == "train" else cfg["seed"] + 1
    ds = ArithmeticDataset(
        num_examples=n,
        min_depth=cfg["data"]["min_depth"],
        max_depth=cfg["data"]["max_depth"],
        number_range=tuple(cfg["data"]["number_range"]),
        seed=seed,
    )
    return DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(split == "train"),
        collate_fn=collate_fn,
    )


def train(cfg: dict) -> None:
    set_seeds(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = VariableRateReasoner(
        vocab_size=VOCAB_SIZE,
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]["dropout"],
        alpha_init=cfg["compression"]["alpha_init"],
        beta_init=cfg["compression"]["beta_init"],
        temperature=cfg["compression"]["temperature"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    log_dir = pathlib.Path(cfg["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training_log.csv"

    lw = cfg["training"]["lm_loss_weight"]
    fw = cfg["training"]["future_loss_weight"]
    cw = cfg["training"]["curvature_loss_weight"]
    log_every = cfg["training"]["log_every"]

    loader = build_loader(cfg, "train")

    with open(log_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "epoch", "total", "lm", "future", "curvature", "avg_lambda"],
        )
        writer.writeheader()

        step = 0
        for epoch in range(cfg["training"]["epochs"]):
            model.train()
            for padded_ids, prob_lengths, _ in loader:
                # padded_ids:   (B, max_len)  full sequence (problem + answer + EOS)
                # prob_lengths: (B,)          token count of the problem part
                padded_ids = padded_ids.to(device)
                prob_lengths = prob_lengths.to(device)

                B, L = padded_ids.shape

                # Standard LM shift: input is all-but-last, target is all-but-first.
                x = padded_ids[:, :-1]   # (B, L-1)
                y = padded_ids[:, 1:]    # (B, L-1)  next-token targets

                logits, stats = model(x, return_stats=True)  # (B, L-1, V)

                # LM loss only on answer+EOS positions (everything after the problem).
                # Position prob_lengths[b]-1 in `x` is the last problem token;
                # its logit predicts the first answer token in `y`.
                seq_len = x.size(1)
                loss_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
                for b in range(B):
                    pl = int(prob_lengths[b].item()) - 1  # last problem token index in x
                    loss_mask[b, pl:] = True
                loss_mask &= (y != PAD_ID)

                if loss_mask.any():
                    lm_loss = F.cross_entropy(logits[loss_mask], y[loss_mask])
                else:
                    lm_loss = torch.tensor(0.0, device=device)

                # --- Compression auxiliary losses ---
                future_loss = torch.tensor(0.0, device=device)
                curvature = torch.tensor(0.0, device=device)
                lambdas = []

                for s in stats:
                    # Future prediction: predict next-layer h from compressed z.
                    # stop-gradient on h_next (DSD-style)
                    future_loss += F.mse_loss(s["h_pred"], s["h_next"].detach())

                    # Curvature penalty: penalise large representation changes when
                    # concentration is high (model is geometrically confident).
                    diff_sq = ((s["h_next"] - s["h"]) ** 2).mean(dim=[1, 2])  # (B,)
                    curvature += (diff_sq * (1.0 - s["_conc"])).mean()

                    lambdas.append(s["lambda"])

                total_loss = lw * lm_loss + fw * future_loss + cw * curvature

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if step % log_every == 0:
                    row = {
                        "step": step,
                        "epoch": epoch,
                        "total": round(total_loss.item(), 6),
                        "lm": round(lm_loss.item(), 6),
                        "future": round(future_loss.item(), 6),
                        "curvature": round(curvature.item(), 6),
                        "avg_lambda": round(float(np.mean(lambdas)), 4),
                    }
                    writer.writerow(row)
                    f.flush()
                    print(
                        f"[epoch {epoch}  step {step:>6}]  "
                        f"loss={row['total']:.4f}  "
                        f"lm={row['lm']:.4f}  "
                        f"future={row['future']:.4f}  "
                        f"λ={row['avg_lambda']:.3f}"
                    )

                step += 1

    ckpt_dir = pathlib.Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model_final.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint → {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
