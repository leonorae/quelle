"""train.py — Training entry point for Variable-Bitrate Reasoning.

Usage:
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

from .data import ArithmeticDataset, VOCAB_SIZE
from .model import VariableRateReasoner


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loader(cfg: dict, split: str):
    from torch.utils.data import DataLoader

    n = cfg["data"]["num_train"] if split == "train" else cfg["data"]["num_test"]
    ds = ArithmeticDataset(
        num_examples=n,
        min_depth=cfg["data"]["min_depth"],
        max_depth=cfg["data"]["max_depth"],
        number_range=tuple(cfg["data"]["number_range"]),
        seed=cfg["training"]["seed"] if split == "train" else cfg["training"]["seed"] + 1,
    )
    return DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=(split == "train"))


def train(cfg: dict) -> None:
    set_seeds(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VariableRateReasoner(
        vocab_size=VOCAB_SIZE,
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        dropout=cfg["model"]["dropout"],
        alpha_init=cfg["compression"]["alpha_init"],
        beta_init=cfg["compression"]["beta_init"],
    ).to(device)

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
            f, fieldnames=["step", "epoch", "total", "lm", "future", "curvature", "avg_lambda"]
        )
        writer.writeheader()

        step = 0
        for epoch in range(cfg["training"]["epochs"]):
            for input_ids, target_ids, _ in loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                logits, stats = model(input_ids, return_stats=True)
                lm_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), target_ids.view(-1)
                )

                future_loss = torch.tensor(0.0, device=device)
                curvature = torch.tensor(0.0, device=device)
                lambdas = []
                for s in stats:
                    future_loss += F.mse_loss(s["h_pred"], s["h_next"].detach())
                    curvature += (
                        (s["h_next"] - s["h"]) ** 2
                    ).mean() * (1 - s["concentration"])
                    lambdas.append(s["lambda"])

                total_loss = lw * lm_loss + fw * future_loss + cw * curvature

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if step % log_every == 0:
                    row = {
                        "step": step,
                        "epoch": epoch,
                        "total": total_loss.item(),
                        "lm": lm_loss.item(),
                        "future": future_loss.item(),
                        "curvature": curvature.item(),
                        "avg_lambda": float(np.mean(lambdas)),
                    }
                    writer.writerow(row)
                    print(
                        f"[{epoch}/{cfg['training']['epochs']}] step={step} "
                        f"loss={total_loss.item():.4f} lm={lm_loss.item():.4f} "
                        f"λ={row['avg_lambda']:.3f}"
                    )

                step += 1

    ckpt_dir = pathlib.Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model_final.pth")
    print(f"Saved checkpoint to {ckpt_dir / 'model_final.pth'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
