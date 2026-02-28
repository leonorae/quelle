"""evaluate.py — Evaluation and baseline comparisons.

Usage:
    python -m src.evaluate --config configs/default.yaml --checkpoint outputs/checkpoints/model_final.pth
"""

# Stub — implement after training is working.

import argparse
import pathlib

import torch
import yaml

from .data import ArithmeticDataset, VOCAB_SIZE
from .model import VariableRateReasoner


DIFFICULTY_NAMES = {0: "easy", 1: "medium", 2: "hard"}


def evaluate_model(model, loader, device):
    """Return dict of accuracy by difficulty and compression stats."""
    # TODO: implement
    raise NotImplementedError


def run_baseline(cfg, baseline_cfg, loader, device):
    """Run a fixed-lambda baseline and return accuracy dict."""
    # TODO: implement
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VariableRateReasoner(
        vocab_size=VOCAB_SIZE,
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # TODO: build test loader, run evaluation, print results, write to RESULTS.md


if __name__ == "__main__":
    main()
