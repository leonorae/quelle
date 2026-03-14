"""visualize.py — PCA / UMAP and scatter plots for geometric analysis.

Usage:
    python -m src.visualize --config configs/default.yaml --checkpoint outputs/checkpoints/model_final.pth
"""

# Stub — implement after evaluation is working.

import argparse
import pathlib

import yaml


def plot_lambda_vs_concentration(stats, output_path):
    """Scatter plot of lambda vs. concentration, colored by difficulty."""
    # TODO: implement with matplotlib
    raise NotImplementedError


def plot_umap(hidden_states, labels, output_path):
    """UMAP projection of hidden states colored by label."""
    # TODO: implement with umap-learn and matplotlib
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    plot_dir = pathlib.Path(cfg["output"]["plot_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)

    # TODO: load model, extract hidden states from small sample, produce plots


if __name__ == "__main__":
    main()
