"""analyze_dim_ordering.py — Soft-Mask Dimension Ordering experiment.

Tests whether VBR's soft dimension mask causes the model to sort
information by importance across dimensions.

Usage (from experiments/archive/variable-bitrate-reasoning/):
    python -m src.analyze_dim_ordering
"""

import collections
import pathlib
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .data import (
    ArithmeticDataset,
    VOCAB_SIZE,
    PAD_ID,
    collate_fn,
    tokenize,
    compute_difficulty,
)
from .model import VariableRateReasoner


DIFFICULTY_NAMES = {0: "easy", 1: "medium", 2: "hard"}


# ---------------------------------------------------------------------------
# Training (trimmed version of train.py, inlined for self-containment)
# ---------------------------------------------------------------------------

def train_model(cfg: dict):
    """Train the VBR model and return it (on CPU)."""
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device = torch.device("cpu")
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

    lw = cfg["training"]["lm_loss_weight"]
    fw = cfg["training"]["future_loss_weight"]
    cw = cfg["training"]["curvature_loss_weight"]

    from torch.utils.data import DataLoader

    ds = ArithmeticDataset(
        num_examples=cfg["data"]["num_train"],
        min_depth=cfg["data"]["min_depth"],
        max_depth=cfg["data"]["max_depth"],
        number_range=tuple(cfg["data"]["number_range"]),
        seed=cfg["seed"],
    )
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                        shuffle=True, collate_fn=collate_fn)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for padded_ids, prob_lengths, _ in loader:
            padded_ids = padded_ids.to(device)
            prob_lengths = prob_lengths.to(device)
            B, L = padded_ids.shape

            x = padded_ids[:, :-1]
            y = padded_ids[:, 1:]

            logits, stats = model(x, return_stats=True)

            seq_len = x.size(1)
            loss_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
            for b in range(B):
                pl = int(prob_lengths[b].item()) - 1
                loss_mask[b, pl:] = True
            loss_mask &= (y != PAD_ID)

            if loss_mask.any():
                lm_loss = F.cross_entropy(logits[loss_mask], y[loss_mask])
            else:
                lm_loss = torch.tensor(0.0, device=device)

            future_loss = torch.tensor(0.0, device=device)
            curvature = torch.tensor(0.0, device=device)

            for s in stats:
                future_loss += F.mse_loss(s["h_pred"], s["h_next"].detach())
                diff_sq = ((s["h_next"] - s["h"]) ** 2).mean(dim=[1, 2])
                curvature += (diff_sq * (1.0 - s["_conc"])).mean()

            total_loss = lw * lm_loss + fw * future_loss + cw * curvature

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        avg_lambdas = []
        model.eval()
        with torch.no_grad():
            for padded_ids, prob_lengths, _ in loader:
                x = padded_ids[:, :-1]
                _, stats = model(x, return_stats=True)
                for s in stats:
                    avg_lambdas.append(s["lambda"])
                break  # just one batch for logging

        print(f"  Epoch {epoch}: loss={epoch_loss/n_batches:.4f}  "
              f"avg_lambda={np.mean(avg_lambdas):.4f}")

    return model


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_dimension_ordering(model, cfg):
    """Run the dimension ordering analysis."""
    device = torch.device("cpu")
    model.eval()

    # Create test dataset
    test_ds = ArithmeticDataset(
        num_examples=500,
        min_depth=cfg["data"]["min_depth"],
        max_depth=cfg["data"]["max_depth"],
        number_range=tuple(cfg["data"]["number_range"]),
        seed=cfg["seed"] + 999,
    )

    d_model = cfg["model"]["d_model"]
    n_layers = cfg["model"]["n_layers"]

    # Collect per-dimension activations and metadata
    # For each example, we get the hidden state at the last non-pad position
    # at each layer (before and after compression).
    print("\n=== Collecting activations ===")

    # Store: per-layer, per-dimension activations and targets
    layer_dim_acts = {l: [] for l in range(n_layers)}  # list of (d_model,) arrays
    answer_values = []
    difficulties = []

    for problem, answer, depth in test_ds.examples:
        prob_ids = tokenize(problem)
        difficulty = compute_difficulty(depth)
        try:
            ans_val = int(answer)
        except ValueError:
            ans_val = 0

        answer_values.append(ans_val)
        difficulties.append(difficulty)

        with torch.no_grad():
            x = torch.tensor([prob_ids], dtype=torch.long, device=device)
            _, stats = model(x, return_stats=True)

        for l in range(n_layers):
            # Take hidden state at last position (the "summary" position)
            h = stats[l]["h"]  # (1, seq_len, d_model)
            last_pos_act = h[0, -1, :].numpy()  # (d_model,)
            layer_dim_acts[l].append(last_pos_act)

    answer_values = np.array(answer_values, dtype=float)
    difficulties = np.array(difficulties, dtype=float)

    # Normalise answer values to [0, 1] for correlation
    ans_min, ans_max = answer_values.min(), answer_values.max()
    if ans_max > ans_min:
        answer_norm = (answer_values - ans_min) / (ans_max - ans_min)
    else:
        answer_norm = np.zeros_like(answer_values)

    # ---------------------------------------------------------------------------
    # (a) Per-dimension correlation with answer value and difficulty
    # ---------------------------------------------------------------------------
    print("\n=== Dimension-wise correlation analysis ===")

    results = {}

    for l in range(n_layers):
        acts = np.array(layer_dim_acts[l])  # (N, d_model)
        corr_answer = np.zeros(d_model)
        corr_difficulty = np.zeros(d_model)

        for d in range(d_model):
            dim_vals = acts[:, d]
            if np.std(dim_vals) > 1e-10:
                corr_answer[d] = abs(np.corrcoef(dim_vals, answer_norm)[0, 1])
                corr_difficulty[d] = abs(np.corrcoef(dim_vals, difficulties)[0, 1])
            else:
                corr_answer[d] = 0.0
                corr_difficulty[d] = 0.0

        results[l] = {
            "corr_answer": corr_answer,
            "corr_difficulty": corr_difficulty,
        }

        print(f"\n  Layer {l}:")
        print(f"    Answer corr   — first 32 dims: {corr_answer[:32].mean():.4f}  "
              f"last 32 dims: {corr_answer[-32:].mean():.4f}")
        print(f"    Difficulty corr — first 32 dims: {corr_difficulty[:32].mean():.4f}  "
              f"last 32 dims: {corr_difficulty[-32:].mean():.4f}")

    # ---------------------------------------------------------------------------
    # (b) Are first-k dims more informative than last d-k?
    # ---------------------------------------------------------------------------
    print("\n=== First-k vs last-k dimension informativeness ===")
    k_values = [16, 32, 48, 64]
    ordering_results = {}

    for l in range(n_layers):
        corr = results[l]["corr_answer"]
        layer_ordering = {}
        for k in k_values:
            first_k = corr[:k].mean()
            last_k = corr[-k:].mean()
            ratio = first_k / last_k if last_k > 1e-10 else float("inf")
            layer_ordering[k] = {"first_k": first_k, "last_k": last_k, "ratio": ratio}
            print(f"  Layer {l}, k={k:3d}: first_k_corr={first_k:.4f}  "
                  f"last_k_corr={last_k:.4f}  ratio={ratio:.2f}")
        ordering_results[l] = layer_ordering

    # ---------------------------------------------------------------------------
    # (c) Correlation profile (plot-ready data)
    # ---------------------------------------------------------------------------
    print("\n=== Correlation profile (dim_index → correlation) ===")
    profile_data = {}
    for l in range(n_layers):
        profile_data[l] = {
            "dim_index": list(range(d_model)),
            "corr_answer": results[l]["corr_answer"].tolist(),
            "corr_difficulty": results[l]["corr_difficulty"].tolist(),
        }
        # Print summary: correlation in 8 equal-sized bins
        bin_size = d_model // 8
        bins = []
        for i in range(8):
            start = i * bin_size
            end = start + bin_size
            bins.append(results[l]["corr_answer"][start:end].mean())
        print(f"  Layer {l} answer-corr bins (8 bins of {bin_size} dims): "
              f"{' '.join(f'{b:.3f}' for b in bins)}")

    # ---------------------------------------------------------------------------
    # (d) Learned alpha and beta values
    # ---------------------------------------------------------------------------
    print("\n=== Learned compression parameters ===")
    alpha_beta = []
    for l, head in enumerate(model.compression_heads):
        alpha = head.alpha.item()
        beta = head.beta.item()
        alpha_beta.append({"layer": l, "alpha": alpha, "beta": beta})
        print(f"  Layer {l}: alpha={alpha:.4f}  beta={beta:.4f}")

    # ---------------------------------------------------------------------------
    # (e) Mean lambda across difficulty levels
    # ---------------------------------------------------------------------------
    print("\n=== Lambda by difficulty level ===")
    lambda_by_diff = {0: [], 1: [], 2: []}

    for problem, answer, depth in test_ds.examples:
        prob_ids = tokenize(problem)
        difficulty = compute_difficulty(depth)
        with torch.no_grad():
            x = torch.tensor([prob_ids], dtype=torch.long, device=device)
            _, stats = model(x, return_stats=True)
        for s in stats:
            lambda_by_diff[difficulty].append(s["lambda"])

    lambda_summary = {}
    for d in sorted(lambda_by_diff):
        vals = lambda_by_diff[d]
        if vals:
            mean_l = np.mean(vals)
            std_l = np.std(vals)
            lambda_summary[d] = {"mean": mean_l, "std": std_l, "n": len(vals)}
            print(f"  {DIFFICULTY_NAMES[d]:8s}  mean_lambda={mean_l:.4f}  "
                  f"std={std_l:.4f}  (n={len(vals)})")

    # ---------------------------------------------------------------------------
    # Compute Spearman rank correlation of dim index vs informativeness
    # ---------------------------------------------------------------------------
    print("\n=== Spearman rank-order test ===")
    from scipy.stats import spearmanr  # type: ignore
    spearman_results = {}
    for l in range(n_layers):
        corr_vals = results[l]["corr_answer"]
        dim_indices = np.arange(d_model)
        rho, pval = spearmanr(dim_indices, corr_vals)
        spearman_results[l] = {"rho": rho, "p": pval}
        direction = "decreasing (early dims more important)" if rho < 0 else "increasing or flat"
        sig = "significant" if pval < 0.05 else "not significant"
        print(f"  Layer {l}: rho={rho:.4f}  p={pval:.4f}  ({direction}, {sig})")

    return {
        "results": results,
        "ordering_results": ordering_results,
        "profile_data": profile_data,
        "alpha_beta": alpha_beta,
        "lambda_summary": lambda_summary,
        "spearman": spearman_results,
    }


def main():
    # Load default config
    cfg_path = pathlib.Path("configs/default.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Override for speed
    cfg["data"]["num_train"] = 2000
    cfg["training"]["epochs"] = 3
    print("=" * 60)
    print("SOFT-MASK DIMENSION ORDERING EXPERIMENT")
    print("=" * 60)
    print(f"Config overrides: num_train=2000, epochs=3")
    print(f"d_model={cfg['model']['d_model']}, n_layers={cfg['model']['n_layers']}")

    # Step 1: Train
    print("\n" + "=" * 60)
    print("PHASE 1: Training VBR model")
    print("=" * 60)
    model = train_model(cfg)

    # Step 2: Analyze
    print("\n" + "=" * 60)
    print("PHASE 2: Dimension ordering analysis")
    print("=" * 60)
    analysis = analyze_dimension_ordering(model, cfg)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    return analysis


if __name__ == "__main__":
    main()
