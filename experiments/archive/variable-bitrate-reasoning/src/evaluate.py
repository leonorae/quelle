"""evaluate.py — Evaluation and baseline comparisons.

Usage (from the experiment root directory):
    python -m src.evaluate --config configs/default.yaml \\
                           --checkpoint outputs/checkpoints/model_final.pth
"""

import argparse
import collections
import copy

import numpy as np
import torch
import yaml

from .data import (
    ArithmeticDataset,
    VOCAB_SIZE,
    EOS_ID,
    detokenize,
    tokenize,
    compute_difficulty,
)
from .model import VariableRateReasoner

DIFFICULTY_NAMES = {0: "easy", 1: "medium", 2: "hard"}


# ---------------------------------------------------------------------------
# Greedy generation
# ---------------------------------------------------------------------------

def generate_answer(model, problem_ids: list[int], max_gen: int = 20, device="cpu") -> str:
    """Greedily decode answer tokens given a list of problem token ids."""
    model.eval()
    with torch.no_grad():
        ids = list(problem_ids)
        for _ in range(max_gen):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(x)                     # (1, seq_len, vocab_size)
            next_id = int(logits[0, -1].argmax().item())
            if next_id == EOS_ID:
                break
            ids.append(next_id)
    return detokenize(ids[len(problem_ids):])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, dataset, device, max_gen=20):
    """Greedy-decode the whole dataset; return accuracy and compression stats.

    Returns:
        acc:        dict[difficulty → float]
        avg_lambda: dict[difficulty → float]
        avg_conc:   dict[difficulty → float]
        correct:    dict[difficulty → int]
        total:      dict[difficulty → int]
    """
    correct = collections.defaultdict(int)
    total = collections.defaultdict(int)
    lambdas_by_diff = collections.defaultdict(list)
    concs_by_diff = collections.defaultdict(list)

    model.eval()
    for problem, answer, depth in dataset.examples:
        prob_ids = tokenize(problem)
        difficulty = compute_difficulty(depth)

        pred = generate_answer(model, prob_ids, max_gen=max_gen, device=device)
        correct[difficulty] += int(pred.strip() == answer.strip())
        total[difficulty] += 1

        with torch.no_grad():
            x = torch.tensor([prob_ids], dtype=torch.long, device=device)
            _, stats = model(x, return_stats=True)
        for s in stats:
            lambdas_by_diff[difficulty].append(s["lambda"])
            concs_by_diff[difficulty].append(s["concentration"])

    acc = {d: correct[d] / total[d] if total[d] else 0.0 for d in total}
    avg_lambda = {d: float(np.mean(v)) for d, v in lambdas_by_diff.items()}
    avg_conc = {d: float(np.mean(v)) for d, v in concs_by_diff.items()}
    return acc, avg_lambda, avg_conc, correct, total


def evaluate_baseline(dataset, fixed_lambda, trained_model, device, max_gen=20):
    """Evaluate with trained weights but a constant (or random) compression rate.

    fixed_lambda: float in [0,1], or None for random uniform per step.
    """
    model = copy.deepcopy(trained_model)

    # Override each compression head to return a constant lambda.
    for head in model.compression_heads:
        def _fixed_forward(h, _lam=fixed_lambda, _dev=device):
            B = h.size(0)
            if _lam is None:
                lam = torch.rand(B, device=_dev)
            else:
                lam = torch.full((B,), _lam, device=_dev)
            conc = torch.zeros(B, device=_dev)
            return lam, conc
        head.forward = _fixed_forward

    acc, _, _, _, _ = evaluate_model(model, dataset, device, max_gen=max_gen)
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-gen", type=int, default=20)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VariableRateReasoner(
        vocab_size=VOCAB_SIZE,
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded: {args.checkpoint}")

    test_ds = ArithmeticDataset(
        num_examples=cfg["data"]["num_test"],
        min_depth=cfg["data"]["min_depth"],
        max_depth=cfg["data"]["max_depth"],
        number_range=tuple(cfg["data"]["number_range"]),
        seed=cfg["seed"] + 999,
    )

    # --- Adaptive model ---
    print("\n=== Adaptive model ===")
    acc, avg_lambda, avg_conc, correct, total = evaluate_model(
        model, test_ds, device, max_gen=args.max_gen
    )
    for d in sorted(acc):
        print(
            f"  {DIFFICULTY_NAMES[d]:8s}  acc={acc[d]:.3f}  "
            f"λ={avg_lambda[d]:.3f}  conc={avg_conc[d]:.3f}  "
            f"({correct[d]}/{total[d]})"
        )

    # Lambda–concentration correlation
    all_lambdas, all_concs = [], []
    model.eval()
    for problem, _, _ in test_ds.examples:
        with torch.no_grad():
            x = torch.tensor([tokenize(problem)], dtype=torch.long, device=device)
            _, stats = model(x, return_stats=True)
        for s in stats:
            all_lambdas.append(s["lambda"])
            all_concs.append(s["concentration"])

    if len(all_lambdas) > 1:
        r = float(np.corrcoef(all_lambdas, all_concs)[0, 1])
        print(f"\n  λ–conc correlation r = {r:.4f}  {'✓' if r < -0.3 else '✗'} (target r < -0.3)")

    # --- Baselines ---
    print("\n=== Baselines (trained weights, overridden λ) ===")
    for bl in cfg["eval"]["baselines"]:
        fixed = bl["lambda"]
        bl_acc = evaluate_baseline(test_ds, fixed, model, device, max_gen=args.max_gen)
        parts = "  ".join(
            f"{DIFFICULTY_NAMES[d]}={bl_acc[d]:.3f}" for d in sorted(bl_acc)
        )
        print(f"  {bl['name']:20s}  {parts}")

    # Adaptive vs best baseline on hard problems
    hard_adaptive = acc.get(2, float("nan"))
    print(f"\n  Adaptive hard acc: {hard_adaptive:.3f}")


if __name__ == "__main__":
    main()
