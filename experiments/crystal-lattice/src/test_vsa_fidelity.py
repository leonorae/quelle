"""VSA Encoding Fidelity Test for Crystal Lattice.

Tests whether HRR encoding in vsa_lattice.py captures molecular topology
well enough to distinguish linear chains from macrocyclic rings.

Tests:
  a) Linear separability (logistic regression on 10kD HVs)
  b) Wormhole effect (cosine similarity shift for ring closures)
  c) Projection survival (logistic regression on 256D projected HVs)
  d) Angle concentration (chains vs rings)

RDKit is NOT required -- SMILES are generated manually.
"""

from __future__ import annotations

import sys
import random
import time

import torch
import torch.nn.functional as F

# Ensure experiment src is importable
sys.path.insert(0, "/home/user/quelle/experiments/crystal-lattice/src")

from vsa_lattice import VSALattice


# ------------------------------------------------------------------
# Manual SMILES generation (no RDKit needed)
# ------------------------------------------------------------------

def generate_chain_smiles(n_carbons: int) -> str:
    """Linear alkane: CCCCC..."""
    return "C" * n_carbons


def generate_ring_smiles(ring_size: int) -> str:
    """Simple cycloalkane: C1CCC...C1"""
    if ring_size < 3:
        raise ValueError("Ring size must be >= 3")
    return "C1" + "C" * (ring_size - 2) + "C1"


def make_dataset(n_chains: int = 100, n_rings: int = 100, seed: int = 42):
    """Generate SMILES and labels (0=chain, 1=ring)."""
    rng = random.Random(seed)
    smiles_list = []
    labels = []

    # Chains: 3-20 carbons, sampled with replacement
    for _ in range(n_chains):
        n = rng.randint(3, 20)
        smiles_list.append(generate_chain_smiles(n))
        labels.append(0)

    # Rings: 3-20 atoms, sampled with replacement
    for _ in range(n_rings):
        n = rng.randint(3, 20)
        smiles_list.append(generate_ring_smiles(n))
        labels.append(1)

    return smiles_list, labels


# ------------------------------------------------------------------
# Logistic regression probe (pure torch)
# ------------------------------------------------------------------

def logistic_probe(X_train, y_train, X_test, y_test, lr=0.01, steps=200):
    """Train a logistic regression classifier and return test accuracy."""
    dim = X_train.shape[1]
    W = torch.zeros(dim, 1, dtype=X_train.dtype, device=X_train.device)
    b = torch.zeros(1, dtype=X_train.dtype, device=X_train.device)
    W.requires_grad_(True)
    b.requires_grad_(True)

    opt = torch.optim.SGD([W, b], lr=lr)
    y_tr = y_train.float().unsqueeze(1)

    for step in range(steps):
        logits = X_train @ W + b
        loss = F.binary_cross_entropy_with_logits(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = (X_test @ W + b).squeeze() > 0
        acc = (preds == y_test).float().mean().item()
    return acc


# ------------------------------------------------------------------
# Test a) Linear separability on raw 10kD HVs
# ------------------------------------------------------------------

def test_linear_separability(hvs: torch.Tensor, labels: torch.Tensor):
    """80/20 split logistic probe on raw hypervectors."""
    n = hvs.shape[0]
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    # Normalize features for stability
    hvs_norm = F.normalize(hvs, dim=1)

    acc = logistic_probe(
        hvs_norm[train_idx], labels[train_idx],
        hvs_norm[test_idx], labels[test_idx],
        lr=0.1, steps=300,
    )
    return acc


# ------------------------------------------------------------------
# Test b) Wormhole effect
# ------------------------------------------------------------------

def test_wormhole_effect(encoder: VSALattice):
    """Compare cosine similarity of closure-atom positional HVs
    in ring vs chain encodings of similar length."""
    results = []
    for size in [6, 8, 10, 12, 14, 16]:
        chain_smi = generate_chain_smiles(size)
        ring_smi = generate_ring_smiles(size)

        chain_hv = encoder.encode_smiles(chain_smi)
        ring_hv = encoder.encode_smiles(ring_smi)

        # Cosine similarity between chain and ring molecule HVs of same size
        cos_sim = F.cosine_similarity(
            chain_hv.unsqueeze(0), ring_hv.unsqueeze(0)
        ).item()

        # Also measure: how much does the ring HV correlate with CLOSURE_TAG?
        ct = encoder.closure_tag.squeeze(0)
        ring_closure_sim = F.cosine_similarity(
            ring_hv.unsqueeze(0), ct.unsqueeze(0)
        ).item()
        chain_closure_sim = F.cosine_similarity(
            chain_hv.unsqueeze(0), ct.unsqueeze(0)
        ).item()

        results.append({
            "size": size,
            "chain_ring_cos": cos_sim,
            "ring_closure_sim": ring_closure_sim,
            "chain_closure_sim": chain_closure_sim,
            "delta_closure": ring_closure_sim - chain_closure_sim,
        })
    return results


# ------------------------------------------------------------------
# Test c) Projection survival (10kD -> 256D)
# ------------------------------------------------------------------

def test_projection_survival(encoder: VSALattice, smiles_list, labels_tensor):
    """Logistic probe on projected 256D vectors."""
    with torch.no_grad():
        projected = encoder.encode_and_project_batch(smiles_list)

    n = projected.shape[0]
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    proj_norm = F.normalize(projected, dim=1)

    acc = logistic_probe(
        proj_norm[train_idx], labels_tensor[train_idx],
        proj_norm[test_idx], labels_tensor[test_idx],
        lr=0.1, steps=300,
    )
    return acc


# ------------------------------------------------------------------
# Test d) Angle concentration
# ------------------------------------------------------------------

def test_angle_concentration(hvs: torch.Tensor, labels: torch.Tensor):
    """Compare mean angle concentration for chains vs rings."""
    chain_mask = labels == 0
    ring_mask = labels == 1
    chain_hvs = hvs[chain_mask]
    ring_hvs = hvs[ring_mask]

    chain_ac = VSALattice.angle_concentration(chain_hvs)
    ring_ac = VSALattice.angle_concentration(ring_hvs)
    cross_ac = VSALattice.angle_concentration(hvs)
    return chain_ac, ring_ac, cross_ac


# ==================================================================
# Main
# ==================================================================

def main():
    torch.manual_seed(42)
    random.seed(42)

    print("=" * 65)
    print("  VSA ENCODING FIDELITY TEST  --  Crystal Lattice")
    print("=" * 65)
    print()

    # --- Generate data ---
    print("[1] Generating 100 chains + 100 rings ...")
    smiles_list, labels = make_dataset(100, 100)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    print(f"    Chains: {sum(1 for l in labels if l == 0)}, "
          f"Rings: {sum(1 for l in labels if l == 1)}")

    # --- Encode ---
    print("[2] Encoding with VSALattice (dim=10000) ...")
    encoder = VSALattice(dim=10_000, proj_dim=256)
    encoder.eval()

    t0 = time.time()
    with torch.no_grad():
        hvs = encoder.encode_batch(smiles_list)
    t_enc = time.time() - t0
    print(f"    Encoded {len(smiles_list)} molecules in {t_enc:.2f}s")
    print(f"    HV shape: {hvs.shape}")
    print()

    # --- Test a) Linear separability ---
    print("[Test A] Linear Separability (logistic probe on 10kD HVs)")
    acc_raw = test_linear_separability(hvs, labels_tensor)
    print(f"    Accuracy: {acc_raw * 100:.1f}%")
    print()

    # --- Test b) Wormhole effect ---
    print("[Test B] Wormhole Effect")
    wormhole_results = test_wormhole_effect(encoder)
    print(f"    {'Size':>4}  {'Chain-Ring cos':>14}  {'Ring-CT sim':>11}  "
          f"{'Chain-CT sim':>12}  {'Delta':>8}")
    for r in wormhole_results:
        print(f"    {r['size']:>4}  {r['chain_ring_cos']:>14.4f}  "
              f"{r['ring_closure_sim']:>11.4f}  "
              f"{r['chain_closure_sim']:>12.4f}  "
              f"{r['delta_closure']:>8.4f}")
    mean_delta = sum(r["delta_closure"] for r in wormhole_results) / len(wormhole_results)
    mean_chain_ring_cos = sum(r["chain_ring_cos"] for r in wormhole_results) / len(wormhole_results)
    print(f"    Mean delta (ring - chain) closure-tag sim: {mean_delta:.4f}")
    print(f"    Mean chain-ring cosine similarity:         {mean_chain_ring_cos:.4f}")
    print()

    # --- Test c) Projection survival ---
    print("[Test C] Projection Survival (logistic probe on 256D projected HVs)")
    acc_proj = test_projection_survival(encoder, smiles_list, labels_tensor)
    print(f"    Accuracy: {acc_proj * 100:.1f}%")
    acc_drop = (acc_raw - acc_proj) * 100
    print(f"    Drop from raw: {acc_drop:+.1f} pp")
    print()

    # --- Test d) Angle concentration ---
    print("[Test D] Angle Concentration")
    chain_ac, ring_ac, cross_ac = test_angle_concentration(hvs, labels_tensor)
    print(f"    Chains (intra-class):  {chain_ac:.6f}")
    print(f"    Rings  (intra-class):  {ring_ac:.6f}")
    print(f"    All    (cross-class):  {cross_ac:.6f}")
    ac_diff = abs(chain_ac - ring_ac)
    print(f"    |chain - ring| diff:   {ac_diff:.6f}")
    print()

    # --- Summary ---
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Raw 10kD probe accuracy:        {acc_raw * 100:.1f}%")
    print(f"  Projected 256D probe accuracy:   {acc_proj * 100:.1f}%")
    print(f"  Accuracy drop:                   {acc_drop:+.1f} pp")
    print(f"  Mean wormhole delta (CT sim):    {mean_delta:.4f}")
    print(f"  Mean chain-ring cos sim:         {mean_chain_ring_cos:.4f}")
    print(f"  Angle conc. chains:              {chain_ac:.6f}")
    print(f"  Angle conc. rings:               {ring_ac:.6f}")
    print("=" * 65)

    # Return results for RESULTS.md generation
    return {
        "acc_raw": acc_raw,
        "acc_proj": acc_proj,
        "acc_drop": acc_drop,
        "mean_delta": mean_delta,
        "mean_chain_ring_cos": mean_chain_ring_cos,
        "chain_ac": chain_ac,
        "ring_ac": ring_ac,
        "cross_ac": cross_ac,
        "wormhole_results": wormhole_results,
        "t_enc": t_enc,
    }


if __name__ == "__main__":
    main()
