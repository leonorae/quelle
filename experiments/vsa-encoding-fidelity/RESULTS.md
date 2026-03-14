# Crystal Lattice: VSA Encoding Fidelity Results

**Date**: 2026-03-14
**Test script**: `src/test_vsa_fidelity.py`
**Encoder**: `VSALattice` (HRR via torchhd, dim=10,000, proj_dim=256)
**Dataset**: 100 linear alkanes (3-20 C) + 100 macrocyclic rings (3-20 C), manually generated SMILES (no RDKit)

---

## Summary Table

| Metric | Value | Interpretation |
|---|---|---|
| **Raw 10kD probe accuracy** | 57.5% | Near chance -- ring vs chain is NOT linearly separable in raw HV space |
| **Projected 256D probe accuracy** | 62.5% | Slightly better than raw (random projection adds no information; likely noise) |
| **Accuracy drop (raw -> proj)** | -5.0 pp | Projection did not degrade; raw accuracy was already near chance |
| **Mean wormhole delta (closure-tag sim)** | 0.0003 | Negligible -- the closure tag is NOT detectable via cosine similarity to the molecule HV |
| **Mean chain-ring cosine similarity** | 0.9522 | Very high -- chains and rings of the same size are nearly identical in HV space |
| **Angle concentration (chains)** | 0.7648 | |
| **Angle concentration (rings)** | 0.6722 | |
| **Angle concentration difference** | 0.0926 | Modest difference; rings are slightly more spread out |

## Wormhole Effect Detail

| Size | Chain-Ring cos | Ring-CT sim | Chain-CT sim | Delta |
|---|---|---|---|---|
| 6 | 0.9244 | -0.0187 | -0.0144 | -0.0043 |
| 8 | 0.9415 | -0.0126 | -0.0162 | +0.0036 |
| 10 | 0.9524 | -0.0168 | -0.0141 | -0.0028 |
| 12 | 0.9601 | -0.0064 | -0.0089 | +0.0025 |
| 14 | 0.9654 | -0.0110 | -0.0119 | +0.0009 |
| 16 | 0.9694 | -0.0083 | -0.0101 | +0.0018 |

---

## Interpretation

### The Wormhole operator is not working as intended.

The core finding is that the ring closure binding (the "Wormhole Operator") adds a single `closure_hv` term to the molecule hypervector via bundling (superposition/addition). Because the molecule HV is already a sum of N atom-position bindings, a single additional term is drowned out by the N existing terms. The closure tag's contribution to the final molecule vector is approximately `1/(N+1)` of the total magnitude, which is negligible for N >= 5.

**Evidence:**
1. **Linear probe at 57.5%** -- barely above the 50% chance baseline. A perfect wormhole signal would make rings trivially separable.
2. **Chain-ring cosine similarity > 0.92 for all sizes** -- the closure binding changes the molecule HV by less than 8% even for small rings.
3. **Closure-tag similarity delta ~ 0.0003** -- the closure tag is undetectable in the molecule HV via direct cosine similarity. The signal is buried in noise.

### Why projection "helps" (62.5% vs 57.5%)

The random (untrained) linear projection from 10kD to 256D occasionally amplifies the faint discriminative signal by chance. This 5pp difference is within noise for a 40-sample test set and should not be interpreted as meaningful.

### Angle concentration difference is real but insufficient

Rings do show measurably lower intra-class angle concentration (0.672 vs 0.765). This likely reflects the structural diversity introduced by different ring sizes interacting with different positional HVs, not the wormhole operator itself. Chains of different lengths are more self-similar because they share longer prefixes.

---

## Recommendations

1. **Amplify the closure signal.** Instead of bundling (adding) the closure HV once, consider:
   - Scaling the closure term by a factor (e.g., `sqrt(N)`) so it has comparable magnitude to the atom-position sum.
   - Using a separate "topology channel" -- a second HV that encodes only ring closure information, concatenated (not bundled) with the atom-position HV.

2. **Use binding (not bundling) for the closure.** The current approach adds the closure term via `torchhd.bundle` (superposition). Binding the closure tag multiplicatively into the molecule HV would create a structurally different representation rather than a slightly perturbed one.

3. **Add bond-type encoding.** Currently only atom identity and position are encoded. Encoding explicit bonds (single, double, aromatic) would provide additional discriminative features.

4. **Re-run after fix.** The current encoding fidelity is insufficient for the CLN to learn ring-closure physics. The wormhole operator must be strengthened before proceeding to Phase 2.

---

## Encoding Performance

- 200 molecules encoded in 0.23s (CPU)
- No RDKit dependency required for this test
