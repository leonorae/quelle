---
status: running
owner: null
dependencies: [crystal-lattice]
---

# VSA Encoding Fidelity

**Hypothesis**: HRR encoding with the Wormhole Operator produces hypervectors
that are linearly separable by topology (ring vs chain) and that the 10kD→256D
projection preserves this separability.

## Background

Crystal-lattice's architecture assumes the VSA encoding provides meaningful
structural information to the CLN. This experiment validates that assumption
before investing in full training.

## Method

1. Generate 200 molecules: 100 linear alkanes, 100 macrocyclic rings
2. Encode with VSALattice (from crystal-lattice/src/vsa_lattice.py)
3. Test linear separability (logistic regression) on raw 10kD and projected 256D
4. Measure Wormhole operator's effect on ring molecule geometry
5. Compare angle concentration between chains and rings

## Results

See RESULTS.md (populated by test_vsa_fidelity.py run).
