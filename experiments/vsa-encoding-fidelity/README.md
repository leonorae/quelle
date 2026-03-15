---
status: needs-fix
owner: null
dependencies: []
---

# VSA Encoding Fidelity

**Question**: Can HRR (Holographic Reduced Representations) encode molecular
topology such that structural properties (ring vs chain, ring size, branching)
are linearly separable in hypervector space?

## Background

This experiment validates whether VSA is a viable representation for molecular
structure. Originally created to validate crystal-lattice's core assumption, it
now stands alone: the question of whether VSA can represent topology is interesting
regardless of what downstream model consumes the representation.

## Current Result: FAILED (2026-03-14)

See `RESULTS.md` for full details.

- Ring vs chain classification: **57.5%** (near chance)
- Closure tag cosine delta: **0.0003** (negligible)
- Chain-ring similarity: **>0.92** (nearly identical)

**Root cause**: Bundling (addition) of a single closure-tag HV into a molecule HV
with N atom-position terms dilutes the signal to ~1/(N+1).

## Next Step: Fix and Re-run

Three candidate fixes, in order of promise:

1. **Multiplicative binding** (not bundling) for closure — creates a structurally
   different representation rather than a slightly perturbed additive one. Most
   principled; changes the algebraic operation, not just the scale.

2. **Separate topology channel** — a second HV encoding only ring closure
   information, concatenated with the atom-position HV. Clean separation but
   doubles the representation size.

3. **Scale closure term by sqrt(N)** — simplest fix, keeps the existing approach
   but amplifies the signal. Least principled; a patch rather than a redesign.

**Success threshold**: Ring vs chain linear probe accuracy >= 85%. If we can't get
there with any fix, VSA-for-topology is likely not viable and this is a clean
negative result.

## If VSA Works

Could be combined with iterative refinement (`cln-iteration-dynamics`) to revisit
the original crystal-lattice architecture. See
`wiki/findings/crystal-lattice-decomposition.md`.

## Literature Gap

No published work applies VSA/HDC to crystal or molecular topology (confirmed in
literature survey, 2026-03-08). A positive result here would be novel. A clean
negative result is also publishable/documentable.

## Code

- `src/test_vsa_fidelity.py` — validation test
- Encoder: `experiments/crystal-lattice/src/vsa_lattice.py`
