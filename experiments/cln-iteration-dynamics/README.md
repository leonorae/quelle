---
status: needs-lit-review
owner: null
dependencies: []
---

# Does Iterative Refinement Help Structured Prediction?

**Question**: When a small model loops over its own output (recurrent refinement),
does it meaningfully improve predictions on structured inputs? At what point do
additional iterations stop helping?

## Background

The crystal-lattice CLN was a 2-layer Transformer looped 8 times with anchor
re-injection. The claim was that iteration relaxes a latent representation toward
a physically valid prediction. But this was never tested — the VSA encoding
failed before CLN training began.

The question generalizes beyond molecular tasks: iterative/recurrent refinement is
a design pattern appearing across multiple architectures (looped transformers,
Huginn, LoopFormer, SpiralFormer — see literature survey).

## Before Building Anything: Literature Review Needed

Recent work on looped/recursive transformers is directly relevant:

1. **"Reasoning with Latent Thoughts" (Saunshi et al., ICLR 2025)**: k-layer
   transformer looped L times nearly matches kL-layer model on reasoning tasks.
   Does this transfer to structured prediction (regression, geometric tasks)?

2. **Ouro, Huginn, LoopFormer, SpiralFormer (2025-2026)**: Scaled looped
   architectures. What do their ablations say about iteration dynamics? Do later
   iterations contribute meaningfully or is most work done early?

3. **Adaptive halting / early exit**: Entropy-based stopping claims 20-30% FLOPs
   savings. Does this imply later iterations are often redundant?

4. **Key question**: Is there already clear evidence about the iteration dynamics
   profile (relaxation curve, useful iteration count, diminishing returns)?

If the literature provides a clear answer, document it and close this experiment.

## If We Run an Experiment

Test iterative refinement on a task where we already have a working representation:

- **Option A**: Looped Transformer on a simple geometric/arithmetic task
  (existing benchmarks from the looped-transformer literature)
- **Option B**: Recurrent refinement on molecular property prediction using
  standard fingerprints (not VSA) — this stays closer to the original motivation
- **Option C**: Apply looping to an existing experiment's model (e.g., add
  recurrence to a probe in behavioral-projections or VVVVVV)

Measure per-iteration diagnostics:
- Prediction change (L2 distance between iteration k and k+1 outputs)
- Output entropy across iterations
- Truncation ablation (accuracy at iteration k for k=1..N)
- Whether the dynamics show relaxation (converging) or oscillation

## Success Criteria

- Characterize the iteration dynamics profile: relaxation, plateau, oscillation?
- Identify the "useful iteration" count for the task
- Determine whether adaptive halting is feasible and beneficial

## Speculative Combination

If iterative refinement works well on structured prediction, it could be combined
with VSA encoding (if that also validates) to revisit the original crystal-lattice
idea. See `wiki/findings/crystal-lattice-decomposition.md`.

## Origin

Decomposed from `crystal-lattice` (2026-03-15). Previously required crystal-lattice
training logs. Now independent — can use any model and representation.
