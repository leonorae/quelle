---
status: lit-review-complete-pending-experiment-design
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

## Literature Review Status

**Completed 2026-03-15**. Full review at
`wiki/findings/iterative-refinement-dynamics-review.md`.

Summary of findings:

- The qualitative dynamics curve is well-established: **log-linear improvement
  in early iterations, then plateau**. Confirmed across looped transformers
  (Saunshi et al., Huginn), protein recycling (AlphaFold2), and diffusion models.
- Adaptive halting saves 20–30% FLOPs in practice (Ouro, CALM).
- Two main failure modes: hidden state collapse; model learning to ignore
  recurrent state. Both known and fixable.
- **The gap**: no published work on small-model looped transformers on structured
  geometric/regression tasks. All large-scale evidence is on language/math tasks
  at ≥100M parameters.
- Recommendation: proceed with a narrow experiment (Option A first), framed as
  validating the known dynamics profile in the small-model structured-prediction
  regime.

The literature does *not* provide a clear enough answer to close this experiment.
It provides strong priors that narrow the experimental question considerably.

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
