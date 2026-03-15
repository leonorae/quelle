---
status: needs-lit-review
owner: null
dependencies: []
---

# Curiosity vs Random Selection in Low-Data Regimes

**Question**: Does diversity-entropy active learning meaningfully outperform random
selection when the sample budget is small (10-50 examples)?

## Background

Active learning is well-studied, but results are regime-dependent. The crystal-lattice
architecture proposed curiosity-driven selection as a core component, but never
validated it independently. This experiment asks the question on its own terms.

## Before Building Anything: Literature Review Needed

Active learning vs random is extensively studied. Before implementing:

1. **Check existing results** for the specific regime: very low budget (10-50 samples),
   structured/molecular inputs, regression or classification tasks.
2. **Key question**: Does the literature already show diminishing returns for active
   learning at very small budgets? (Some results suggest random is competitive when
   the budget is tiny because the model hasn't learned enough to score candidates
   meaningfully.)
3. **If the literature is clear**: Document the finding and close this experiment.
   No need to re-derive known results.
4. **If the literature is ambiguous for our regime**: Design a minimal test. Use the
   simplest model and representation that works — not the full crystal-lattice stack.

## If We Run an Experiment

The test should be domain-agnostic or use a well-benchmarked domain:

- **Model**: Any small regressor/classifier (not necessarily CLN)
- **Representation**: Any working representation (molecular fingerprints, not
  necessarily VSA)
- **Selection strategies**: Random, uncertainty sampling, diversity-entropy (coreset),
  and ideally one more baseline (e.g., max-entropy)
- **Metric**: Accuracy/MAE at budget = {10, 25, 50} samples
- **Repetitions**: Multiple seeds, report variance

## Success Criteria

- Clear evidence that one strategy dominates at the relevant budget, OR
- Clear evidence that the difference is negligible (also a useful result)

## Speculative Combination

If curiosity-driven selection wins convincingly, it could be applied to any
low-data experiment in this repo (not just molecular tasks). Note in
`wiki/findings/crystal-lattice-decomposition.md`.

## Origin

Decomposed from `crystal-lattice` (2026-03-15). Previously depended on
crystal-lattice completing Phases 1-3. Now independent.
