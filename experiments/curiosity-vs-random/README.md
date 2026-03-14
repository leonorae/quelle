---
status: planned
owner: null
dependencies: [crystal-lattice]
---

# Curiosity vs Random Baseline

**Hypothesis**: Diversity-entropy active learning selects better training samples
than random selection, leading to higher Phase 3 accuracy with the same sample
budget.

## Background

Crystal-lattice's core claim is that curiosity-driven selection of "stepping
stones" achieves 90%+ accuracy on ~50 samples vs 1000+ for random sampling.
This experiment directly tests whether the selection strategy matters.

## Method

Run crystal-lattice Phase 1 + 2 + 3 twice:

| Condition | Phase 2 selection | Everything else |
|---|---|---|
| Curiosity | Diversity-entropy top-k | Identical |
| Random | Random top-k from valid candidates | Identical |

Compare Phase 3 metrics: ring classification accuracy, distance MAE.

## Success criteria

- Curiosity condition achieves >10% higher accuracy than random on Phase 3
  macrocycles with same number of training samples
- If no difference: the curiosity loop is decorative and should be simplified

## Implementation

Add `--selection-mode random` flag to curiosity_loop.py. Run both conditions
with same seed for Phase 1 (shared starting point).
