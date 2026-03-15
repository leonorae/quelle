---
status: decomposed
owner: null
dependencies: []
---

# Crystal Lattice — Decomposed

> **2026-03-15**: This experiment has been decomposed into independent research
> threads. See rationale in `wiki/findings/crystal-lattice-decomposition.md`.

## Original Thesis

"Small Data + Physical Grounding > Big Data + Token Guessing" — via a stack of
VSA encoding, iterative CLN refinement, curiosity-driven active learning, and
LLM-generated molecular mutations applied to macrocyclic ring closure prediction.

## Why Decomposed

The architecture coupled four independently unvalidated ideas into a linear
dependency chain. The first layer (VSA encoding) failed validation
(`vsa-encoding-fidelity` RESULTS.md, 2026-03-14). Rather than fix the base and
rebuild upward, we decompose: each idea gets validated independently on whatever
substrate is cheapest, and combination is speculative/contingent.

The linear model (A requires B requires C requires D) was constraining dynamism.
The interesting questions are independent.

## Decomposed Threads

| Thread | Experiment | Status |
|---|---|---|
| Can VSA encode molecular topology? | `vsa-encoding-fidelity` | needs fix + re-run |
| Does iterative refinement help structured prediction? | `cln-iteration-dynamics` | needs lit review |
| Does curiosity-driven selection beat random? | `curiosity-vs-random` | needs lit review |
| Can LLMs propose useful molecular modifications? | (no experiment yet) | low priority |

## Speculative Recombination

If individual threads produce positive results, recombination paths include:

- **VSA + CLN**: If VSA encoding works and iterative refinement helps, combine
  them on the original macrocyclic task. This is the closest to the original plan.
- **Curiosity + any model**: If active learning beats random, it applies to any
  low-data prediction task, not just this architecture.
- **CLN + learned embeddings**: If iterative refinement helps but VSA doesn't,
  use standard molecular fingerprints as input instead.

These are noted, not planned. Gate on individual results.

## Preserved Code

All implementation code remains in `src/` for reference:
- `vsa_lattice.py` — HRR encoding with Wormhole binding
- `data_generator.py` — RDKit molecule generation
- `resonator.py` — CLN with iterative loop
- `curiosity_loop.py` — Ollama + diversity-entropy active learning
- `test_vsa_fidelity.py` — validation test

The code may be reused by decomposed experiments as needed.
