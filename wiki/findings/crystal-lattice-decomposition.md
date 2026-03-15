---
title: "Crystal-Lattice Decomposition: From Linear Stack to Independent Threads"
date: 2026-03-15
type: decision
tags: [crystal-lattice, experiment-design, decomposition]
---

# Crystal-Lattice Decomposition

## What Happened

The crystal-lattice experiment proposed a four-layer architecture for molecular
reasoning: VSA encoding → CLN iterative refinement → curiosity-driven selection →
LLM-generated mutations. The hypothesis was "Small Data + Physical Grounding >
Big Data + Token Guessing."

On 2026-03-14, the first layer failed validation. `vsa-encoding-fidelity` showed
that the Wormhole operator (ring closure binding) produces hypervectors that are
not linearly separable by topology — 57.5% accuracy (near chance) on ring vs
chain classification. Root cause: additive bundling dilutes the closure signal
to ~1/(N+1) of the total vector magnitude.

## Why Decompose (Not Fix-and-Continue)

The failure prompted a reassessment of the experiment structure itself:

1. **The stack coupled four independently unvalidated ideas.** Each layer depended
   on the one below. If any failed, everything above was meaningless. This is a
   fragile design for exploratory research.

2. **The questions are actually independent.** "Can VSA encode topology?" is not
   the same question as "Does iterative refinement help?" or "Does active learning
   beat random?" Each can be tested on its own terms, with whatever substrate is
   cheapest.

3. **The linear model constrained dynamism.** You couldn't explore curiosity-vs-random
   without first fixing VSA, training the CLN, running three phases, etc. This
   serialization was artificial — imposed by the architecture, not by the research
   questions.

4. **Fastest path to knowledge.** Two of the three downstream questions
   (iterative refinement, active learning) may be resolvable by literature review
   alone. The looped-transformer literature (Ouro, Huginn, LoopFormer) and active
   learning literature are mature enough that building from scratch may be
   unnecessary.

## The Decomposed Threads

### 1. VSA for Molecular Topology (`vsa-encoding-fidelity`)

**Question**: Can HRR encode molecular structure such that topology is linearly
separable?

**Status**: Needs fix (multiplicative binding) and re-run.

**Why it matters independently**: No published work applies VSA to molecular
topology (confirmed by literature survey). A positive result is novel; a clean
negative is informative.

### 2. Iterative Refinement for Structured Prediction (`cln-iteration-dynamics`)

**Question**: Does looping a small model over its output improve structured
predictions? What do the dynamics look like (relaxation, plateau, oscillation)?

**Status**: Needs literature review. Recent looped-transformer papers may
already answer this.

**Why it matters independently**: This is a fundamental architectural question.
It applies to any prediction task, not just molecular geometry.

### 3. Curiosity-Driven vs Random Selection (`curiosity-vs-random`)

**Question**: At very small sample budgets (10-50), does diversity-entropy
selection outperform random?

**Status**: Needs literature review. Active learning is well-studied; the
answer may already exist for this regime.

**Why it matters independently**: If curiosity wins, it applies to any low-data
experiment in this repo, not just molecular tasks.

### 4. LLM as Molecular Hypothesis Generator

**Question**: Can an LLM propose structurally useful molecular modifications?

**Status**: Deferred. Low priority until other threads produce results.

## Speculative Recombination Paths

These are noted for future reference, gated on individual results:

| If this works... | ...and this works... | Then consider... |
|---|---|---|
| VSA encoding (fix) | Iterative refinement | Original crystal-lattice CLN on VSA representations |
| VSA encoding (fix) | Curiosity selection | VSA + active learning (skip CLN) |
| Iterative refinement | Any representation | CLN on molecular fingerprints (skip VSA) |
| Curiosity selection | Any model | Active learning for any low-data experiment |

**No combination is planned. Each is contingent on individual thread results.**

## What This Means for the Repo

The dependency graph changes from:

```
crystal-lattice → curiosity-vs-random
                → cln-iteration-dynamics
                → (blocked everything)
```

To:

```
vsa-encoding-fidelity   (independent)
curiosity-vs-random     (independent)
cln-iteration-dynamics  (independent)
```

All three can proceed in parallel. Two may resolve by literature review alone.
The repo moves from one blocked chain to three independently actionable threads.
