---
date: 2026-03-15
scope: VVVVVV, behavioral-projections
type: decomposition
---

# VVVVVV Reframe: From Architecture Experiment to Interpretability Instrument

## What happened

The multi-ve table experiment (VVVVVV Phases 1-4: k>1 tables, symmetry-breaking,
assumptive pressures, task evaluation) was never formally killed. It eroded
through accumulated evidence that the premises don't hold:

### Three reasons the architecture experiment is dead

**1. Gate input may be reading spike channels, not content.**
The nanochat ve gate reads fixed first-32 channels of residual. Sun et al.
establishes that spike channels occupy specific, consistent indices. If those
indices fall in the first 32, the gate sees a near-constant signal regardless of
document content — the multi-table routing is broken at the input. Empirically
unverified for nanochat (relu² vs SwiGLU profile may differ), but the risk is
structural. Phase 0 Q0.1 tests exactly this.

**2. ve_local is theoretically weak.**
Attention already does local conditioning implicitly. ve_local is potentially
redundant with what residual V and Dherin's implicit rank-1 MLP updates compute.
The only table with strong theoretical justification is ve_document (adds a
timescale structurally inaccessible to attention). ve_static already exists.
ve_local is probably noise.

**3. The interesting question shifted.**
The multi-ve table experiment was framed as architecture optimization ("does this
improve loss?"). The conversation moved toward the factorization class as an
interpretability question — not "does this improve loss" but "what structure can
be made legible?" This reframe makes the architecture experiment the wrong level
to work at.

### Status

- **Multi-ve architecture (Phases 1-4)**: `conjectured dead`
- **Phase 0 diagnostics (Q0.1-3)**: `still worth running` — answers matter for
  interpretability angle too, not just architecture
- **Ve-table-as-interpretability-instrument**: `open` — new thread

## What's worth doing

### 1. Phase 0 diagnostics — YES, run them

The three Phase 0 questions (spike channel overlap, BOS stability, ve functional
load) are worth answering regardless of whether multi-ve proceeds. They
characterize the existing ve mechanism as an interpretability object:

- **Q0.1** (spike channels): Tells us what the gate is actually reading — content
  signal or near-constant spike artifact. This matters for understanding the
  factorization, not just for routing k>1 tables.
- **Q0.2** (BOS stability): Tests whether BOS carries document-level information.
  Relevant to the timescale question independent of architecture.
- **Q0.3** (ve ablation): Ground truth for how much work the factored component
  does. Directly informative for the factorization taxonomy.

**When**: After d12 checkpoint is trained. Same timeline as before.

### 2. Ve table content analysis — YES, new thread

The existing ve table is a `[vocab × kv_dim]` matrix. Each row is a token type's
context-collapsed retrieval prior. This is:

- Already factored (by construction)
- Already legible (it's a matrix, not entangled activations)
- Directly comparable to behavioral-projections' bisimulation probe

**Concrete questions**:
- Do ve table rows cluster by syntactic function (function words vs content words)?
- Does cosine structure in ve space correlate with bisimulation distance?
- What's the effective rank of the ve table? (How many "types" of retrieval prior
  exist?)
- Do ve layers differ in what they encode? (alternating layers have different tables)

**When**: Can start as soon as a checkpoint exists. Cheap — no training, just
matrix analysis. Natural home is behavioral-projections (it's a projection
comparison), not a new experiment.

### 3. Multi-ve architecture (Phases 1-4) — NO, not now

The architecture experiment requires:
- Training multiple models with k>1 tables
- Symmetry-breaking pressure design
- Task evaluation pipeline

This is expensive, the theoretical motivation is weakened (gate-input problem,
ve_local redundancy), and the interpretability framing yields more insight per
compute dollar. If ve table analysis reveals structure that clearly benefits from
multiple tables, the architecture question can be revisited — but as a response
to data, not a speculative design.

**Kill condition**: If Phase 0 Q0.1 shows >50% spike-channel overlap (gate reads
noise), the multi-table routing design is definitively broken and Phases 1-4 are
dead, not just conjectured dead.

**Revival condition**: If ve table analysis shows clear multi-modal structure
(distinct clusters of retrieval priors that a single table conflates), the
architecture question becomes empirically motivated rather than theoretical.

### 4. Phase 0 gated decisions (D5) — REINTERPRET

DECISIONS.md D5 gates architectural moves on Phase 0 results:
- Learned projection gate (§6.1): gated on Q0.1
- BOS-conditioned table (§6.2): gated on Q0.2

These were architecture gates. Under the reframe, the same Phase 0 results now
gate interpretability conclusions instead:
- Q0.1 low overlap → gate reads noise → ve factorization is doing something
  interesting despite an uninformative gate signal (the table itself carries the
  load, not the gating)
- Q0.2 BOS varies → document-level signal exists → timescale separation is
  empirically present, not just theoretically motivated

## Connection to behavioral-projections

The ve table may partially answer what the four projections are measuring. If
bisimulation distance (projection #1) correlates with ve table cosine structure,
then the ve table is a free readout of behavioral similarity — already factored
by the architecture, no probing required.

This makes ve table analysis a natural **Phase 0.6** of behavioral-projections:
cheap, requires only a checkpoint, and tests whether the factorization already
provides what the projection suite is trying to learn.

## Updated dependency graph

```
VVVVVV Phase 0 diagnostics (ready to run)
    ├─→ ve table content analysis (behavioral-projections Phase 0.6)
    ├─→ probe-signal-comparison (needs d12 checkpoint)
    └─→ [if revival condition met] multi-ve architecture (Phases 1-4)

behavioral-projections (active)
    ├─→ frame_ratio curve (blocked on tuned lens)
    └─→ ve table comparison (Phase 0.6, needs checkpoint)
```
