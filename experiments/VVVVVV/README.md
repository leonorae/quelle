---
experiment: VVVVVV
title: Multi-Timescale Value Embedding
status: planning
owner: claude-sonnet-4-6
created: 2026-03-09
depends_on:
  - wiki/concepts/value-embeddings
  - nanochat/gpt.py (external — karpathy/nanochat fork)
---

# VVVVVV — Multi-Timescale Value Embedding

## Hypothesis (revised 2026-03-15)

Value information in transformer residual streams decomposes into components
varying at different timescales (token-type-intrinsic → document-level →
local-context → hyper-local). The current nanochat architecture carries one
static value embedding table per ve-layer, gated by a fixed 32-channel slice
of the residual.

**Original question** (Phases 1-4): does increasing k cause spontaneous table
specialisation? `conjectured dead` — see below.

**Current question** (Phase 0 + interpretability): what does the existing ve
table contain, and what does that tell us about representation geometry? The ve
table is a `[vocab × kv_dim]` matrix of context-collapsed retrieval priors —
already factored, already legible. Analyzing it may be more valuable than
training multiple tables. See `wiki/findings/vvvvvv-reframe.md`.

## Status

`planning` — Phase 0 diagnostics implemented; awaiting d12 checkpoint.
Phases 1-4 (multi-ve architecture) `conjectured dead` pending Phase 0 results.

## Phases

| Phase | Goal | Tool | Status |
|---|---|---|---|
| 0 | Diagnostic baseline — characterise existing single-table ve | nanochat full pipeline | ready to run |
| — | *Ve table content analysis (moved to behavioral-projections Phase 0.6)* | matrix analysis | planned |
| 1 | ~~Collapse detection — does k>1 spontaneously specialise?~~ | autoresearch | conjectured dead |
| 2 | ~~Minimal symmetry-breaking~~ | autoresearch | conjectured dead |
| 3 | ~~Assumptive pressures~~ | nanochat full | conjectured dead |
| 4 | ~~Task evaluation~~ | nanochat + SFT | conjectured dead |

**Revival condition for Phases 1-4**: Ve table analysis reveals clear multi-modal
structure (distinct clusters that a single table conflates), making the
architecture question empirically motivated. See `wiki/findings/vvvvvv-reframe.md`.

## Phase 0 Questions

Three concrete diagnostics before touching k:

- **Q0.1**: Where do spike channels land? Do the highest-magnitude residual
  channels fall in [:32] (the current gate's read window)? If >50% overlap:
  gate is reading informative channels. If <20%: gate is reading near-noise.
- **Q0.2**: Is the BOS residual document-level? If BOS varies across documents
  but is stable within them, it is usable as a document-level conditioning
  signal. If near-constant across documents (pure sink), BOS conditioning
  carries no document-level signal.
- **Q0.3**: Functional load of existing ve. Zero out the single ve table and
  measure val_bpb degradation. This is the ground truth for "how much work is
  ve doing" before touching k.

## Dependencies

- `nanochat/gpt.py` — GPT, CausalSelfAttention, has_ve(), GPTConfig
- Existing ve mechanism: `self.value_embeds` (nn.Embedding per ve-layer),
  `self.ve_gate` (nn.Linear(32, n_kv_head)), alternating `has_ve()` pattern
- Run scale: d12 or d16 for Phase 0 (fast iteration)

## Key Prior Results

- ve is load-bearing: any reduction (low-rank, sharing, projections) hurt
  (nanochat discussion #481)
- Alternating ve placement won over every-layer or U-shaped
- ve weight decay 0.001–0.003 improves; 0.005 regresses (autoresearch #43)
- BOS token develops near-constant post-norm representation in intermediate
  layers (Sun et al. arXiv:2603.05498) — verify empirically in Phase 0
- nanochat uses relu² not SwiGLU — spike channel profile may differ from
  Sun et al.; measure before assuming

## Why Phases 1-4 eroded (2026-03-15)

Three accumulated reasons:

1. **Gate-input problem**: If spike channels (relu²) fall in [:32], the gate
   reads near-constant signal — multi-table routing is broken at input. Phase 0
   Q0.1 tests this directly.
2. **ve_local is theoretically weak**: Attention already does local conditioning.
   Only ve_document has strong justification (adds timescale inaccessible to
   attention). ve_static already exists.
3. **Question shifted**: From "does this improve loss?" (architecture) to "what
   structure can be made legible?" (interpretability). The factorization class
   (`wiki/concepts/factorization-taxonomy.md`) reframes ve as one instance of a
   general move. The existing table is more interesting to analyze than to multiply.

**Kill condition**: Phase 0 Q0.1 shows >50% spike overlap → gate reads noise →
Phases 1-4 definitively dead.

## Open Questions

See `ve_implementation_plan.md` §10, `DECISIONS.md`, and
`wiki/findings/vvvvvv-reframe.md`.
