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

## Hypothesis

Value information in transformer residual streams decomposes into components
varying at different timescales (token-type-intrinsic → document-level →
local-context → hyper-local). The current nanochat architecture carries one
static value embedding table per ve-layer, gated by a fixed 32-channel slice
of the residual. Multiple tables (k > 1), each conditioned on different context
signals, can factor out stable components and reduce interference in the
residual stream.

The core question: does increasing k cause spontaneous table specialisation
(each table captures a different timescale), or do the tables collapse toward
a single shared representation?

## Status

`planning` — Phase 0 diagnostics implemented; awaiting first training run.

## Phases

| Phase | Goal | Tool | Status |
|---|---|---|---|
| 0 | Diagnostic baseline — characterise existing single-table ve | nanochat full pipeline | planning |
| 1 | Collapse detection — does k>1 spontaneously specialise? | autoresearch | not started |
| 2 | Minimal symmetry-breaking — minimum pressure to escape saddle | autoresearch | not started |
| 3 | Assumptive pressures — geometric, routing, informational | nanochat full | not started |
| 4 | Task evaluation — functional differences on dissociated tasks | nanochat + SFT | not started |

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

## Open Questions

See `ve_implementation_plan.md` §10 and `DECISIONS.md`.
