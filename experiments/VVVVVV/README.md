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
static value embedding table per ve-layer, gated by a fixed 12-channel slice
of the residual (empirically reduced from 32, autoresearch #43). Multiple tables
(k > 1), each conditioned on different context
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
  channels fall in [:12] (the current gate's read window)? If >50% overlap:
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
  `self.ve_gate` (nn.Linear(12, n_kv_head)), alternating `has_ve()` pattern
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

### Scale dependence

All "ve is load-bearing" evidence (nanochat discussion #481, autoresearch #43)
is at nanochat's speedrun scales (d12–d26). Two competing hypotheses for *why*
ve helps:

- **Capacity-hack hypothesis**: ve is a cheap token-type-specific prior on
  value projections that a small attention mechanism doesn't have capacity to
  learn from context. Benefit decays with scale as weight matrices have room
  to encode the same patterns directly. If true, ve is a small-model artifact
  with limited generalization interest.

- **Interference-reduction hypothesis**: ve offloads stable token-type patterns
  from the residual stream into a dedicated table, reducing superposition
  interference. If true, benefit should be scale-stable or increasing, because
  a wider residual stream carrying more competing signals has more to gain from
  factoring out static components.

Phase 0 Q0.3 (ablation delta) establishes functional load at d12, but cannot
distinguish these hypotheses. Distinguishing them requires replicating Q0.3
at multiple scales. This is not planned for current phases but should inform
how strongly to generalize Phase 0 results.

### relu² specificity

nanochat uses relu² (squared ReLU). This produces genuinely sparse,
high-magnitude spike channels. SwiGLU (standard in Llama/Mistral-class
production models) is smooth and bounded — its channel magnitude profile is
qualitatively different. If the gate mechanism depends on spike channel
structure (Q0.1 hypothesis), results may not transfer to SwiGLU architectures.
This is a hard limit on generalizability that Phase 0 cannot address — it is a
known prior to carry into interpretation.

### Phase 0 as mechanism characterisation, not just hypothesis gating

The Phase 0 questions are written as gates for VVVVVV Phase 1+. They are also
the first mechanistic characterisation of why ve helps at all — a question that
is separately valuable regardless of whether VVVVVV proceeds. Whatever the gate
is reading (Q0.1), and whatever functional load ve carries (Q0.3), is new
information about a result that has been empirically established but not
explained.

### Q0.3 metric validity — ClimbMix bpb may not capture ve's contribution

val_bpb on ClimbMix is a proxy. It may be wrong — possibly near-zero or
negative — not because ve is inert, but because the metric is insensitive to
what ve actually does.

- **Distribution specificity:** gate_window=12 was tuned on ClimbMix; the gate
  and its read channels have been shaped by this distribution. The ablation
  measures load in that context. Whether the same functional load exists on
  other distributions is unknown.
- **Sequence length:** ClimbMix val sequences are ~2048 tokens. If ve's primary
  contribution is to multi-timescale long-range coherence, a 2048-token window
  may not expose it — the effect would appear small even if ve is load-bearing
  over longer documents.
- **Negative delta is possible:** the table is tied to token identity, not
  context. Zeroing it could cause the model to fall back to a different
  (possibly adequate) path for certain short-range predictions, masking the
  true cost. Alternatively, ve may be adding a slight bias in directions that
  happen to hurt certain val sequences — a small negative delta would not mean
  ve is harmful, only that the metric is coarse.

Treat Q0.3 as directional evidence, not ground truth. If delta is small or
near-zero, do not conclude ve is unimportant — consider whether a more targeted
eval (longer documents, or documents with explicit long-range structure) would
be more sensitive.

### Gate channel placement — the arbitrary first-indices problem

The gate reads `x[:, :, :12]` — the first 12 channels of the residual stream.
This slice is arbitrary. There is no reason channels 0–11 are more informative
than channels 100–111 a priori; the choice originated as a simplification
(discussed in karpathy's original nanochat notes). Gradient pressure from the
ve pathway does push the network to route gate-useful information into channels
0–11, but:

- This competes with whatever channels 0–11 are already doing for attention/MLP.
- If the highest-magnitude (spike) channels land inside the gate window (Q0.1
  high overlap), the gate may be reading channels that are already doing heavy
  lifting for next-token prediction. "Informative" here means high-magnitude,
  not necessarily maximally discriminative for gating.
- If a spike channel is inside the gate window, it is potentially doing double
  duty: carrying residual-stream content *and* carrying gate signal. This could
  represent wasted capacity — the channel's bandwidth is split between two
  objectives.
- Conversely, a "near-noise" result (low overlap) might mean the gate is reading
  channels that are relatively free of competing residual content — which could
  actually be efficient, with dedicated gate-signaling channels that don't
  interfere with the main prediction pathway.

**The ideal gate input** would be low-magnitude for the main pathway but
high-variance for the gating objective — channels that act as dedicated
gate-signaling lanes. Whether the network spontaneously creates these under the
fixed-slice constraint is an empirical question; Q0.1 provides partial evidence.

**The cheap fix:** replace the fixed `[:12]` slice with a learned linear
projection W_gate ∈ ℝ^(12×d_model). This lets the model learn which combination
of channels to read, removing the arbitrary-index problem entirely. Cost at
d12 (d_model=256): 12 × 256 = 3,072 extra parameters — negligible. This is
already flagged as a candidate in §6.1 of the implementation plan and in D5 of
DECISIONS.md; it is gated on Q0.1 results showing near-noise (learned projection
becomes necessary), but is also worth considering even in the "informative" case
because the overlap metric doesn't distinguish clean gate-lanes from busy shared
channels.
