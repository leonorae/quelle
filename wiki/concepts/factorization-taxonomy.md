# Factorization Taxonomy

> Date: 2026-03-15
> Source: chat analysis (Apophenia + agent synthesis)

## The general move

Identify a component of a computation that varies along fewer dimensions than
the full computation. Give it a dedicated pathway: `f(a, b) → g(a) + h(a, b)`,
where `g` captures the component depending only on the lower-dimensional
argument.

Shared property: the factored component becomes **legible** after extraction.
Token embedding matrices are readable. RoPE frequencies are explicit. The ve
table is inspectable. Legibility is not a side effect — it's a signal that the
factorization found real structure.

## Inventory

| Factorization | What's peeled off | Axis removed | Side |
|---|---|---|---|
| Token embedding | type-level identity | token-type vs position/context | input |
| RoPE | relative position | from Q and K | weight |
| LayerNorm | scale and shift | magnitude vs direction | weight |
| Untied embeddings | input vs output distributions | encoding vs decoding priors | weight |
| GQA | K/V rank | head-specific vs shared content | weight |
| **Value embeddings (ve)** | **type-level retrieval prior** | **token-type vs instance context** | **output** |
| x0 residual | initial identity | accumulated transform vs origin | input |
| Bias removal | constant offset | from weight matrices | weight |

## The output-side distinction

Most factorizations operate at input or weight level. Ve is unusual: it's an
**output-side factorization** — it changes what the computation produces (what
gets handed over during retrieval), not what it receives or stores. This appears
genuinely underexplored in the literature.

## QKV scale asymmetry

(Apophenia's intuition, confirmed by analysis.)

V's type-level component is large relative to instance-level deviation —
function words, delimiters have strong type-level priors, weak contextual
modulation, high redundancy in the unfactored case. QK's instance-level
component dominates because routing is inherently relational — what you're
looking for only means something relative to context.

Factorization recovers more from V because there's more to recover.

## "Context-collapsed", not context-free

The stable component in ve isn't context-free in the sense of context-ignorant.
The table was learned from millions of contexts. It's **context-collapsed**: the
distribution over all contexts a token has appeared in, compressed into a single
vector. The table entry is the model's prior over what that token type tends to
contribute, averaged functionally across the training distribution.

The factorization gives that prior a dedicated home rather than forcing it to be
reconstructed from scratch every forward pass.

## The deeper point

Without these factorizations, the model must represent calculations that apply
across many contexts redundantly, entangled, carrying both the stable prior and
the instance-level deviation through every layer at full bandwidth cost. The
factorizations are a progressive articulation of structure that was always
latent — they don't impose structure, they reveal it by giving it a cleaner
place to live.

## Connections

- **VVVVVV**: ve is the motivating instance. The multi-table architecture
  question (k>1) is now [conjectured dead]; the interpretability question (what
  does the existing table contain?) is [open]. See `wiki/findings/vvvvvv-reframe.md`.
- **behavioral-projections**: the ve table is a `[vocab × kv_dim]` matrix of
  token-type retrieval priors. If the bisimulation probe measures functional
  identity, the ve table may be a cheaper readout of part of the same thing —
  already factored, already legible, no probing required.
- **Variable-bitrate-reasoning (archived)**: concentration as diagnostic, not
  control signal, is an instance of this pattern — the factored signal is
  informative but doesn't drive optimization.
