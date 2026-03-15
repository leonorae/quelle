# Quelle Wiki

The wiki is the living knowledge base for this monorepo. It accumulates
concepts, findings, and working notes across all experiments and sessions.

> **Agents**: Read `STATUS.md` at session start. Consult wiki only when you
> need deeper background on a concept or finding.

---

## Structure

```
wiki/
├── README.md          ← this index
├── concepts/          ← background theory and shared definitions
├── findings/          ← cross-experiment synthesis and key results
├── agents/            ← per-agent working notes and session logs
└── humans/            ← human-authored decisions, open questions
```

---

## Concepts

| Page | Description |
|------|-------------|
| [geometric-self-awareness.md](concepts/geometric-self-awareness.md) | Angle concentration, cosine similarity, and what geometry tells us about representation certainty |
| [ris-scoring.md](concepts/ris-scoring.md) | Reasoning Integrity Score rubric, LLM judge ensemble protocol, and cost estimates |
| [variable-bitrate-compression.md](concepts/variable-bitrate-compression.md) | Adaptive bandwidth allocation in representation space |
| [gumbel-softmax.md](concepts/gumbel-softmax.md) | Differentiable discrete sampling — how and why |
| [dsd-future-prediction.md](concepts/dsd-future-prediction.md) | Dead-Stop-Detach (stop-gradient) future state prediction objective |
| [related-work.md](concepts/related-work.md) | Summaries of six papers directly informing `variable-bitrate-reasoning` |
| [factorization-taxonomy.md](concepts/factorization-taxonomy.md) | General factorization moves, output-side distinction, QKV scale asymmetry |

## Findings

Cross-experiment synthesis lives in `findings/`. Browse the directory directly;
no manual index is maintained here.

## Agent Notes

| Agent | Page |
|-------|------|
| claude-sonnet-4-6 | [claude-sonnet-4-6.md](agents/claude-sonnet-4-6.md) |
| claude-opus-4-6 | [claude-opus-4-6.md](agents/claude-opus-4-6.md) |

## Human Notes

| Topic | Page |
|-------|------|
| [decisions.md](humans/decisions.md) | Design decisions and rationale log |
| [latent-research-context.md](humans/latent-research-context.md) | Research positioning and feasibility assessment |

---

## Conventions

- Every page should have a one-line description at the top.
- Use relative links between wiki pages.
- Keep pages focused. If a page grows beyond ~200 lines, split it.
- Tag pages with relevant experiment slugs so they can be cross-referenced.
