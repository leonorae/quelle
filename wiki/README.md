# Quelle Wiki

The wiki is the living knowledge base for this monorepo. It accumulates
concepts, findings, and working notes across all experiments and sessions.

> **Agents**: Read this index at the start of every session. Update it when you
> add new pages.

---

## Structure

```
wiki/
├── README.md          ← this index
├── concepts/          ← background theory and shared definitions
├── findings/          ← cross-experiment synthesis and key results
├── agents/            ← per-agent working notes and session logs
└── humans/            ← human-authored decisions, roadmap, open questions
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
| [constraint-forcing-function.md](concepts/constraint-forcing-function.md) | Using architectural constraints deliberately to force discovery of network hacks; generalising those hacks |

## Findings

| Page | Experiments | Summary |
|------|-------------|---------|
| *(none yet)* | | |

## Agent Notes

| Agent | Page |
|-------|------|
| claude-sonnet-4-6 | [claude-sonnet-4-6.md](agents/claude-sonnet-4-6.md) |

## Human Notes

| Topic | Page |
|-------|------|
| [roadmap.md](humans/roadmap.md) | Project roadmap and planned experiments |
| [decisions.md](humans/decisions.md) | Design decisions and rationale log |
| [latent-research-context.md](humans/latent-research-context.md) | Research positioning and feasibility assessment for `variable-bitrate-reasoning` |

---

## Conventions

- Every page should have a one-line description at the top.
- Use relative links between wiki pages.
- Keep pages focused. If a page grows beyond ~200 lines, split it.
- Tag pages with relevant experiment slugs so they can be cross-referenced.
