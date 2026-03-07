# Project Roadmap

> Human-authored. Update this when priorities shift.

**Last updated**: 2026-03-07

---

## Current Focus

1. **Scaffold repo structure** — experiments/, wiki/, CLAUDE.md ✓
2. **Run `variable-bitrate-reasoning`** — code complete; run full training and fill in RESULTS.md

## Planned Experiments (not yet started)

| Slug | Idea | Priority |
|------|------|----------|
| `gumbel-compression-ablation` | Ablate Gumbel-Softmax variants against the soft-sigmoid baseline | Medium |
| `manifold-capability-probing` | Do distinct capabilities occupy separable manifolds? | Low |
| `multi-task-vbr` | Extend variable-bitrate approach to multiple task types | Low |

## Infrastructure Todos

- [ ] Shared tokenizer utility (when multiple experiments need the same vocab)
- [ ] Dashboard (future — see CLAUDE.md)
- [ ] CI for linting experiment code

## Done

- Repo initialized with CLAUDE.md, experiments/, wiki/ scaffold
- `variable-bitrate-reasoning` implemented (2026-03-07) — awaiting first run
