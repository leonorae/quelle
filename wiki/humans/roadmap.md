# Project Roadmap

> Human-authored. Update this when priorities shift.

**Last updated**: 2026-02-28

---

## Current Focus

1. **Scaffold repo structure** — experiments/, wiki/, CLAUDE.md ✓
2. **Implement `variable-bitrate-reasoning`** — see experiment README for checklist

## Planned Experiments (not yet started)

| Slug | Idea | Priority |
|------|------|----------|
| `geometric-self-awareness-reasoning` | Do angle concentration + trajectory geometry predict reasoning quality (RIS) on GSM8K? Training-free, interpretable error detection. | **High** |
| `gumbel-compression-ablation` | Ablate Gumbel-Softmax variants against the soft-sigmoid baseline | Medium |
| `manifold-capability-probing` | Do distinct capabilities occupy separable manifolds? | Low |
| `multi-task-vbr` | Extend variable-bitrate approach to multiple task types | Low |

## Infrastructure Todos

- [ ] Shared tokenizer utility (when multiple experiments need the same vocab)
- [ ] Dashboard (future — see CLAUDE.md)
- [ ] CI for linting experiment code

## Done

- Repo initialized with CLAUDE.md, experiments/, wiki/ scaffold
