---
experiment: behavioral-projections
title: "Behavioral Projections: Learned Linear Maps on LLM Activations"
status: scaffolding
owner: claude-opus-4-6
created: 2026-03-11
depends_on:
  - transformers
  - EleutherAI/pythia-410m
  - Qwen/Qwen2.5-7B-Instruct
tags: [projections, interpretability, bisimulation, geometry]
---

# Behavioral Projections

## Hypothesis

Four learned linear projections on frozen LLM activations — each optimized for
a different behavioral objective — decompose activation space into content,
computation, sensitivity, and transformation. The intersection of their null
spaces isolates endogenous scaffolding (the geometric analog of attention sinks
and register tokens).

If the bisimulation probe achieves R² > 0.3 on held-out pairs, a low-rank
linear map captures meaningful behavioral geometry. If iterative peeling
saturates in few iterations, that geometry is concentrated.

## Status

**Phase 0** (activation caching): pipeline validated on 221-prompt bootstrap set (Pythia-410m, CPU)
**Phase 0.5** (tuned lens baseline): complete on bootstrap — ρ≈0.88, R²≈0.5–0.6 at middle layers
**Phase 0.5b** (pairwise lens): implemented, ready to run
**Phase 1** (bisimulation probe): Ridge validated; learned projection + rank sweep implemented
**Phase 1.5** (three-condition comparison): implemented — `src/compare_conditions.py`
**Phases 2–4**: stubbed

## Phases

| Phase | Name | Gate condition | Status |
|-------|------|---------------|--------|
| 0 | Activation caching | — | validated (bootstrap) |
| 0.5 | Tuned Lens baseline | Phase 0 cached | complete (bootstrap) |
| 0.5b | Pairwise Tuned Lens | Phase 0 cached | implemented |
| 1 | Bisimulation probe | Phase 0 cached for Pythia-410m | implemented |
| 1.5 | Three-condition comparison | Phases 0.5, 0.5b, 1 | implemented |
| 2 | Iterative peeling | Phase 1 R² measured | stub |
| 3 | Contrastive prompt discrimination | Phase 0 cached | stub |
| 4 | Perturbation-sensitivity map | Phase 0 cached | stub |
| 5 | Intersection analysis | ≥2 projections trained | stub (in analysis.py) |

Phase 0.5 determines whether Phase 1 is worth building. If the Tuned Lens
per-layer R² curve shows decent pairwise KL prediction at middle layers,
independent decoding already captures behavioral distance and the bisimulation
probe is redundant. See DECISIONS.md D13.

## Models

1. **Pythia-410m** — pipeline development, CPU-friendly. 24 layers, d=1024.
2. **Qwen2.5-7B-Instruct** — primary target. 32 layers, d=3584. Requires 3090 24GB.
3. **Qwen2.5-14B/32B** (quantized) — stretch goal, Colab A100.

No GPT-2. Pythia-410m covers the same niche with better tooling.

## Hardware

- **3090 Ti 24GB** (borrowed): Qwen2.5-7B inference + caching
- **CPU**: unlimited for analysis on cached activations
- **Colab Pro A100**: larger models

## Dependencies

```
torch>=2.0
transformers>=4.40
safetensors
scikit-learn
numpy
tqdm
pyyaml
```

## Key references

- `wiki/concepts/behavioral-projections.md` — theoretical foundations
- `experiments/projections/behavioral-projections-draft.md` — original spec (on projections branch)
- Zhang et al. (ICLR 2021) — bisimulation metrics in RL
- Gurnee (2024) — SAE dark matter / reconstruction pathology
