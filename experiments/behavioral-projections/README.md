---
status: active
owner: null
dependencies: []
---

# Behavioral Projections

**Thesis**: Learned linear projections on frozen LLM activations can measure the
implicit multi-timescale structure that transformers simulate but can't make
explicit. The projection suite provides a diagnostic vocabulary that existing
interpretability tools (logit lens, SAE, probing) miss.

## The Four Projections

| # | Projection | What it measures | Status |
|---|---|---|---|
| 1 | Output-distribution probe (bisimulation) | Model's own behavioral distance metric | Phase 1 implemented |
| 2 | Contrastive prompt discrimination | Same-prompt representation clustering | Phase 3 stubbed |
| 3 | Layer-contrastive (delta) projection | What each layer transition does | New — see 0.5c |
| 4 | Perturbation-sensitivity map | Amortized activation patching | Phase 4 stubbed |

## Key Innovation: Lens Delta Decomposition (Phase 0.5c)

For adjacent tuned lens translators P_l, P_{l+1}:

```
total_delta  = P_{l+1}(h_{l+1}) - P_l(h_l)
state_delta  = P_{l+1}(h_{l+1}) - P_{l+1}(h_l)    # activation changed, same lens
frame_delta  = P_{l+1}(h_l) - P_l(h_l)             # same activation, lens changed
frame_ratio  = ‖frame_delta‖ / ‖total_delta‖       # per layer pair
```

**frame_ratio** separates "the model updated what it knows" (state) from "the
model changed how it categorizes what it has" (frame/ontology). Layers where
frame_delta dominates are ontology reorganization — the model isn't adding
information, it's changing representational frame.

**Validated**: L0→L1 of Pythia-410m, frame_ratio=0.035 (state-dominated, expected
for early layers).

## Priority Order

1. **Finish frame_ratio curve** across all 24 Pythia-410m layers
   - Blocked on tuned lens training completion
   - `src/lens_delta_analysis.py` spec written
2. **Three-condition comparison** (C1/C2/C3) on real corpus
3. **About-to-be-wrong validation**: partition by confident-correct /
   confident-wrong / uncertain, compare frame_ratio curves
4. **Projection suite on pretrained LNN** (parallel, independent)
5. **Spectral-to-topology mapping on small LNNs** (parallel, independent)
6. **Design between-regime LNN predictors** (synthesis — after 1-5)

Steps 1-3: existing infrastructure. Steps 4-5: parallel, independently
publishable. Step 6: the synthesis.

## Target Models

- **Development**: Pythia-410m (24 layers, manageable)
- **Primary**: Qwen2.5-7B-Instruct
- **Hardware**: 3090 Ti 24GB (Qwen inference + caching), CPU for analysis,
  Colab Pro A100 for larger models

## Longer-Term: Hybrid Architecture (Speculative)

Transformer layers do within-regime processing via attention. At regime
boundaries (detected by frame_ratio peaks), small LNN predictors model
cross-regime transitions in continuous time. The discrepancy between LNN
prediction and actual next-regime representation is a surprise signal.

Key insight: LNN wiring topology → graph Laplacian eigenspectrum → dynamical
modes. frame_ratio curve specifies needed timescale structure → constrains
wiring topology. Architecture designed from diagnostic, not searched.

**Not implemented. Gated on steps 1-5 producing clean results.**

## Connections

- `probe-signal-comparison`: geometric metrics vs tuned lens is a subset
  of the projection suite comparison
- `tools/analysis/geometry/`: concentration as baseline comparison for
  bisimulation probe
- VVVVVV: ve decomposition is the same timescale separation problem
- SAE dark matter: intersection of projection null spaces isolates
  endogenous scaffolding (geometric analog of attention sinks)

## Unvalidated Claims (to test)

- [ ] frame_ratio peaks are stable across models and prompts
- [ ] frame_ratio peaks correspond to functionally meaningful regime boundaries
- [ ] LNN spectral-to-topology mapping preserves dynamical properties
- [ ] bisimulation metric diverges from task-conditioned embeddings at scale
- [ ] frame_ratio curve predicts about-to-be-wrong before output distribution
