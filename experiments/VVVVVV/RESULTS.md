---
experiment: VVVVVV
status: planning
last_updated: 2026-03-09
---

# VVVVVV — Results

*Fill in after each phase completes.*

---

## Phase 0 — Diagnostic Baseline

**Run config:** (fill in: model scale, dataset, training steps, checkpoint path)

### Q0.1 — Spike Channel Overlap

| ve-layer | top-32 channel indices | overlap with [:32] | interpretation |
|---|---|---|---|
| (fill after run) | | | |

**Threshold:** >50% → gate reads informative channels. <20% → gate reads near-noise.

**Finding:** (fill in)

**Implication for Phase 6.1 (learned projection gate):** (fill in after Q0.1)

---

### Q0.2 — BOS Residual Stability

| ve-layer | within-doc mean cosine | cross-doc mean cosine | classification |
|---|---|---|---|
| (fill after run) | | | |

**Classification key:**
- within >> cross: BOS carries document signal → BOS conditioning viable
- both high: pure sink (near-constant) → BOS conditioning = fixed bias only
- both low: no stable BOS signal

**Finding:** (fill in)

**Implication for Phase 6.2 (BOS-conditioned table):** (fill in after Q0.2)

---

### Q0.3 — ve Functional Load

| Condition | val_bpb | delta vs baseline |
|---|---|---|
| Baseline (ve enabled) | (fill) | — |
| ve ablated (zeroed) | (fill) | (fill) |

**Finding:** (fill in)

**Implication:** (fill in — sets expectations for Phase 1 ablation deltas)

---

## Phase 1 — Collapse Detection

*(not started)*

---

## Phase 2 — Minimal Symmetry-Breaking

*(not started)*

---

## Phase 3 — Assumptive Pressures

*(not started)*

---

## Phase 4 — Task Evaluation

*(not started)*
