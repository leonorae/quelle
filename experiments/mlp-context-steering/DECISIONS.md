# MLP Context-Steering — Implementation Decisions

Decisions specific to this experiment. Repo-level decisions live in
`wiki/humans/decisions.md`.

---

## 2026-03-15

### D1 — TransformerLens, not raw hooks

**Decision:** Use TransformerLens `HookedTransformer` for all interventions.

**Rationale:** This experiment requires hook-based MLP input replacement at
arbitrary layers with activation propagation through subsequent layers.
TransformerLens has first-class support for this via `run_with_hooks` and
`hook_fn` signatures. It also has Pythia-410m out of the box. The existing
repo infrastructure uses nanochat's `register_forward_hook` — that pattern
doesn't support interventional ablation (modifying activations mid-forward-pass
and propagating the change).

**Implication:** This is the first experiment in the repo using TransformerLens.
`pip install transformer-lens` is a new dependency. Do not add TransformerLens
to `tools/` or make it a repo-wide dependency — it lives in this experiment's
requirements only.

---

### D2 — erank (entropy-based), not PCA participation ratio

**Decision:** Use entropy-based effective rank:
```
erank(H) = exp(entropy(σ / sum(σ)))
```
where σ are singular values of H.

**Rationale:** The existing `tools/analysis/geometry/effective_dimensionality`
uses PCA participation ratio (count of components explaining X% of variance).
That's a threshold-dependent step function. Erank is continuous and
differentiable in the spectrum — better for detecting gradual rank changes
across layers. The Dong et al. paper uses erank specifically.

**Implication:** Do not import from `tools/analysis/geometry/`. Implement erank
in `src/metrics.py`. If erank proves useful across experiments later, it can
be promoted to tools then.

---

### D3 — Ablation propagates through all subsequent layers

**Decision:** When ablating MLP input at layer l, the modified activation
propagates through layers l+1, ..., L. This is NOT a per-layer isolated
measurement.

**Rationale:** The hypothesis is about cumulative context-steering. Isolating
each layer's ablation would miss the cascading effect. The all-layers-ablated
condition is the primary test; per-layer isolation is a Phase 3 control (3b).

**Implementation:** Use TransformerLens `run_with_hooks` with hooks registered
at every layer simultaneously. Each hook modifies the MLP input in-place (or
returns modified value), and TransformerLens propagates the result forward.

---

### D4 — Two ablation variants, both mandatory

**Decision:** Run both strong and surgical ablation. Neither is optional.

- **Strong:** `x_ablated[pos] = mean_over_positions(x)` for all pos
- **Surgical:** `x_ablated[pos] = (x[pos] - attn_out[pos]) + mean(attn_out)`

**Rationale:** The strong ablation is a confounded but clean test — it removes
ALL per-position variation. The surgical ablation isolates the
attention-derived context-steering specifically (the Dherin mechanism). If
strong shows effect but surgical doesn't, the MLP's non-attention input
variation matters more than context-steering. If both show effect, the
attention-derived component is load-bearing.

---

### D5 — Prompt corpus: sample ~500 diverse prompts

**Decision:** Start with ~500 diverse prompts from a standard corpus (e.g., The
Pile validation set, OpenWebText). If the 7,304-prompt corpus from another
experiment is easily available, use that instead.

**Rationale:** 500 prompts is enough for correlation analysis with reasonable
confidence intervals. The experiment is compute-light (forward passes only),
so the bottleneck is activation caching memory, not FLOPS. 500 prompts ×
24 layers × residual stream size is manageable.

**Override condition:** If cross-prompt correlations (Phase 2c) have wide
confidence intervals at 500 prompts, scale up.

---

### D6 — Sink intensity: BOS attention fraction, averaged across heads

**Decision:** `sink_intensity(l) = mean over heads of (attention mass on
position 0, averaged across query positions)`.

**Rationale:** Attention sinks in the literature are predominantly BOS-directed.
Position 0 is the canonical sink target. Averaging across heads gives a
per-layer summary; per-head breakdowns are available for follow-up but not
the primary metric.

**TRAP potential:** Some prompts may not have BOS at position 0 depending on
tokenizer behavior. Verify TransformerLens's Pythia tokenizer prepends BOS.
If not, use whatever token is at position 0 and document.

---

### D7 — Sequence length: use model's native context (2048)

**Decision:** Use full 2048-token sequences. Pad shorter prompts or truncate
longer ones.

**Rationale:** Rank dynamics and sink formation are sequence-length-dependent.
Using the full context window matches the model's training distribution and
gives the strongest signal. Short sequences would underestimate both effects.

---

### D8 — Output format: per-prompt .pt files + aggregate JSON

**Decision:** Save per-prompt measurements as individual `.pt` files in
`outputs/baseline/` and `outputs/ablated_{variant}/`. Aggregate statistics
in `outputs/summary.json`.

**Rationale:** Per-prompt files enable post-hoc re-analysis without re-running
forward passes. Aggregate JSON is the input to `src/summarize.py` and
`RESULTS.md`. Individual `.pt` files are gitignored; summary JSON is committed.

---

### D9 — Phase ordering: 0 → 1 → 2 → 3, no parallelism

**Decision:** Phases are sequential. Phase 2 requires Phase 0 and 1 outputs.
Phase 3 is a follow-up, not a prerequisite.

**Rationale:** Phase 2 literally computes deltas between Phase 0 and Phase 1
measurements. Phase 3 controls can run after Phase 2 results are known — if
the main signal is null, some controls become unnecessary.

**Override condition:** If Phase 0 reveals something unexpected (e.g., erank
is flat across layers, no sinks form), pause and reassess before running
Phase 1.
