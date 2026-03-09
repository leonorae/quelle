# VVVVVV — Implementation Decisions

Decisions specific to this experiment. Repo-level decisions live in
`wiki/humans/decisions.md`.

---

## 2026-03-09

### D1 — Phase 0 probe implementation as standalone module

**Decision:** Phase 0 diagnostics are implemented as a single standalone
module (`src/phase0_diagnostics.py`) importable against any nanochat
checkpoint. No modifications to nanochat's `gpt.py` for Phase 0.

**Rationale:** The diagnostic phase must not touch the architecture under
study. Hook-based probing (register_forward_hook) lets us instrument the
existing model without modifying it. This also means Phase 0 can run against
any existing nanochat checkpoint without retraining.

**Implication:** `src/phase0_diagnostics.py` imports nanochat's `gpt.py`
and `has_ve()`. The nanochat repo must be importable from the experiment's
working directory (submodule, symlink, or PYTHONPATH extension).

---

### D2 — Run scale: d12 for Phase 0

**Decision:** Phase 0 runs at d12 (nanochat's small config) unless d16 is
needed for signal quality.

**Rationale:** The plan specifies "d12 or d16 for fast iteration." d12 is
faster; the diagnostic questions (channel overlap, BOS stability, ablation
delta) do not require large models.

**Override condition:** If Phase 0 Q0.1 and Q0.2 signals are too noisy at
d12 (e.g., cosine similarities have high variance across batches), escalate
to d16 before drawing conclusions.

---

### D3 — n_batches defaults

**Decision:** Default `n_batches=50` for `probe_spike_channels`, `n_batches=20`
for `probe_bos_stability`, as specified in the implementation plan.

**Rationale:** Spike channel measurement averages absolute activations — 50
batches gives stable estimates. BOS stability uses cosine similarity which
converges faster — 20 batches is sufficient. The plan sets these explicitly;
do not optimise without empirical justification.

---

### D4 — Phase 0 Q0.2: cross-doc measurement uses different sequences in same batch

**Decision:** "Cross-document" cosine similarity is computed between BOS
residuals from *different sequences in the same batch*, not across separate
batches.

**Rationale:** nanochat's dataloader concatenates documents and packs them
into fixed-length sequences. BOS tokens appear at document boundaries within
a batch. Pairs within the same batch are computationally cheap and
statistically equivalent to cross-batch pairs for this measurement.

---

### D5 — Open questions gated on Phase 0

The following architectural decisions are explicitly deferred until Phase 0
results are in hand:

- **Learned projection gate (§6.1):** Implement only if Phase 0 Q0.1 shows
  <20% spike-channel overlap with [:32]. If overlap is high, current gate
  may already be reading informative channels.
- **BOS-conditioned table (§6.2):** Implement only if Phase 0 Q0.2 shows
  BOS varies across documents. If BOS is a pure sink, this conditioning adds
  no document-level signal.

These are hard gates, not soft preferences. Do not preempt Phase 0.
