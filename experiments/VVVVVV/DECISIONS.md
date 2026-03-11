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
  <20% spike-channel overlap with [:12]. If overlap is high, current gate
  may already be reading informative channels. See D13 for why the overlap
  criterion alone may be insufficient even in the high-overlap case.
- **BOS-conditioned table (§6.2):** Implement only if Phase 0 Q0.2 shows
  BOS varies across documents. If BOS is a pure sink, this conditioning adds
  no document-level signal.

These are hard gates, not soft preferences. Do not preempt Phase 0.

---

## 2026-03-09 (training setup)

### D6 — nanochat as git submodule at repo root

**Decision:** `karpathy/nanochat` is added as a git submodule at `quelle/nanochat/`
(repo root), not inside the experiment directory.

**Rationale:** nanochat is a shared dependency across all VVVVVV phases and
potentially future experiments. Placing it at the repo root avoids duplication
and makes it available to the wiki and other tools.

**Implication:** Run `git submodule update --init` after cloning. Pin the
submodule to a specific commit before beginning Phase 1 to ensure reproducibility.

---

### D7 — Training data in outputs/nanochat_base/ (gitignored)

**Decision:** `NANOCHAT_BASE_DIR` defaults to `experiments/VVVVVV/outputs/nanochat_base/`.
All nanochat intermediates (parquet data shards, tokenizer, checkpoints) are stored
there and gitignored.

**Rationale:** Keeps experiment artifacts self-contained. Large files (parquet
shards, model checkpoints) must not be committed. The default can be overridden
via the `NANOCHAT_BASE_DIR` env var for shared compute setups where data lives
on a separate disk.

---

### D8 — 30 ClimbMix shards for Phase 0 d12 baseline

**Decision:** Default `N_SHARDS=30` in `setup.sh`.

**Rationale:** nanochat docs say ~170 shards for GPT-2-level training (speedrun).
d12 with 6000 steps × 524288 tokens/step ≈ 3.1B tokens; 30 shards should cover
this with room to spare. If training data runs out before 6000 steps, increase
`N_SHARDS` and re-run `setup.sh` (download is idempotent).

---

### D10 — T4 training workarounds: window-pattern L and no torch.compile

**Decision:** `train_d12.sh` passes `--window-pattern L` and sets
`TORCHDYNAMO_DISABLE=1`.

**Rationale:**
- Tesla T4 (SM 75, pre-Ampere) cannot run Flash Attention 3. nanochat falls back
  to PyTorch SDPA.
- SDPA cannot implement sliding window attention, so the default `SSSL` pattern
  causes it to materialise full 2048×2048 attention matrices for every layer.
  At batch=32 this is ~3 GB per layer, causing OOM during the forward pass.
- With `--window-pattern L` all layers use full causal attention, which SDPA
  handles via its flash-attention-like fused kernel without materialising the
  full matrix.
- `torch.compile` hangs for 50+ minutes on T4 during triton kernel compilation
  without producing a single training step. `TORCHDYNAMO_DISABLE=1` runs in
  eager mode, which is slightly slower per step but actually runs.

**Implication:** Trained model uses full causal attention (no sliding window).
This is fine for Phase 0 diagnostics; architectural variants are gated on Phase 0
results anyway (D5).

---

### D12 — PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

**Decision:** Export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `train_d12.sh`.

**Rationale:** After halving device batch size to 16 (D11), the logits tensor
needs 4 GiB. The T4 had only 1.11 GiB free but 3.96 GiB reserved-but-unallocated
in fragmented blocks the allocator couldn't combine into a contiguous region.
`expandable_segments:True` lets the allocator grow segments non-contiguously,
resolving the fragmentation. No functional change to training.

---

### D11 — device-batch-size 16 for T4

**Decision:** `train_d12.sh` passes `--device-batch-size=16`.

**Rationale:** The logits tensor shape is [batch, seq, vocab] = [B, 2048, 32768].
At the default `--device-batch-size=32` this is 32×2048×32768×4 bytes = 8 GiB,
which exceeds the T4's ~4.7 GiB of free VRAM after model and optimizer states
are loaded. Halving to 16 reduces logits to 4 GiB. Gradient accumulation steps
automatically double (8→16) so total batch size (524,288 tokens) is preserved.

---

### D9 — run_phase0.py uses nanochat's checkpoint_manager.build_model()

**Decision:** `run_phase0.py` loads checkpoints via `nanochat.checkpoint_manager.build_model()`
rather than raw `torch.load()`.

**Rationale:** nanochat does not save a single `ckpt["model"]`/`ckpt["config"]` dict.
It saves `model_{step:06d}.pt` (flat state dict) and `meta_{step:06d}.json` (config)
separately. `build_model()` handles both, applies key-name patching for old
checkpoints, and returns an initialised eval-mode model. Using it directly avoids
reimplementing fragile checkpoint-loading logic.

---

## 2026-03-11 (config)

### D15 — ve_weight_decay: OPEN, not yet pinned

**Status:** Blocked. `--ve-weight-decay` is not a recognised CLI arg in
`nanochat/scripts/base_train.py` (`unrecognized arguments` error at runtime).
The flag has been removed from `train_d12.sh` pending resolution.

**Context:** autoresearch #43 finding ("ve weight decay 0.001–0.003 improves;
0.005 regresses") refers to AdamW optimizer group configuration inside nanochat's
training code — not an exposed CLI flag. The nanochat submodule is currently
empty (not initialised), so the actual default and the correct mechanism are
unknown.

**Required action (before Phase 0 run is valid):**
1. Initialise the nanochat submodule and inspect `scripts/base_train.py` for how
   ve embedding tables are assigned to optimizer groups and what weight decay is
   applied to them.
2. If the current default is already in 0.001–0.003: document it in this entry
   and proceed. No code change needed.
3. If not: add a `--ve-weight-decay` CLI flag to `base_train.py` (patch nanochat
   submodule), apply it to the ve optimizer group, and restore the flag in
   `train_d12.sh`.

**Desired value once mechanism is understood:** 0.001 — conservative lower bound
of the known-good range; least regularization known to improve. 0.002 (midpoint)
is equally arbitrary; 0.003 risks unnecessary shrinkage for a diagnostic baseline.

---

### D16 — configs/d12_baseline.yaml as documentation, no config loader

**Decision:** Create `configs/d12_baseline.yaml` documenting the full effective
configuration. No config loader.

**Rationale:** A config loader would require either wrapping nanochat's CLI
(fragile — nanochat is a submodule with its own arg surface) or reimplementing
nanochat's argument parsing. For a single Phase 0 training run, this adds
complexity with no benefit. The YAML file serves as a human record of what the
effective config is, including nanochat defaults we're implicitly relying on.

A loader becomes worth building in Phase 1 if we run sweeps (k variants,
gate_window variants, etc.) that require programmatic configuration. Revisit
then.

---

## 2026-03-11

### D13 — gate_window=12: empirical origin, gradient dynamics, and open limitations

**Decision recorded:** gate_window updated from 32 to 12 throughout experiment
scripts and probes. Codified as the default in `run_phase0.py` with auto-detection
from `model.config.ve_gate_channels`.

**Origin:** autoresearch #43 reduced the gate's fixed read window from 32 to 12
empirically during post-training hyperparameter search. This reduction improved
val metrics on ClimbMix. It was not derived from a principled analysis of which
channels carry gate-useful information.

**Gradient dynamics of the fixed slice:**
The gate reads `x[:, :, :gate_window]` — a fixed prefix of the residual stream.
Channels 0–11 receive gradient from both the standard next-token prediction path
and the ve gate path. Channels 12+ receive no gradient from the gate. This creates
differential pressure: the model learns to route gate-useful information into
channels 0–11, competing with whatever those channels already carry.

Implications:
- If spike channels (high mean |activation|) land inside the gate window, they
  are doing double duty: carrying residual content *and* carrying gate signal.
  This may represent wasted capacity — bandwidth split between two objectives.
- Conversely, if gate channels are low-magnitude for the main pathway but
  high-variance for gating, the fixed slice has spontaneously created dedicated
  gate-signaling lanes. Q0.1's overlap metric cannot distinguish these two cases.
- The optimal gate input is channels that are maximally discriminative for gating
  with minimal interference to the residual prediction path — not necessarily the
  highest-magnitude channels.

**Arbitrary first-indices problem:**
The `[:12]` slice is arbitrary. The original decision to use the first k channels
rather than a learned projection was a simplification (noted in karpathy's nanochat
discussion; exact reference needed). Whether channels 0–11 happen to be good gate
inputs is an empirical question, not a design guarantee.

**Domain specificity:**
gate_window=12 was tuned on ClimbMix (webtext-like). The optimal window may differ
for other distributions — corpora with more structural heterogeneity (code, math,
multilingual) may require more dimensions to distinguish gating contexts where the
same token has qualitatively different roles.

**Cheap fix if Q0.1 is inconclusive or unsatisfactory:**
A learned linear projection W_gate ∈ ℝ^(12×d_model) replaces the fixed slice.
At d12 (d_model=256): 3,072 extra parameters — negligible. Removes the
arbitrary-index problem and allows gradient to select the optimal gate subspace
rather than defaulting to the first 12 residual coordinates. Gated on Phase 0
Q0.1 results per D5, but worth considering even in the "informative" case.

---

### D14 — Q0.3 metric validity caveat: ClimbMix bpb may not capture ve's contribution

**Decision recorded:** Q0.3 results must be interpreted with explicit awareness
that the metric may be insensitive to ve's actual functional role.

**Concern:** val_bpb on ClimbMix val sequences is a proxy for ve's contribution.
It may be near-zero, small, or even slightly negative without this meaning ve is
unimportant. Specific failure modes:

1. **Sequence length / context window:** If ve's benefit is primarily multi-timescale
   coherence over long documents, the ~2048-token eval window may not expose it.
   The ablation cost would appear small.
2. **Distribution fit:** The gate and ve table were trained on ClimbMix. val_bpb
   on ClimbMix val is the metric the network was directly optimised for. This
   could overstate ve's contribution (metric the network was optimised on) or
   understate it (if ve's benefit is orthogonal to next-token prediction on
   short webtext sequences).
3. **Negative delta is possible:** zeroing ve may cause the model to route around
   it for some predictions, producing a near-zero or slightly negative measured
   delta. This would not mean ve is harmful — it would mean the ablation is
   incomplete (other parameters partially compensate within the eval).

**Interpretation rule:** Q0.3 delta is directional evidence. Small or near-zero
does not close the question. If delta is small, consider whether a targeted eval
on longer documents or documents with explicit long-range structure would be more
sensitive before drawing conclusions about ve's functional load.
