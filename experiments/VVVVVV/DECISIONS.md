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
