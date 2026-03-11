# Agent Notes — claude-sonnet-4-6

## Summary

Scaffolded the Quelle monorepo (CLAUDE.md, experiments/, wiki/) and fully
implemented `variable-bitrate-reasoning` (model, data, train, evaluate).
Smoke-tested successfully. Awaiting first full training run.

> When this file exceeds ~80 lines, move old sessions to
> `archive/claude-sonnet-4-6-YYYY-MM.md` and update this summary.

---

## Session 2026-02-28

**Branch**: `claude/setup-experiments-directory-x5kqf`

**Work done**:
- Created `CLAUDE.md` — org structure, naming conventions, agent protocols,
  commit format, gitignore defaults.
- Scaffolded `experiments/variable-bitrate-reasoning/` — README, RESULTS,
  .gitignore, configs/default.yaml, src stubs (model, data, train, evaluate,
  visualize), placeholder directories.
- Scaffolded `wiki/` — index README, four concept pages, agent notes, human
  notes (roadmap, decisions).

**Experiments touched**: `variable-bitrate-reasoning` (planning → skeleton committed)

**Status at end of session**: Scaffold complete. Implementation stubs in place.
No code has been run yet. Ready for an implementation agent to pick up.

~~Handoff Notes~~ — all stubs implemented as of 2026-03-07 session (see below).

~~Open Questions~~ — both resolved; see experiment README and decisions.md 2026-03-07.

---

## Session 2026-03-07

**Branch**: `claude/setup-experiments-directory-x5kqf` (merged)

**Work done**:
- `src/data.py`: `_gen_expr()` recursive binary-tree generator; `collate_fn`
  concatenates problem + answer + EOS tokens, pads to batch max length, returns
  `(padded_ids, prob_lengths, difficulties)`.
- `src/model.py`: per-example `concentration()` → `(B,)` tensor; `compress()`
  with per-example `lambda_t` broadcast over dimension axis; `VariableRateReasoner`
  using individual `TransformerEncoderLayer` modules for intermediate state access;
  causal + float padding masks; post-norm `LayerNorm`; `_init_weights`.
- `src/train.py`: causal LM training with answer-only loss mask (via
  `prob_lengths`), curvature penalty weighted by per-example `(1 - conc)`,
  gradient clipping (max_norm=1.0), CSV logging flushed every step.
- `src/evaluate.py`: greedy generation, per-difficulty accuracy, λ–concentration
  Pearson r with r < -0.3 criterion, baseline comparison via deep-copy +
  compression-head override.
- `configs/default.yaml`: added `model.max_seq_len: 128`; fixed `lr` to
  `0.0001` (PyYAML parses `1e-4` as a string).

**Experiments touched**: `variable-bitrate-reasoning` (planning → running)

**Status at end of session**: All `src/` stubs implemented except `visualize.py`.
Smoke-tested: 8-step training run passes, forward + backward verified.
Ready to run full training (`python3.11 -m src.train --config configs/default.yaml`).

---

## Session 2026-03-09

**Branch**: `claude/setup-vvvvv-phase-0-OWK85`

**Work done**:
- Read `experiments/VVVVVV/ve_implementation_plan.md` in full.
- Created experiment skeleton: `README.md`, `RESULTS.md`, `DECISIONS.md`,
  `.gitignore`, directory stubs (`src/`, `outputs/`, `configs/`).
- Implemented `src/phase0_diagnostics.py` — three Phase 0 probes:
  - `probe_spike_channels()`: hook-based measurement of mean |activation| per
    channel at each ve-layer; computes overlap fraction with gate's [:32] window.
  - `probe_bos_stability()`: collects BOS token residuals per ve-layer, computes
    within-batch vs cross-batch cosine similarity, classifies as document-signal /
    pure-sink / ambiguous.
  - `eval_with_ve_ablated()`: saves, zeros, evals, restores all value_embeds;
    returns baseline bpb, ablated bpb, delta.
  - `run_phase0()`: convenience runner that calls all three and optionally writes
    JSON output.
- Implemented `src/run_phase0.py` — CLI runner with checkpoint loading, fallback
  DataLoader (numpy memmap), eval_fn, and implication guidance in output.
- Updated `STATUS.md` (added VVVVVV to active experiments, Open Questions).
- Documented implementation decisions in `DECISIONS.md` (D1–D5).

**Experiments touched**: `VVVVVV` (stub → Phase 0 probes implemented)

**Status at end of session**: Phase 0 probe code complete. Architecture
unchanged (nanochat/gpt.py not modified). Awaiting: nanochat checkpoint at
d12 or d16 scale to run `python src/run_phase0.py --checkpoint ... --data_dir ...`.

## Session 2026-03-11

**Branch**: `claude/fix-nanochat-training-error-d3nlZ`

**Work done**: Conceptual discussion, no code changes.
- Updated `experiments/VVVVVV/README.md` — expanded Open Questions with:
  scale-dependence of ve benefit (capacity-hack vs. interference-reduction),
  relu² specificity and SwiGLU non-transferability, Phase 0 as mechanism
  characterisation independent of VVVVVV hypotheses.
- Created `wiki/concepts/constraint-forcing-function.md` — research methodology
  page documenting the pattern: constrained architecture → forced discovery →
  hack characterisation → generalisation or architectural motivation. Includes
  historical examples (induction heads, attention sinks → registers, GQA) and
  application to VVVVVV. Includes soft-register-for-inference-context as a
  future direction stub.
- Updated `wiki/README.md` index.

**Key discussion points**:
- ve gate reduced 32→12 channels empirically and improved; this suggests either
  gate reads a few spike channels (small minority) or gate is doing something
  very low-dimensional (near magnitude threshold).
- If BOS is a pure sink, register tokens (Darcet et al.) are the minimal
  architectural fix — free BOS from sink pressure, let gradient flow to ve gate
  create pressure for BOS to encode document structure.
- MoE analogy: VVVVVV wants document-conditional routing (coarser than token);
  standard MoE does per-token routing. The gap is exactly the BOS sink problem.
- nanochat constraints are valued as forcing functions, not goals. The research
  interest is in what hacks the constrained network discovers, not in nanochat
  results per se.
- Scale dependence is the critical unknown: "ve is load-bearing" is established
  only at nanochat scales. Capacity-hack hypothesis predicts decay with scale;
  interference-reduction hypothesis predicts stable or increasing benefit.
- relu² spike channels may not generalise to SwiGLU — a hard limit on
  transferability of any gate mechanism that relies on spike structure.

**Experiments touched**: `VVVVVV` (README updated), wiki (new concept page)

**Status at end of session**: No code changes. Awaiting d12 training run for
Phase 0 diagnostics.

## Session 2026-03-11 (continued)

**Branch**: `claude/fix-nanochat-training-error-d3nlZ`

**Work done**:
- Fixed gate_window 32→12 throughout: `run_phase0.py` (default + auto-detect
  from model config), `colab_train.ipynb` (GATE_WINDOW config + pass-through),
  `README.md` hypothesis / Q0.1 / dependencies, `DECISIONS.md` D5 cross-ref.
- Added DECISIONS.md D13: gate_window empirical origin, gradient pressure
  dynamics, arbitrary-first-indices problem, domain specificity, learned
  projection as cheap fix.
- Added DECISIONS.md D14: Q0.3 metric validity caveat — bpb delta may be small
  or negative due to sequence length, distribution fit, or ablation routing;
  treat as directional evidence only.
- Extended README Open Questions: full write-up of Q0.3 metric validity concern
  and gate channel placement / double-duty / ideal channel analysis.

**Key discussion**:
- Spike channel inside gate window = double duty (residual content + gate signal)
  = possible wasted capacity. "Informative" overlap ≠ efficient gate.
- Ideal gate channels: low-magnitude for main pathway, high-variance for gating.
  Q0.1's overlap metric can't distinguish this from busy shared channels.
- Fixed `[:12]` slice is arbitrary (karpathy nanochat discussion, ref needed).
  Learned projection (3,072 params at d12) removes the problem cleanly.
- Q0.3 metric risk: ClimbMix bpb on 2048-token sequences may not be sensitive to
  multi-timescale ve contribution; delta could be near-zero or negative without
  ve being unimportant.
- Broader concern: autoresearch tuned gate_window=12 on ClimbMix metrics — the
  gate and its read channels are shaped by that distribution. Results may not
  transfer to other corpora without re-tuning.

## Open Questions

- Q0.1: Do spike channels (relu²) fall in [:12]? And if so, are those channels
  doing double duty (residual + gate) vs dedicated gate-signaling lanes? The
  overlap fraction alone doesn't distinguish.
- Q0.2: Is BOS residual document-varying or near-constant? → BOS conditioning
  viability.
- Q0.3: Is val_bpb on ClimbMix the right metric, or do we need longer-document
  or domain-shifted eval to see ve's actual functional load?
- Scale dependence: does ve benefit decay with model size? (not addressed in
  current phases — carry as interpretive caveat on Phase 0 results)
- gate_window domain specificity: is 12 right for ClimbMix only, and what would
  a learned projection gate reveal about the actual optimal subspace?
