# Agent Notes — claude-sonnet-4-6

## Summary

Scaffolded the Quelle monorepo (CLAUDE.md, experiments/, wiki/) and
implemented `variable-bitrate-reasoning` (model, data, train, evaluate).
Smoke-tested successfully. Built Phase 0 diagnostics for VVVVVV.
Earlier sessions archived in `archive/claude-sonnet-4-6-2026-03.md`.

> When this file exceeds ~80 lines, move old sessions to
> `archive/claude-sonnet-4-6-YYYY-MM.md` and update this summary.

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

## Open Questions

- Q0.1: Do spike channels (relu²) fall in [:32]? → determines whether learned
  projection gate (§6.1) is needed.
- Q0.2: Is BOS residual document-varying or near-constant? → determines whether
  BOS-conditioned table (§6.2) is viable.
