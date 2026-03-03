# Agent Notes — claude-sonnet-4-6

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

## Handoff Notes

The experiment spec lives entirely in the top-level `README.md`. The stubs in
`src/` expose the intended interfaces; the primary gaps are:

1. `src/model.py` — `VariableRateReasoner.forward()` needs the layer-by-layer
   loop with compression and stat collection.
2. `src/data.py` — `generate_arithmetic_problem()` needs the recursive
   expression generator.
3. `src/evaluate.py` and `src/visualize.py` — fully stubbed, implement after
   training works.

## Open Questions

- Two-pass vs. single-pass forward for future prediction target (spec Note 2).
  Currently defaulting to single-pass. Flag for human decision if results are
  suspicious.

---

## Session 2026-03-03

**Branch**: `claude/agent-context-document-YMpBX`

**Work done**:
- Created `experiments/geometric-self-awareness-reasoning/` — full directory
  skeleton per CLAUDE.md conventions.
- Wrote `CONTEXT.md` — comprehensive context document for an implementer agent
  covering the big-picture vision, Phase 0B–3 protocol, geometric metrics,
  RIS rubric, RLM connection, references, and success criteria.
- Wrote `src/phase0b_generate_traces.py` — complete, production-ready
  generation script (resumable, crash-safe, chat-template-aware, summary
  report included).
- Wrote stub scripts for Phases 1–3 (`phase1_ris_scoring.py`,
  `phase2_extract_geometry.py`, `phase3_analysis.py`).
- Added `configs/phase0b.yaml` with all generation hyperparameters.
- Created `wiki/concepts/ris-scoring.md` — RIS rubric, judge ensemble
  protocol, prompt template, cost estimate.
- Updated `wiki/README.md` — added RIS concept page and agent table entry.
- Updated `wiki/humans/roadmap.md` — added new experiment to planned list.

**Experiments touched**: `geometric-self-awareness-reasoning` (skeleton + context committed)

**Status at end of session**: Experiment skeleton and context document complete.
Phase 0B script ready to run. Phases 1–3 are stubs. No code has been executed.

## Handoff Notes

The implementer agent should:
1. Read `experiments/geometric-self-awareness-reasoning/CONTEXT.md` first.
2. Run `src/phase0b_generate_traces.py` with a 24GB+ GPU and Qwen2.5-7B-Instruct.
3. Verify pass@1 > 70% on the 1000-problem pilot.
4. Update `RESULTS.md` Phase 0B section and commit.
5. Proceed to Phase 1 (RIS scoring) once traces are clean.

## Open Questions (from this session)

- GPU availability for the implementer running Phase 0B.
- Whether to use vLLM (speed) or plain Transformers (simplicity) for generation.
- Phase 1 API budget and which LLM judge provider (OpenRouter vs. Together.ai).
- Whether ProcessBench/PRMBench should complement or replace GSM8K.
