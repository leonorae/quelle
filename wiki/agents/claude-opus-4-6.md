# Agent Notes — claude-opus-4-6

## Summary

Session protocol analysis and refinement. No experiments touched.

> When this file exceeds ~80 lines, move old sessions to
> `archive/claude-opus-4-6-YYYY-MM.md` and update this summary.

---

## Session 2026-03-15

**Branch**: `claude/parallel-experiments-orchestration-eCATm`

**Work done**:
- Audited actual session practices against CLAUDE.md prescriptions.
- Researched external best practices (session-handoff tools, Beads, centralized
  context management, multi-agent frameworks). Assessed most as overengineered
  for this repo's scale and ethos.
- Rewrote `wiki/agents/README.md`:
  - Added strict startup checklist (3 items: STATUS.md, experiment README,
    recent agent sessions on same experiment).
  - Added strict session-end checklist (4 items: experiment status, STATUS.md,
    agent notes, RESULTS.md).
  - Added handoff notes guidance: be specific or don't bother.
  - Kept conventions section, updated active agents table.
- Archived `claude-sonnet-4-6.md` sessions 2026-02-28 and 2026-03-07 to
  `archive/claude-sonnet-4-6-2026-03.md`. Trimmed active file to ~45 lines.
- Added repo-level decision to `wiki/humans/decisions.md` documenting the
  protocol refinement rationale and scheduling a reassessment for 2026-04-15.

**What I chose NOT to do** (and why):
- No JSON schemas for session logs — adds tooling dependency, no demonstrated
  need at current agent count (1-2 sequential).
- No automated handoff chaining — the repo is small enough that reading 2 files
  (STATUS.md + agent notes) gives full context.
- No enforcement of WIP checkpoints — the ~30min commit cadence in CLAUDE.md is
  aspirational. In practice, agents commit once per session. This is a known gap
  but not worth adding process for until sessions start crashing and losing work.
- Did not touch CLAUDE.md — the protocol there is fine as aspirational guidance.
  The strict parts live in `wiki/agents/README.md` where agents actually look.

**Experiments touched**: none

**Status at end of session**: Protocol refinements committed. No blockers.
