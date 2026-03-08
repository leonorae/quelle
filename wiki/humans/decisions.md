# Design Decisions Log

> Repo-level and structural decisions. Implementation decisions for specific
> experiments live in `experiments/<slug>/DECISIONS.md`.

---

## 2026-02-28 — Separate experiments/ and wiki/

**Decision**: Keep all runnable code under `experiments/` and all written
knowledge under `wiki/`. No exceptions.

**Rationale**: In a multi-agent monorepo, agents will scan for code to run and
documents to read. Mixing the two leads to confusion about what is canonical,
what is stale, and what is executable. Strict separation makes both concerns
easier to navigate.

---

## 2026-02-28 — Stub-first experiment scaffolding

**Decision**: Create `src/*.py` stubs with `raise NotImplementedError` rather
than empty files.

**Rationale**: Stubs make the intended interface explicit. An agent picking up
the experiment can see what functions exist and what their signatures are without
having to infer them from the spec. Empty files give no signal.

---

## 2026-03-07 — STATUS.md briefing card

**Decision**: Add a root-level `STATUS.md` that agents read at session start
instead of traversing the full wiki. Update it at the end of each session.

**Rationale**: Reduces mandatory startup reading from ~5 files to 2
(`CLAUDE.md` + `STATUS.md`). The wiki remains available as a reference but is
not required reading for every session.

---

## 2026-03-07 — Rolling agent logs with archive

**Decision**: Keep agent log files under ~80 lines. When exceeded, move old
sessions to `wiki/agents/archive/<id>-YYYY-MM.md` and replace with a
cumulative summary paragraph.

**Rationale**: Agent logs grow linearly with sessions. Future agents only need
the current summary and recent sessions for context; full history is preserved
in archive files but does not consume context window.

---

## 2026-03-07 — Per-experiment DECISIONS.md

**Decision**: Implementation decisions live in `experiments/<slug>/DECISIONS.md`.
Only repo-level structural decisions live in `wiki/humans/decisions.md`.

**Rationale**: Prevents the global decisions file from accumulating
experiment-specific implementation details that are irrelevant to agents working
on other experiments.
