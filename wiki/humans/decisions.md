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

---

## 2026-03-15 — Session protocol: strict checklists, loose everything else

**Decision**: Added strict startup and session-end checklists to
`wiki/agents/README.md`. These are the only mandatory process steps. Everything
else in CLAUDE.md's agent session protocol remains aspirational guidance.

**What's strict** (3 startup items, 4 end items — see `wiki/agents/README.md`):
- Read STATUS.md, experiment README, recent agent sessions.
- Update experiment status, STATUS.md, agent notes, RESULTS.md.

**What's deliberately left loose**:
- WIP checkpoint cadence (~30min commits). Aspirational. Not enforced.
- wiki/findings/ contributions. Judgment call per session.
- Handoff note depth. "Be specific or don't bother" — no template.

**What was considered and rejected**:
- JSON schema validation for session logs (tooling overhead, no demonstrated need).
- Automated handoff chaining (repo is small enough that 2 files give full context).
- Centralized context management systems (enterprise solution for a 1-2 agent
  research repo).

**Rationale**: The trends in multi-agent session management (2026) skew toward
enterprise coordination tooling — schema validation, MCP servers, centralized
state stores. These solve real problems at scale but add process overhead that
fights research velocity at this repo's size. The existing pattern (STATUS.md +
agent notes + experiment READMEs) works. The gap was not tooling but clarity:
which steps are mandatory vs aspirational. Now that's explicit.

**Reassess**: 2026-04-15, or sooner if: (a) sessions start crashing and losing
work (→ enforce WIP checkpoints), (b) a third agent type joins and handoffs
degrade (→ consider structured handoff format), or (c) the repo grows past ~10
active experiments (→ consider richer indexing).
