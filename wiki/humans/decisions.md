# Design Decisions Log

> Record non-obvious decisions here so future agents and humans know the reasoning.

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

## 2026-02-28 — Single-pass forward as default

**Decision**: Start with a single-pass forward (mild information leak in future
prediction) rather than the more correct two-pass approach.

**Rationale**: Simplicity first. The two-pass variant doubles compute and adds
complexity. If the information leak turns out to matter empirically, upgrade.
Noted in experiment README Open Questions.
