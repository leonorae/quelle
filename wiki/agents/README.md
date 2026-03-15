# Agent Notes

Each agent working in this repo keeps a file here named after their agent ID.

---

## Startup Checklist (strict)

Every agent, every session. No exceptions.

1. **Read `STATUS.md`** — current experiments, blockers, dependency graph.
2. **Read `experiments/<slug>/README.md`** for whatever you're working on.
3. **Scan `wiki/agents/` for recent sessions on the same experiment** — avoid
   redoing work or missing context from a prior agent.

That's it. If you need deeper background: `wiki/concepts/`, `wiki/findings/`,
experiment `DECISIONS.md`. But don't read speculatively — read when a question
arises.

## Session End Checklist (strict)

Before closing, in this order:

1. **Update experiment `README.md` status** if it changed (YAML frontmatter).
2. **Update `STATUS.md`** if anything changed — new experiments, status
   transitions, resolved questions, dependency graph shifts.
3. **Append a session entry to `wiki/agents/<your-id>.md`** — what you did,
   what changed, what's unfinished. Include the branch name.
4. **Update `RESULTS.md`** if you produced or analyzed results. Write it
   assuming the reader has no access to raw data.

If you have non-obvious findings that matter across experiments, write them to
`wiki/findings/`. This is a judgment call, not a requirement.

## Handoff Notes

When ending a session mid-task, include in your session entry:

- **What's unfinished** and where you stopped (file paths, line numbers).
- **What commands to run** to pick up where you left off.
- **What you tried that didn't work** — saves the next agent from repeating it.

Generic handoffs ("continuing work on X") are worse than no handoff. Be specific
or don't bother.

## Conventions

- File name: `<agent-id>.md` (e.g. `claude-sonnet-4-6.md`, `claude-opus-4-6.md`)
- Append-only during session. Write observations as they happen, not only at
  session end. If the session crashes, the notes survive.
- Archive when file exceeds ~80 lines: move old sessions to
  `archive/<id>-YYYY-MM.md`, keep a summary paragraph at top of active file.
- `## Open Questions` section for anything needing human or cross-agent input.

## Active Agents

| Agent | File | Last Active |
|-------|------|-------------|
| claude-sonnet-4-6 | [claude-sonnet-4-6.md](claude-sonnet-4-6.md) | 2026-03-09 |
| claude-opus-4-6 | [claude-opus-4-6.md](claude-opus-4-6.md) | 2026-03-15 |
