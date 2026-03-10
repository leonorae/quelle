# CLAUDE.md — Quelle

Primary reference for all agents and humans. Read before creating files or committing.

---

## What this repo is

Frontier research monorepo. Truth-seeking; no metrics, quotas, or need to appear
directed. Failures and pivots are data. Question assumptions and steer at any time.

Three strictly separate concerns:

| Location | Contains |
|---|---|
| `experiments/` | Runnable code, configs, outputs, per-experiment notes |
| `wiki/` | Knowledge: concepts, findings, cross-experiment synthesis, agent notes |
| `tools/` | Shared utilities with demonstrated reuse across ≥2 experiments |

**Never mix these.** If it runs, it belongs in `experiments/` or `tools/`. If it explains,
it belongs in `wiki/`.

---

## Directory layout

```
quelle/
├── CLAUDE.md
├── STATUS.md                 ← current briefing (read this on session start)
├── README.md
├── nanochat/                 ← git submodule (karpathy/nanochat)
│
├── experiments/<slug>/
│   ├── README.md             ← hypothesis, status, owner, dependencies
│   ├── RESULTS.md            ← post-run findings
│   ├── DECISIONS.md          ← implementation decisions for this experiment
│   ├── src/
│   ├── configs/
│   ├── data/                 ← gitignore large files
│   ├── outputs/              ← gitignore large files
│   └── notebooks/
│
├── tools/
│   ├── training/             ← compute deployment (Colab, etc.), experiment-agnostic
│   └── analysis/             ← shared probe/viz utilities, organised by probe target
│       ├── residual/
│       ├── attention/
│       └── ...
│
└── wiki/
    ├── README.md             ← index
    ├── library/              ← papers, references, philosophical grounding
    ├── concepts/             ← theory, shared definitions
    ├── findings/             ← cross-experiment synthesis
    ├── agents/               ← per-agent working notes and handoffs
    │   └── archive/
    └── humans/               ← human-authored notes, decisions, roadmap
```

---

## Antidirectives

Things agents must **not** do without explicit instruction:

- **Do not move code to `tools/`** until it is needed by ≥2 separate experiments.
  Keep it in `experiments/<slug>/src/` until then.
- **Do not preempt phase-gated decisions.** If a DECISIONS.md entry says "implement
  only if X", do not implement it speculatively. Wait for the gate condition.
- **Do not create sibling experiments for hyperparameter variants.** Variants live
  inside the same experiment directory as separate configs or result subdirectories.
- **Do not commit large binary files** (models, datasets, checkpoints). Gitignore
  them and document where they are stored.
- **Do not generalise prematurely.** Three similar lines of code across one experiment
  is not sufficient reason to abstract. Wait for genuine cross-experiment reuse.

---

## Experiments

Slug format: `lowercase-hyphen-separated-descriptive`

On starting: create the directory skeleton and fill in `README.md` (hypothesis,
status, owner, dependencies) before writing any code. Commit the skeleton.

Log all non-obvious implementation choices in `DECISIONS.md`.
Repo-level decisions go in `wiki/humans/decisions.md`.

---

## Tools

`tools/training/` — compute deployment utilities. Cross-experiment by design from
the start; create entries here when building deployment infrastructure.

`tools/analysis/` — shared probe and visualisation utilities. Subdirectories
organised by what they examine (residual stream, attention, gates, …).
**Do not add here until a utility is needed by a second experiment.**
Until then, keep analysis code in `experiments/<slug>/src/`.

---

## Commit format

```
<type>(<scope>): <short description>
```
Types: `feat` `fix` `data` `results` `docs` `refactor` `chore`
Scope: experiment slug, `wiki`, `tools`, or `repo`

Keep experiment changes and wiki updates in separate commits.

---

## Agent session protocol

**Start:** Read `STATUS.md`. Check `experiments/<slug>/README.md` for the active
experiment. Consult `wiki/` only if you need deeper background.

**End:** Update experiment `README.md` status. Update `STATUS.md` if anything
changed. Write non-obvious findings to `wiki/findings/`. Append a short entry to
`wiki/agents/<your-id>.md` (rotate to archive when file exceeds ~80 lines). Commit.

**Questions:** `<!-- QUESTION: ... -->` in the relevant file, or `## Open Questions`
in `wiki/agents/<your-id>.md`.

---

## wiki/ vs code comments

| wiki/ | code comments |
|---|---|
| Conceptual explanations | Why a specific line works the way it does |
| Cross-experiment patterns | Inline TODOs |
| Design decisions and rationale | Type/arg documentation |
| Reading lists, references | Implementation notes for one function |

---

## Dashboard (planned)

Will read YAML frontmatter from `experiments/*/README.md` and `wiki/findings/*.md`.
Keep those frontmatter blocks intact and parseable.

Wishlist: StumpWM/Emacs integration, chat logging, PDF annotation, question↔finding
linkage, context-aware agentic experiment management.
