# CLAUDE.md — Quelle Repository Guide

This file is the primary reference for all agents (and humans) working in this
monorepo. Read it before creating files, directories, or committing changes.

---

## Repository Purpose

**Quelle** is a shared monorepo for frontier research. Multiple agents and
humans collaborate here over time. The two core concerns are kept strictly
separate:

| Concern | Location | What belongs here |
|---|---|---|
| **Experiments** | `experiments/` | Runnable code, configs, results, per-experiment notes |
| **Knowledge** | `wiki/` | Concepts, findings, glossary, agent notes, cross-experiment synthesis |

Never mix the two. If you are writing code that runs, it goes in `experiments/`.
If you are writing down what you learned, it goes in `wiki/`.

---

## Research Philosophy Primer
TODO: write this section.
for now, Claude agents should consult my user data for context on my methodology and philosophy when faced with making directorial decisions. If strictly implementing a defined specification, this is less important, particularly if you need your whole context for the task. However, I welcome questioning and re-steering at any time. I do not have metrics or quotas to meet, I do not have a need to appear directed or justify failures, I seek only truth.

## Directory Layout

```
quelle/
├── CLAUDE.md                        ← you are here
├── README.md                        ← human-facing project overview / current spec
│
├── experiments/                     ← one subdirectory per experiment
│   └── <slug>/
│       ├── README.md                ← what this experiment tests, status
│       ├── RESULTS.md               ← findings after the run (fill in post-hoc)
│       ├── src/                     ← source code
│       ├── configs/                 ← hyperparameters, environment specs
│       ├── data/                    ← generated or downloaded data (gitignore large files)
│       ├── outputs/                 ← checkpoints, logs, plots (gitignore large files)
│       └── notebooks/               ← exploratory notebooks (optional)
│
└── wiki/                            ← living knowledge base
    ├── README.md                    ← wiki index / table of contents
    ├── library/                     ← papers, documents, philosophical grounding, inspiration
    ├── concepts/                    ← theory, background, shared definitions
    ├── findings/                    ← cross-experiment synthesis and key results
    ├── agents/                      ← per-agent working notes and handoffs
    └── humans/                      ← human-authored notes, decisions, roadmap
```

---

## Experiment Naming

Slugs must be lowercase, hyphen-separated, descriptive:

```
experiments/variable-bitrate-reasoning/
experiments/manifold-capability-probing/
experiments/gumbel-compression-ablation/
```

Use a new directory for each distinct experiment. Variants (different seeds,
hyperparameters) live under the same experiment directory as separate config
files or result subdirectories, not as sibling experiments.

---

## Starting a New Experiment

1. Create `experiments/<slug>/` with the structure above.
2. Fill in `experiments/<slug>/README.md` immediately (even if partial). Include:
   - **Hypothesis** — what you are testing
   - **Status** — `planning | running | complete | abandoned`
   - **Owner** — agent ID or human handle
   - **Depends on** — other experiments or wiki concepts this builds on
3. Commit the skeleton before writing any code.
4. Log significant decisions in `wiki/agents/<your-id>.md`.

---

## Committing

- Commit message format: `<type>(<scope>): <short description>`
  - Types: `feat`, `fix`, `data`, `results`, `docs`, `refactor`, `chore`
  - Scope: experiment slug or `wiki` or `repo`
  - Example: `feat(variable-bitrate-reasoning): add compression head`
- Keep experiments and wiki updates in separate commits where practical.
- Never commit large binary files (models, datasets). Use `.gitignore` inside
  the experiment directory and document where artifacts are stored.

---

## Agent Protocols

### Starting a session
- Read this file (`CLAUDE.md`).
- Read `wiki/README.md` for current state of knowledge.
- Check the relevant `experiments/<slug>/README.md` for status.
- Append a session entry to `wiki/agents/<your-id>.md`.

### Ending a session
- Update `experiments/<slug>/README.md` with current status.
- Write any non-obvious findings to `wiki/findings/`.
- Commit all changes before ending.

### Asking for help
- Leave a clearly marked `<!-- QUESTION: ... -->` comment in the relevant file,
  or open a section in `wiki/agents/<your-id>.md` headed `## Open Questions`.

---

## What Goes in `wiki/` vs. Code Comments

| Use `wiki/` for | Use code comments for |
|---|---|
| Conceptual explanations | Why a specific line works the way it does |
| Cross-experiment patterns | Inline TODOs |
| Design decisions and rationale | Type/arg documentation |
| Reading lists, external references | Implementation notes specific to one function |

---

## Gitignore Defaults

Each experiment directory should contain its own `.gitignore` covering at
minimum:

```
__pycache__/
*.pyc
*.pth
*.ckpt
data/raw/
outputs/checkpoints/
outputs/cache/
.env
```

---

## Dashboard (future)

A dashboard aggregating experiment status and wiki activity is planned. When
it arrives, it will read structured frontmatter from `experiments/*/README.md`
and `wiki/findings/*.md`. Please keep the YAML frontmatter block at the top
of those files as shown in the templates below so the dashboard can parse
them without changes.

wishlist:
- StumpWM/emacs env integration, chat logging, PDF annotating/tagging, (Agentic research OS)
- fast way for me to associate open questions with research, detection of a question being answered (or relevant findings)
- context-aware agentic management of these
