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

---

## 2026-03-07 — Per-example concentration and λ

**Decision**: `concentration()` returns a `(batch,)` tensor (one value per
sequence), and `compress()` broadcasts the per-example `lambda_t` vector across
the dimension axis.

**Rationale**: A batch-level scalar gives every sequence in the batch the same
compression rate regardless of its own geometry. Per-example λ is the minimum
correct implementation of the hypothesis: different problems have different
geometric structure and should receive different compression rates.

---

## 2026-03-07 — Causal LM training (concatenate problem + answer)

**Decision**: `collate_fn` concatenates problem tokens + answer tokens + EOS
into a single sequence; the transformer uses a causal (upper-triangular -inf)
mask; LM loss is computed only at answer+EOS positions, controlled by
`prob_lengths`.

**Rationale**: Matches standard autoregressive LM training and enables greedy
generation at eval time without architectural changes. Answer-only loss avoids
training the model to predict random arithmetic expression characters, which
adds noise without signal.

---

## 2026-03-07 — Float padding mask (matched type with causal mask)

**Decision**: The padding mask is a float tensor (0.0 for valid, -inf for PAD
positions) matching the float causal mask, rather than a boolean tensor.

**Rationale**: PyTorch 2.x raises a `UserWarning` when `src_mask` and
`src_key_padding_mask` have different types (float vs bool). Using matched float
masks suppresses the warning without changing behaviour.
