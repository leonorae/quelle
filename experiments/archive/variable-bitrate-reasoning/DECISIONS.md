# Implementation Decisions — variable-bitrate-reasoning

> Non-obvious implementation decisions for this experiment. Repo-level
> decisions live in `wiki/humans/decisions.md`.

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
