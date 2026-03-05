# RIS Scoring (Reasoning Integrity Score)

> Background concept for the LLM judge ensemble used to evaluate reasoning
> traces in the `geometric-self-awareness-reasoning` experiment.

**Relevant experiments**: `geometric-self-awareness-reasoning`

---

## What Is RIS?

The **Reasoning Integrity Score (RIS)** is a rubric introduced by Advani et al.
2026 ("Teaching Models to Teach Themselves") to quantify *how correctly* a
model reasons, independent of whether the final answer is right.

The key motivation: models can produce the correct final answer by using
flawed reasoning steps ("right for wrong reasons"). Surface-level accuracy
metrics miss this; RIS does not.

---

## Rubric

RIS is evaluated per reasoning step and overall on a **1–5 scale**:

| Score | Meaning |
|-------|---------|
| 5 | Correct logic, correct computation, clearly expressed |
| 4 | Correct logic, minor computational slip that doesn't affect conclusion |
| 3 | Mostly correct, but contains a non-critical error or unjustified leap |
| 2 | Partially correct; a significant error that could mislead |
| 1 | Incorrect or circular reasoning; wrong premise or invalid inference |

An overall score is the mean of step scores, weighted by step complexity.

---

## Judge Ensemble Protocol

A single judge is noisy. We use three models and average their scores:

| Judge Model | Access | Notes |
|------------|--------|-------|
| `deepseek-ai/DeepSeek-V3` | OpenRouter / Together.ai | Strong at math reasoning |
| `Qwen/Qwen2.5-72B-Instruct` | OpenRouter / Together.ai | High agreement with human raters |
| `meta-llama/Llama-3.1-70B-Instruct` | OpenRouter / Together.ai | Diverse prior |

The ensemble mean `ris_ensemble` is used as the ground-truth signal in
Phase 3 analysis.

---

## Judge Prompt Template

```
You are a math reasoning evaluator. Score the following reasoning step
on a scale of 1–5 using the RIS rubric:

  5 = Correct logic and computation, clearly expressed
  4 = Correct logic, minor computational slip
  3 = Mostly correct, but contains a non-critical error or unjustified leap
  2 = Significant error that could mislead
  1 = Incorrect or circular reasoning

Problem:
{question}

Reasoning step (step {step_number} of {total_steps}):
{step_text}

Respond with only: SCORE: <integer 1–5>
REASON: <one sentence>
```

---

## Cost Estimate

- ~500 tokens per call (prompt + response).
- OpenRouter: ≈ $0.002 per call.
- 1000 traces × ~5 steps × 3 judges = 15 000 calls ≈ **$30 total**.
- To reduce cost: score overall (not per-step) → 3000 calls ≈ **$6 total**.

---

## References

- Advani et al. 2026 — "Teaching Models to Teach Themselves" (primary source
  of the RIS rubric; exact citation TBD once paper is public).
- G-Eval framework (Liu et al. 2023) — a related LLM-as-judge methodology for
  NLG evaluation.
