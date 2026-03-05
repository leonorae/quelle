# Context for Implementer Agent: Geometric Self-Awareness for Reasoning Integrity

> **Who this is for**: Any agent (or human) picking up this experiment for the
> first time. Read this before touching any code.
>
> **Date written**: 2026-03-03
>
> **Related files**:
> - `README.md` — experiment overview, phase status, success criteria
> - `configs/phase0b.yaml` — generation hyperparameters
> - `src/phase0b_generate_traces.py` — Phase 0B implementation
> - `wiki/concepts/geometric-self-awareness.md` — angle concentration background
> - `wiki/concepts/ris-scoring.md` — RIS rubric and judge ensemble protocol

---

## 1. The Big Picture

We are building toward a **self-organizing hierarchical learning system** where
models use internal geometric signals (angle concentration, manifold structure)
to guide their own improvement. The ultimate goal is a system that can:

- Monitor its own uncertainty via geometry.
- Decide when to request teacher help or delegate sub-tasks.
- Evolve its own policies for when to compress, when to explore, and when to
  consolidate.
- Operate across multiple timescales (fast token prediction, medium reasoning
  steps, slow meta-learning).

The core insight: **geometry is the lingua franca** that unifies internal
state, teacher alignment, and evolutionary optimization. Recent work (Advani
et al. 2026, Confidence Manifold 2026, Geometry of Reasoning 2025) validates
that correctness and reasoning quality are encoded in geometric properties of
hidden states.

---

## 2. The Immediate Experiment

We are testing a foundational hypothesis:

> **Angle concentration and trajectory geometry (velocity, curvature, manifold
> dimensionality) in hidden states correlate with human-judged reasoning quality
> (RIS) and can predict reasoning flaws before they manifest in final answers.**

If true, this provides a **training-free, interpretable** way to detect "right
for wrong reasons" errors and enable real-time intervention.

### 2.1 Experimental Phases

| Phase | Description |
|-------|-------------|
| **0B** | Generate ~1000 reasoning traces from Qwen2.5-7B-Instruct on GSM8K |
| **1** | Score each trace with an LLM judge ensemble (RIS rubric) |
| **2** | Extract geometric features: angle concentration, velocity, curvature, manifold dim, centroid distance |
| **3** | Correlate geometry with RIS, train predictors, test temporal early-warning |

### 2.2 Why This Matters

- **For science**: Shows that internal geometry reflects reasoning *process*,
  not just answer correctness.
- **For practice**: Enables training-free error detection and intervention in
  resource-constrained settings.
- **For our larger vision**: Validates geometry as a self-awareness signal,
  which will later power hierarchical teaching loops and evolutionary
  optimization.

---

## 3. Phase 0B: Data Generation (Your First Task)

### 3.1 Goal

Produce a JSONL file of 1000 reasoning traces from a capable 7B-scale model
on GSM8K math problems. Each trace includes step-by-step reasoning and a final
answer, formatted so Phase 1 scoring and Phase 2 geometry extraction can
consume it directly.

### 3.2 Requirements

| Parameter | Value |
|-----------|-------|
| Dataset | GSM8K training split, first 1000 problems |
| Model | `Qwen/Qwen2.5-7B-Instruct` (or `meta-llama/Llama-3.1-8B-Instruct`) |
| GPU VRAM | 24GB+ required for bfloat16; use quantized (GPTQ/AWQ) if lower |
| temperature | 0.7 |
| top_p | 0.95 |
| max_new_tokens | 512 |
| do_sample | True |
| Output file | `data/traces_gsm8k_qwen.jsonl` |

### 3.3 Prompt Template

```
Solve the following math problem step by step. Show all your work clearly.
End your response with "Final answer: [number]" on its own line.

Problem: {question}
```

Consistency is critical — every trace must use the same prompt so geometric
comparisons are valid.

### 3.4 Output Schema (one JSON object per line)

```jsonc
{
  "id": "gsm8k_0",              // zero-indexed, e.g. "gsm8k_999"
  "question": "...",            // original GSM8K question text
  "ground_truth": 42,           // integer extracted from GSM8K answer field
  "generated_trace": "...",     // full model output (reasoning + final answer)
  "extracted_answer": 42,       // integer parsed from "Final answer: N", or null
  "is_correct": true,           // extracted_answer == ground_truth
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "generation_config": {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_new_tokens": 512,
    "do_sample": true
  }
}
```

### 3.5 Implementation Steps

The script `src/phase0b_generate_traces.py` is the canonical implementation.
Run it as:

```bash
python src/phase0b_generate_traces.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --n_problems 1000 \
    --output data/traces_gsm8k_qwen.jsonl
```

The script handles:
1. Loading GSM8K via HuggingFace `datasets`.
2. Loading model + tokenizer with `bfloat16` and `device_map="auto"`.
3. Generating traces with the standard prompt.
4. Extracting and verifying final answers via regex.
5. Writing JSONL output incrementally (crash-safe).
6. Printing a summary report (accuracy, mean trace length, answer distribution).

### 3.6 Environment Setup

```bash
pip install torch transformers accelerate datasets tqdm
# Optional for faster inference:
pip install vllm
```

Python 3.10+ required.

### 3.7 Deliverables for Phase 0B

- `data/traces_gsm8k_qwen.jsonl` — 1000 trace objects.
- `outputs/phase0b_report.txt` — accuracy, mean token length, answer histogram.
- Update `RESULTS.md` Phase 0B section.
- Commit with message: `data(geometric-self-awareness-reasoning): generate 1000 GSM8K traces`

### 3.8 Optional Extensions

- Also generate from `mistralai/Mistral-7B-Instruct-v0.3` for multi-model
  comparison. Output to `data/traces_gsm8k_mistral.jsonl`.
- If GSM8K accuracy is < 60%, switch to a quantized variant
  (`Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`) or a stronger model.
- ProcessBench and PRMBench provide pre-annotated step-level error labels and
  could skip Phase 1, but we prefer GSM8K for simplicity and controllability.

---

## 4. Phase 1: RIS Scoring (Next, Not Your Immediate Task)

Phase 1 scores each trace using an LLM judge ensemble:

- **Models**: DeepSeek-V3, Qwen-72B, Llama-3.1-70B (via OpenRouter or
  Together.ai).
- **Rubric**: Advani et al. 2026 RIS rubric — per-step and overall scores on
  a 1–5 scale. See `wiki/concepts/ris-scoring.md` for the prompt template.
- **Output**: augmented JSONL with `ris_scores` field per trace.

This phase is not yet implemented. A stub lives in `src/phase1_ris_scoring.py`.

---

## 5. Phase 2: Geometric Feature Extraction (After Phase 1)

Re-run the same model with `output_hidden_states=True` on each stored trace.
For each reasoning step (identified by delimiter in the generated text):

| Feature | Formula | Notes |
|---------|---------|-------|
| Angle concentration | Mean off-diagonal cosine similarity of token hidden states | See `wiki/concepts/geometric-self-awareness.md` |
| Velocity | L2 norm of difference between consecutive step centroids | Measures how fast the representation moves |
| Curvature | Menger curvature of three consecutive step centroids | Detects abrupt direction changes |
| Manifold dimensionality | PLS or MLE on hidden states within a step | Proxy for effective degrees of freedom |
| Centroid distance | Distance from step centroid to correct/incorrect class manifold | Requires fitting manifolds on labeled data |

Implementations will use `scikit-learn` (PLS, PCA, logistic regression
baselines) and `torch` for hidden state processing.

---

## 6. Phase 3: Analysis (After Phase 2)

- Pearson/Spearman correlation: each geometric feature vs. RIS score.
- Logistic regression: predict step-level flaws from geometry.
- Temporal early-warning: does geometry at step *t* predict flaws at step
  *t+k* (k = 1, 2, 3)?
- Visualizations: scatter plots (geometry vs. RIS), trajectory plots, UMAP
  of hidden states colored by RIS.

---

## 7. Connection to Higher Layers: RLM

We are aware of the **Recursive Language Models (RLM)** library
(arXiv:2512.24601; code at https://github.com/alexzhang13/rlm), which
implements hierarchical inference: a main model delegates sub-tasks to child
models via a REPL. RLMs achieve state-of-the-art on long-context tasks.

Our geometric self-awareness could integrate into RLMs as
**uncertainty-guided recursion**: the main model uses concentration signals to
decide when to spawn a sub-call, and the REPL could expose geometric summaries
of sub-results. This is a longer-term goal, but it validates that our
direction aligns with cutting-edge research.

---

## 8. Key References

| Reference | Why it matters |
|-----------|----------------|
| Advani et al. 2026 — "Teaching Models to Teach Themselves" | Establishes the "right for wrong reasons" problem and defines the RIS metric |
| Confidence Manifold (arXiv:2602.08159) | Correctness lives in a low-dimensional subspace; centroid distance is a zero-shot predictor |
| Geometry of Reasoning (arXiv:2510.09782) | Defines velocity, curvature, flow similarity for reasoning trajectories |
| RLM (arXiv:2512.24601) | Recursive Language Models — longer-term integration target |

---

## 9. Success Criteria

| Phase | Criterion |
|-------|-----------|
| 0B | 1000 clean traces, pass@1 > 70% |
| 3 | Pearson r > 0.5 between geometric metrics and RIS |
| 3 | AUC > 0.8 for flaw prediction |
| 3 | Temporal early-warning AUC > 0.7 |

---

## 10. Open Questions

- GPU availability: do you have a 24GB+ VRAM card? If not, use a quantized
  model or a cloud instance (A100/H100).
- Speed vs. simplicity: prefer `vllm` for 10× faster generation, plain
  `transformers` for easier debugging.
- Phase 1 API budget: OpenRouter costs ~$0.002/call; 1000 traces × 3 judges
  ≈ $6 total.
- Should we run ProcessBench/PRMBench instead of (or alongside) GSM8K?

<!-- QUESTION: Which judge models are available to you for Phase 1? Do you
have OpenRouter or Together.ai API keys? -->
