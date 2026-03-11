# HANDOFF: Behavioral Projections — Activation Caching & Projection Pipeline

## For: Code agent (Claude Code / implementer)
## From: Opus research session, 2026-03-11
## Operator: Apophenia (GitHub: leonorae)
## Priority: Get activations cached, build projection pipeline, run bisimulation probe

---

## What this is

We've developed a framework for analyzing LLM activations using four learned linear projections, each optimized for a different behavioral objective derived from the model's own computation. The theoretical work, literature review, and experimental design are complete. We need the code.

Reference doc: `behavioral-projections-draft.md` (should be in the repo or outputs — this is the primary spec).

## Hardware

- **Local**: AMD Radeon RX 5700 (8GB, ROCm spotty), 3090 Ti (24GB, borrowed overnight access)
- **Colab Pro**: A100 available for larger model runs
- **CPU**: Unlimited time for analysis on cached activations

## Model priority

1. **Pythia-410m** — fast iteration, CPU-friendly, already producing Slicer renders. Use for pipeline development.
2. **Qwen2.5-7B-Instruct** — primary target. Fits on 3090. Active fine-tune/LoRA ecosystem. Agent-relevant. This is where real results come from.
3. **Qwen2.5-14B/32B** (quantized) — stretch goal on Colab Pro for scaling analysis.

Skip GPT-2. We have manifests from it (see Slicer work) but it's not worth further investment.

## Phase 0: Activation caching

### Goal
Extract and store per-layer hidden states from diverse prompts. Everything downstream runs on cached data.

### Spec
```
For each model:
  For each prompt in prompt_set:
    Run forward pass
    Extract hidden states at every layer (residual stream, post-LayerNorm)
    Store: {prompt_id, layer_idx, hidden_state (float16), output_logits (top-k or full)}
    Also store: full output distribution (or top-100 logprobs) for bisimulation metric
```

### Storage format
HDF5 or safetensors. Key structure: `/{model}/{prompt_id}/layer_{i}` → tensor. Separate file for output distributions: `/{model}/{prompt_id}/logits` → tensor.

Float16 is fine for hidden states. Output distributions need higher precision for KL computation — float32 or store as log-probs.

### Prompt set (~5-10k prompts)

The prompt set should include diverse categories. Suggested structure:

**Agent-relevant decisions** (~1000):
- Code review scenarios (approve/reject/refactor decisions)
- Architecture decisions (which approach to take)
- Debugging prompts (identify the bug in this code)
- Planning prompts (break this task into steps)

**Consensus diagnostic prompts** (~500):
- The forced-choice questions from our consensus-diagnostic.jsx artifact
- Variants with bias perturbations pre-applied
- Questions where models should agree vs should disagree

**Saddle / ambiguity prompts** (~500):
- Prompts with contradictory context
- Prompts near decision boundaries
- The cipher tasks from our saturation-curves.jsx experiments

**Standard benchmarks** (~2000):
- MMLU subset (for calibration against known results)
- GSM8K subset (reasoning)
- HumanEval subset (coding)

**Semantic diversity** (~2000):
- Concrete vs abstract concepts
- Common vs unusual/novel prompts
- The probe sets from Slicer (beauty, entropy, dogs, etc.)

**Perturbation pairs** (~1000):
- For each of ~200 base prompts, generate 5 perturbations:
  - Rephrase (same meaning, different words)
  - Irrelevant context added
  - Authority bias added
  - Negation of framing
  - Register/formality shift
- Store base and perturbed with shared prompt_group_id

### Implementation notes
- Use `transformers` with `output_hidden_states=True`
- For Qwen2.5-7B: load in float16, batch size 1-4 depending on VRAM
- Activations come free with inference — if we're running the model anyway, cache everything
- For perturbation pairs, run base and all perturbations in sequence to minimize overhead

### Pythia-410m quick cache
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m", output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")

# For each prompt:
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden_dim) per layer
    logits = outputs.logits  # (batch, seq, vocab)
```

For the projection experiments we typically want the hidden state at the **last token position** (the one predicting the next token), at each layer. But also store the full sequence for potential position-wise analysis later.

## Phase 1: Bisimulation probe (highest priority)

### Goal
Learn a linear map P where ‖P(h₁) − P(h₂)‖ predicts KL(f(h₁) ‖ f(h₂)).

### Spec
```
Input: pairs of hidden states (h_i, h_j) from the same layer
Target: KL divergence between their output distributions
Model: Linear projection P (matrix, d_hidden → d_proj), trained with:
  Loss = |‖P(h_i) - P(h_j)‖₂ - KL(f(h_i) ‖ f(h_j))|
  or MSE between projected distance and KL divergence
```

### Implementation
- Sample pairs from cached activations. Include: same-prompt-different-perturbation pairs (should have low KL), different-prompt pairs (variable KL), and perturbation pairs (controlled KL shift).
- Train per-layer (separate P for each layer) initially.
- Projection dimension d_proj: start with d_hidden (no compression), then sweep lower dimensions to find minimum rank that preserves the metric.
- Use Ridge regression as baseline: the weight matrix of a Ridge model predicting KL from activation differences IS the bisimulation projection. Then try learned MLP for comparison.
- Validate: on held-out pairs, does projected distance predict actual KL? Measure R² and rank correlation.

### Key analysis
- **Effective rank** of P at each layer (via SVD). How many dimensions carry the behavioral metric?
- **Column space of P** vs **SAE feature directions** (if SAE features available for the model). Do they align?
- **Null space of P**: directions where you can perturb the activation without changing the behavioral distance. This is the model's "slack" — representational capacity not used for computation.
- Compare the column space of P across layers. Does the model reorganize what it considers behaviorally important at each layer?

## Phase 2: Iterative residual peeling

### Goal
Characterize whether the "dark matter" (what the bisimulation probe can't see) is high-rank linear or genuinely nonlinear.

### Spec
```
1. Train P_1 on activations → KL divergence (this is Phase 1)
2. Compute residual: r_i = h_i - P_1^T @ P_1 @ h_i (project out P_1's column space)
3. Train P_2 on residuals → KL divergence
4. Compute new residual, repeat
5. Plot: variance in KL explained by {P_1, P_1+P_2, P_1+P_2+P_3, ...}
```

### Key output
The curve of cumulative KL variance explained vs number of projections. Fast saturation = dark matter is linear and the first probe captures most of it. Slow saturation = genuinely nonlinear structure.

## Phase 3: Contrastive prompt discrimination

### Goal
Learn a projection where same-prompt pairs cluster and different-prompt pairs separate.

### Augmentation strategy (important)
Rather than dropout/noise, use **register token augmentation**:
- Run same prompt with and without prepended padding/register-style tokens
- Content representation should be invariant; administrative representation shifts
- The contrastive projection learns to ignore bookkeeping variation

### Spec
```
Positive pairs: (h_prompt_A_variant_1, h_prompt_A_variant_2) — same prompt, different augmentation
Negative pairs: (h_prompt_A, h_prompt_B) — different prompts
Loss: InfoNCE / NT-Xent
```

### Implementation
- Can also use perturbation pairs from the prompt set: same base prompt with rephrase/register augmentation as positives
- Monitor for dimensional collapse (track effective rank of projected embeddings throughout training)
- Use projector head (2-layer MLP) during training, discard for analysis (SimCLR pattern)

## Phase 4: Perturbation-sensitivity map

### Goal
Learn W such that ‖W·δ‖ predicts |Δ output| for perturbation δ.

### Spec
```
For each cached activation h:
  Sample random perturbation directions δ (unit vectors)
  Compute perturbed output: f(h + ε·δ) for small ε
  Measure: ΔKL = KL(f(h) ‖ f(h + ε·δ))
  Train W: ‖W·δ‖₂ ≈ ΔKL
```

### Note
This requires additional forward passes (can't be done purely from cached activations — you need to run the perturbed activations through the remaining layers). For Pythia-410m this is cheap. For Qwen2.5-7B, batch the perturbation forward passes during the 3090 session.

Alternative: use the Jacobian directly. `torch.autograd.functional.jacobian` on the output w.r.t. a specific layer's activation gives the exact local sensitivity. Then W = J^T @ J gives the sensitivity matrix. But this is expensive for large models.

Practical approximation: sample ~100 random perturbation directions per activation, measure the output change for each, fit W via least squares. This is a stochastic approximation of the Jacobian that scales better.

## Repo structure suggestion

```
quelle/
├── experiments/
│   ├── behavioral-projections/
│   │   ├── cache_activations.py    # Phase 0
│   │   ├── bisimulation_probe.py   # Phase 1
│   │   ├── iterative_peeling.py    # Phase 2
│   │   ├── contrastive_prompt.py   # Phase 3
│   │   ├── perturbation_map.py     # Phase 4
│   │   ├── analysis.py             # Cross-projection intersection analysis
│   │   └── prompts/
│   │       ├── agent_decisions.jsonl
│   │       ├── consensus_diagnostic.jsonl
│   │       ├── perturbation_pairs.jsonl
│   │       └── benchmarks.jsonl
│   ├── variable-bitrate-reasoning/  # existing
│   ├── crystal-lattice/             # existing
│   └── slicer/                      # existing Slicer code
├── data/
│   ├── activations/                 # cached activation tensors
│   │   ├── pythia-410m/
│   │   └── qwen2.5-7b/
│   └── projections/                 # trained projection matrices
├── wiki/
│   └── findings/
│       └── behavioral-projections-draft.md
└── CLAUDE.md
```

## Dependencies

```
torch>=2.0
transformers>=4.40
safetensors
h5py
scikit-learn  # for Ridge baseline
numpy
tqdm
```

## What success looks like

1. **Cached activations** for Pythia-410m (all prompts, all layers) and Qwen2.5-7B (same).
2. **Bisimulation probe** trained per-layer for both models. R² > 0.5 on held-out pairs would be a strong positive result. Even R² > 0.3 is publishable if the effective rank analysis is clean.
3. **Iterative peeling curve** showing how many projections are needed to capture most behavioral variance.
4. **A finding**: either the bisimulation metric has surprisingly low effective rank (most of the model's behavioral geometry is concentrated in a small subspace), or it doesn't (the model uses its full capacity). Either result is interesting.

## Context from the research arc

This work emerges from a multi-session research program. Key prior results:

- **Saturation curves**: In-context learning in Claude has measurable implicit rank. More demonstrations can degrade performance (L-system task). Models externalize reasoning at capacity boundaries.
- **Saddle detection**: Models balanced between competing implicit updates externalize reasoning and break system prompt to analyze inconsistency.
- **Slicer**: Ridge regression from LLM activations to CLIP space. R² vs nn_recall@5 are in direct conflict — reconstruction fidelity and structural preservation are fundamentally different objectives. Alpha acts as a compressor. Cross-model (GPT-2, Pythia-410m) confirmation that this is a property of the LLM→external-space linear projection, not model-specific.
- **SAE dark matter connection**: Gurnee (2024) found SAE reconstruction errors are pathological — they hurt the model more than random perturbations of the same magnitude. The bisimulation probe is designed to find the equivalent phenomenon for behavioral projections.

## Critical constraints

- **No GPT-2**. Skip it. Pythia-410m for prototyping, Qwen2.5-7B for real results.
- **Float16 for activations, float32 for logits/KL computation.**
- **Cache the output distribution** (at minimum top-100 logprobs with their indices), not just the argmax. The bisimulation probe needs pairwise KL, which requires the full distribution or a good approximation.
- **Perturbation pairs must share a prompt_group_id** so the analysis can find them.
- **The projection training should use Ridge as baseline first** before any neural approaches. If Ridge works, the finding is stronger (a simple linear map captures behavioral geometry). If Ridge fails but MLP succeeds, the finding is that the metric has nonlinear structure.

## Questions the code agent should ask Apophenia

1. Where is the quelle repo currently? (local path, or need to clone?)
2. Is the 3090 available tonight? (determines whether to start with Pythia CPU caching or go straight to Qwen2.5)
3. HuggingFace token for gated models? (Qwen2.5 might need it)
4. Preferred prompt set sources — use existing prompt files from Slicer, or generate fresh?
5. Storage budget — how much disk space for activation caches? (Qwen2.5-7B, 32 layers, 4096-dim, 10k prompts ≈ 40-80GB depending on sequence length and precision)

---

*"The evaluation of the system's own computation is the application of the system to its task." — derived from SICP, applied to LLM geometric self-awareness*
