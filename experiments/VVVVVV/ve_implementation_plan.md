# Multi-Timescale Value Embedding: Implementation Plan
**For: Implementer agent / Claude Code**
**Context: Fork of `karpathy/nanochat` → modify `nanochat/gpt.py` and supporting scripts**

---

## 0. ORIENTATION — READ FIRST

You are implementing a research experiment, not a product. **Correctness of measurement > performance of model.** Every phase produces logged metrics. Do not skip phases. Do not optimise the architecture before the diagnostic phases are complete.

### What exists (do not modify without flagging)
- `nanochat/gpt.py` — contains `GPT`, `CausalSelfAttention`, `MLP`, `Block`, `GPTConfig`
- Value embedding mechanism already present: `self.value_embeds` (dict of `nn.Embedding`), `self.ve_gate` (`nn.Linear(32, n_kv_head)`), `has_ve(layer_idx, n_layer)` alternating pattern
- x0 residual mechanism: `self.resid_lambdas`, `self.x0_lambdas`, `x0 = x` saved before loop
- Optimizer split: Muon for matrix params, AdamW for embeddings/ve/scalars
- `karpathy/autoresearch` — single-GPU, single-file, 5-min experiment loop. Use for Phase 1 sweeps. **Do not use for Phase 3 task evaluation** (too short, too small scale)

### What the experiment is about
Current nanochat ve: **one** static embedding table per ve-layer, gated by a fixed 32-channel slice of the residual. We are asking: what happens when you have **k > 1** tables, each potentially conditioned on different context signals, with gates that can read beyond the fixed slice?

Core hypothesis: value information decomposes into components varying at different timescales (token-type-intrinsic → document-level → local-context → hyper-local). The residual V carries everything; multiple ve tables can factor out stable components, reducing interference in the residual stream.

### Key prior results to carry in
- Karpathy (nanochat discussion #481): ve is load-bearing. Any reduction (low-rank, sharing, projections) hurt. Alternating placement won over every-layer or U-shaped. Models "love the ve capacity."
- autoresearch session #43: ve weight decay 0.001–0.003 improves; 0.005 regresses. Carry this into all phases.
- ResFormer (Zhou et al., arXiv:2410.17897, ACL 2025): cross-layer value residual connection mitigates attention concentration. SVFormer variant shares first-layer value across all layers, ~50% KV cache reduction. **nanochat's ve differs**: it's a token-ID lookup (context-free), not a layer-1 output reuse (still context-dependent). Keep this distinction sharp.
- Sun et al. (arXiv:2603.05498, Mar 2026): massive activations / attention sinks. BOS token develops near-constant post-norm representation in intermediate layers. Spike channels are consistent across prompts. Pre-norm + SwiGLU is the architectural cause; nanochat uses relu² which may produce similar but differently-profiled outliers — **verify empirically in Phase 0**.

---

## 1. PHASE 0 — DIAGNOSTIC BASELINE
**Goal:** Characterise the existing single-table ve before touching k. Answer three questions.
**Tool:** nanochat full pipeline (not autoresearch — need checkpoint access)
**Duration:** One full training run at d12 or d16 (fast iteration scale)

### Q0.1 — Where do spike channels land?
Nanochat uses relu² not SwiGLU. The spike channel mechanism (Sun et al. 2603.05498 §3.1) may still apply but with different step-up block positions and amplification profile.

```python
# After training, at each ve-layer, extract residuals and measure:
# Which channel indices have highest mean absolute activation?
# Do any of these fall in [:32] (the current gate's read window)?

def probe_spike_channels(model, dataloader, n_batches=50):
    channel_magnitudes = {}  # layer_idx -> mean |activation| per channel
    hooks = []
    for i, block in enumerate(model.transformer.h):
        if has_ve(i, model.config.n_layer):
            def hook(m, inp, out, idx=i):
                # inp[0] is x before norm — post-residual hidden state
                channel_magnitudes[idx] = inp[0].abs().mean(dim=[0,1]).cpu()
            hooks.append(block.register_forward_hook(hook))
    # run n_batches, collect, remove hooks
    # Log: top-32 channel indices by magnitude vs first-32 indices
    # KEY QUESTION: overlap fraction
```

**Threshold:** If >50% of top-32 channels by magnitude overlap with indices [:32], the current gate is reading informative channels non-accidentally. If <20%, the gate is reading near-noise.

### Q0.2 — Is the BOS residual document-level?
The BOS token residual should accumulate a diffuse document-level summary if the sink mechanism is active. Test: measure cosine similarity of BOS residual *within* a document (across positions) vs *across* documents.

```python
def probe_bos_stability(model, dataloader):
    # For each batch, extract BOS token residual at each ve-layer
    # Measure: mean cosine_sim(BOS_pos0_doc_A, BOS_pos0_doc_B) — cross-doc
    # Measure: mean cosine_sim(BOS_layer_i_doc_A, BOS_layer_j_doc_A) — cross-layer same doc
    # If within-doc similarity >> cross-doc similarity: BOS carries document signal
    # If both similar: BOS has collapsed to near-constant regardless of document (pure sink)
    # If both low: no stable BOS signal at all
```

**Why this matters:** If BOS is near-constant across documents (pure sink), conditioning a ve gate on it gives no document-level signal — it's just a fixed bias. If it varies across documents but is stable within them, it's a usable document-level conditioning signal.

### Q0.3 — Functional load of existing ve
Baseline counterfactual: zero out the single ve table during eval and measure val_bpb degradation. This is the ground truth for "how much work is ve doing" before we touch k.

```python
def eval_with_ve_ablated(model, dataloader):
    # Temporarily zero all value_embeds weights
    # Run eval loop, record val_bpb
    # Restore weights
    # Delta from baseline = ve functional contribution
```

Log all three as baseline entries. These are your reference points for everything that follows.

---

## 2. PHASE 1 — COLLAPSE DETECTION (no pressure)
**Goal:** Determine whether k>1 tables spontaneously specialise or collapse under task loss alone.
**Tool:** autoresearch (5-min runs sufficient for collapse dynamics; cheap to sweep k)
**k values to test:** 2, 3, 4 (run each independently from identical init)

### 2.1 Architecture changes

```python
# In GPTConfig, add:
n_value_tables: int = 1  # k; set >1 for multi-table experiments

# In GPT.__init__, replace:
self.value_embeds = nn.ModuleDict({
    str(i): nn.Embedding(padded_vocab_size, kv_dim) 
    for i in range(config.n_layer) if has_ve(i, config.n_layer)
})

# With:
self.value_embeds = nn.ModuleDict({
    f"{i}_{j}": nn.Embedding(padded_vocab_size, kv_dim)
    for i in range(config.n_layer) if has_ve(i, config.n_layer)
    for j in range(config.n_value_tables)
})

# In CausalSelfAttention.__init__, add k gates:
self.ve_gates = nn.ModuleList([
    nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
    for _ in range(config.n_value_tables)
]) if has_ve(layer_idx, config.n_layer) else None

# In forward:
if ve_list is not None:  # ve_list is now a list of k tensors
    for j, (ve_j, gate_j) in enumerate(zip(ve_list, self.ve_gates)):
        ve_j = ve_j.view(B, T, self.n_kv_head, self.head_dim)
        g_j = 2 * torch.sigmoid(gate_j(x[..., :self.ve_gate_channels]))
        v = v + g_j.unsqueeze(-1) * ve_j
# Note: additive mixture, not convex combination. Result can lie outside convex hull.
```

**Init:** Each table initialised with different random seed but same distribution as current ve. This is the minimum symmetry-breaking — different init, no other pressure. Gate weights init to zero (neutral, same as current).

**Optimizer:** Add all new ve tables to the AdamW embedding group with weight decay 0.001–0.003 (per autoresearch session #43 finding). Do not apply Muon to embedding tables.

### 2.2 Collapse detection metric hierarchy

Run this measurement suite at every checkpoint (or every N steps, N tunable):

#### Level 1 — Cosine gate (cheap, run always)
```python
def cosine_gate(model):
    # For each ve-layer, compute pairwise cosine similarity between table weight matrices
    # shape: (vocab, kv_dim) for each table
    # Flatten to (vocab * kv_dim,) and compute cosine
    results = {}
    for i in range(n_layers):
        if has_ve(i, n_layers):
            tables = [model.value_embeds[f"{i}_{j}"].weight.detach().flatten() 
                      for j in range(k)]
            for a, b in combinations(range(k), 2):
                sim = F.cosine_similarity(tables[a], tables[b], dim=0).item()
                results[f"ve_cos_l{i}_t{a}t{b}"] = sim
    return results
# If ALL pairwise cosine > 0.95: flag as COLLAPSED, log, skip expensive metrics this step
# If ANY pairwise cosine < 0.7: proceed to erank
# 0.7–0.95: ambiguous, proceed to erank
```

⚠️ **Cosine is necessary but not sufficient.** High cosine → almost certainly collapsed. Low cosine → not collapsed in weight space but functional collapse still possible. Always proceed to erank if cosine is ambiguous.

#### Level 2 — Effective rank (system-level, moderate cost)
```python
def effective_rank(model, layer_idx, k):
    # Stack all k table weight matrices: shape (k, vocab, kv_dim)
    # Reshape to (k, vocab * kv_dim) — or subsample vocab for speed
    W = torch.stack([
        model.value_embeds[f"{layer_idx}_{j}"].weight.detach()
        for j in range(k)
    ])  # (k, vocab, kv_dim)
    W_flat = W.view(k, -1)  # (k, vocab*kv_dim)
    # SVD on (k, vocab*kv_dim) — only k singular values possible
    S = torch.linalg.svdvals(W_flat)
    S_norm = S / S.sum()
    erank = torch.exp(-(S_norm * torch.log(S_norm + 1e-8)).sum()).item()
    return erank
    # erank = 1.0: complete collapse (rank-1)
    # erank = k: full diversity (all k directions independent)
    # erank in (1, k): partial specialisation
```

Note: erank is showing up in the Slicer project too (Ridge map null space analysis). Same underlying question — how much of the available representational capacity is actually being used. The null space of the stacked table matrix is the "wasted" capacity. If k=4 tables have erank=2, two dimensions are a null space in table weight manifold.

#### Level 3 — Gate output correlation (usage pattern, moderate cost)
```python
def gate_correlation(model, dataloader, n_batches=20):
    # For each ve-layer, collect gate output vectors g_j (B, T, n_kv_head) for each j
    # Measure Pearson correlation between g_a and g_b across (B*T*n_kv_head) elements
    # High correlation: tables being used identically regardless of weight similarity
    # This is observational — correlation ≠ causation
    # Two tables can correlate because both respond to same input feature (not collapse)
    # Use as flag: if corr > 0.95 AND cosine > 0.9: strong collapse evidence
```

#### Level 4 — Dropout ablation probe (causal, expensive, use at key checkpoints)
```python
def dropout_ablation(model, dataloader, layer_idx, table_idx):
    # Zero out table j weights, run eval, measure val_bpb delta
    # This is the CAUSAL test: if removing table j changes nothing, j is redundant
    # Run at: checkpoint 0 (init), 25%, 50%, 75%, 100% of training
    # Not every step — expensive (k+1 forward passes per checkpoint)
    original_weight = model.value_embeds[f"{layer_idx}_{table_idx}"].weight.data.clone()
    model.value_embeds[f"{layer_idx}_{table_idx}"].weight.data.zero_()
    ablated_loss = eval_loss(model, dataloader)
    model.value_embeds[f"{layer_idx}_{table_idx}"].weight.data = original_weight
    return ablated_loss  # compare to baseline loss
```

⚠️ MI (mutual information between table outputs) is observational like correlation but captures nonlinear dependence. More expensive to estimate. Use only if gate correlation gives ambiguous results. Dropout ablation is always more definitive — prefer it over MI for causal questions.

### 2.3 What to look for

**Phase transition:** If tables are differentiating, you'll see a point in training where erank increases from ~1.0 toward k, gate correlations drop, and ablation deltas become nonzero. This is the signal. The threshold isn't a fixed number — it's the presence or absence of this transition.

**Collapse pattern:** Tables stay at erank ~1.0 throughout. Gate correlations stay high. Ablation delta ≈ 0 for all but one table. → Pressure is load-bearing, not just regularisation.

**Partial collapse:** erank stabilises at some value between 1 and k. Some tables load-bearing, some not. → Interesting — which tables specialise and which don't?

**K-dependence hypothesis:** k=2 may specialise more cleanly than k=4 (fewer degrees of freedom to exploit). Test all k values before concluding about collapse.

---

## 3. PHASE 2 — MINIMAL SYMMETRY-BREAKING
**Goal:** Determine minimum pressure needed to escape symmetric saddle without imposing functional content.
**Tool:** autoresearch for pressure coefficient sweeps

### 3.1 Perturbation approach
```python
# Different random init per table (already done in Phase 1)
# Additionally: small additive noise to table weights at init
noise_scales = [0.0, 0.001, 0.01, 0.1]  # sweep
for j in range(k):
    model.value_embeds[f"{i}_{j}"].weight.data += torch.randn_like(...) * noise_scale
```

**Key question:** What is the minimum noise scale that produces persistent differentiation (erank > 1.5 at end of training)? Below this scale, the symmetric saddle captures the optimiser. Above it, tables differentiate.

### 3.2 Annealing test
If noise at init produces differentiation: does the differentiation persist after the perturbation is gone? Yes → pressure only needed to escape saddle, task loss drives the rest. No → loss landscape pulls tables back toward collapse, pressure must be persistent.

```python
# Train with cosine similarity penalty (weight λ_div) for first 20% of steps
# Anneal λ_div to zero
# Continue training and monitor erank — does it hold or regress?
```

### 3.3 Success criterion for Phase 2
Phase 2 is complete when you've identified:
- Minimum noise scale for persistent differentiation (or confirmed no noise scale works)
- Whether annealed pressure is sufficient or persistent pressure is required
- These answers determine what Phase 3 pressure types are worth testing

---

## 4. PHASE 3 — ASSUMPTIVE PRESSURES
**Goal:** Test specific diversity objectives that encode different priors about what "diverse" means.
**Tool:** Full nanochat runs (not autoresearch — need enough training for task evaluation)
**Run each independently. Ablate each. Keep collapse detection metrics running.**

Each pressure type encodes a distinct assumption. Run them independently so you can attribute effects.

### P1 — Geometric pressure (cosine similarity penalty)
```python
# Added to training loss
def cosine_diversity_loss(model, layer_indices, k, λ=0.01):
    loss = 0.0
    for i in layer_indices:
        tables = [model.value_embeds[f"{i}_{j}"].weight for j in range(k)]
        for a, b in combinations(range(k), 2):
            ta = F.normalize(tables[a].flatten(), dim=0)
            tb = F.normalize(tables[b].flatten(), dim=0)
            loss += (ta * tb).sum()  # penalise alignment
    return λ * loss
```
**Assumption encoded:** Useful decomposition is geometrically orthogonal in weight space.
**What it does NOT guarantee:** Orthogonal weights → functionally different outputs. Use dropout ablation to verify function follows geometry.

### P2 — Routing pressure (gate entropy maximisation)
```python
def gate_entropy_loss(gate_outputs, λ=0.01):
    # gate_outputs: list of k tensors, each (B, T, n_kv_head)
    # Stack and softmax across k dimension to get routing distribution
    stacked = torch.stack(gate_outputs, dim=-1)  # (B, T, n_kv_head, k)
    routing = F.softmax(stacked, dim=-1)
    entropy = -(routing * torch.log(routing + 1e-8)).sum(dim=-1).mean()
    return -λ * entropy  # maximise entropy = minimise negative entropy
```
**Assumption encoded:** Useful decomposition routes different token types to different tables.
**Note:** This penalises the gate, not the tables. Tables can remain similar if gates route them differently. Check both cosine AND gate correlation under this pressure.

### P3 — Informational pressure (output MI minimisation)
Expensive to estimate well. Use only if P1 and P2 give ambiguous results or if you want to compare.
Estimate via MINE (Mutual Information Neural Estimation) or binned histogram approximation on held-out data.
**Assumption encoded:** Useful decomposition carries non-redundant information in outputs.
**Note:** This is still observational — low MI between outputs doesn't prove counterfactual load-bearing. Always verify with dropout ablation.

### 4.1 Ablation mechanism for all pressures
```python
# Each pressure has a coefficient that can be set to zero
# λ_geom = 0.0  → no geometric pressure
# λ_route = 0.0 → no routing pressure
# etc.
# After full training: zero out each λ, continue training N more steps, check if erank holds
# This tests: is pressure load-bearing throughout training, or just needed to escape early saddle?
```

### 4.2 Dynamic pressure scheduling hypothesis
*Speculative — test only after static pressures are characterised*

Different pressure types may be most useful at different training phases:
- Early: symmetry-breaking (escape saddle)
- Mid: possibly reduce pressure and let loss differentiate
- Late: pressure may constrain specialisation the loss is already driving

```python
# Simple linear schedule to test:
λ(t) = λ_max * max(0, 1 - t / T_anneal)
# T_anneal = 20% of total steps is a reasonable first guess
```

---

## 5. PHASE 4 — TASK EVALUATION
**Goal:** Test whether specialisation (if achieved) produces measurable functional differences on carefully designed tasks.
**Tool:** Full nanochat with SFT stage. Tasks designed to dissociate timescales.

### 5.1 Task hierarchy (sharpest first)

**T1 — Adversarial ICL (sharpest diagnostic)**
Document establishes a pattern (document-level prior). Final example has strong local context suggesting a different completion. Correct answer: follow document-level pattern.

Prediction: ve-augmented model with document-level conditioning more robust to local context noise. Baseline has no separate channel for document-level prior.

Format: k-shot prompt where first k-1 examples establish pattern, example k introduces conflicting local signal.

**T2 — Cross-domain terminology discrimination**
Priority terms (in order of diagnostic sharpness):
1. **kernel** — linear algebra (null space) vs ML (RKHS) vs OS vs functional analysis vs image processing. Completely unrelated definitions.
2. **normal** — distribution vs perpendicular vector vs normal subgroup vs normal force. Five independent definitions.
3. **field** — physics (vector/scalar field) vs abstract algebra (algebraic field). Unrelated despite name.
4. **schema** — Kant (mediating representation between pure concepts and sensible intuitions, Critique of Pure Reason §III) vs database (structural specification) vs ML (format spec). Family resemblance makes this harder and more interesting than pure homonyms — partial conflation is subtly wrong, not obviously wrong.
5. **entropy** — thermodynamic vs information-theoretic. Genuinely related (Boltzmann) but not identical. Hardest case.

Task formats:
- **Document-level:** 500–1000 token document establishing field, then question requiring field-specific definition
- **Adversarial:** Document establishes field A, final paragraph introduces field B terminology, question asks about the term
- **Zero-shot:** Single sentence, no document context (tests token-type ve in isolation)

**T3 — Long-context coherence**
Measure style drift as a function of document length. BOS-conditioned ve predicts less drift.
Proxy: perplexity on continuation given established style, measured at 512 / 1024 / 2048 / 4096 tokens.

**T4 — Code generation (secondary)**
Token-type ve should accelerate syntactic scaffolding learning.
Measure: steps to reach fixed val_bpb on code-heavy eval set. Prediction: fewer steps with ve than without, at small compute budgets.

**T5 — Lies (important, underrepresented in training data)**
Lies are structurally different from irony: locally, a lie looks identical to a true statement. No local cue. Document-level signal only (established narrator unreliability). This is the cleanest test of whether document-conditioned gate does real work — irony has recoverable local cues, lies structurally do not.

Dataset: narratives with established unreliable narrator, followed by factually false statements consistent with narrator's motivation. Baseline model must recover unreliability from local context alone (it can't). Document-conditioned gate can in principle flag the whole document as unreliable.

---

## 6. EXTENDED ARCHITECTURE VARIANTS
*Implement only after Phase 1–2 results are in hand. Do not preempt.*

### 6.1 Learned projection gate (replace fixed slice)
```python
# Current:
gate = 2 * torch.sigmoid(self.ve_gate(x[..., :32]))

# Variant: learned d → r projection
self.ve_gate_proj = nn.Linear(config.n_embd, 32, bias=False)  # per table per layer
gate = 2 * torch.sigmoid(self.ve_gate(self.ve_gate_proj(x)))
```
**Motivation:** Spike channels may not fall in first 32 indices. Learned projection lets gate read any direction in residual. Cost: one small matrix per table per ve-layer. Strictly more expressive.

**Test conditioned on Phase 0 Q0.1 result:** If spike channels overlap strongly with [:32], current gate may already be working. If not, learned projection gate is indicated.

### 6.2 BOS-conditioned table
```python
# In forward pass, extract BOS token residual at each ve-layer
bos_residual = x[:, 0, :]  # (B, d_model) — BOS is always position 0 (dataloader guarantees this)
# Use as additional gate input for one of the k tables
gate_doc = 2 * torch.sigmoid(self.ve_gate_doc(bos_residual))  # (B, n_kv_head)
# Apply across all positions: gate_doc.unsqueeze(1) → (B, 1, n_kv_head) → broadcast over T
v = v + gate_doc.unsqueeze(1).unsqueeze(-1) * ve_doc.view(B, T, self.n_kv_head, self.head_dim)
```
**Prerequisite:** Phase 0 Q0.2 must show BOS residual varies across documents. If BOS is near-constant across documents (pure sink), this conditioning carries no document-level signal.

**Note from dataloader:** nanochat's dataloader guarantees every sequence starts with BOS (BOS-aligned design decision). This is load-bearing for the BOS conditioning approach.

---

## 7. IMPLEMENTATION CHECKLIST

```
Phase 0:
[ ] Implement probe_spike_channels() — log top channel indices at each ve-layer
[ ] Implement probe_bos_stability() — within-doc vs cross-doc BOS cosine similarity
[ ] Implement eval_with_ve_ablated() — baseline ve functional contribution
[ ] Run at d12 or d16 for speed
[ ] Log all three results before proceeding

Phase 1:
[ ] Add n_value_tables to GPTConfig (default=1 to preserve baseline)
[ ] Implement k-table value_embeds ModuleDict
[ ] Implement k ve_gates ModuleList per attention layer
[ ] Add all new ve tables to AdamW group with wd=0.001–0.003
[ ] Implement cosine_gate() — run every checkpoint
[ ] Implement effective_rank() — run when cosine is ambiguous (< 0.95)
[ ] Implement gate_correlation() — run when erank is ambiguous
[ ] Implement dropout_ablation() — run at 0%, 25%, 50%, 75%, 100% of training
[ ] Run k=2, 3, 4 independently
[ ] Log phase transition (if any) — when does erank start increasing?

Phase 2:
[ ] Implement noise perturbation at init (sweep [0.0, 0.001, 0.01, 0.1])
[ ] Implement annealing schedule for cosine similarity penalty
[ ] Test: does differentiation persist after pressure removed?

Phase 3:
[ ] Implement cosine_diversity_loss() with λ coefficient
[ ] Implement gate_entropy_loss() with λ coefficient
[ ] Implement ablation mechanism (zero out λ, continue training)
[ ] Run each pressure independently (not combined) first
[ ] Verify with dropout_ablation() that geometric diversity → functional diversity

Phase 4:
[ ] Implement adversarial ICL task (T1)
[ ] Implement cross-domain terminology eval (T2, priority: kernel > normal > field > schema > entropy)
[ ] Implement long-context coherence measurement (T3)
[ ] Collect enough lies-containing data for T5

Extensions (after Phase 1–2):
[ ] Learned projection gate (conditioned on Phase 0 Q0.1)
[ ] BOS-conditioned table (conditioned on Phase 0 Q0.2)
```

---

## 8. KEY REFERENCES

| Source | What it tells you | Where to find it |
|--------|------------------|-----------------|
| nanochat `gpt.py` | Current ve implementation, gate channels, has_ve(), optimizer grouping | `nanochat/gpt.py` |
| nanochat discussion #481 | "Models love ve capacity." Any reduction hurt. Alternating won. | github.com/karpathy/nanochat/discussions/481 |
| autoresearch session #43 | ve WD 0.001–0.003 optimal. Transformer init 0.68x sweet spot. | github.com/karpathy/autoresearch/discussions/43 |
| ResFormer (Zhou et al.) | Cross-layer value residual. SVFormer. Attention concentration mitigation. | arXiv:2410.17897 |
| Sun et al. spike/sink | Massive activations mechanism. BOS near-constant. Pre-norm + SwiGLU causation. | arXiv:2603.05498 |
| autoresearch `program.md` | How to run 5-min experiments. Branch protocol. Results format. | github.com/karpathy/autoresearch/blob/master/program.md |
| modded-nanogpt | Origin of ve and per-layer scalars (upstream of nanochat) | github.com/KellerJordan/modded-nanogpt |

---

## 9. WHAT NOT TO DO

🚫 Do not modify `prepare.py` in autoresearch — it's fixed infrastructure  
🚫 Do not apply Muon to embedding tables — AdamW only  
🚫 Do not run Phase 3 task evaluation in autoresearch — 5 min is not enough  
🚫 Do not combine all pressures simultaneously — run independently first  
🚫 Do not treat ResFormer's ve and nanochat's ve as the same thing — ResFormer reuses layer-1 computed values (context-dependent); nanochat uses token-ID lookup (context-free)  
🚫 Do not skip Phase 0 — the BOS conditioning and learned gate extensions are conditional on its results  
🚫 Do not use cosine similarity alone as collapse criterion — necessary but not sufficient  
🚫 Do not add more architecture before collapse detection is characterised  

---

## 10. OPEN QUESTIONS (do not resolve by assumption — flag and log)

❓ Do spike channels in nanochat (relu²) fall in first 32 indices? → Phase 0 Q0.1  
❓ Is BOS residual document-varying or document-invariant? → Phase 0 Q0.2  
❓ What is the minimum k that produces spontaneous specialisation? → Phase 1  
❓ Is pressure load-bearing throughout training or only needed to escape early saddle? → Phase 2  
❓ Do geometric diversity (low cosine) and functional diversity (nonzero ablation delta) co-occur? → Phase 3  
❓ What is the natural threshold for "collapsed" at this training scale? → Emerges from Phase 1 dynamics, not predetermined
