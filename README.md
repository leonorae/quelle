# Implementation Specification: Variable-Bitrate Reasoning with Geometric Self-Awareness

## For the Implementation Agent

This document specifies a minimal, feasible experiment to test whether a language model can learn to adaptively compress its internal representations based on geometric self-awareness, using only laptop-scale compute. The implementation should prioritize simplicity and clarity over optimization.

---

## Core Idea

A single transformer processes multi-step arithmetic problems. At each step, it:

1. Computes **angle concentration** (a measure of geometric certainty) from its hidden states.
2. Uses this concentration to determine **how much to compress** the representation before passing it to the next step.
3. Is trained to **predict its own future state** through the compressed channel, with stop-gradient to prevent collapse.

The compression policy emerges from the interaction of geometry and task loss—**no RL required**.

---

## Architecture

### Model Components

```
Input tokens → [Transformer Layer] → hidden state h_t
                                         ↓
                               [Compression Head] → compressed z_t (variable dimension)
                                         ↓
                               [Predictor Head] → predicted h_{t+1}
```

**Transformer**: 4 layers, 4 attention heads, hidden dimension d=128. Standard pre-norm architecture.

**Compression Head**: Linear projection with **differentiable dimension selection**:
- Input: h_t ∈ ℝᵈ
- Output: z_t ∈ ℝᵏ where k = ⌈d·(1-λ_t)⌉
- λ_t ∈ [0,1] is the compression rate (higher = more compression)

**Predictor Head**: Linear layer mapping z_t back to ℝᵈ, trained to match h_{t+1}.

**Angle Concentration Calculator**: Computes mean cosine similarity between all pairs of token representations in the sequence:
```python
def concentration(h):
    # h: (batch, seq_len, d)
    h_norm = F.normalize(h, dim=-1)
    sim = torch.bmm(h_norm, h_norm.transpose(1, 2))  # (batch, seq_len, seq_len)
    mask = ~torch.eye(sim.shape[1], dtype=bool, device=sim.device)
    return sim[:, mask].mean()  # scalar per batch
```

**Compression Policy**:
```python
lambda_t = torch.sigmoid(alpha * (concentration - beta))
```
where alpha and beta are learnable scalars (initialized: alpha=2.0, beta=0.5).

### Differentiable Dimension Selection

To make k differentiable, use **Gumbel-Softmax** over dimension indices:

```python
def compress(h, lambda_t, temperature=1.0):
    # h: (batch, seq_len, d)
    d = h.shape[-1]
    k = int(d * (1 - lambda_t.item()))  # for actual compression
    k = max(1, min(d-1, k))
    
    # Differentiable dimension weights
    dim_weights = torch.linspace(0, 1, d, device=h.device)
    keep_prob = torch.sigmoid((dim_weights - lambda_t) / temperature)
    
    # Weighted masking (approximates selecting first k dimensions)
    z = h * keep_prob.unsqueeze(0).unsqueeze(0)
    return z, k
```

For forward pass, actually truncate to k dimensions; for backward, use the continuous approximation.

---

## Training Objective

### Loss Function
```python
total_loss = lm_loss + 0.1 * future_loss + 0.01 * curvature_penalty
```

**LM Loss**: Standard next-token prediction cross-entropy.

**Future Loss** (DSD-inspired):
```python
future_loss = F.mse_loss(predictor(z_t), sg(h_{t+1}))
```
where `sg` = stop-gradient. This forces z_t to preserve information needed for future steps.

**Curvature Penalty** (simplified from pseudocode):
```python
def curvature_penalty(h_t, h_t_next):
    # Encourage smooth transitions when concentration is high
    diff = h_t_next - h_t
    return (diff ** 2).mean() * (1 - concentration)  # penalize less when uncertain
```

**No explicit compression regularizer**—the future loss and limited capacity already create pressure to compress.

---

## Task: Multi-Step Arithmetic

### Data Format
```
Input:  "((3 + 5) × 2) + 7 = ?"
Target: "23"
```

Generate 10,000 examples with:
- Operations: +, -, ×
- Numbers: 1-20
- Depth: 2-5 steps
- Parentheses to force order

### Difficulty Labels (for analysis only)
- Easy: 2 steps, small numbers
- Medium: 3-4 steps
- Hard: 5 steps, requires holding multiple intermediates

### Data Loader
```python
class ArithmeticDataset(Dataset):
    def __init__(self, num_examples=10000):
        self.examples = [generate_arithmetic_problem() for _ in range(num_examples)]
    
    def __getitem__(self, idx):
        problem, answer = self.examples[idx]
        # Tokenize: simple digit-by-digit with special tokens for operators
        input_ids = tokenize(problem)  # includes " = ?"
        target_ids = tokenize(answer)
        return input_ids, target_ids, compute_difficulty(problem)
```

---

## Training Loop

```python
model = VariableRateReasoner(d_model=128, n_layers=4, n_heads=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dataset = ArithmeticDataset(10000)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for batch in loader:
        input_ids, target_ids, difficulties = batch
        
        # Forward pass with compression stats
        logits, stats = model(input_ids, return_stats=True)
        
        # Losses
        lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        future_loss = 0
        curvature = 0
        for layer_stats in stats:
            future_loss += F.mse_loss(
                layer_stats['h_pred'], 
                layer_stats['h_next'].detach()  # stop-gradient
            )
            curvature += ((layer_stats['h_next'] - layer_stats['h']) ** 2).mean() * \
                         (1 - layer_stats['concentration'])
        
        total_loss = lm_loss + 0.1 * future_loss + 0.01 * curvature
        
        # Backprop
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Log
        if step % 100 == 0:
            print(f"Step {step}: loss={total_loss.item():.4f}, "
                  f"lm={lm_loss.item():.4f}, future={future_loss.item():.4f}, "
                  f"avg_lambda={np.mean([s['lambda'] for s in stats]):.3f}")
```

---

## Measurements and Analysis

### During Training (logged every 100 steps)
- Loss components
- Average lambda per layer
- Average concentration per layer
- Correlation between lambda and concentration (running estimate)

### After Training: Evaluation on Held-Out Test Set (1000 examples)

**Primary Metrics:**
1. **Accuracy** overall and by difficulty level
2. **Compression policy**:
   - Plot lambda_t vs. concentration (scatter, color by difficulty)
   - Compute correlation coefficient
   - Compare lambda distribution on steps that require intermediate storage vs. output steps

**Geometric Analysis** (on a small sample, ~100 examples):
- Extract hidden states from all layers
- Run PCA/UMAP, color by:
   - Lambda at that step
   - Whether step required intermediate storage
   - Correctness
- Train linear probe to predict "requires intermediate storage" from hidden states

**Baseline Comparisons** (run 3 seeds each):
- Fixed high compression (lambda=0.8)
- Fixed low compression (lambda=0.2)
- Random compression (lambda sampled uniform)
- No compression (full hidden state passed)

---

## Implementation Checklist

- [ ] Tokenizer for arithmetic expressions (digits, operators, parentheses, "=", "?")
- [ ] Dataset generator with difficulty labels
- [ ] Transformer with configurable layers/dimensions
- [ ] Compression head with differentiable dimension selection (Gumbel-Softmax)
- [ ] Concentration computation
- [ ] Learnable alpha/beta parameters
- [ ] Training loop with all losses
- [ ] Logging infrastructure
- [ ] Evaluation script with metrics
- [ ] Baseline implementations
- [ ] Visualization code (PCA/UMAP, scatter plots)

---

## Expected Runtime

| Component | Time |
|-----------|------|
| Dataset generation | 5 minutes |
| Training (10 epochs, 10k examples) | 2-4 hours on laptop CPU |
| Evaluation + analysis | 1-2 hours |
| **Total** | **~6 hours** |

---

## Notes for the Implementer

1. **Differentiability**: The dimension selection is the trickiest part. If Gumbel-Softmax causes issues, start with a hard threshold but accept that lambda won't get gradients. You can still analyze the learned policy post-hoc.

2. **Future prediction target**: The simplified code uses the actual next hidden state from the forward pass. This creates a slight information leak (the model sees the future to predict it). For a cleaner experiment, run two forward passes: one without compression to get target states, one with compression for prediction. This doubles compute but may be worth it.

3. **Hyperparameters**: Start with the values given; adjust if training is unstable. The future loss weight (0.1) and curvature weight (0.01) are guesses—tune if needed.

4. **Logging**: Save all stats (lambda, concentration, losses) to a file for later analysis. Use tensorboard or simple CSV.

5. **Reproducibility**: Set random seeds (42) for all components.

---

## Extensions for Future Work (Not Required Now)

- Replace simple compression regularizer with UL-style diffusion prior
- Add per-capability manifolds (as in pseudocode)
- Extend to multiple tasks
- Implement self-training on high-concentration trajectories

---

## Success Criteria

The experiment is successful if:

1. Training converges (loss decreases stably).
2. Lambda shows **negative correlation** with concentration (r < -0.3).
3. Adaptive compression **outperforms fixed baselines** on hard problems.
4. UMAP reveals distinct clusters for different step types.
5. Linear probes predict storage requirement from hidden states (>60% accuracy).

Even if some criteria are not met, the results will inform next steps.

---

Proceed with implementation. Report any blockers or unexpected behavior.
