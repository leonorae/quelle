# Variable-Bitrate Compression

> How adaptive bandwidth allocation in representation space works and why it
> differs from fixed compression.

**Relevant experiments**: `variable-bitrate-reasoning`

---

## Analogy: Audio/Video Codecs

Variable-bitrate (VBR) codecs allocate more bits to complex regions (fast motion,
high detail) and fewer to simple regions (silence, static background). The key
insight is that *not all information is equally hard to compress at every moment*.

The same principle applies to transformer hidden states: at some steps (e.g.,
outputting a final answer), the representation may be geometrically tight and
easy to compress; at others (e.g., mid-computation with many open sub-problems),
it may require full fidelity.

## Compression Policy

In `variable-bitrate-reasoning`, compression is parameterized by a rate λ ∈ [0,1]:

```
λ_t = sigmoid(α · (concentration_t - β))
```

- **α** controls sharpness of the policy.
- **β** is the concentration threshold at which compression flips from low to high.
- Both are learnable. Initialized at α=2.0, β=0.5.

## Differentiable Dimension Selection

Selecting *k = ⌈d·(1-λ)⌉* dimensions is non-differentiable. The workaround is
a soft masking via a sigmoid over dimension indices:

```python
dim_weights = torch.linspace(0, 1, d)       # [0, 1/d, 2/d, ..., 1]
keep_prob   = sigmoid((dim_weights - λ) / T) # soft step at position λ
z = h * keep_prob                            # weighted hidden state
```

During the forward pass, dimensions are actually truncated to k (hard). The
soft weights provide gradients during backprop. Temperature T controls
sharpness (lower T → more step-like, less smooth gradient).

## Alternatives and Tradeoffs

| Method | Differentiable | Expressive | Notes |
|--------|---------------|-----------|-------|
| Soft sigmoid mask (used here) | Yes | Medium | Simple, may not zero dims |
| Gumbel-Softmax over dim indices | Yes | High | More complex, noisier |
| Hard threshold, straight-through | Partial | Medium | Biased gradient |
| Learned projection to fixed k | Yes | High | No adaptive k |

## Open Questions

- Does the soft mask actually learn to concentrate important information in the
  first k dimensions, or does it spread it across all dims with small weights?
- Would a learned permutation (sorting by importance) before masking help?
