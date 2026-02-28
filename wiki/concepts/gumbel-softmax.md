# Gumbel-Softmax

> Differentiable approximation to discrete sampling — the mechanism used for
> differentiable dimension selection in variable-bitrate compression.

---

## Problem

Many interesting operations require choosing a discrete index (e.g., "keep the
top-k dimensions"). Discrete choices are non-differentiable, breaking
backpropagation.

## Gumbel-Max Trick

To sample from a categorical distribution with logits **l**:

```
i* = argmax_i (l_i + g_i)   where g_i ~ Gumbel(0, 1)
```

This is equivalent to sampling from softmax(l) but expressed as an argmax.

## Gumbel-Softmax Relaxation

Replace argmax with softmax at temperature T:

```
y_i = exp((l_i + g_i) / T) / Σ_j exp((l_j + g_j) / T)
```

- As T → 0, y approaches a one-hot vector (argmax).
- As T → ∞, y approaches uniform.
- For moderate T, y is a continuous approximation that supports gradients.

## Straight-Through Estimator

In the forward pass, use the hard (argmax) sample. In the backward pass, use
the soft (Gumbel-Softmax) gradient. This gives exact forward behavior with
approximate gradients.

## In This Repo

`variable-bitrate-reasoning` uses a simplified version: rather than sampling
over dimension indices, it applies a sigmoid soft-mask parameterized by λ. This
avoids the sampling noise of Gumbel while retaining differentiability. See
[variable-bitrate-compression.md](variable-bitrate-compression.md).

## References

- Jang et al. (2017), "Categorical Reparameterization with Gumbel-Softmax"
- Maddison et al. (2017), "The Concrete Distribution"
