# DSD — Dead-Stop-Detach Future Prediction

> The stop-gradient future-state prediction objective used to train the
> compression bottleneck without representation collapse.

---

## Motivation

A compression bottleneck can collapse: if z is too small, the predictor can
learn to ignore it and predict the mean of h_{t+1}. Stop-gradient (sg / detach)
breaks this by preventing the target from adapting to the predictor.

## Objective

```python
future_loss = MSE(predictor(z_t), sg(h_{t+1}))
```

- `z_t` is the compressed representation at step t.
- `h_{t+1}` is the next hidden state (from the full-dimensional path).
- `sg` stops gradients from flowing through `h_{t+1}`.

The predictor must learn to reconstruct the future state from the compressed
bottleneck. This forces z_t to preserve information about what comes next.

## Information Leak Warning

In a single-pass implementation, h_{t+1} is computed using the same model that
produced h_t. This creates a mild information leak: the model can "cheat" by
producing h_{t+1} that is easy to predict from z_t. Two mitigations:

1. **Two-pass training**: run one forward pass without compression to get target
   states, then a second pass with compression for prediction. Doubles compute.
2. **Accept the leak**: it may not matter empirically. Test and report.

## Collapse Prevention vs. BYOL / SimSiam

This is structurally similar to BYOL's online/target network scheme and
SimSiam's stop-gradient. The key difference: here we are not doing
contrastive or self-supervised pre-training — the future state is determined
by the task, not by augmentation.

## References

- Grill et al. (2020), "Bootstrap Your Own Latent" (BYOL)
- Chen & He (2021), "Exploring Simple Siamese Representation Learning" (SimSiam)
