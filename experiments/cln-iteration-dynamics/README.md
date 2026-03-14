---
status: planned
owner: null
dependencies: [crystal-lattice]
---

# CLN Iteration Dynamics

**Hypothesis**: The CLN (Continuous Latent Navigator) uses its iterative loop
meaningfully — later iterations perform qualitatively different computation
than early ones, visible in per-iteration diagnostics.

## Background

Crystal-lattice uses a 2-layer Transformer looped 8 times. The "resonance"
metaphor claims the CLN iteratively relaxes the VSA lattice into a physically
valid prediction. But it's possible the model does all computation in
iteration 1 and coasts — making the loop wasteful.

## Method

After any crystal-lattice training run, analyze the per-iteration CLNDiagnostics
(already logged during training):

1. Plot entropy, integrity, alpha, latent_norm across iterations 1-8
2. Compute the "useful iteration" count: at which iteration do predictions stop
   changing meaningfully?
3. Compare learned alpha_t trajectory: does it decay (resonance) or stay flat?
4. Ablation: at inference time, truncate the loop at iteration k for k=1..8.
   How does accuracy degrade?

## Success criteria

- Diagnostics show a clear "relaxation" pattern (e.g., entropy decreases,
  integrity increases monotonically)
- Truncation ablation shows accuracy loss when removing late iterations
- If diagnostics are flat: the loop is overkill, simplify to fewer iterations

## Results

See RESULTS.md (populated by analyze_cln_dynamics.py).
