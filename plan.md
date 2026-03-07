# Crystal Lattice — Implementation Plan

## Guiding Principle

The primary deliverable is **observability of the internalization process**, not
molecular accuracy. Every design decision should be evaluated by: "does this
make the dynamics of learning more legible?"

---

## Phase 0: Instrumented Scaffold

**Goal**: Data pipeline, VSA encoding, and dense logging infrastructure.

### 0a. `data_generator.py` — The Physics Oracle
- Generate linear alkanes (C₁–C₂₀) with RDKit: SMILES → 3D embedding → pairwise
  distances.
- Generate macrocyclic rings (sizes 12–20, with and without substituents) via
  programmatic ring closure (NOT LLM-generated). Use `RingClosureMutator` or
  manual SMILES construction.
- Energy filter: discard conformers with internal energy > 2σ from mean for that
  ring size.
- Output format: `(smiles, atom_positions_3d, head_tail_distance, metadata)`.
- Include a `curriculum_schedule()` function that yields batches in controlled
  order (linear → small rings → large rings → substituted rings).

### 0b. `vsa_lattice.py` — Structural Memory
- TorchHD-based HRR encoding of SMILES.
- Atom-type hypervectors (C, N, O, H, etc.) bound with positional hypervectors.
- **Closure operator**: when ring closure detected in SMILES, bind
  `pos[i] * pos[j] * CLOSURE_TAG` into the molecule hypervector.
- Return both the composite hypervector and the individual atom encodings (the
  CLN may need both).

### 0c. `logger.py` — Geometric Trace Recorder
- At every CLN iteration, record:
  - Latent entropy (distribution of hidden state activations)
  - Anchor cosine similarity (hidden vs. VSA anchor)
  - Angle concentration (mean pairwise cosine sim of token representations)
  - Current alpha value
  - Prediction error (head-tail distance)
  - L2 norm of hidden state
- Per-training-step, record:
  - All of the above, averaged and per-sample
  - Gradient norms (total and per-module)
  - Raw hidden states for a fixed probe set of molecules (for trajectory
    visualization later)
- Storage: HDF5 or simple `.npz` files. Keep it lightweight — this will be read
  by analysis scripts, not production code.

**Visualization note**: Store raw hidden states for a small probe set (5 linear,
5 rings) at every training checkpoint. This enables 3D trajectory animation
later with no re-running. PCA/UMAP projections can be computed post-hoc.

---

## Phase 1: Linear Baseline

**Goal**: Establish what "successful internalization of simple physics" looks
like geometrically.

### 1a. `resonator.py` — The CLN
- Architecture: single-layer GRU or S4 block, ~50M params.
- Input: VSA hypervector (10,000-D, projected down to model dim).
- Iterative loop: `hidden = CLN(hidden + alpha * vsa_anchor)` for N iterations.
- **Alpha is a learned scalar, initialized at 1.0, logged at every step.**
- Output head: predict head-to-tail Euclidean distance from final hidden state.
- Loss: MSE on distance prediction.

### 1b. Training
- Train on 100 linear alkanes (C₁–C₂₀, multiple conformers each).
- Train until convergence. Expected: >95% accuracy (trivial physics).
- **Primary output**: geometric trace of the learning process.
  - How does alpha evolve? (Expect: may decrease as the model internalizes the
    linear distance rule)
  - What does the latent trajectory look like at convergence? (Expect: smooth,
    low-entropy, high anchor similarity)
  - How many CLN iterations does the model actually "use"? (Later iterations may
    become no-ops if the physics is simple enough)

### 1c. Baseline Probing
- Pass a few ring molecules (unseen) through the trained model.
- Record geometric traces. Expected: high entropy, low anchor similarity,
  incorrect distance predictions.
- This gives us the geometric signature of "the model does NOT understand this
  constraint" — the contrast case for Phase 2.

---

## Phase 2: Controlled Ring Introduction

**Goal**: Watch the latent trajectory destabilize and re-crystallize as new
physics is introduced.

### 2a. Curriculum
- Introduce rings programmatically in controlled batches:
  1. Size-12 unsubstituted rings (10 samples)
  2. Size-16 unsubstituted rings (10 samples)
  3. Size-20 unsubstituted rings (10 samples)
  4. Rings with substituents (10 samples)
- Each batch: fine-tune the Phase 1 model for a fixed number of steps, then
  evaluate on held-out rings of that size AND all previous sizes.

### 2b. Observations
- At each curriculum step, record full geometric traces.
- Key questions:
  - Does alpha shift when rings are introduced? Does it re-stabilize?
  - Is there a "phase transition" visible in the latent trajectory — a moment
    where the representation qualitatively reorganizes?
  - Does the Gated Externalizer's structural integrity score drop when new
    ring sizes are introduced and recover after training?
  - Does the model's behavior on previously-learned ring sizes degrade
    (catastrophic forgetting) or remain stable?

### 2c. Gated Externalizer
- Implement entropy and structural integrity metrics.
- Log continuously. Define concrete thresholds post-hoc based on Phase 1
  calibration (not pre-set to 0.5).
- **No automatic response policy yet** — just instrument and observe. The right
  response policy is something we learn from the data.

---

## Phase 3: Internalization Analysis

**Goal**: Extract generalizable findings about the internalization process.

### 3a. Geometric Signature Extraction
- From Phase 1-2 traces, characterize:
  - The "crystallization signature": what does the geometric trajectory look like
    when the model transitions from confusion to understanding?
  - The "alpha story": does alpha's learned trajectory correspond to meaningful
    stages of internalization?
  - Whether concentration/entropy/integrity form a low-dimensional summary of
    learning state.

### 3b. Predictiveness Test
- Using the geometric signals from early training, can we predict:
  - Whether the model will generalize to a new ring size (before seeing it)?
  - How many samples the model needs for a new constraint (from the shape of
    the destabilization-recovery curve)?

### 3c. Visualization (conditional)
- If Phase 1-2 traces show legible structure in 2D projections, build 3D
  animated trajectory visualization.
- Use the fixed probe set's hidden states across training checkpoints.
- Tool: matplotlib animation or plotly for interactive 3D.

---

## Phase 4 (Future): Curiosity Loop

**Deferred** — not part of this implementation round. To be designed after
Phase 3 analysis, informed by what we learn about internalization dynamics.

- Question: does curiosity-driven data selection produce faster or qualitatively
  different internalization trajectories vs. curriculum ordering?
- Requires Phase 3 to define what "faster" and "different" mean geometrically.

---

## Implementation Order

| Step | File | Depends on | Est. complexity |
|------|------|-----------|-----------------|
| 1 | `src/data_generator.py` | nothing | Medium |
| 2 | `src/vsa_lattice.py` | nothing | Medium |
| 3 | `src/logger.py` | nothing | Low |
| 4 | `src/resonator.py` | vsa_lattice, logger | Medium-High |
| 5 | `src/gated_externalizer.py` | resonator, logger | Low |
| 6 | `src/train.py` | all above | Medium |
| 7 | `src/evaluate.py` | train, logger | Low |
| 8 | `src/visualize.py` | logger outputs | Low-Medium |

Steps 1, 2, 3 can be built in parallel. Step 4 depends on 2+3. Steps 5-8
are sequential.

---

## Control Arm (noted for future)

To prove the curiosity loop's value (when we get there), we'll need a control:
same architecture, same total sample count, random ring selection instead of
curiosity-driven. Design this into the training script's config from the start
so it's trivial to run later.

---

## Open Questions (to resolve during implementation)

1. **VSA dimensionality**: 10,000-D is standard for TorchHD but may be overkill
   for this domain. Consider 4,096 or even 1,024 and test whether binding
   fidelity degrades.
2. **CLN iteration count**: Fixed at N? Or learned/adaptive? Start fixed (e.g.,
   15), observe whether later iterations become no-ops, then decide.
3. **Projection from VSA dim to CLN dim**: Linear? Learned? This is a potential
   bottleneck and information-loss point.
4. **What "entropy of hidden state" means precisely**: Shannon entropy of what
   distribution? Softmax over activations? Binned histogram? Define this
   concretely before implementing.
