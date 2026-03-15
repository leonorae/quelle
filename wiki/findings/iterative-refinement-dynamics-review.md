---
date: 2026-03-15
type: literature-review
scope: cln-iteration-dynamics
---

# Iterative Refinement Dynamics: Literature Review

## Verdict

The literature characterizes iteration dynamics **substantially but not completely**.
The improvement-vs-iteration curve shape is well-established across multiple
communities (log-linear in early iterations, then plateau), task dependence is
documented, and adaptive halting is understood at the algorithmic level. What
remains open — and is the specific gap that justifies a narrow version of
`cln-iteration-dynamics` — is how these dynamics behave on **small models on
structured/geometric/regression tasks** rather than language benchmarks. All
large-scale evidence (Huginn 3.5B, Ouro 1.4B–2.6B, Saunshi et al.) uses
language tasks or synthetic algorithmic tasks at moderate-to-large scale. The
protein folding and GNN communities provide structured-prediction evidence but
on very different architectures.

**Recommendation**: Do not close the experiment. Narrow it to Option A or B from
the README (small model, structured task, per-iteration diagnostics on a task
where you have a working representation). Frame the contribution as
*validation in the small-model structured-prediction regime*, not discovery of
the phenomenon. The dynamics questions (relaxation vs. oscillation,
useful-iteration count, log-linear vs. log-step shape) are worth measuring once
concretely on your target task. The literature gives strong priors for what to
expect; the experiment tests whether those priors hold at small scale and on
non-language inputs.


## Looped Transformers

### Universal Transformer (Dehghani et al., ICLR 2019)

The original looped transformer. A k-layer transformer block is applied
repeatedly (up to a maximum N), with per-position Adaptive Computation Time
(ACT) halting. The key finding on halting: the model learns to ponder *more* on
ambiguous or semantically important tokens (e.g., ends of sentences, relevant
facts in bAbI) and *less* on straightforward ones. On structured algorithmic
tasks (parity, addition, sorting) and language tasks, ACT improved accuracy
over fixed-depth versions. Turing completeness under ACT was established
theoretically. ACT showed a consistent pattern: hard positions need more steps,
easy positions halt early — but the work did not publish systematic per-step
accuracy curves.

**Reference**: arXiv:1807.03819. ICLR 2019.

### Saunshi et al. — "Reasoning with Latent Thoughts" (ICLR 2025)

The most directly relevant theoretical/empirical study for this experiment.

Core finding: a k-layer transformer looped L times nearly matches a kL-layer
non-looped model on reasoning tasks (addition, p-hop induction, math problems),
significantly outperforming a k-layer non-looped model. The key empirical
observation is that **accuracy scales as the logarithm of effective depth**
(loop count). This log-linear scaling holds across reasoning tasks; gains are
larger for tasks requiring more reasoning steps. On memorization tasks
(closed-book QA), looping helps much less — confirming that iteration buys
*computation*, not *parameter capacity*.

Per-iteration contribution curve shape (inferred from the log-depth scaling):
each doubling of loop count produces a roughly constant improvement in accuracy,
meaning absolute marginal returns per iteration decay — the kth loop contributes
less than the (k-1)th. No direct per-step accuracy curve is plotted, but the
log-depth scaling directly implies this shape.

**Reference**: arXiv:2502.17416. ICLR 2025.

### Huginn (Geiping et al., 2025)

3.5B parameter recurrent-depth model (2-layer prelude, 4-layer recurrent core,
2-layer coda), trained to randomly sample iteration count 1–32 during training.
Average inference depth: 32 iterations, giving ~132 effective layers.

Iteration ablation on GSM8K: from near-zero accuracy at 1 iteration to ~42%
at 32 iterations. Task-dependent saturation: HellaSwag saturates around 8
iterations; GSM8K continues improving to 32+. This is the clearest published
demonstration of task-dependent saturation depth in a large recurrent transformer.

Training instabilities were encountered: hidden state collapse at k=8 iterations
(identical representations across tokens), and a mode where the model learned to
ignore the recurrent state so test-time compute added nothing. These were
resolved via careful normalization, adapter mechanisms, and reducing from k=8
to k=4 steps. This is important for `cln-iteration-dynamics`: instability modes
are known failure cases to watch for, not just a theoretical concern.

**Reference**: arXiv:2502.05171. Geiping et al., University of Maryland, 2025.

### Ouro (ByteDance, 2025)

1.4B–2.6B parameter looped LM trained to 7.7T tokens. Uses 4 recurrent steps
(R4) with entropy-regularized adaptive halting. Achieves ~4B equivalent
performance with 1.4B parameters on reasoning benchmarks.

Training stability: initial experiments with k=8 recurrent steps produced loss
spikes and gradient oscillations; reducing to k=4 was the fix. A two-stage
entropy regularization objective teaches when to halt: Stage 1 uses KV
regularization toward a uniform prior (exploration); Stage 2 learns an adaptive
exit policy based on performance improvement. Attempting RL alignment of the
adaptive exit mechanism failed due to infrastructure challenges with dynamic
computation graphs.

The adaptive halting results: 20–30% FLOPs savings are achievable by early exit
on simple tokens/inputs, consistent across benchmarks.

**Reference**: https://ouro-llm.github.io/. HuggingFace: ByteDance/Ouro-1.4B,
ByteDance/Ouro-2.6B.

### LoopFormer (Jeddi et al., arXiv:2602.11451, 2026)

Addresses a key failure mode of standard looped models: when trained with a
fixed loop count, early exit causes representation stagnation (flat trajectories,
high CKA between successive hidden states). Standard looped baselines show
minimal representation change after a few iterations.

LoopFormer introduces shortcut-consistency training with timestep conditioning
(inspired by diffusion transformer adaLN), aligning shorter-loop trajectories
to full-loop targets. Result: perplexity and reasoning scale gracefully across
1–L loop budgets without retraining. Representation trajectory diagnostics
(curvature, anisotropy, entropy) show LoopFormer evolving through a structured
arc — rising curvature and entropy in mid-depth, tapering near final step.

Key implication for `cln-iteration-dynamics`: a model trained at a fixed loop
count may *not* produce well-structured iteration dynamics at shorter counts.
Training strategy matters for whether the curve shape is informative.

**Reference**: arXiv:2602.11451.

### SpiralFormer (Yu et al., arXiv:2602.11698, 2026)

Proposes multi-resolution recursion: the shared transformer core processes
compressed (downsampled) representations at earlier loops and full-resolution
representations at later loops. This induces hierarchical functional
specialization across iterations, confirmed by attention entropy probes showing
systematic cross-loop shifts. Better Pareto front vs. FLOPs than vanilla looped
baselines at 160M–1.4B scale.

**Reference**: arXiv:2602.11698.

### Two-Scale Latent Dynamics (Pappone et al., NeurIPS 2025)

The most geometrically precise study of what iteration actually does inside a
looped block. Trained on GPT-2 scale, over training the authors measure step
norms and consecutive-step angles:

- Within a looped block, updates become **smaller and increasingly orthogonal**
  to one another over the course of training, indicating local fine-grained
  refinements (tight spiral geometry in PCA space).
- Across blocks (the non-looped prelude/coda), the representation undergoes
  larger drift.
- This two-scale geometry motivates a second-order early-exit criterion (halt
  when the norm of the difference between consecutive loop updates stops
  changing), which outperforms step-norm and KL-based baselines.

For `cln-iteration-dynamics`, this is the key paper on *what convergence looks
like mechanistically*: not fixed-point convergence in general, but a progressive
stabilization of update direction (orthogonality) and magnitude (shrinking norms).

**Reference**: arXiv:2509.23314. NeurIPS 2025.

### Hidden State Trajectory Types

Synthesizing across the literature (primarily from the looped transformer PCA
studies), three dynamic regimes have been observed in looped models:

1. **Fixed-point convergence**: ordinary/low-information tokens reach a stable
   representation after a few iterations. Visible as convergence to a point in
   PCA space.
2. **Orbiting**: tokens requiring complex reasoning (e.g., numbers in math
   problems, structural tokens) exhibit rotational trajectories in PCA space —
   a limit cycle, not convergence. Suggests the model is performing iterative
   computation rather than relaxation.
3. **Sliders**: some key tokens drift consistently along a direction across
   iterations, possibly supporting counting or evidence accumulation.

These are empirical observations from large pretrained models. Whether they
appear in small models on structured tasks is unknown.

### ANIRA / Understanding Dynamic Compute Allocation (Moosa et al., arXiv:2602.08864, 2026)

Introduced complexity-controlled synthetic evaluation (algorithmic and synthetic
language tasks with parameterized difficulty) to test whether token-level
adaptive halting actually correlates with ground-truth token difficulty. Two
variants: ANIRA-E (depth decided from prelude representation) and ANIRA-O
(online halting between iterations). Found that adaptive allocation does correlate
with token difficulty on synthetic tasks but that the correlation is task-dependent.

**Reference**: arXiv:2602.08864.

### Mixture-of-Recursions (Bae et al., arXiv:2507.10524, 2025)

Token-level dynamic recursion depth via lightweight router. A 118M MoR model
outperforms a 315M vanilla transformer on few-shot accuracy, with 25% less
memory and ~2x inference speedup. The router is trained jointly with the model
and learns to assign more iterations to harder tokens. Key result: adaptive
depth is learnable from scratch.

**Reference**: arXiv:2507.10524.

### Relaxed Recursive Transformers (arXiv:2410.20672, 2024)

Layer-tying with per-layer LoRA adaptors (low-rank, not full weight sharing).
Allows weight sharing while retaining per-iteration differentiation. Recursive
Gemma 1B matches Gemma 2B on most benchmarks. Combined with early-exit:
2–3x inference throughput gains.

**Reference**: arXiv:2410.20672.

### Loop as a Bridge (arXiv:2601.10242, 2026)

Investigates whether looped transformers (specifically Ouro) develop
introspective awareness across loop iterations — i.e., can they detect
concept injections in early/intermediate loops? Findings: no. Concept
injections are detected mainly when injected in the *final* loop. Increasing
loop iterations narrows accuracy gaps, but partly through degradation of
representation-based probes rather than genuine introspection. This is an
important negative result: more loops is not the same as richer intermediate
representations.

**Reference**: arXiv:2601.10242.


## Structured Prediction (Protein Folding, GNNs, Diffusion)

### AlphaFold2 Recycling (Jumper et al., Nature 2021)

AlphaFold2 recycles predictions by feeding the predicted backbone coordinates,
pair representations, and first MSA row back as inputs to the next pass. Default:
3 recycling iterations during both training and inference. Training uses a random
number of recycling iterations per example (not always the maximum), so each
recycling output is trained to incrementally improve.

Ablation (Figure 4b): recycling was marked "relatively important" — contributing
meaningfully to the overall accuracy. Rerunning CASP14 with 12 recycling cycles
(vs. default 3) improved average TM-score from ~0.89 to 0.898 for
MSA-poor targets. Easy targets (pLDDT > 85) often converge within 3 cycles or
fewer; hard targets benefit from up to ~20 cycles, with diminishing returns
beyond. The curve shape is roughly: rapid improvement in cycles 1–3, slow
continued improvement 4–12, plateau beyond 12.

**Reference**: Jumper et al., Nature 596, 583–589 (2021).

### RoseTTAFold2 Recycling (Baek et al., 2023)

RF2 added recycling (0–3 iterations during training, gradients only from final
iteration). Adding recycling + AF2 distillation improved GDT-TS by 3.1 on
CASP14 targets. The original RoseTTAFold (2021) could produce comparable
predictions in a single iteration due to its three-track parallel architecture —
suggesting that recycling benefits depend on whether the architecture needs
iterative depth for its core computation.

**Reference**: Baek et al., biorXiv:2023.05.24.542179.

### GNN Message Passing: Over-Smoothing (Li et al., 2018; many follow-ups)

GNNs iterate message passing across neighborhood hops. The over-smoothing
phenomenon: after enough iterations, node representations converge toward a
stationary distribution (degree-weighted or uniform, depending on normalization),
becoming indistinguishable. This is the GNN analogue of the oscillation/stagnation
failure mode. In practice:

- 2–3 message-passing iterations are typical; 4–6 occasionally used.
- Performance peaks early and *degrades* at higher iteration counts due to
  over-smoothing (not just plateau — active degradation).
- Residual connections, normalization (PairNorm, BatchNorm), and gating
  (GRU-style) are required to sustain improvement beyond ~4 iterations.
- This is a hard constraint: GNNs without anti-over-smoothing mechanisms cannot
  benefit from additional iterations.

This is a well-known failure mode with a different character from the looped
transformer case. In transformers, the analogous failure is hidden state collapse
(Huginn's failure #1) or the model learning to ignore the recurrent state
(Huginn's failure #2).

**Reference**: Li et al. (2018), "Deeper Insights into GCNs"; broad literature
on over-smoothing. Recent survey: arXiv:2006.13318.

### Diffusion Models: Denoising Step Count

Diffusion models are discrete-time iterative refinement processes. DDPM
(Ho et al., 2020) trains with 1000 diffusion steps; DDIM (Song et al., 2021)
shows that 100–50 steps produce comparable quality. FID scores plateau
between 20–50 steps on most benchmarks. Diminishing returns appear rapidly.
A "StepSaver" analysis (arXiv:2408.02054) finds that SSIM between consecutive-step
images shows a sharp drop early then flattens — the first significant drop
identifies the minimum viable step count for a given prompt difficulty.

The Pareto structure: most information is recovered in the first 10–20% of the
nominal step count. This mirrors the looped-transformer log-depth scaling.

**Reference**: Ho et al., NeurIPS 2020. Song et al. (DDIM), ICLR 2021.
StepSaver: arXiv:2408.02054.

### Iterative Amortized Inference (Marino, Yue & Mandt, ICML 2018)

VAE encoder iteratively refines approximate posterior estimates by encoding
gradients. Trained with 16 inference iterations; at test time remains stable
over hundreds of iterations. Iterative inference converges faster to better
ELBO estimates than conventional optimizers (momentum-based), despite having
less derivative information. The convergence profile: rapid improvement in the
first ~16 steps, then stable (not degrading) plateau. The key finding: the
learned iterative refinement generalizes — it was not over-fit to the specific
iteration count it was trained at.

**Reference**: Marino, Yue & Mandt (2018), ICML. arXiv:1807.09356.


## The Iteration Dynamics Profile

Synthesizing across all sources, the following picture emerges:

**Shape**: The improvement-vs-iteration curve is approximately **log-linear in
the early phase**, followed by a plateau. That is, doubling the iteration count
produces a roughly constant additive improvement in accuracy/quality, meaning
absolute marginal returns decay approximately as 1/k for the kth iteration. This
holds in:
- Looped transformers (Saunshi et al.: explicit log-depth accuracy scaling)
- Huginn (GSM8K: ~0% at k=1, ~34% at k=8, ~42% at k=32 — consistent with
  log-linear)
- AlphaFold2 recycling (fast improvement 1–3 cycles, slow 4–12, plateau beyond)
- Diffusion models (plateau at 20–50 steps from 1000 nominal)

**Convergence geometry** (Pappone et al., NeurIPS 2025): within a looped block,
updates shrink in norm and become increasingly orthogonal over the course of
training. The representation traces a contracting spiral, not a straight-line
convergence to a fixed point. This is a more nuanced picture than simple
relaxation.

**Failure modes** that change the shape:
1. **Hidden state collapse** (Huginn failure #1): all token representations
   become identical. Curve is flat from early iterations. Fix: normalization
   + adaptor mechanisms.
2. **Ignoring the recurrent state** (Huginn failure #2): model stops using
   the iterative input. Iterations add compute but not quality.
3. **Over-smoothing** (GNNs): performance peaks then *degrades* with more
   iterations. This is an active reversal, not a plateau. Occurs when
   residual connections and gating are absent.
4. **Representation stagnation** (LoopFormer baseline analysis): standard
   looped models trained at a fixed loop count show near-zero representation
   change (flat CKA) at shorter loop counts — the model has not learned to
   use intermediate iterations productively.

**Relaxation vs. oscillation**: both regimes are observed empirically. Ordinary
tokens tend toward fixed-point convergence (relaxation). Reasoning-relevant
tokens show orbital (limit-cycle) behavior. Over-smoothing in GNNs is a
degenerate fixed point. True divergence/oscillation (increasing error with
iteration) has not been reported in well-trained looped transformers but occurs
during *training* instability (loss spikes at k=8 in Ouro, Huginn).


## Adaptive Halting

Three main approaches, ordered by sophistication:

**ACT (Graves 2016)**: per-symbol halting probability predicted at each step;
processing stops when cumulative halting sum reaches 1 − ε. A ponder cost in
the loss discourages excessive computation. Results: model learns to allocate
more iterations to harder/more ambiguous tokens. Deterministic and
differentiable, low implementation overhead.

**Universal Transformer ACT (Dehghani et al., 2019)**: ACT applied per-position
in the sequence. Visualization on bAbI: the network allocates more ponder time
to facts relevant to the question than to irrelevant context. Mixed results:
improved accuracy on algorithmic and structured tasks, marginally degraded on
MT.

**PonderNet (Banino et al., NeurIPS 2021)**: reformulates halting as a
probabilistic generative model. Halting event is a geometric process;
loss is the expectation of reconstruction loss plus KL toward a geometric prior.
Unbiased, low-variance gradient estimates (unlike ACT's mean-field approximation).
More principled regularization: the prior discourages pondering longer than
necessary without hard constraints. Works on parity, bAbI, and paired associative
inference tasks.

**Ouro entropy-regularized halting**: Two-stage; Stage 1 encourages exploration
via KL toward a uniform prior on exit steps; Stage 2 learns an exit policy from
performance signal. Achieves 20–30% FLOPs savings at inference. RL alignment
of the exit mechanism failed.

**CALM (Schuster et al., NeurIPS 2022)**: Early exit in standard (non-looped)
transformers based on softmax confidence of intermediate-layer predictions.
Achieves 3x speedup with full performance maintained; the dynamic oracle
suggests theoretical speedups of 5.2x. Key finding: only ~1/3 of layers are
needed on average across tokens for typical text generation tasks — most tokens
are "easy" and could exit after few layers. This is effectively a halting
distribution over depth, not over loop count, but the inference is similar.

**Learned iteration counts across models**: The consistent finding across ACT,
PonderNet, CALM, MoR, and Ouro is that the *distribution* of iteration counts
is bimodal or heavy-tailed: most tokens/inputs use very few iterations
(the "easy" mass), while a small fraction requires significantly more (the
"hard" tail). The mean useful iteration count is typically 20–50% of the
maximum budget. This is task- and model-dependent.


## Task Dependence

The literature provides fairly clear evidence on what predicts useful iteration
count:

**Requires more iterations**:
- Multi-hop reasoning (p-hop induction, GSM8K, OlympiadBench)
- Structured algorithmic tasks (sorting, parity, addition) — with the caveat
  that these saturate at different counts: addition ≈ 4–8 loops, more complex
  reasoning ≈ 16–32
- Protein structures with few homologs (low MSA depth) — more recycling cycles
  needed
- Graph prediction tasks requiring long-range dependencies (but over-smoothing
  limits naive depth scaling)

**Requires few iterations**:
- Simple token prediction / language modeling where context is sufficient
  (CALM: many tokens exit after 1–2 layers out of 8)
- HellaSwag, OpenBookQA: Huginn saturates around 8 loops
- High-confidence protein structures (AF2: early-exit criteria based on pLDDT)
- Memorization tasks (Saunshi et al.: looping does not help closed-book QA)

**General predictor**: tasks that can be cast as iterative algorithms (repeated
application of a rule, message passing, gradient descent, coordinate descent)
benefit from more iterations. Tasks that require retrieval from parametric
memory do not benefit proportionally.

**Structured prediction vs. language**: the protein folding and GNN communities
suggest that structured prediction tasks *do* have a well-defined convergence
point — once the structure is determined, additional iterations do not improve
it further. This contrasts with open-ended reasoning tasks where more iterations
may keep improving. This distinction is directly relevant to `cln-iteration-dynamics`
if the target task is regression or geometry (which has a ground-truth answer
that imposes a natural convergence criterion).


## Implications for cln-iteration-dynamics Experiment

**What the literature already establishes** (does not need to be re-measured):
- The qualitative shape of the improvement curve: log-linear early, then plateau.
- Failure modes: collapse, state-ignoring, over-smoothing.
- That adaptive halting is feasible and saves 20–30% compute in practice.
- That task difficulty predicts useful iteration count.
- The two-scale geometry of looped block dynamics (Pappone et al.).

**What is not established and motivates the experiment**:

1. **Small model, structured task**: all published evidence comes from models
   ≥100M parameters on language or math tasks. Whether a small (say, 2–12
   layer) looped transformer on a structured geometric or molecular prediction
   task shows the same log-linear curve, or saturates much sooner, or collapses
   more easily, is not known. The failure modes (collapse, state-ignoring) are
   more likely at small scale.

2. **Regression outputs vs. discrete token outputs**: most looped transformer
   work measures accuracy on classification/generation tasks. Measuring
   per-iteration L2 distance of regression outputs, entropy of continuous
   predictions, and convergence of output structure is a different diagnostic
   regime and has not been published for looped transformers.

3. **Whether the crystal-lattice design pattern works at all**: the 2-layer
   transformer looped 8 times with anchor re-injection was a specific design
   choice. The literature suggests anchor re-injection is novel (not standard
   in the looped transformer literature) and its effect on dynamics is untested.

**Concrete recommendations**:

- Run Option A from the README first: looped transformer on one of the synthetic
  tasks from Saunshi et al. (p-hop induction or addition) at small scale. This
  validates that the experimental infrastructure produces dynamics consistent
  with the published literature before moving to novel territory.

- Measure the diagnostics specified in the README (L2 between iterations,
  output entropy, truncation accuracy at each k). These are not available in
  existing papers at the token/output level for structured tasks.

- Watch for Huginn's failure mode #2 specifically: if the model's accuracy vs.
  iteration count curve is flat, it has learned to ignore the recurrent state.
  This is the first diagnostic to check.

- Do not implement adaptive halting in the initial experiment. Establish the
  fixed-loop dynamics first. Halting is a follow-on.

- The useful iteration count for small models on structured tasks is likely low
  (2–8 loops), based on the pattern that simpler tasks saturate earlier. Plan
  experimental budgets accordingly.


## References

1. Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, Ł. (2019).
   Universal Transformers. ICLR 2019. https://arxiv.org/abs/1807.03819

2. Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks.
   arXiv:1603.08983. https://arxiv.org/abs/1603.08983

3. Banino, A., Balaguer, J., & Blundell, C. (2021). PonderNet: Learning to
   Ponder. ICML 2021 Workshop on Automated Machine Learning. arXiv:2107.05407.
   https://arxiv.org/abs/2107.05407

4. Schuster, T., Fisch, A., Gupta, J., Dehghani, M., Bahri, D., Tran, V.,
   Tenenholtz, N., & Miculivicius, P. (2022). Confident Adaptive Language
   Modeling (CALM). NeurIPS 2022. arXiv:2207.07061.
   https://arxiv.org/abs/2207.07061

5. Saunshi, N., Dikkala, N., Li, Z., Kumar, S., & Reddi, S.J. (2025). Reasoning
   with Latent Thoughts: On the Power of Looped Transformers. ICLR 2025.
   arXiv:2502.17416. https://arxiv.org/abs/2502.17416

6. Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., et al. (2025). Scaling
   up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.
   arXiv:2502.05171. https://arxiv.org/abs/2502.05171 (Huginn)

7. Zhu, et al. (2025). Ouro: Looped Language Models. ByteDance.
   https://ouro-llm.github.io/
   HuggingFace: https://huggingface.co/ByteDance/Ouro-1.4B

8. Jeddi, A., et al. (2026). LoopFormer: Elastic-Depth Looped Transformers for
   Latent Reasoning via Shortcut Modulation. arXiv:2602.11451.
   https://arxiv.org/abs/2602.11451

9. Yu, C., et al. (2026). SpiralFormer: Looped Transformers Can Learn
   Hierarchical Dependencies via Multi-Resolution Recursion. arXiv:2602.11698.
   https://arxiv.org/abs/2602.11698

10. Pappone, F., Crisostomi, D., & Rodolà, E. (2025). Two-Scale Latent Dynamics
    for Recurrent-Depth Transformers. NeurIPS 2025. arXiv:2509.23314.
    https://arxiv.org/abs/2509.23314

11. Anonymous (2026). Loop as a Bridge: Can Looped Transformers Truly Link
    Representation Space and Natural Language Outputs? arXiv:2601.10242.
    https://arxiv.org/abs/2601.10242

12. Bae, S., et al. (2025). Mixture-of-Recursions: Learning Dynamic Recursive
    Depths for Adaptive Token-Level Computation. arXiv:2507.10524.
    https://arxiv.org/abs/2507.10524

13. Moosa, I.M., et al. (2026). Understanding Dynamic Compute Allocation in
    Recurrent Transformers. arXiv:2602.08864.
    https://arxiv.org/abs/2602.08864

14. Jumper, J., et al. (2021). Highly accurate protein structure prediction with
    AlphaFold. Nature 596, 583–589.
    https://www.nature.com/articles/s41586-021-03819-2

15. Baek, M., et al. (2023). Efficient and accurate prediction of protein
    structure using RoseTTAFold2. biorXiv:2023.05.24.542179.
    https://biorxiv.org/content/10.1101/2023.05.24.542179

16. Marino, J., Yue, Y., & Mandt, S. (2018). Iterative Amortized Inference.
    ICML 2018. arXiv:1807.09356. https://arxiv.org/abs/1807.09356

17. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic
    Models. NeurIPS 2020. arXiv:2006.11239.
    https://arxiv.org/abs/2006.11239

18. Song, J., et al. (2021). Denoising Diffusion Implicit Models (DDIM).
    ICLR 2021. arXiv:2010.02502. https://arxiv.org/abs/2010.02502

19. StepSaver (2024). Predicting Minimum Denoising Steps for Diffusion Model
    Image Generation. arXiv:2408.02054.
    https://arxiv.org/abs/2408.02054

20. Fang, M., et al. (2024). Relaxed Recursive Transformers. arXiv:2410.20672.
    https://arxiv.org/abs/2410.20672
