# Review: Project Crystal Lattice -- Literature-Grounded Refinement

**Reviewer**: claude-opus-4-6
**Date**: 2026-03-08
**Companion survey**: `wiki/findings/geometric-and-recursive-reasoning-architectures-survey.md`

---

## Overall Assessment

The proposal is philosophically coherent and architecturally inventive. The core thesis -- that physical grounding + curiosity-driven active learning + compositional structure (VSA) can massively reduce data requirements -- is well-motivated and testable. The three-component stack (VSA skeleton / CLN muscle / RLM curiosity) is clean.

However, the recent literature (2024-2026) reveals several convergent lines of work that both **validate** key design choices and **suggest concrete refinements** that could strengthen the experiment. This review covers what the literature supports, what it complicates, and what it opens up.

---

## 1. What the Literature Validates

### 1a. VSA as structural backbone -- confirmed viable

**LARS-VSA** (Mejri et al., Georgia Tech, 2024) demonstrates that VSA-based self-attention in high-dimensional bipolar space matches or exceeds Transformers on abstract rule learning with lower compute. **NVSA** (IBM Research) shows computation-in-superposition scales to CNNs/Transformers with 244x faster probabilistic abduction. Your choice of HRR as the structural memory is architecturally sound.

The "Wormhole Operator" (Closure Tag binding) is a natural use of VSA's binding algebra. However, note that **no published work applies VSA/HDC to crystal or molecular structures** -- this is genuinely novel territory. That's exciting but also means there's no prior art to debug against.

### 1b. Iterative latent refinement -- strongly supported

The proposal's CLN (iterative relaxation loop with anchor re-injection) is now backed by an explosion of work on **recursive/looped transformers**:

- **"Reasoning with Latent Thoughts"** (Saunshi et al., ICLR 2025): Proves a k-layer transformer looped L times nearly matches a kL-layer model. Looped models implicitly generate "latent thoughts" equivalent to chain-of-thought.
- **Ouro** (ByteDance, 2025): 1.4B looped LM matches 12B standard models. Entropy-regularized objective for learned depth.
- **LoopFormer** (Feb 2026): Elastic-depth via shortcut modulation -- single model works across variable compute budgets.

Your CLN is effectively a looped/recursive architecture. This is now a well-validated paradigm.

### 1c. Curiosity-driven active learning -- sound approach

The Diversity-Entropy coreset sampling strategy is a reasonable active learning criterion. The literature on molecular CoT (ChemCoTBench, IBM's multi-agent CoT) confirms that breaking molecular reasoning into explicit intermediate steps improves both performance and interpretability.

---

## 2. What the Literature Complicates

### 2a. The CLN model choice: GRU/S4 vs. looped Transformer

The proposal suggests a "tiny Recurrent Transformer or single-layer S4/Mamba" as a pragmatic shortcut. The literature now offers a sharper trade-off:

| Option | Pros | Cons | Literature support |
|--------|------|------|-------------------|
| **GRU** | Fast, minimal params | No attention, weak at long-range | Legacy |
| **S4/Mamba** | Linear complexity, strong on sequences | Not equivariant, periodic structure unclear | HiM (2025) shows Mamba + hyperbolic works |
| **Looped Transformer (2-4 layers)** | Proven latent reasoning, adaptive depth | Heavier per-step | Ouro, LoopFormer, Relaxed Recursive (ICLR 2025) |
| **Looped S4/Mamba** | Best of both? | Untested combination | SpiralFormer (2026) hints at this |

**Suggestion**: The looped transformer literature is now strong enough that even a 2-layer transformer looped 5-10 times may be more principled than a single GRU. The **Relaxed Recursive Transformer** (ICLR 2025) shows that depth-wise LoRA relaxation of weight-tied loops outperforms vanilla models at minimal parameter cost. Consider this as the CLN backbone -- it gives you adaptive computation for free.

### 2b. 10,000-D hypervectors -- dimensionality question

Standard VSA literature uses 10,000-D, which is appropriate for symbolic binding tasks. But your downstream task (predicting a scalar: head-to-tail Euclidean distance) is very low-dimensional. The question is whether the CLN can efficiently extract a scalar from a 10,000-D space.

**Suggestion**: Consider a projection layer from VSA space to a lower-dimensional "reasoning manifold" (say 128-256D) before the CLN loop. The VSA provides the compositional structure; the CLN works in a denser space. This mirrors how **LARS-VSA** uses a relational bottleneck between the high-D VSA encoding and the reasoning module.

### 2c. Using Llama-3.2-1B as the mutation generator

The RLM (curiosity stressor) uses a 1B LLM to generate macrocyclic SMILES mutations. There are two concerns:

1. **Llama-3.2-1B's chemistry knowledge is shallow.** ChemCrow and CSLLM show that even GPT-4 needs external chemistry tools to generate valid molecules reliably. A 1B model will produce many invalid SMILES.
2. **CrystaLLM** (Nature Comms, Dec 2024) shows that LLMs can learn crystallographic grammar when fine-tuned on domain data. An off-the-shelf 1B model hasn't seen enough chemistry.

**Suggestion**: Either (a) add an RDKit validity check immediately after generation (before the entropy scoring -- discard invalid SMILES first), or (b) use a simpler programmatic mutation strategy (systematic ring size variation + substituent placement) to guarantee chemical validity. The LLM-as-mutator is a nice idea philosophically but may be the weakest link in practice.

### 2d. The "90%+ accuracy on 50 samples" claim

This is speculative (acknowledged as such), but the literature suggests caution:

- GeoGramBench shows frontier LLMs achieve <50% on geometric program reasoning.
- "The Geometry of Thought" (2025) finds that scientific/mathematical reasoning geometry is inherently diffuse and resistant to compression.
- The jump from linear alkanes to macrocycles with bulky substituents is a significant distributional shift.

**Suggestion**: Define "accuracy" more precisely. Mean absolute error in Angstroms? Relative error? Binary "does it predict closure vs. extension"? The simpler the metric, the more likely the 50-sample claim holds. I'd recommend starting with binary classification (ring vs. chain) before regressing to exact distance.

---

## 3. What the Literature Opens Up (Refinement Opportunities)

### 3a. Equivariance as a free lunch

**EquiCSP** (Dec 2025) and the **Geometric Algebra Transformer** (NeurIPS 2024) show that building symmetry invariance into the architecture -- rather than hoping the model learns it from data -- dramatically improves sample efficiency for structural prediction.

Your proposal deliberately avoids equivariance (the VSA + CLN stack is not equivariant). But the hypothesis is about sample efficiency. Equivariance is the single most proven method for achieving sample efficiency on geometric tasks.

**Refinement**: Consider making the CLN's projection head SE(3)-invariant, even if the VSA core and CLN loop are not. Concretely: after the CLN loop outputs a latent state, pass it through an invariant pooling layer before predicting the distance. This doesn't require rewriting the architecture -- it's a 5-line change that ensures the model can't "cheat" by memorizing orientations.

### 3b. Hyperbolic latent geometry for the CLN

**HypLoRA** (NeurIPS 2025) shows that LLM embeddings exhibit high hyperbolicity -- they're naturally tree-like. **Hierarchical Mamba** (2025) shows Mamba + Poincare ball geometry improves multi-hop reasoning. **HELM** (2025) introduces mixture-of-curvature experts.

Your proposal mentions "Manifold Curvature" conceptually ("the CLN must learn that curvature increases to bring head and tail into the same neighborhood"). The literature suggests making this literal:

**Refinement**: Run the CLN loop in a Poincare ball or Lorentzian manifold rather than flat Euclidean space. The `geoopt` library provides Poincare ball optimizers for PyTorch. This gives the "curvature" metaphor actual geometric teeth -- closures naturally correspond to regions of higher curvature, and the hyperbolic metric makes "bringing distant things close" representationally cheaper.

This is a more ambitious change, but it directly tests the proposal's core metaphor.

### 3c. Anchor re-injection as a resonance mechanism

The proposal specifies: `hidden = transformer(hidden + alpha * vsa_anchor)`. The recursive reasoning literature suggests a refinement:

**LoopFormer** (Feb 2026) uses "shortcut modulation" -- learned residual connections that allow information to skip iterations. **Relaxed Recursive Transformers** use depth-wise LoRA to allow each iteration to behave slightly differently.

**Refinement**: Instead of a fixed `alpha`, make it iteration-dependent: `alpha_t = sigmoid(linear(hidden_t))`. This lets the model learn when to lean on the VSA anchor (early iterations, high uncertainty) vs. when to trust its own refinement (late iterations, convergence). This is essentially a learned resonance decay.

### 3d. Crystal CoT as a longer-term extension

The survey reveals that explicit crystal chain-of-thought protocols don't exist yet. ChemCoTBench formalizes molecular CoT but not crystal/periodic structure CoT. This is a gap.

If Crystal Lattice succeeds, a natural follow-up is to decompose the prediction into interpretable steps: SMILES -> topology -> ring size -> conformational flexibility -> distance estimate. Each step would produce an intermediate VSA state that could be inspected. This connects to the broader project's interest in geometric self-awareness (the model knowing what it knows at each step).

### 3e. The "sensing organ" and confidence geometry

The Gated Externalizer (entropy + structural integrity metrics) maps directly to the wiki's `geometric-self-awareness` concept and the angle concentration metric. The literature on "Geometry of Thought" and "Confidence Manifold" (arXiv:2602.08159) provides a formal framework:

**Refinement**: Instead of a hard threshold (integrity < 0.5), use the angle concentration metric from the `geometric-self-awareness` wiki page as the integrity signal. This connects Crystal Lattice to the variable-bitrate-reasoning experiment and creates a shared measurement vocabulary across the research program.

---

## 4. Suggested Revised Architecture (Minimal Changes)

If incorporating only the lowest-risk refinements:

1. **CLN**: 2-layer Transformer looped 5-10x with depth-wise LoRA relaxation (instead of GRU/S4)
2. **VSA -> CLN bridge**: Add a 10,000-D -> 256-D learned projection before the CLN loop
3. **Anchor alpha**: Make iteration-dependent via `alpha_t = sigmoid(linear(hidden_t))`
4. **Output head**: SE(3)-invariant pooling before distance prediction
5. **RLM validity**: Add RDKit SMILES validity check before entropy scoring
6. **Integrity metric**: Use angle concentration (pairwise cosine similarity) instead of raw cosine to anchor
7. **Evaluation**: Start with binary ring-closure detection, then graduate to distance regression

If going bolder:

8. **Hyperbolic CLN**: Run the loop in Poincare ball space via `geoopt`
9. **Adaptive halting**: Use entropy-based stopping criteria (as in Ouro/LoopFormer) instead of fixed iteration count

---

## 5. Open Questions (for Nora)

Responding to the embedded question ("What are we making brittle for practical reasons and what is the idealized, continuous version?"):

**Brittle by design (experimental controls)**:
- SMILES as input representation (discrete, lossy encoding of 3D structure)
- RDKit as ground-truth oracle (approximation of real physics)
- Fixed ring size range (12-20 atoms)
- LLM-based mutation (chemically unreliable)

**The idealized continuous version** would be:
- Direct 3D point cloud or electron density as input
- DFT or MD simulation as oracle
- Continuous molecular geometry space (no SMILES discretization)
- Physics-guided generative model for mutations (diffusion on conformational manifold)

The gap between these is the experiment's contribution: showing that VSA + iterative refinement can bridge discrete representations to continuous physical understanding.

**Additional question raised by the literature**: The experiment currently tests "does the model learn ring closure physics?" But a deeper question is "does the model's latent geometry reflect physical geometry?" The trajectory metrics from `geometric-self-awareness` (velocity, curvature, manifold dimension across CLN iterations) could answer this. Consider logging them even if they're not the primary metric -- they're free to compute and may reveal whether the CLN is genuinely "relaxing" a manifold or just memorizing a lookup.

---

## 6. Suggested Reading Priority

For refining the experiment before implementation, ranked by relevance:

1. **LARS-VSA** (arXiv:2405.14436) -- closest architecture to your VSA+reasoning stack
2. **"Reasoning with Latent Thoughts"** (ICLR 2025) -- theoretical foundation for your CLN loop
3. **Relaxed Recursive Transformers** (ICLR 2025) -- practical guide for looped architectures
4. **EquiCSP** (arXiv:2512.07289) -- equivariant diffusion for crystal structure, different approach to same domain
5. **CrystaLLM** (Nature Comms, 2024) -- LLMs learning crystallographic grammar
6. **HypLoRA** (NeurIPS 2025) -- if considering hyperbolic CLN
7. **LoopFormer** (Feb 2026) -- adaptive depth and elastic computation
