---
title: "Literature Survey: Geometric, Recursive, and Structured Reasoning Architectures (2024-2026)"
date: 2026-03-08
type: literature-survey
tags: [geometric-reasoning, molecular-cot, VSA, hyperbolic-geometry, recursive-reasoning, crystal-structure]
---

# Literature Survey: Geometric, Recursive, and Structured Reasoning Architectures (2024-2026)

Surveyed 2026-03-08. Covers five topic areas with relevance to crystal/lattice structure understanding by neural networks.

---

## 1. Geometric Reasoning Architectures

Neural networks that reason about geometric/spatial structure, beyond molecular simulation.

### Key Papers

**AlphaGeometry / AlphaGeometry 2** (Trinh et al., Google DeepMind, Nature 2024; updated 2025)
- Neuro-symbolic system: neural language model + symbolic deduction engine for Olympiad geometry.
- AG2 uses Gemini-based LM trained on 10x more synthetic data; solves 84% of IMO geometry problems (2000-2024), surpassing average gold medalist.
- Key insight: the LM proposes auxiliary constructions that open new deduction paths for the symbolic engine. Demonstrates that geometric reasoning benefits from a hybrid approach where neural models handle creative leaps and symbolic systems handle rigorous deduction.

**Geometric Algebra Transformer (GATr)** (Brehmer et al., 2024)
- Scalable Transformer using projective geometric algebra representations, E(3)-equivariant.
- Extended to Lorentz-equivariant variant (L-GATr) for high-energy physics (NeurIPS 2024).
- Represents geometric objects (points, lines, planes) natively in the algebra, enabling geometric reasoning within attention.

**Geometric Reasoning in the Embedding Space** (2025, MDPI)
- GNNs and Transformers trained on geometric constraint satisfaction self-organize point embeddings into 2D grid structures. During inference, models iteratively construct hidden geometric figures within embedding spaces.
- Demonstrates that neural networks can develop genuine spatial representations, not just statistical correlations.

**GeoGramBench** (submitted NeurIPS 2025)
- Benchmark for geometric program reasoning: translating procedural drawing code into geometric reasoning.
- Even frontier LLMs achieve <50% accuracy at highest abstraction level, revealing a significant gap in geometric reasoning.

**G2VLM** (2025)
- Mixture-of-Transformer-Experts with dedicated geometric and semantic perception experts.
- Learns 3D geometry from 2D inputs via two-stage training.

**EquiCSP: Equivariant Diffusion for Crystal Structure Prediction** (Dec 2025)
- Novel equivariant diffusion model addressing lattice permutation equivariance.
- Rigorously maintains periodic translation equivariance during noising.
- Significantly surpasses existing CSP models.

**Equivariant NNs for Molecular Crystal Lattice Energy** (ACS Omega, Sep 2024)
- Allegro (equivariant NN) trained on molecular clusters for crystal lattice energy prediction.
- Eliminates need for periodic DFT; integrated with USPEX for crystal structure search.

### Crystal/Lattice Relevance
- Geometric algebra transformers could natively represent crystallographic symmetry operations.
- Equivariant architectures (GATr, EquiCSP) directly encode the symmetry groups relevant to crystal lattices.
- The self-organizing spatial representations observed in embedding spaces suggest networks can learn lattice structure implicitly.

---

## 2. Molecular Chain-of-Thought

Chain-of-thought reasoning applied to molecular, crystal, and materials science problems.

### Key Papers

**IBM: Interpretable Molecular Modeling via LLM Agents with CoT** (MRS Fall 2025)
- Multi-agent framework using CoT to infer intermediate properties (HOMO, LUMO, dipole) before final prediction.
- Improves both performance and interpretability for organic dye property prediction.

**ChemCrow** (Bran et al., Nature Machine Intelligence, 2024)
- LLM agent integrating 18 chemistry tools with GPT-4.
- Uses Thought/Action/Observation reasoning loop; autonomously planned syntheses of insect repellent, organocatalysts, and a novel chromophore.

**ChemCoTBench** (2025)
- First reasoning framework bridging molecular structure understanding with arithmetic-inspired operations (addition, deletion, substitution).
- 22,000-instance dataset with expert-annotated chains of thought for chemistry.
- Formalizes chemical problem-solving into transparent step-by-step workflows.

**CrystaLLM** (Nature Communications, Dec 2024)
- Autoregressive LLM trained on CIF files for crystal structure generation.
- Generates plausible crystal structures for unseen inorganic compounds.

**CrysText** (ChemRxiv, 2024-2025)
- Text-conditioned crystal structure generation from natural language prompts.
- LLaMA-3.1-8B and Mistral-7B fine-tuned with QLoRA on MP-20 benchmark.

**LLM-Prop** (npj Computational Materials, 2025)
- Predicts crystal properties from text descriptions.
- Outperforms GNN baselines: +8% band gap prediction, +65% unit cell volume.

**CSLLM** (Nature Communications, Jul 2025)
- Three specialized LLMs predicting synthesizability (98.6% accuracy), synthetic methods, and precursors for 3D crystals.

**MatLLMSearch** (Cornell, 2025)
- LLM-guided evolutionary crystal structure generation with DFT verification.

**MatPC** (ACS Applied Materials & Interfaces, 2025)
- Prompt-engineered LLMs + crystal structure prediction + DFT for semantic-driven material design.

**Prompt Engineering for Chemistry** (ACS Central Science, 2025)
- Survey of prompt engineering techniques for MOFs, batteries, autonomous experiments.

### Crystal/Lattice Relevance
- CrystaLLM and CrysText demonstrate that LLMs can learn crystallographic grammar (space groups, Wyckoff positions, lattice parameters) from CIF text.
- CoT reasoning could decompose crystal structure prediction into interpretable steps: composition -> prototype selection -> symmetry assignment -> coordinate refinement.
- The gap between LLM performance on molecular vs. crystal reasoning (more structured, periodic) is an open question.

---

## 3. VSA / Hypervector Architectures

Vector Symbolic Architectures and hyperdimensional computing for reasoning and representation.

### Key Papers

**LARS-VSA** (Mejri et al., Georgia Tech, May 2024)
- Hyperdimensional computing architecture for learning abstract rules from limited samples.
- Implements context-based self-attention in bipolar high-dimensional space.
- Combines relational bottleneck with explicit vector binding; robust to curse of compositionality.
- Matches or exceeds Transformer/Abstractor on mathematical reasoning with lower computational overhead.

**Rel-SAR: Systematic Abductive Reasoning via Diverse Relation Representations in VSA** (Jan 2025)
- VSA for abstract visual reasoning (Raven's Progressive Matrices-style tasks).
- Uses diverse relation representations in hyperdimensional space for abductive reasoning.

**NVSA: Neuro-Vector-Symbolic Architecture** (IBM Research, ongoing)
- Extends computation-in-superposition to CNNs and Transformers.
- 244x faster probabilistic abduction inference vs. SOTA.
- Enables transparent learning-to-reason with single-pass training.

**HDC Framework for Stochastic Computation** (Journal of Big Data, Oct 2024)
- HDC as probabilistic model for computation beyond classification.
- Exploits geometry and algebra of high-dimensional spaces for metaphorical/analogical reasoning.

**The Blessing of Dimensionality** (Fanizzi & d'Amato, 2025)
- Survey of HDC/VSA for analogical reasoning and knowledge graphs.
- Argues for transparent, error-tolerant neurosymbolic models.

**One-Shot Graph Representation Learning Using HDC** (Feb 2024)
- HDGL: leverages injectivity of node representations for one-shot graph learning.
- Semi-supervised node labeling with hyperdimensional embeddings.

### Crystal/Lattice Relevance
- VSA's algebraic binding/bundling operations map naturally to crystallographic operations: binding could represent atom-site associations, bundling could represent unit cell composition.
- The compositional, interference-robust nature of hypervectors could encode periodic structures without explicit periodicity handling.
- HDC graph learning (HDGL) could provide efficient crystal graph representations.
- VSA's capacity for analogical reasoning could enable "reasoning by analogy" across crystal structure types.
- **Open gap**: No published work directly applies VSA/HDC to crystal or lattice structures.

---

## 4. Latent Geometry for Reasoning

Using geometric structure in latent spaces (especially non-Euclidean) to improve reasoning.

### Key Papers

**HypLoRA** (NeurIPS 2025)
- Low-rank adaptation of LLMs directly on the hyperbolic manifold.
- Avoids distortion from exp/log map round-trips.
- Significant improvements on arithmetic and commonsense reasoning benchmarks.
- Key finding: LLM token embeddings exhibit high hyperbolicity (latent tree-like structure).

**Hierarchical Mamba (HiM)** (2025)
- Integrates Mamba2 (state-space model) with Poincare ball / Lorentzian manifold.
- Learnable curvature enables effective long-range and multi-hop reasoning.

**Hyperbolic LLMs (HELM)** (2025)
- First fully hyperbolic LLM variants.
- HELM-MiCE uses mixture-of-curvature experts to reflect geometric variation in token distributions.
- Hybrid-curvature architectures may unlock new capabilities by matching data geometry.

**HyperKGR** (EMNLP 2025)
- Hyperbolic GNN for knowledge graph reasoning.
- Embeds recursive learning trees in hyperbolic space; hierarchical message passing aligns with reasoning paths.

**Hyperbolic Geometric Latent Diffusion** (ICML 2024)
- GGBall: graph generation in hyperbolic latent space.
- Preserves structural hierarchies via continuous Poincare ball embeddings.

**HyperMR** (LREC 2024)
- Multi-hop reasoning in adjustable hyperbolic spaces.
- Maps different reasoning stages to different curvatures.

**HypMix** (CIKM 2024)
- Addresses graphs with mixed hierarchical and non-hierarchical structures.
- Handles poly-hierarchical structures (multiple parent trees).

**"The Geometry of Thought"** (2025)
- Scientific/mathematical reasoning maintains diffuse, exploratory geometry ("Liquid phase") that resists compression even with 9x parameter scaling.
- Global dimensionality unchanged across scales for science/math domains.

**Hyperbolic Deep Learning for Foundation Models: A Survey** (KDD 2025)
- Argues Euclidean geometry constrains foundation model representational capacity.
- Hyperbolic space provides provably low-distortion embeddings of tree-like and power-law structures.

### Crystal/Lattice Relevance
- Crystal structures have inherent hierarchical organization: atoms -> coordination polyhedra -> unit cells -> supercells -> crystal systems.
- Space group hierarchies (230 space groups organized into crystal systems, Bravais lattices, point groups) are naturally tree-like -- ideal for hyperbolic embedding.
- Mixed-curvature approaches (HELM, HiM) could handle crystals with both hierarchical (symmetry) and non-hierarchical (bonding network) structure.
- Hyperbolic latent spaces could provide more efficient encoding of crystal structure databases than Euclidean alternatives.

---

## 5. Recursive Reasoning Architectures

Architectures applying reasoning steps recursively or iteratively.

### Key Papers

**"Reasoning with Latent Thoughts"** (Saunshi et al., ICLR 2025)
- Foundational result: k-layer transformer looped L times nearly matches kL-layer model on reasoning tasks.
- Proves looped models implicitly generate "latent thoughts" and can simulate T steps of CoT with T loops.
- Establishes dichotomy between reasoning (benefits from looping) and memorization (does not).

**Relaxed Recursive Transformers** (ICLR 2025)
- Relaxes weight tying via depth-wise LoRA while preserving compactness.
- Recursive Gemma 1B outperforms similar-sized vanilla models.
- Continuous Depth-wise Batching enables 2-3x inference throughput gains.

**Ouro (Ouroboros)** (ByteDance, 2025)
- Pre-trained Looped Language Models at scale (7.7T tokens).
- 1.4B and 2.6B models match up to 12B SOTA LLMs.
- Entropy-regularized objective for learned depth allocation.

**Huginn** (Geiping et al., 2025)
- Recurrent depth approach with dynamic resource allocation via RNN-like iteration.
- Matches larger static-depth models with improved efficiency.

**LoopFormer** (Jeddi et al., Feb 2026)
- Elastic-depth looped Transformer via shortcut modulation.
- Single model performs well across variable compute budgets without retraining.
- Shortcut-consistency objective aligns shorter routes to full-route representations.

**SpiralFormer** (Feb 2026)
- Multi-resolution recursion for hierarchical dependencies.
- Recursive looping provides substrate for iterative refinement ("thinking time" in latent space).

**HRM: Hierarchical Reasoning Model** (2025)
- Dual-module: high-level (abstraction/slow reasoning) + low-level (fast/local computation).
- Alternating modules form "hierarchical convergence."

**ReSSFormer** (Oct 2025)
- Integrates recurrence, sparse attention, and self-organizing structure discovery.
- Reuses recurrent block with memory aggregation instead of stacking layers.

### Theoretical Properties
- **Turing completeness**: Looped architectures enable universal computation on finite-precision, bounded-width setups.
- **Algorithmic simulation**: Can exactly simulate gradient descent, Newton's method, dynamic programming, graph algorithms.
- **Length generalization**: Looped transformers with adaptive stopping dramatically outperform standard transformers on arithmetic at test lengths exceeding training lengths.
- **Adaptive inference**: Entropy/confidence-based halting yields 20-30% FLOPs savings with no accuracy loss.

### Crystal/Lattice Relevance
- Crystal structure determination is inherently iterative (refinement cycles in Rietveld analysis, DFT relaxation).
- Looped architectures could naturally model iterative structure refinement, where each loop corresponds to a refinement step.
- Adaptive computation depth maps to the variable difficulty of crystal structure prediction (simple vs. complex unit cells).
- The ability to simulate fixed-point algorithms is directly relevant to self-consistent field calculations in crystal electronic structure.
- Hierarchical reasoning (HRM) mirrors the multi-scale nature of crystal properties (local bonding -> unit cell -> bulk).

---

## Cross-Cutting Themes and Synthesis

### Convergence Points
1. **Geometry as inductive bias**: All five areas share the insight that encoding geometric structure (whether in the architecture, the latent space, or the reasoning process) improves neural network performance on structured problems.
2. **Iterative refinement**: Recursive architectures and CoT reasoning both implement iterative refinement -- the former in latent space, the latter in token space. Crystal structure prediction naturally requires both.
3. **Hierarchy and composition**: VSA binding/bundling, hyperbolic embeddings, and recursive architectures all handle hierarchical composition, which is fundamental to crystal structure (atoms compose into motifs, motifs into unit cells, unit cells into lattices).
4. **Symmetry**: Equivariant architectures explicitly encode symmetry; geometric algebra transformers represent symmetry operations natively. Crystal structures are defined by their symmetry groups.

### Open Opportunities for Crystal/Lattice Understanding
- **Equivariant looped transformers**: Combining equivariance (Topic 1) with recursive depth (Topic 5) for iterative crystal structure refinement.
- **Hyperbolic crystal embeddings**: Using hyperbolic geometry (Topic 4) to embed space group hierarchies and crystal structure databases.
- **VSA for crystallographic reasoning**: Applying VSA binding operations (Topic 3) to represent atom-site associations and symmetry operations compositionally.
- **Crystal CoT**: Developing explicit chain-of-thought protocols (Topic 2) for crystal structure prediction: composition -> Bravais lattice -> space group -> Wyckoff positions -> coordinates.
- **Geometric latent reasoning**: Using geometric algebra representations in a looped architecture's latent space to reason about crystal symmetry.
