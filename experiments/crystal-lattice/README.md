# Project "Crystal Lattice" -- Curiousity Driven Resonator
**Theme**: curiosity-driven latent exploration with topological grounding.

**Hypothesis**: a low data base, physically grounded reasoning (a "Neural Resonator" or "sensing organ"), and curiousity-driven self-directed learning could beget a massive efficiency breakthrough in training and learning strategies.

Subtesting/Base for:
- Unown neurons (neuro-symbolic with deep grounding)
- Active learning
- Geometric (molecular reasoning) self-awareness

**Context**: a hacker-implementation of a philosophical paradigm shift for AI architectures. theoretical grounding for continuous, geometrically self-aware, self-improving, AGI. Hypothetical anti-target: big data and discrete, prescribed token guessing.

**The "Why"**: We are trying to prove that **Small Data + Physical Grounding > Big Data + Token Guessing.**
*   The **VSA** provides the "Skeleton."
*   The **CLN** provides the "Muscle" (interpolation).
*   The **RLM** provides the "Curiosity" (where to practice).

---

# PROJECT BRIEF: Project Crystal Lattice
## Architecture: Multi-Scale Recursive Manifold (MSRM)
**Objective**: Build a "Neural Resonator" that internalizes 3D physical constraints (Macrocyclic Steric Clashes) through an active exploration loop of self-generated "Stepping Stones."

---

## 1. THE ARCHITECTURAL STACK (REFINED)

### **A. VSA Core (The Structural Memory)**
*   **Library**: `TorchHD` (using Holographic Reduced Representations / HRR).
*   **Role**: Encodes the 1D SMILES string into a 10,000-D hypervector lattice.
*   **The "Wormhole" Operator**: If a ring closure is detected (e.g., SMILES `1...1`), the VSA must bind the indices of both `1`s with a unique `CLOSURE_TAG` hypervector. This signals topological adjacency to the Navigator.

### **B. CLN (The Continuous Latent Navigator)**
*   **Model**: A tiny Recurrent Transformer or a single-layer `S4/Mamba` block (~50M params).
*   **Role**: Iteratively updates the latent state to "relax" the VSA lattice into a physically valid 3D manifold.
*   **Anchor Resonance**: The original VSA hypervector is added back into the latent state at every iteration to prevent "Latent Drift."

### **C. Gated Externalizer (The Sensing Organ)**
*   **Metrics**:
    1.  **Entropy**: Measures the "blurriness" of the Navigator’s hidden state.
    2.  **Structural Integrity**: Measures the Cosine Similarity between the current state and the VSA Anchor.
*   **Action**: Triggers a "Structural Warning" if integrity < 0.5.


---

## 2. THE EXPERIMENT: "CURIOSITY-DRIVEN PRACTICE"

### **Phase 1: Seed Crystallization (The Alphabet)**
*   **Action**: Train the HRM (CLN + VSA) on 100 simple **Linear Alkanes** (C-C-C chains).
*   **Goal (Grounding)**: Predict the 3D Euclidean distance between the first and last atom.
*   **Result**: The model will be perfect at "Length = N * 1.5Å" but will fail to understand that a ring can bring the head and tail within 2Å.
*   **Baseline**: Accuracy should be >95% because the physics is simple and additive.
To prevent the CLN from treating a ring like a failed linear chain, we modify the **VSA Encoding**:
*   **Mechanism**: When the RLM identifies a ring closure (e.g., the `1` in `C1...C1`), the VSA Core binds the first and last atom hypervectors with a unique **"Closure Tag"** hypervector.
*   **Role**: This acts as a "Wormhole" in latent space, signaling to the Navigator that position $A$ and position $B$ are adjacent despite their distance in the string.


    **Questions from Nora**: What are we making brittle for practical reasons and what is the idealized, continuous version? Question even the act of generating our data with a known tool: this is an experimental control, not an architecturally ideal one.

### **Phase 2: The Curiosity Loop (Identifying Blind Spots) / Self-Generated Stepping Stones (Active Learning)**
*   **The Stressor (RLM)**: Prompt `Ollama` (Llama-3.2-1B) to "mutate these linear SMILES chains into macrocyclic rings of sizes 12 to 20."
*   **Measurement and The Diversity-Entropy Filter**:
    1.  Pass the RLM’s generated rings through the Stage 1 HRM.
    2.  Record: $Score = \text{Latent Entropy} \times (1 - \text{Similarity to Alphabet})$.
    3.  Record: **VSA Diversity** (is this a new ring size?).
*   4. **Filtering**: Instead of just picking the "most confusing" molecules, we implement **Coreset Sampling**:
*   **Calculation**: $Score = \text{Latent Entropy} \times (1 - \text{Max Cosine Similarity to existing training set})$.
*   **Role**: This ensures the "Stepping Stones" explore **new continents** of the manifold rather than obsessively digging into a single "weird" but redundant geometry.
Select the 10 rings with the highest **Diversity-Entropy** score.
*   **The Physics Teacher**: Run **RDKit** on these 10 rings to get 3D Ground Truth. Use an **Energy Filter** to discard any "strained junk" (unphysical conformations).
    **Energy Filter**: Discard any where internal energy is > 2σ from the mean (ignore "synthetic junk").

### **Phase 3: Super-Resolution Training**
*   **Action**: Fine-tune the HRM on these 10 high-quality "Stepping Stones."
*   **Final Exam**: Test on a 20-atom macrocycle with a bulky tert-butyl group. Test on a new set of macrocycles. Does the "Head-to-Tail" distance prediction now account for the 3D ring closure?
*   **Success**: The HRM predicts the clash correctly because it learned the "Physics of the Loop" from the stepping stones.

---

## 2. Speculative Efficiency & Resource Needs

| Metric | Random Sampling | **Project Crystal Lattice** |
| :--- | :--- | :--- |
| **Training Data Needed** | 1,000+ rings | **~25-50 rings** |
| **Compute Time** | 12+ hours | **< 1 hour** |
| **Reasoning Type** | Probabilistic Interpolation | **Structural Derivation** |

### **Trade-offs for the Shortcut**
*   **The Shortcut**: Using a single `GRU` or `S4` layer as a proxy for a full CLN.
*   **Trade-off**: You lose the high-level semantic reasoning of a transformer, but you **gain extreme training speed**. For a "No-Moneys" experiment, a **Latent GRU-VSA** is the most efficient way to prove the "Neural Adhesion" concept.


## 3. IMPLEMENTATION GUIDE FOR CODING AGENT

### **Module 1: `data_generator.py` (The Physics Teacher)**
```python
# Task: Use RDKit to generate SMILES -> 3D Distances.
# CRITICAL: Implement the Energy Filter.
# If mol.GetProp('ENERGY') > mean + 2std, discard it.
# We only practice on "Stable Physics."
```

### **Module 2: `vsa_lattice.py` (The Structural Bond)**
```python
# Task: Use TorchHD to bind atom types and positions.
# CRITICAL: Implement the 'Cycle Operator'.
# When index i and j share a ring-id, mol_hv += (pos_i * pos_j * CLOSURE_TAG).
```
> "Treat the **VSA Closure Tag** as a hard constraint. The CLN must learn that when the 'Closure Tag' is present, the **Manifold Curvature** must increase to bring the 'Head' and 'Tail' vectors into the same neighborhood."

### **Module 3: `resonator.py` (The Navigator)**
```python
# Task: Implement the iterative CLN loop.
# CRITICAL: Use 'Anchor Re-injection'.
# hidden = transformer(hidden + alpha * vsa_anchor).
# Add an 'Anchor Loss' that probes the distance at step 5, 10, 15.
```

### **Module 4: `curiosity_loop.py` (The Practice Engine)**
```python
# Task: Integration script.
# 1. RLM (Ollama) generates 50 mutants.
# 2. HRM scores all 50 by Latent Entropy.
# 3. Request RDKit Ground Truth for Top 10.
# 4. Update HRM weights.
```

**Failure Management**:
*   If training plateaus: **Increase Anchor Loss weight.** This pulls the model back to physics.
*   If the model hallucinations: **Tighten the Energy Filter.** Your stepping stones are too "strained."

---

## 5. FINAL VERDICT ON EFFECTIVENESS
By focusing on **Diversity-Entropy Sampling**, you should achieve **90%+ accuracy** using fewer than **50 training samples**. This would be a massive efficiency breakthrough, proving that the model "inhabits" the physical manifold rather than just memorizing it.

**Target Resource Usage**:
*   **Compute**: 1x Local GPU (e.g., RTX 3060) or Colab Free.
*   **Time**: < 2 hours for total MVE.
*   **Cost**: $0.00.

**Agent Action**: "Start by implementing the `vsa_lattice.py` with the HRR-FFT binding logic. That is the seed of the crystal." 💎⚛️🚀

**Potential Verdict**: We have moved from a model that "talks about molecules" to a **Curiosity-Driven Resonator**. By adding the **Energy Filter** and **Diversity-Entropy Score**, you are protecting the model from the "Hall of Mirrors" and ensuring every single "Stepping Stone" builds real physical intuition.

This could serve as a baseline methodology for a meta-learning system, as a model for Curriculum Design, as a model for a particular kind of prehension organ design.

**The lattice is hardened. Start the loop.** 💎⚛️🚀
