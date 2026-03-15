---
date: 2026-03-15
type: literature-review
scope: factorization-taxonomy
---

# Output-Side Factorizations in Transformers: Literature Scan

## Verdict

The claim is **substantially correct but needs nuancing**. There is no body of work that
frames output-side factorizations as a *design principle* or recognizes value embeddings
as a named member of a factorization taxonomy. The engineering practice exists
(modded-nanogpt, nanochat), the empirical benefit is established, but:

1. No paper analyses *what the value table learns* — the interpretability gap is real.
2. No theoretical work asks *why the output side benefits from type-level priors* or
   frames the QKV asymmetry as a factorization asymmetry.
3. Adjacent work (value residual learning, SwitchHead, MLA) addresses the value side for
   efficiency or compression reasons, not because it recognises a type-level prior on the
   output side.

The gap is not "nobody has done output-side tricks" but "nobody has framed output-side
type-level factorization as a principled thing, nor analysed what is stored there."

---

## Prior Art Found

### Value Embeddings in modded-nanogpt / nanochat (2024–2025)

The most direct prior art. Keller Jordan's modded-nanogpt introduced per-layer value
embedding tables (`nn.Embedding(vocab_size, kv_dim)`) added to the V vectors in
attention. Nanochat (Karpathy, launched Oct 2025) adopted the same design with a
learned gate (`3 * sigmoid(linear(x))` scoping output to (0, 3)) and a lower LR for
the embedding parameters.

snimu's blog posts document engineering ablations: adding more value embedding modules
monotonically reduces loss; they are applied in an alternating/shared pattern across
layers; they are described as "a powerful way to add bias to LLMs." Logit-lens
analysis is mentioned but contains **no analysis of what token-type structure the table
encodes**. The mechanism is treated as an empirical improvement, not a conceptual one.

- modded-nanogpt repo: https://github.com/KellerJordan/modded-nanogpt
- snimu, "adding value embeddings" (Oct 2025): https://snimu.github.io/2025/10/07/modded-nanogpt-value-embeddings.html
- snimu, "analyzing value-embedding lambdas" (Aug 2025): https://snimu.github.io/2025/08/11/modded-nanogpt-lambdas.html
- nanochat gpt.py: https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py

### Value Residual Learning — ResFormer / SVFormer (Zhou et al., Oct 2024 / ACL 2025)

Proposes passing a residual of the **first layer's value states** to all deeper layers
(ResFormer) or sharing a single value state from layer 1 across all layers (SVFormer).
Motivation is preventing attention concentration, not type-level priors. SVFormer
reduces KV cache ~50%. Achieves equivalent loss with 16% fewer parameters.

This is an output-side intervention but at the cross-layer level (preserving
first-layer value residuals), not a token-type lookup. The paper does not ask what the
first layer's values encode or why they are worth preserving; it treats them empirically.

- arXiv 2410.17897: https://arxiv.org/abs/2410.17897
- ACL 2025: https://aclanthology.org/2025.acl-long.1375/

### Bigram Subnetworks (arXiv 2504.15471, Apr 2025)

Identifies minimal subnetworks that implement bigram (token-identity-to-next-token)
predictions, concentrated in the **first MLP layer**, comprising <0.2% of parameters.
This is mechanistic evidence for a type-level prior on the output side living in the
early MLP, not in the value projection. The framing is about subnetwork identification,
not factorization design.

Directly relevant: if bigram structure lives in early MLP, value embeddings may be
providing a cheaper / more direct channel for the same computation.

- arXiv 2504.15471: https://arxiv.org/abs/2504.15471

---

## Related but Distinct

### Large Memory Layers with Product Keys (Lample et al., NeurIPS 2019)

Product-key memory (PKM) replaces FFN layers with a sparse lookup: query → top-k
product-key matches → retrieved value vectors → summed output. This is a large
output-side lookup, but it replaces the FFN, not the V projection in attention. The
retrieval is fully context-dependent (query-gated), so there is no static type-level
component. The framing is "memory capacity" not "type-level prior."

- arXiv 1907.05242: https://arxiv.org/abs/1907.05242

### Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al., EMNLP 2021)

Interprets FFN layers as key-value memories where keys detect input patterns and values
promote output-vocabulary distributions. This is the closest theoretical framing to a
type-level output-side prior, but it is a post-hoc interpretation of FFN weights, not a
factorization design principle. The value vectors are distributed across neurons, not
factored into a lookup table.

- arXiv 2012.14913: https://arxiv.org/abs/2012.14913

### SwitchHead (Csordás et al., NeurIPS 2024)

Applies MoE to the V and O projections specifically (not Q/K), using 5 hard-routed
experts with only 2 attention heads. Explicitly notes that MoE on V+O outperforms MoE
on Q+K. This is empirical evidence for asymmetric value-side expressivity, directly
relevant to the QKV asymmetry claim. Not framed as "type-level factorization" — framed
as parameter-efficient attention.

- arXiv 2312.07987: https://arxiv.org/abs/2312.07987

### DeepSeek MLA — Multi-Head Latent Attention (DeepSeek-V2, May 2024)

Low-rank factorization of the KV cache: compress K and V through a shared low-rank
bottleneck `C_KV`, then decompress per head. Reduces KV cache by 93.3%. The factored
structure applies symmetrically to K and V for cache efficiency; no asymmetric
treatment of V as carrying type-level information. Framing: compression, not priors.

- arXiv 2405.04434: https://arxiv.org/abs/2405.04434

### Attention Sinks and Value-State Drains

A substantial empirical literature (StreamingLLM, Active-Dormant Attention Heads,
Value-State Gated Attention) documents that delimiter / initial tokens become attention
sinks with near-zero value norms. This is the complementary phenomenon: when the model
has no dedicated type-level output channel, it compensates by suppressing value states
at sink tokens to implement a "no-op." Value-State Gated Attention (VGA, arXiv
2510.09017) mitigates this with a learnable gate on value states. None of this work
connects the phenomenon to the absence of an output-side type-level factorization,
though the connection is suggestive.

- StreamingLLM: https://hanlab.mit.edu/blog/streamingllm
- Active-Dormant paper: https://arxiv.org/html/2410.13835v2
- VGA: https://arxiv.org/html/2510.09017

---

## QKV Asymmetry

### KVSlimmer (arXiv 2603.00907, Mar 2025)

The most directly relevant theoretical work. Establishes via spectral analysis that:
- **Q and K projections have concentrated spectral energy** → feature homogeneity across
  tokens → keys and queries are mergeable / compressible with low loss.
- **V projections have dispersed spectral energy** → feature heterogeneity → values
  resist merging and must be treated separately.

This is the strongest theoretical support for the factorization asymmetry claim:
Q/K are naturally low-rank / shared across token types; V is not. KVSlimmer's
conclusion is to compress K more aggressively than V. Our interpretation of the same
finding is that V benefits from a dedicated type-level channel precisely because its
heterogeneity is load-bearing.

- arXiv 2603.00907: https://arxiv.org/abs/2603.00907

### AsymKV (Cui and Xu, 2025)

Empirically established K homogeneity vs V heterogeneity in the KV cache before
KVSlimmer's theoretical treatment. Same finding, empirical rather than spectral.
Referenced by KVSlimmer.

### SwitchHead asymmetry finding

Csordás et al. find empirically that MoE on V+O outperforms MoE on Q+K. This is
consistent with V carrying more per-token specificity (and therefore benefiting more
from additional capacity), but the paper does not theorise about why.

### Key-Value Transformer (Borji, arXiv 2305.19129, 2023)

Tests removing Q entirely (KV-only transformer with symmetric attention maps). Finds
it competitive in some settings. Consistent with Q being more redundant than V, but
framed as parameter reduction, not type-level analysis.

- arXiv 2305.19129: https://arxiv.org/abs/2305.19129

### No direct theoretical treatment of the type-level/relational split

No paper we found directly argues: "Q/K compute relational scores so their type-level
component is small; V retrieves content so its type-level component is large; therefore
a static V lookup is more valuable than a static Q or K lookup." The asymmetry is
established empirically and spectrally, but the conceptual framing (type-level vs
relational) appears absent from the literature.

---

## Implications

1. **The factorization taxonomy is genuinely novel framing.** The engineering insight
   (value embeddings work) is known in the modded-nanogpt community; the framing as a
   member of a factorization taxonomy, distinct from input-side and weight-side
   factorizations, does not appear anywhere in the literature.

2. **KVSlimmer / AsymKV provide the theoretical grounding.** The spectral argument
   (dispersed V energy → heterogeneity → V resists compression) can be read in reverse
   as: V benefits disproportionately from a dedicated type-level channel. This is a
   citable justification for why output-side factorization works.

3. **The interpretability gap is the real open question.** There are zero papers
   analysing what a value embedding table encodes. The bigram subnetworks paper (2504.15471)
   hints that token-identity-to-output mappings live in early MLP layers; value embeddings
   may be a dedicated, cheaper path for the same computation. Testing this is tractable.

4. **Attention sink literature is a natural foil.** The no-op mechanism that emerges in
   standard transformers (suppressed value norms at sink tokens) is arguably the model
   compensating for the absence of a static output channel. VGA and related work try to
   fix the symptom; a value embedding table may address the cause.

5. **SwitchHead is the closest structural cousin.** If citing related architectural
   work, SwitchHead's V+O MoE is the nearest precedent for "additional capacity on the
   value/output side specifically." It is not a static lookup but shares the intuition
   that V is where additional expressivity pays off.

---

## References

- Jordan, K. et al. **modded-nanogpt** (2024–2025). https://github.com/KellerJordan/modded-nanogpt
- Karpathy, A. **nanochat** (2025). https://github.com/karpathy/nanochat
- snimu. "modded-nanogpt medium world record: adding value embeddings" (Oct 2025). https://snimu.github.io/2025/10/07/modded-nanogpt-value-embeddings.html
- snimu. "Analyzing value-embedding-, UNet-, and x0-lambdas" (Aug 2025). https://snimu.github.io/2025/08/11/modded-nanogpt-lambdas.html
- Zhou et al. **Value Residual Learning For Alleviating Attention Concentration In Transformers** (Oct 2024 / ACL 2025). https://arxiv.org/abs/2410.17897
- Csordás et al. **SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention** (NeurIPS 2024). https://arxiv.org/abs/2312.07987
- DeepSeek-AI. **DeepSeek-V2** (May 2024). https://arxiv.org/abs/2405.04434
- Lample et al. **Large Memory Layers with Product Keys** (NeurIPS 2019). https://arxiv.org/abs/1907.05242
- Geva et al. **Transformer Feed-Forward Layers Are Key-Value Memories** (EMNLP 2021). https://arxiv.org/abs/2012.14913
- Borji, A. **Key-Value Transformer** (2023). https://arxiv.org/abs/2305.19129
- Anonymous. **KVSlimmer: Theoretical Insights and Practical Optimizations for Asymmetric KV Merging** (Mar 2025). https://arxiv.org/abs/2603.00907
- Lieberum et al. **Bigram Subnetworks: Mapping to Next Tokens in Transformer Language Models** (Apr 2025). https://arxiv.org/abs/2504.15471
- Han et al. **Active-Dormant Attention Heads** (Oct 2024). https://arxiv.org/html/2410.13835v2
- Anonymous. **Value-State Gated Attention** (Oct 2025). https://arxiv.org/html/2510.09017
