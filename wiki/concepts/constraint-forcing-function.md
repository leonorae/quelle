# Constraint as Forcing Function

> Cross-experiment methodology note. Relevant to: `VVVVVV`, any experiment
> using a constrained architecture variant.

---

## The core pattern

When a network is given insufficient architecture to solve a problem cleanly,
it finds a hack. The hack is the data. Training on a constrained model is not
just an approximation of the real experiment — it is a separate experiment whose
output is the discovered workaround.

Two kinds of research value flow from this:

1. **The hack generalises.** The constrained network found a capability that
   was latent in the architecture class. Studying the hack gives you a mechanism
   you can transplant, extend, or use as an existence proof for the larger
   architecture.

2. **The hack fails or is trivial.** The network couldn't find a solution, or
   found only a degenerate one (near-identity, magnitude threshold). This is
   empirical evidence that the architecture class genuinely lacks a needed
   inductive bias. The failure motivates a specific architectural addition, with
   a clear account of what the addition provides that the constrained version
   could not learn.

Both outcomes are informative. Neither is a failed experiment.

---

## How to run this deliberately

1. **Design the constraint.** The constraint should be tight enough to create
   pressure but not so tight that the network has no degrees of freedom. A fixed
   small read-window (e.g. a gate reading only `[:12]` of the residual) is a
   good constraint: it forces the network to route useful signal into specific
   channels, but doesn't prevent it from doing so.

2. **Run diagnostics to catch the hack in the act.** Before interpreting results,
   measure what the network actually put in the constrained interface. Don't
   assume it learned what you hoped. Channel magnitude probes, cosine similarity
   across contexts, ablation deltas — these are the tools for reading the
   discovered solution.

3. **Classify the hack.** Is it:
   - A structured solution (network routed meaningful signal into the constraint)?
   - A degenerate solution (network bypassed the constraint by making it near-identity)?
   - A failure (constraint actively prevented learning something useful)?

4. **Ask whether the hack generalises.** Does the mechanism depend on
   architecture-specific conditions (small scale, relu², particular training
   distribution)? Or is it a general principle that would appear in other settings?

---

## Historical examples

**Induction heads** (Olsson et al. 2022): The constraint of next-token prediction
with attention only forced the development of a two-layer copying circuit as the
cheapest solution to in-context pattern completion. Nobody designed induction
heads. They were found by studying what trained transformers actually do under
pressure. The mechanism transferred: induction heads appear across model families
and scales, and became a key lens for understanding in-context learning.

**Attention sinks → register tokens** (Xiao et al. 2023, Darcet et al. 2023):
The network discovered a hack — use structural tokens (BOS, EOS) as attention
dump targets for heads with nothing to attend to. This stabilised training but
destroyed semantic content in those tokens. Understanding the hack led to register
tokens: dedicated sink tokens that preserve the hack's stability benefit while
freeing semantic tokens to carry real content. The architectural primitive
generalised beyond the original setting.

**KV cache growth → grouped query attention**: Memory pressure during inference
forced study of what structure exists in K and V matrices. Found that K/V heads
are more redundant than Q heads. Hack (sharing K/V across query groups) became
GQA, now standard.

---

## Relationship to mechanistic interpretability

Mech interp typically runs *backward*: observe trained behaviour, decompose into
circuits, understand the discovered solution post-hoc. The constraint-forcing
approach runs *forward*: design the pressure, observe what gets discovered, then
decompose that specific discovery.

The two approaches are complementary. Mech interp gives richer circuit-level
understanding; constraint-forcing gives more control over what kinds of
discoveries are likely to appear. Using both — run the constrained experiment,
then apply circuit analysis to whatever the network found — gives the clearest
mechanistic picture.

---

## Application to VVVVVV

The ve gate's fixed `[:12]` read window is a constraint. The network has to work
with whatever it puts in those 12 channels. Phase 0 diagnostics (Q0.1–Q0.3) are
not only checking VVVVVV's hypotheses — they are reading off the discovered
solution. Whatever the gate is actually reading is the network's hack, and that
hack is worth understanding regardless of whether it matches the multi-timescale
hypothesis.

Possible discovered solutions and their implications:
- **Gate reads spike channels**: hack is structural-anchor detection; implies
  ve is doing value-side attention-sink routing; generalises if spike channels
  exist in other architectures (less likely with SwiGLU).
- **Gate reads low-magnitude channels**: hack failed; gate is near-identity;
  implies the constraint (fixed small window, no learned projection) prevents
  useful conditioning; motivates learned projection gate.
- **Gate reads some other structured signal**: most interesting case; the network
  found a third option; warrants full circuit-level analysis.

---

## Related future directions

**Soft register for inference-context**: Extend the same pattern to *across*
forward-pass state. Give the model a fixed-size persistent buffer with a learned
read/write interface; observe what it discovers to store there. The constraint
(fixed buffer size, specific interface) forces discovery of what is worth
persisting. The hack reveals what the model treats as durable context. This is
a separate experiment with different infrastructure requirements (stateful
inference loop) but follows the same methodology.
