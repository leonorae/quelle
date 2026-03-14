# Decisions — Behavioral Projections

## D1: Phase ordering (2026-03-14)

Steps 1-3 use existing infrastructure and share cached activations.
Steps 4-5 are parallel and independently publishable.
Step 6 (hybrid architecture) is gated on 1-5 producing clean results.
Do not implement step 6 speculatively.

## D2: frame_ratio validation requirements (2026-03-14)

Before treating frame_ratio peaks as "regime boundaries," must verify:
1. Stability across random seeds (same model, different prompts)
2. Stability across prompt types (code, prose, math)
3. Consistency across model sizes (Pythia family)

If peaks are prompt-dependent, they're data-dependent features, not
architectural regime boundaries. This changes interpretation significantly.

## D3: LNN spectral claim (2026-03-14)

The spectral-to-topology mapping is exact for linear systems.
For LNNs (nonlinear), this requires linearization — validate on small
LNNs (step 5) before designing architecture around it.
If linearization doesn't preserve needed properties, the hybrid
architecture needs a different design principle.

## D4: Model choices (2026-03-14)

Pythia-410m for development (fast, 24 layers, well-studied).
Qwen2.5-7B-Instruct as primary target.
No GPT-2 (less interpretability infrastructure, older tokenizer).
