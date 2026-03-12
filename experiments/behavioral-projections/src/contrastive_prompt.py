"""Phase 3: Contrastive prompt discrimination.

Learn a projection where same-prompt pairs cluster, different-prompt pairs
separate. Uses register-token augmentation (prepend padding tokens) rather
than arbitrary noise.

Objective: InfoNCE / NT-Xent loss.
    Positive pairs: (h_prompt_A_variant_1, h_prompt_A_variant_2)
    Negative pairs: (h_prompt_A, h_prompt_B)
    L = -log(exp(sim(a,p)/τ) / Σ exp(sim(a,n_k)/τ))

Augmentation: Run same prompt with/without prepended register-style tokens.
Content representation should be invariant; administrative representation shifts.

Uses SimCLR pattern: train with 2-layer MLP projector head, discard for analysis.
Monitor for dimensional collapse (track effective rank of projected embeddings).

Requires: Phase 0 cache + perturbation pairs with group_id linkage.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import torch


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE contrastive loss.

    Args:
        anchor: (batch, d_proj)
        positive: (batch, d_proj)
        negatives: (batch, n_neg, d_proj)
        temperature: scaling factor
    Returns:
        scalar loss
    """
    raise NotImplementedError("Phase 3 — implement after Phase 0 cache available")


def train_contrastive_projection(
    cache_dir: Path,
    layer: int,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train contrastive projection with register-token augmentation.

    Uses perturbation pairs (same group_id) as positives.
    Different prompts as negatives.

    Returns:
        projection: (d_proj, d_hidden) tensor
        metrics: {loss_final, effective_rank, ...}
    """
    raise NotImplementedError("Phase 3 — implement after Phase 0 cache available")
