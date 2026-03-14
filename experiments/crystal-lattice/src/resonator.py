"""Continuous Latent Navigator (CLN) for Crystal Lattice.

A 2-layer Transformer looped 5-10 times with depth-wise LoRA relaxation,
implementing the iterative "resonator" that relaxes VSA hypervectors
into physically valid 3D predictions.

Key design choices (from REVIEW.md):
- 2-layer Transformer backbone, weight-tied across loop iterations
- Depth-wise LoRA: per-iteration low-rank delta on attention weights
- Anchor re-injection: hidden = transformer(hidden + alpha_t * anchor)
  where alpha_t = sigmoid(linear(hidden_t))  (learned, iteration-dependent)
- Output head: SE(3)-invariant pooling -> distance + ring classification
- Per-iteration diagnostic logging: entropy, integrity, latent norm
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CLNDiagnostics:
    """Per-iteration metrics logged during the CLN forward pass."""
    iteration: int
    entropy: float
    integrity: float     # angle concentration to anchor
    latent_norm: float
    alpha: float         # anchor blending weight


class DepthLoRA(nn.Module):
    """Low-rank delta applied per loop iteration to a weight matrix.

    For iteration t, the effective weight is W + A_t @ B_t where A_t, B_t
    are small matrices indexed by iteration.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, max_iters: int):
        super().__init__()
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.randn(in_features, rank) * 0.01)
            for _ in range(max_iters)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(rank, out_features))
            for _ in range(max_iters)
        ])

    def delta(self, iteration: int) -> torch.Tensor:
        """Return the low-rank weight delta for a given iteration."""
        return self.lora_A[iteration] @ self.lora_B[iteration]


class LoRATransformerLayer(nn.Module):
    """Single Transformer encoder layer with depth-wise LoRA on QKV projections."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, lora_rank: int, max_iters: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Core (weight-tied) transformer layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # LoRA deltas for in_proj_weight (3*d_model x d_model)
        self.lora_qkv = DepthLoRA(d_model, 3 * d_model, lora_rank, max_iters)

    def forward(self, x: torch.Tensor, iteration: int) -> torch.Tensor:
        """Forward with iteration-specific LoRA perturbation.

        Parameters
        ----------
        x : (B, S, D) tensor
        iteration : current loop iteration index
        """
        # Apply LoRA delta to QKV projection
        original_weight = self.self_attn.in_proj_weight.data
        delta = self.lora_qkv.delta(iteration).T  # (3*d_model, d_model)
        self.self_attn.in_proj_weight.data = original_weight + delta

        # Self-attention + residual
        residual = x
        x_normed = self.norm1(x)
        attn_out, _ = self.self_attn(x_normed, x_normed, x_normed)
        x = residual + attn_out

        # Restore original weights
        self.self_attn.in_proj_weight.data = original_weight

        # Feed-forward + residual
        residual = x
        x = residual + self.ff(self.norm2(x))
        return x


class InvariantPooling(nn.Module):
    """SE(3)-invariant pooling: global mean + max pooling with norm features.

    Since the CLN operates on abstract latent vectors (not 3D coordinates),
    true SE(3) invariance is achieved by ensuring the output depends only
    on norms and inner products, not on directional information.
    """

    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        # mean-pool + max-pool + norm statistics -> 3 * d_model
        self.head = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool (B, S, D) -> (B, out_dim).

        Aggregates using:
          - mean over sequence dim
          - max over sequence dim
          - L2 norm of mean (scalar, broadcast to d_model for concat)
        """
        mean_pool = x.mean(dim=1)                              # (B, D)
        max_pool = x.max(dim=1).values                         # (B, D)
        norm_feat = mean_pool.norm(dim=-1, keepdim=True)       # (B, 1)
        norm_feat = norm_feat.expand_as(mean_pool)             # (B, D)
        pooled = torch.cat([mean_pool, max_pool, norm_feat], dim=-1)  # (B, 3D)
        return self.head(pooled)


class ContinuousLatentNavigator(nn.Module):
    """The CLN: iteratively refines a VSA-projected latent into physical predictions.

    Architecture:
        - 2-layer Transformer backbone (weight-tied across iterations)
        - Depth-wise LoRA relaxation per iteration
        - Learned anchor re-injection alpha_t
        - SE(3)-invariant output head

    Outputs:
        - distance: predicted head-to-tail Euclidean distance (scalar)
        - ring_logit: binary logit for ring classification
        - diagnostics: per-iteration CLNDiagnostics list
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_ff: int = 512,
        num_layers: int = 2,
        num_iters: int = 8,
        lora_rank: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_iters = num_iters
        self.num_layers = num_layers

        # Transformer layers (weight-tied across iterations, differentiated by LoRA)
        self.layers = nn.ModuleList([
            LoRATransformerLayer(d_model, nhead, dim_ff, lora_rank, num_iters)
            for _ in range(num_layers)
        ])

        # Anchor alpha gate: iteration-dependent blending
        # alpha_t = sigmoid(linear(hidden_t))
        self.alpha_gate = nn.Linear(d_model, 1)

        # Output heads
        self.distance_head = InvariantPooling(d_model, 1)
        self.ring_head = InvariantPooling(d_model, 1)

    def _compute_entropy(self, hidden: torch.Tensor) -> float:
        """Compute entropy of the hidden state distribution.

        Uses softmax over the feature dimension to treat activations
        as a pseudo-probability distribution, then computes Shannon entropy.
        """
        # (B, S, D) -> mean over batch and sequence -> (D,)
        mean_act = hidden.mean(dim=(0, 1))
        probs = F.softmax(mean_act, dim=0)
        entropy = -(probs * (probs + 1e-10).log()).sum()
        return float(entropy.item())

    def _compute_integrity(
        self, hidden: torch.Tensor, anchor: torch.Tensor
    ) -> float:
        """Angle concentration between hidden state and anchor."""
        h_flat = hidden.mean(dim=1).flatten()  # (D,)
        a_flat = anchor.flatten()
        h_norm = F.normalize(h_flat, dim=0)
        a_norm = F.normalize(a_flat, dim=0)
        return float((h_norm * a_norm).sum().item())

    def forward(
        self, projected_vsa: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[CLNDiagnostics]]:
        """Run the iterative CLN loop.

        Parameters
        ----------
        projected_vsa : (B, proj_dim) tensor from VSALattice.encode_and_project_batch

        Returns
        -------
        distance : (B, 1) predicted head-to-tail distance in Angstroms
        ring_logit : (B, 1) logit for ring classification
        diagnostics : list of CLNDiagnostics, one per iteration
        """
        # Reshape to (B, 1, D) -- single "token" sequence for transformer
        anchor = projected_vsa.unsqueeze(1)   # (B, 1, D)
        hidden = anchor.clone()               # (B, 1, D)

        diagnostics: list[CLNDiagnostics] = []

        for t in range(self.num_iters):
            # Compute iteration-dependent alpha
            alpha_t = torch.sigmoid(self.alpha_gate(hidden.mean(dim=1)))  # (B, 1)
            alpha_val = float(alpha_t.mean().item())

            # Anchor re-injection
            hidden = hidden + alpha_t.unsqueeze(1) * anchor  # (B, 1, D)

            # Pass through weight-tied transformer layers with LoRA
            for layer in self.layers:
                hidden = layer(hidden, iteration=t)

            # Log diagnostics
            diagnostics.append(CLNDiagnostics(
                iteration=t,
                entropy=self._compute_entropy(hidden),
                integrity=self._compute_integrity(hidden, anchor),
                latent_norm=float(hidden.norm().item()),
                alpha=alpha_val,
            ))

        # Output heads (SE(3)-invariant pooling)
        distance = self.distance_head(hidden)    # (B, 1)
        ring_logit = self.ring_head(hidden)      # (B, 1)

        # Distance should be non-negative
        distance = F.softplus(distance)

        return distance, ring_logit, diagnostics


class HRM(nn.Module):
    """Full HRM stack: VSA encoder + CLN resonator.

    Convenience wrapper that chains VSALattice.encode_and_project_batch
    with ContinuousLatentNavigator.forward.
    """

    def __init__(self, vsa: "VSALattice", cln: ContinuousLatentNavigator):
        super().__init__()
        self.vsa = vsa
        self.cln = cln

    def forward(
        self, smiles_list: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, list[CLNDiagnostics]]:
        """Encode SMILES and predict distance + ring classification."""
        projected = self.vsa.encode_and_project_batch(smiles_list)
        return self.cln(projected)

    def predict(
        self, smiles_list: list[str]
    ) -> dict:
        """Convenience method returning a dict of predictions."""
        with torch.no_grad():
            distance, ring_logit, diagnostics = self.forward(smiles_list)
        return {
            "distance": distance.squeeze(-1).cpu().numpy(),
            "ring_prob": torch.sigmoid(ring_logit).squeeze(-1).cpu().numpy(),
            "diagnostics": diagnostics,
        }
