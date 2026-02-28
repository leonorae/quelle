# model.py — VariableRateReasoner
#
# Stub — implement according to experiments/variable-bitrate-reasoning/README.md
# and the top-level specification in README.md.

import torch
import torch.nn as nn
import torch.nn.functional as F


def concentration(h: torch.Tensor) -> torch.Tensor:
    """Mean pairwise cosine similarity across token positions.

    Args:
        h: (batch, seq_len, d_model)
    Returns:
        scalar tensor
    """
    h_norm = F.normalize(h, dim=-1)
    sim = torch.bmm(h_norm, h_norm.transpose(1, 2))  # (B, S, S)
    mask = ~torch.eye(sim.shape[1], dtype=torch.bool, device=sim.device)
    return sim[:, mask].mean()


def compress(h: torch.Tensor, lambda_t: torch.Tensor, temperature: float = 1.0):
    """Differentiable soft compression with Gumbel-Softmax approximation.

    Args:
        h:           (batch, seq_len, d_model)
        lambda_t:    scalar — compression rate in [0, 1]
        temperature: Gumbel-Softmax temperature
    Returns:
        z:  compressed tensor (same shape, high dims zeroed/down-weighted)
        k:  integer — effective kept dimensions
    """
    d = h.shape[-1]
    k = max(1, min(d - 1, int(d * (1 - lambda_t.item()))))

    dim_weights = torch.linspace(0, 1, d, device=h.device)
    keep_prob = torch.sigmoid((dim_weights - lambda_t) / temperature)
    z = h * keep_prob.unsqueeze(0).unsqueeze(0)
    return z, k


class CompressionHead(nn.Module):
    """Maps hidden state to learnable compression rate via angle concentration."""

    def __init__(self, alpha_init: float = 2.0, beta_init: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, h: torch.Tensor):
        conc = concentration(h)
        lambda_t = torch.sigmoid(self.alpha * (conc - self.beta))
        return lambda_t, conc


class PredictorHead(nn.Module):
    """Predicts h_{t+1} from compressed z_t."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


class VariableRateReasoner(nn.Module):
    """Transformer with per-layer adaptive compression."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        alpha_init: float = 2.0,
        beta_init: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.compression_heads = nn.ModuleList(
            [CompressionHead(alpha_init, beta_init) for _ in range(n_layers)]
        )
        self.predictor_heads = nn.ModuleList(
            [PredictorHead(d_model) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, return_stats: bool = False):
        """
        Args:
            input_ids:    (batch, seq_len)
            return_stats: if True, also return per-layer compression stats
        Returns:
            logits: (batch, seq_len, vocab_size)
            stats:  list of dicts (only when return_stats=True)
        """
        # TODO: implement layer-by-layer forward with compression
        raise NotImplementedError("Implement the forward pass.")
