# model.py — VariableRateReasoner

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometric utilities
# ---------------------------------------------------------------------------

def concentration(h: torch.Tensor) -> torch.Tensor:
    """Per-example mean pairwise cosine similarity across token positions.

    Args:
        h: (batch, seq_len, d_model)
    Returns:
        (batch,) — values in [-1, 1]; higher means representations are more
        geometrically aligned (concentrated).
    """
    B, S, _ = h.shape
    if S <= 1:
        return torch.zeros(B, device=h.device)

    h_norm = F.normalize(h, dim=-1)
    sim = torch.bmm(h_norm, h_norm.transpose(1, 2))  # (B, S, S)

    # Off-diagonal mask: True where i != j
    mask = ~torch.eye(S, dtype=torch.bool, device=h.device)  # (S, S)
    # Flatten spatial dims to select off-diagonal elements per batch element
    sim_off = sim.reshape(B, S * S)[:, mask.reshape(S * S)]  # (B, S*(S-1))
    return sim_off.mean(dim=1)  # (B,)


def compress(h: torch.Tensor, lambda_t: torch.Tensor, temperature: float = 1.0):
    """Differentiable soft dimension masking.

    A sigmoid gate over dimension indices keeps the first ~(1-λ)·d dimensions
    at full weight and fades the rest toward zero.  This is differentiable with
    respect to lambda_t, so gradients flow back through the compression policy.

    Args:
        h:          (batch, seq_len, d_model)
        lambda_t:   (batch,) compression rate in [0, 1]
        temperature: sharpness of the sigmoid cutoff (lower = sharper)
    Returns:
        z: (batch, seq_len, d_model) — soft-masked representation
        k: mean effective kept dimensions across the batch (for logging)
    """
    d = h.shape[-1]
    # dim_weights[i] = i / (d-1), ranging from 0 to 1
    dim_weights = torch.linspace(0, 1, d, device=h.device)  # (d,)

    # keep_prob[b, dim] = sigmoid((dim_weights[dim] - lambda_t[b]) / T)
    # When lambda_t[b]=0.5, dims in the lower half are kept, upper half faded.
    keep_prob = torch.sigmoid(
        (dim_weights.unsqueeze(0) - lambda_t.unsqueeze(-1)) / temperature
    )  # (B, d)

    z = h * keep_prob.unsqueeze(1)  # (B, S, d)
    k = int(d * (1 - lambda_t.mean().item()))
    return z, k


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class CompressionHead(nn.Module):
    """Computes per-example compression rate λ from angle concentration."""

    def __init__(self, alpha_init: float = 2.0, beta_init: float = 0.5):
        super().__init__()
        # α controls slope; β is the concentration threshold for λ=0.5
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, h: torch.Tensor):
        conc = concentration(h)                              # (B,)
        lambda_t = torch.sigmoid(self.alpha * (conc - self.beta))  # (B,)
        return lambda_t, conc


class PredictorHead(nn.Module):
    """Predicts h_{t+1} from the compressed representation z_t (stop-gradient target)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class VariableRateReasoner(nn.Module):
    """Causal transformer with per-layer adaptive compression.

    At each layer:
      1.  Apply transformer layer to current representation x → h
      2.  Compute angle concentration of h → λ (per example)
      3.  Soft-mask h with λ → z  (the compressed representation)
      4.  Predict next-layer h from z  (auxiliary future-prediction loss)
      5.  Pass z as input to the next layer
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        alpha_init: float = 2.0,
        beta_init: float = 0.5,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Individual layers so we can capture intermediate hidden states.
        # pre-norm (norm_first=True) matches the spec's "standard pre-norm architecture".
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

        self.compression_heads = nn.ModuleList(
            [CompressionHead(alpha_init, beta_init) for _ in range(n_layers)]
        )
        self.predictor_heads = nn.ModuleList(
            [PredictorHead(d_model) for _ in range(n_layers)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

    def forward(self, input_ids: torch.Tensor, return_stats: bool = False):
        """
        Args:
            input_ids:    (batch, seq_len)
            return_stats: if True, return per-layer compression statistics
        Returns:
            logits: (batch, seq_len, vocab_size)
            stats:  list[dict] — one entry per layer (only when return_stats=True)
        """
        B, S = input_ids.shape
        device = input_ids.device

        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        # Causal mask: -inf above the diagonal so position i only attends to ≤ i.
        causal_mask = torch.triu(
            torch.full((S, S), float("-inf"), device=device), diagonal=1
        )
        # Padding mask (float, same type as causal_mask): -inf for PAD positions.
        pad_mask = torch.zeros(B, S, device=device)
        pad_mask[input_ids == 0] = float("-inf")

        hidden_states: list[torch.Tensor] = []
        preds: list[torch.Tensor] = []
        lambdas: list[torch.Tensor] = []
        concentrations: list[torch.Tensor] = []

        for layer, comp_head, pred_head in zip(
            self.layers, self.compression_heads, self.predictor_heads
        ):
            h = layer(x, src_mask=causal_mask, src_key_padding_mask=pad_mask)

            lambda_t, conc = comp_head(h)        # (B,), (B,)
            z, _ = compress(h, lambda_t, temperature=self.temperature)
            h_pred = pred_head(z)

            hidden_states.append(h)
            preds.append(h_pred)
            lambdas.append(lambda_t)
            concentrations.append(conc)

            x = z  # compressed representation feeds the next layer

        logits = self.lm_head(self.norm(x))  # (B, S, vocab_size)

        if not return_stats:
            return logits

        n = len(hidden_states)
        stats = []
        for i in range(n):
            # h_next: the hidden state of the following layer (target for future prediction).
            # For the last layer, reuse its own output — effectively a no-op loss there.
            h_next = hidden_states[i + 1] if i + 1 < n else hidden_states[-1]
            stats.append({
                "h":            hidden_states[i],
                "h_next":       h_next,
                "h_pred":       preds[i],
                "lambda":       lambdas[i].mean().item(),      # scalar for logging
                "concentration": concentrations[i].mean().item(),
                # Tensors needed for loss computation:
                "_lambda_t":    lambdas[i],       # (B,)
                "_conc":        concentrations[i], # (B,)
            })

        return logits, stats
