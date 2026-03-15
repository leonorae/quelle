"""Tests for metrics.py — pure tensor functions, no model dependency.

Covers:
- erank: identity matrix, rank-1 matrix, 3D input, numerical edge cases
- sink_intensity: uniform attention, full-sink attention, 3D input
- max_activation: basic cases, negative values, multi-dimensional
- collect_layer_metrics: integration of all three metrics
"""

import sys
from pathlib import Path

import pytest
import torch

# Add experiment src to path (directory name has hyphens, not a valid Python package)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import (
    collect_layer_metrics,
    erank,
    max_activation,
    sink_intensity,
)


# ---------------------------------------------------------------------------
# erank tests
# ---------------------------------------------------------------------------


class TestErank:
    """Tests for the entropy-based effective rank."""

    def test_identity_matrix(self):
        """Identity matrix should have erank equal to its dimension.

        All singular values are 1, so the distribution is uniform,
        and exp(H) = n.
        """
        n = 8
        X = torch.eye(n)
        result = erank(X)
        assert torch.isclose(result, torch.tensor(float(n)), atol=1e-4), (
            f"Identity({n}) should have erank={n}, got {result.item():.4f}"
        )

    def test_rank_one_matrix(self):
        """Rank-1 matrix should have erank close to 1.

        A single nonzero singular value gives H=0, so exp(H)=1.
        """
        # Outer product: rank 1
        v = torch.randn(10, 1)
        X = v @ v.T  # shape (10, 10), rank 1
        result = erank(X)
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-4), (
            f"Rank-1 matrix should have erank≈1, got {result.item():.4f}"
        )

    def test_rank_two_equal_singular_values(self):
        """Matrix with exactly 2 equal nonzero singular values should have erank≈2."""
        # Construct a matrix with exactly two equal singular values
        U = torch.eye(10, 2)  # (10, 2) orthonormal columns
        S = torch.tensor([5.0, 5.0])  # equal singular values
        V = torch.eye(8, 2)   # (8, 2) orthonormal columns
        X = U @ torch.diag(S) @ V.T  # (10, 8), rank 2, equal singular values
        result = erank(X)
        assert torch.isclose(result, torch.tensor(2.0), atol=1e-4), (
            f"Rank-2 (equal svs) should have erank≈2, got {result.item():.4f}"
        )

    def test_3d_input_flattens(self):
        """3D input (batch, seq_len, d_model) should be flattened to 2D."""
        B, T, D = 2, 5, 8
        X = torch.eye(D).unsqueeze(0).expand(B, -1, -1)  # (2, 8, 8)
        result = erank(X)
        # After flattening: (16, 8) — the identity block repeated
        # Singular values still uniform across 8 dimensions
        assert result.item() > 1.0, "3D erank should be > 1"
        assert torch.isfinite(result), "erank should be finite"

    def test_erank_bounded_by_min_dimension(self):
        """erank should never exceed min(rows, cols)."""
        X = torch.randn(20, 10)
        result = erank(X)
        assert result.item() <= 10.0 + 1e-4, (
            f"erank should be ≤ min(20,10)=10, got {result.item():.4f}"
        )

    def test_erank_positive(self):
        """erank should always be positive."""
        X = torch.randn(5, 3)
        result = erank(X)
        assert result.item() > 0, f"erank should be > 0, got {result.item()}"

    def test_zero_matrix(self):
        """All-zero matrix should have erank=1 (edge case, all svs zero)."""
        X = torch.zeros(5, 5)
        result = erank(X)
        # With all singular values zero (filtered out), we return 1.0
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-4)

    def test_wrong_dimensions_raises(self):
        """1D and 4D+ tensors should raise ValueError."""
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            erank(torch.randn(10))
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            erank(torch.randn(2, 3, 4, 5))

    def test_erank_increases_with_rank(self):
        """Higher-rank matrices should have higher erank (monotonicity)."""
        eranks = []
        for r in [1, 3, 6, 10]:
            # Build matrix with exactly r equal singular values
            U = torch.eye(20, r)
            S = torch.ones(r)
            V = torch.eye(15, r)
            X = U @ torch.diag(S) @ V.T
            eranks.append(erank(X).item())

        for i in range(len(eranks) - 1):
            assert eranks[i] < eranks[i + 1], (
                f"erank should increase with rank: {eranks}"
            )

    def test_erank_dtype_float64(self):
        """erank should work with float64 tensors."""
        X = torch.eye(5, dtype=torch.float64)
        result = erank(X)
        assert torch.isclose(result, torch.tensor(5.0, dtype=torch.float64), atol=1e-6)


# ---------------------------------------------------------------------------
# sink_intensity tests
# ---------------------------------------------------------------------------


class TestSinkIntensity:
    """Tests for attention sink intensity measurement."""

    def test_uniform_attention(self):
        """Uniform attention should give sink_intensity = 1/seq_len."""
        T = 10
        H = 4
        # Uniform: each query attends equally to all keys
        attn = torch.ones(1, H, T, T) / T
        result = sink_intensity(attn, sink_token_index=0)
        expected = 1.0 / T
        assert torch.isclose(result, torch.tensor(expected), atol=1e-5), (
            f"Uniform attention: expected sink_intensity={expected}, got {result.item()}"
        )

    def test_full_sink_attention(self):
        """All attention on sink token should give sink_intensity = 1."""
        T = 8
        H = 2
        attn = torch.zeros(1, H, T, T)
        attn[:, :, :, 0] = 1.0  # All mass on token 0
        result = sink_intensity(attn, sink_token_index=0)
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-5)

    def test_no_sink_attention(self):
        """No attention on sink token should give sink_intensity = 0."""
        T = 8
        H = 2
        attn = torch.zeros(1, H, T, T)
        # Spread attention equally over tokens 1..T-1 (none on token 0)
        attn[:, :, :, 1:] = 1.0 / (T - 1)
        result = sink_intensity(attn, sink_token_index=0)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)

    def test_3d_input(self):
        """3D attention tensor (H, T, T) should work by adding batch dim."""
        T = 6
        H = 3
        attn = torch.ones(H, T, T) / T
        result = sink_intensity(attn, sink_token_index=0)
        expected = 1.0 / T
        assert torch.isclose(result, torch.tensor(expected), atol=1e-5)

    def test_nonzero_sink_index(self):
        """Sink on a non-zero token index should work."""
        T = 5
        H = 2
        sink_idx = 3
        attn = torch.zeros(1, H, T, T)
        attn[:, :, :, sink_idx] = 1.0
        result = sink_intensity(attn, sink_token_index=sink_idx)
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-5)

    def test_batched_attention(self):
        """Multiple batch elements should be averaged."""
        T = 4
        H = 1
        # Batch element 0: all on sink
        # Batch element 1: uniform
        attn = torch.zeros(2, H, T, T)
        attn[0, :, :, 0] = 1.0
        attn[1, :, :, :] = 1.0 / T
        result = sink_intensity(attn, sink_token_index=0)
        # Average: (1.0 + 0.25) / 2 = 0.625
        expected = (1.0 + 1.0 / T) / 2.0
        assert torch.isclose(result, torch.tensor(expected), atol=1e-5), (
            f"Batched: expected {expected}, got {result.item()}"
        )

    def test_wrong_dimensions_raises(self):
        """2D and 5D+ tensors should raise ValueError."""
        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            sink_intensity(torch.randn(5, 5))
        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            sink_intensity(torch.randn(1, 2, 3, 4, 5))

    def test_output_range(self):
        """Sink intensity should be in [0, 1] for valid attention patterns."""
        T = 10
        H = 4
        B = 3
        # Random valid attention (softmax-like: positive, sums to 1)
        raw = torch.rand(B, H, T, T)
        attn = raw / raw.sum(dim=-1, keepdim=True)
        result = sink_intensity(attn, sink_token_index=0)
        assert 0.0 <= result.item() <= 1.0, (
            f"Sink intensity should be in [0,1], got {result.item()}"
        )


# ---------------------------------------------------------------------------
# max_activation tests
# ---------------------------------------------------------------------------


class TestMaxActivation:
    """Tests for maximum absolute activation."""

    def test_simple_case(self):
        """Basic positive tensor."""
        X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = max_activation(X)
        assert torch.isclose(result, torch.tensor(6.0))

    def test_negative_dominates(self):
        """Absolute value means negative values can dominate."""
        X = torch.tensor([[1.0, -10.0], [3.0, 2.0]])
        result = max_activation(X)
        assert torch.isclose(result, torch.tensor(10.0))

    def test_3d_tensor(self):
        """Works on 3D activation tensors."""
        X = torch.zeros(2, 5, 8)
        X[1, 3, 7] = 42.0
        result = max_activation(X)
        assert torch.isclose(result, torch.tensor(42.0))

    def test_scalar_output(self):
        """Result should be a scalar tensor."""
        X = torch.randn(3, 4, 5)
        result = max_activation(X)
        assert result.dim() == 0, f"Expected scalar, got {result.dim()}D"

    def test_all_zeros(self):
        """All-zero tensor should have max_activation = 0."""
        X = torch.zeros(5, 5)
        result = max_activation(X)
        assert torch.isclose(result, torch.tensor(0.0))

    def test_single_element(self):
        """Single-element tensor."""
        X = torch.tensor([[-7.5]])
        result = max_activation(X)
        assert torch.isclose(result, torch.tensor(7.5))


# ---------------------------------------------------------------------------
# collect_layer_metrics integration tests
# ---------------------------------------------------------------------------


class TestCollectLayerMetrics:
    """Integration tests for the convenience bundling function."""

    def test_returns_all_keys(self):
        """Output dict should contain exactly the three expected keys."""
        B, T, D, H = 2, 8, 16, 4
        resid = torch.randn(B, T, D)
        attn = torch.rand(B, H, T, T)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        result = collect_layer_metrics(resid, attn)
        assert set(result.keys()) == {"erank", "sink_intensity", "max_activation"}

    def test_values_are_scalar_tensors(self):
        """All values should be scalar (0-dim) tensors."""
        B, T, D, H = 2, 8, 16, 4
        resid = torch.randn(B, T, D)
        attn = torch.rand(B, H, T, T)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        result = collect_layer_metrics(resid, attn)
        for key, val in result.items():
            assert isinstance(val, torch.Tensor), f"{key} should be a Tensor"
            assert val.dim() == 0, f"{key} should be scalar, got {val.dim()}D"

    def test_values_are_finite(self):
        """All metric values should be finite (no nan/inf)."""
        B, T, D, H = 3, 10, 32, 8
        resid = torch.randn(B, T, D)
        attn = torch.rand(B, H, T, T)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        result = collect_layer_metrics(resid, attn)
        for key, val in result.items():
            assert torch.isfinite(val), f"{key} is not finite: {val.item()}"

    def test_custom_sink_token(self):
        """Passing a different sink_token_index should affect sink_intensity."""
        B, T, D, H = 1, 6, 8, 2
        resid = torch.randn(B, T, D)
        # All attention on token 3
        attn = torch.zeros(B, H, T, T)
        attn[:, :, :, 3] = 1.0

        result_default = collect_layer_metrics(resid, attn, sink_token_index=0)
        result_custom = collect_layer_metrics(resid, attn, sink_token_index=3)

        assert torch.isclose(result_default["sink_intensity"], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(result_custom["sink_intensity"], torch.tensor(1.0), atol=1e-5)

    def test_erank_matches_standalone(self):
        """collect_layer_metrics erank should match standalone erank()."""
        B, T, D, H = 2, 8, 16, 4
        resid = torch.randn(B, T, D)
        attn = torch.rand(B, H, T, T)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        bundled = collect_layer_metrics(resid, attn)
        standalone = erank(resid)
        assert torch.isclose(bundled["erank"], standalone, atol=1e-6)
