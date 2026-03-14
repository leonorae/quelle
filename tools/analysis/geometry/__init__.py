"""Geometric diagnostic tools for analyzing transformer hidden states.

Shared across experiments. Operates on tensors — no model dependency.
"""

from .concentration import (
    concentration,
    concentration_per_layer,
    representation_velocity,
    effective_dimensionality,
    geometric_summary,
)
