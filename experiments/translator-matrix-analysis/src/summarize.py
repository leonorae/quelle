"""Reduce outputs/ to RESULTS.md.

Reads analysis outputs and generates a structured summary suitable for
pasting into chat-based analysis sessions.

Usage:
    python -m src.summarize --output-dir outputs

See CLAUDE.md: RESULTS.md should be written assuming the reader has
NO access to raw data or outputs/.
"""

from __future__ import annotations


def summarize() -> str:
    """Generate RESULTS.md content from analysis outputs."""
    # TODO: Implement after results exist
    # Read outputs/svd/scalar_metrics.json
    # Read outputs/frame_deltas/regime_boundaries.json
    # Read outputs/atlas/clustering.json
    # Format as markdown with tables, epistemic tags, open questions
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
