"""Reduce outputs/ to RESULTS.md.

Reads aggregate analysis outputs and writes a structured summary
suitable for pasting into chat (no raw data access assumed).

Usage:
    python -m src.summarize
"""

from __future__ import annotations

import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
RESULTS_PATH = EXPERIMENT_DIR / "RESULTS.md"


def generate_results_md() -> str:
    """Generate RESULTS.md content from analysis outputs.

    Structure (per CLAUDE.md protocol):
    - Key metrics (table format)
    - Unexpected observations (tagged [observed])
    - What changed from prior expectations
    - Open questions raised by the data
    """
    # TODO: Implement once Phase 2 outputs exist
    # Load outputs/analysis/*.json
    # Format into markdown tables and text
    raise NotImplementedError


if __name__ == "__main__":
    content = generate_results_md()
    RESULTS_PATH.write_text(content)
    print(f"Wrote {RESULTS_PATH}")
