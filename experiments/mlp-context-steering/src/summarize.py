"""Result summarization for MLP context steering.

TODO: Implement after results exist. Should read metric dicts from
baseline, ablation, and control runs and produce a summary suitable
for RESULTS.md.
"""


def summarize_results(baseline, ablated, controls=None):
    """Summarize experiment results.

    TODO: Implement after Phase 1-3 results are collected.

    Args:
        baseline: Baseline metrics from collect_baseline().
        ablated: Ablated metrics from collect_ablated().
        controls: Optional control metrics from Phase 3.

    Returns:
        String: formatted summary for RESULTS.md.
    """
    raise NotImplementedError("summarize_results not yet implemented")
