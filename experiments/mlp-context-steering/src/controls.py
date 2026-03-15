"""Phase 3 controls — permutation baselines.

TODO: Collection loops are stubs. make_permutation_hook is implemented
in ablation.py.
"""

from torch import Tensor


def collect_permutation_control(model, prompts, config):
    """Collect metrics with permuted MLP outputs (control condition).

    TODO: Implement.

    Guidance:
    - Same structure as collect_ablated(), but use make_permutation_hook()
      instead of make_strong_ablation_hook().
    - This preserves activation magnitudes and statistics but destroys
      positional information.
    - If MLP ablation effects are due to specific computation (not just
      perturbation magnitude), permutation controls should show DIFFERENT
      metric shifts than zeroing.

    Args:
        model: TransformerLens HookedTransformer.
        prompts: List of token tensors.
        config: Experiment config dict.

    Returns:
        Dict[int, Dict[str, Tensor]]: layer -> metrics mapping.
    """
    raise NotImplementedError("collect_permutation_control not yet implemented")


def compare_ablation_vs_permutation(ablated_metrics: dict, permutation_metrics: dict) -> dict:
    """Compare ablation and permutation metric shifts.

    TODO: Implement.

    If zeroing and permutation produce similar metric shifts, the effect
    is due to generic perturbation, not MLP-specific computation.
    If they differ, the MLP is doing something computationally specific.

    Args:
        ablated_metrics: From collect_ablated().
        permutation_metrics: From collect_permutation_control().

    Returns:
        Dict with comparison statistics per layer.
    """
    raise NotImplementedError
