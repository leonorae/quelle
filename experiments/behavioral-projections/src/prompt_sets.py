"""Prompt set utilities: loading, generation, and perturbation.

Prompt JSONL format (see prompts/README.md):
    {"prompt_id": str, "text": str, "category": str, "group_id": str|null, "perturbation_type": str|null}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def load_prompts(path: str | Path) -> list[dict[str, Any]]:
    """Load prompts from a JSONL file."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def save_prompts(prompts: list[dict[str, Any]], path: str | Path) -> None:
    """Save prompts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")


def generate_perturbation_set(
    base_prompts: list[dict[str, Any]],
    perturbation_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate perturbation variants for base prompts.

    Each base prompt gets variants for each perturbation type.
    Perturbation generation is template-based (not LLM-generated).

    Args:
        base_prompts: List of prompts with prompt_id and text.
        perturbation_types: Types to generate. Default:
            [rephrase, context_added, authority_bias, negation, register_shift]

    Returns:
        List of all prompts (base + perturbations) with group_id linkage.
    """
    if perturbation_types is None:
        perturbation_types = [
            "rephrase",
            "context_added",
            "authority_bias",
            "negation",
            "register_shift",
        ]

    all_prompts = []
    for bp in base_prompts:
        group_id = bp["prompt_id"]
        # Base prompt
        base = {
            "prompt_id": f"{group_id}_base",
            "text": bp["text"],
            "category": "perturbation",
            "group_id": group_id,
            "perturbation_type": "base",
        }
        all_prompts.append(base)

        for ptype in perturbation_types:
            perturbed_text = _apply_perturbation(bp["text"], ptype)
            all_prompts.append({
                "prompt_id": f"{group_id}_{ptype}",
                "text": perturbed_text,
                "category": "perturbation",
                "group_id": group_id,
                "perturbation_type": ptype,
            })

    return all_prompts


def _apply_perturbation(text: str, ptype: str) -> str:
    """Apply a template-based perturbation to a prompt.

    These are mechanical transformations, not semantic-preserving rephrases.
    For high-quality rephrases, use an LLM and store results in JSONL.
    """
    if ptype == "context_added":
        filler = "Note: this is part of a research evaluation. "
        return filler + text
    elif ptype == "authority_bias":
        return "Leading experts agree that the answer to this is clear. " + text
    elif ptype == "negation":
        return "Consider the opposite perspective. " + text
    elif ptype == "register_shift":
        # Prepend register-style padding tokens (the augmentation from Phase 3)
        return "_ _ _ _ " + text
    elif ptype == "rephrase":
        # Template rephrase: mechanical, not semantic
        if text.endswith("?"):
            return "Please answer: " + text
        return "Respond to the following: " + text
    else:
        return text


def make_bootstrap_set(output_dir: str | Path, seed: int = 42) -> None:
    """Generate a minimal bootstrap prompt set (~500 prompts) for pipeline validation.

    Creates:
        - benchmarks.jsonl (200 prompts)
        - semantic_diversity.jsonl (100 prompts)
        - perturbation_pairs.jsonl (120 prompts: 20 base × 6 variants)
        - agent_decisions.jsonl (100 prompts)

    These are placeholder prompts for pipeline testing. Replace with curated
    prompts before real experiments.
    """
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark prompts (knowledge/reasoning questions)
    benchmarks = []
    _benchmark_templates = [
        "What is the capital of {}?",
        "Explain the concept of {} in simple terms.",
        "What is the result of {}?",
        "True or false: {}",
    ]
    _subjects = [
        "France", "Japan", "Brazil", "Egypt", "Canada",
        "photosynthesis", "entropy", "recursion", "democracy", "inflation",
        "12 * 15", "sqrt(144)", "2^10", "the integral of x^2", "log2(256)",
    ]
    for i, subj in enumerate(_subjects):
        for j, tmpl in enumerate(_benchmark_templates):
            idx = i * len(_benchmark_templates) + j
            if idx >= 200:
                break
            benchmarks.append({
                "prompt_id": f"bench_{idx:03d}",
                "text": tmpl.format(subj),
                "category": "benchmark",
                "group_id": None,
            })
    save_prompts(benchmarks[:200], output_dir / "benchmarks.jsonl")

    # Semantic diversity (concrete vs abstract)
    semantic = []
    _concrete = [
        "Describe a {}.", "What does a {} look like?", "How is a {} used?",
        "Where would you find a {}?", "What is a {} made of?",
    ]
    _abstract = [
        "What is {}?", "How would you explain {} to a child?",
        "Why is {} important?", "Give an example of {}.",
        "How does {} relate to everyday life?",
    ]
    _concrete_nouns = ["bicycle", "lighthouse", "telescope", "violin", "glacier",
                       "compass", "cathedral", "volcano", "submarine", "windmill"]
    _abstract_nouns = ["justice", "nostalgia", "entropy", "freedom", "ambiguity",
                       "causality", "emergence", "irony", "symmetry", "resilience"]
    idx = 0
    for noun in _concrete_nouns:
        for tmpl in _concrete:
            if idx >= 50:
                break
            semantic.append({
                "prompt_id": f"sem_{idx:03d}",
                "text": tmpl.format(noun),
                "category": "semantic",
                "group_id": None,
            })
            idx += 1
    for noun in _abstract_nouns:
        for tmpl in _abstract:
            if idx >= 100:
                break
            semantic.append({
                "prompt_id": f"sem_{idx:03d}",
                "text": tmpl.format(noun),
                "category": "semantic",
                "group_id": None,
            })
            idx += 1
    save_prompts(semantic[:100], output_dir / "semantic_diversity.jsonl")

    # Perturbation pairs (20 base × 6 variants = 120)
    base_for_perturbation = [
        {"prompt_id": f"pertbase_{i:02d}", "text": benchmarks[i * 10]["text"]}
        for i in range(min(20, len(benchmarks) // 10))
    ]
    perturbation_prompts = generate_perturbation_set(base_for_perturbation)
    save_prompts(perturbation_prompts, output_dir / "perturbation_pairs.jsonl")

    # Agent decision prompts
    agent = []
    _agent_templates = [
        "Review this code and suggest improvements:\n```python\n{}\n```",
        "Debug this error: {}",
        "Plan the implementation of: {}",
        "What architecture would you use for {}?",
        "Explain the tradeoffs between {} and {}.",
    ]
    _code_snippets = [
        "def f(x): return x if x > 0 else -x",
        "for i in range(len(lst)): result.append(lst[i] * 2)",
        "data = json.loads(open('file.json').read())",
        "if x == None: return False",
        "try: result = compute()\nexcept: pass",
    ]
    _errors = [
        "TypeError: 'NoneType' object is not iterable",
        "KeyError: 'missing_key'",
        "IndexError: list index out of range",
        "RecursionError: maximum recursion depth exceeded",
        "MemoryError during large matrix multiplication",
    ]
    _tasks = [
        "a REST API with authentication",
        "a real-time chat system",
        "a distributed task queue",
        "an image processing pipeline",
        "a recommendation engine",
    ]
    idx = 0
    for snippet in _code_snippets:
        agent.append({
            "prompt_id": f"agent_{idx:03d}",
            "text": _agent_templates[0].format(snippet),
            "category": "agent",
            "group_id": None,
        })
        idx += 1
    for error in _errors:
        agent.append({
            "prompt_id": f"agent_{idx:03d}",
            "text": _agent_templates[1].format(error),
            "category": "agent",
            "group_id": None,
        })
        idx += 1
    for task in _tasks:
        for tmpl in _agent_templates[2:]:
            if "{}" in tmpl and tmpl.count("{}") == 2:
                # tradeoff template needs two args
                other = random.choice([t for t in _tasks if t != task])
                agent.append({
                    "prompt_id": f"agent_{idx:03d}",
                    "text": tmpl.format(task, other),
                    "category": "agent",
                    "group_id": None,
                })
            elif "{}" in tmpl:
                agent.append({
                    "prompt_id": f"agent_{idx:03d}",
                    "text": tmpl.format(task),
                    "category": "agent",
                    "group_id": None,
                })
            idx += 1
    save_prompts(agent[:100], output_dir / "agent_decisions.jsonl")

    total = len(benchmarks[:200]) + len(semantic[:100]) + len(perturbation_prompts) + len(agent[:100])
    print(f"Generated bootstrap set: {total} prompts in {output_dir}")


if __name__ == "__main__":
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else "prompts"
    make_bootstrap_set(output)
