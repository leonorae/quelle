"""Build perturbation families corpus component (D8 component 3, D9).

Selects ~400 base prompts from benchmark and semantic diversity components,
generates 5 perturbation variants per base. register_shift is always template-based;
other types use diverse templates as a placeholder until LLM-generated rephrasings
are available (see D9).

    python -m src.build_perturbations [--output-dir prompts] [--n-bases 400] [--seed 42]

Produces: perturbation_families.jsonl (~2400 prompts: 400 bases × 6 variants including base)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# --- Perturbation templates ---
# Multiple variants per type to avoid mechanical monotony.
# D9: These are placeholders. Replace with LLM-generated rephrasings when available.

REPHRASE_TEMPLATES = [
    "Please answer the following: {}",
    "I'd like to know: {}",
    "Can you help me understand: {}",
    "Here's my question: {}",
    "I'm curious about this: {}",
    "Consider this question: {}",
    "Could you address this: {}",
    "I need your input on: {}",
]

CONTEXT_ADDED_TEMPLATES = [
    "As part of a research evaluation, please respond: {}",
    "This is for an educational assessment. {}",
    "In the context of a comprehensive review: {}",
    "For a knowledge survey we're conducting: {}",
    "As an expert being consulted: {}",
    "During a formal examination: {}",
]

AUTHORITY_BIAS_TEMPLATES = [
    "Leading experts agree that this has a clear answer. {}",
    "According to recent peer-reviewed research: {}",
    "A Nobel laureate recently noted the importance of this. {}",
    "The scientific consensus is well-established here. {}",
    "Top researchers in this field emphasize: {}",
    "Authoritative sources confirm: {}",
]

NEGATION_TEMPLATES = [
    "Consider the opposite perspective. {}",
    "What if the common assumption is wrong? {}",
    "Challenge the standard view: {}",
    "Argue against the conventional wisdom: {}",
    "Take the contrarian position on: {}",
    "What's the strongest counterargument? {}",
]

# register_shift: mechanical padding tokens (D9: always template-based)
REGISTER_SHIFT_PREFIXES = [
    "_ _ _ _ ",
    "_ _ _ _ _ _ _ _ ",
    ". . . . ",
    "--- ",
    ">> ",
]

PERTURBATION_TYPES = {
    "rephrase": REPHRASE_TEMPLATES,
    "context_added": CONTEXT_ADDED_TEMPLATES,
    "authority_bias": AUTHORITY_BIAS_TEMPLATES,
    "negation": NEGATION_TEMPLATES,
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_bases(
    benchmark_path: Path,
    semantic_path: Path,
    gsm8k_path: Path,
    n_bases: int,
    seed: int,
) -> list[dict]:
    """Select base prompts from existing components, balanced across sources."""
    rng = random.Random(seed)

    sources = {}
    if benchmark_path.exists():
        sources["mmlu"] = load_jsonl(benchmark_path)
    if gsm8k_path.exists():
        sources["gsm8k"] = load_jsonl(gsm8k_path)
    if semantic_path.exists():
        sources["semantic"] = load_jsonl(semantic_path)

    if not sources:
        raise FileNotFoundError("No source JSONL files found. Run build_corpus.py and build_semantic_diversity.py first.")

    # Balanced allocation across sources
    per_source = n_bases // len(sources)
    remainder = n_bases - per_source * len(sources)

    bases = []
    for i, (name, records) in enumerate(sources.items()):
        n = per_source + (1 if i < remainder else 0)
        n = min(n, len(records))
        sampled = rng.sample(records, n)
        bases.extend(sampled)

    rng.shuffle(bases)
    return bases[:n_bases]


def generate_perturbations(bases: list[dict], seed: int) -> list[dict]:
    """Generate perturbation families from base prompts."""
    rng = random.Random(seed)
    all_prompts = []

    for base_idx, base in enumerate(bases):
        group_id = f"pert_{base_idx:04d}"
        base_text = base["text"]

        # Include the base prompt
        all_prompts.append({
            "prompt_id": f"{group_id}_base",
            "text": base_text,
            "category": "perturbation",
            "group_id": group_id,
            "perturbation_type": "base",
            "source_id": base.get("prompt_id"),
        })

        # Template-based perturbations (one per type)
        for ptype, templates in PERTURBATION_TYPES.items():
            tmpl = rng.choice(templates)
            all_prompts.append({
                "prompt_id": f"{group_id}_{ptype}",
                "text": tmpl.format(base_text),
                "category": "perturbation",
                "group_id": group_id,
                "perturbation_type": ptype,
            })

        # register_shift (always mechanical)
        prefix = rng.choice(REGISTER_SHIFT_PREFIXES)
        all_prompts.append({
            "prompt_id": f"{group_id}_register_shift",
            "text": prefix + base_text,
            "category": "perturbation",
            "group_id": group_id,
            "perturbation_type": "register_shift",
        })

    return all_prompts


def build_perturbations(output_dir: Path, n_bases: int, seed: int) -> list[dict]:
    """Build perturbation families corpus component."""
    benchmark_path = output_dir / "benchmarks_mmlu.jsonl"
    gsm8k_path = output_dir / "benchmarks_gsm8k.jsonl"
    semantic_path = output_dir / "semantic_diversity_full.jsonl"

    print(f"  Selecting {n_bases} base prompts...")
    bases = select_bases(benchmark_path, semantic_path, gsm8k_path, n_bases, seed)
    print(f"  Selected {len(bases)} bases from available components")

    print("  Generating perturbation variants...")
    prompts = generate_perturbations(bases, seed)
    n_families = len(bases)
    n_variants = len(prompts) - n_families  # exclude base copies

    out_path = output_dir / "perturbation_families.jsonl"
    _save_jsonl(prompts, out_path)
    print(f"  Perturbations: {n_families} families × 6 = {len(prompts)} prompts → {out_path}")
    print(f"  NOTE: Using template perturbations. Replace with LLM-generated rephrasings per D9.")

    return prompts


def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build perturbation families corpus")
    parser.add_argument("--output-dir", default="prompts")
    parser.add_argument("--n-bases", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print("Building perturbation families...")
    build_perturbations(output_dir, args.n_bases, args.seed)


if __name__ == "__main__":
    main()
