"""Merge all corpus components into a single JSONL file.

    python -m src.merge_corpus [--prompts-dir prompts] [--output prompts/corpus_full.jsonl]

Reads:
    benchmarks_mmlu.jsonl
    benchmarks_gsm8k.jsonl
    semantic_diversity_full.jsonl
    perturbation_families.jsonl
    periphery_probes.jsonl

Produces: corpus_full.jsonl with manifest summary printed to stdout.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


COMPONENTS = [
    "benchmarks_mmlu.jsonl",
    "benchmarks_gsm8k.jsonl",
    "semantic_diversity_full.jsonl",
    "perturbation_families.jsonl",
    "periphery_probes.jsonl",
]


def merge(prompts_dir: Path, output: Path) -> None:
    all_prompts = []
    prompt_ids = set()

    for component_file in COMPONENTS:
        path = prompts_dir / component_file
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue

        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                # Deduplicate by prompt_id
                pid = record["prompt_id"]
                if pid in prompt_ids:
                    print(f"  WARNING: duplicate prompt_id '{pid}' in {component_file}, skipping")
                    continue
                prompt_ids.add(pid)

                # Tag source component
                record["source_file"] = component_file
                all_prompts.append(record)
                count += 1

        print(f"  {component_file}: {count} prompts")

    # Write merged file
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for p in all_prompts:
            f.write(json.dumps(p) + "\n")

    # Print manifest
    categories = Counter(p["category"] for p in all_prompts)
    print(f"\n  Total: {len(all_prompts)} prompts → {output}")
    print("  By category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    # Check perturbation group integrity
    groups = Counter(p.get("group_id") for p in all_prompts if p.get("group_id"))
    group_sizes = Counter(groups.values())
    if groups:
        print(f"  Perturbation groups: {len(groups)} families, size distribution: {dict(group_sizes)}")


def main():
    parser = argparse.ArgumentParser(description="Merge corpus components")
    parser.add_argument("--prompts-dir", default="prompts")
    parser.add_argument("--output", default="prompts/corpus_full.jsonl")
    args = parser.parse_args()

    print("Merging corpus components...")
    merge(Path(args.prompts_dir), Path(args.output))


if __name__ == "__main__":
    main()
