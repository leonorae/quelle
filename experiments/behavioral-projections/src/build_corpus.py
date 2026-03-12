"""Build the full benchmark corpus (D8 components 1-2: MMLU + GSM8K).

Downloads curated MMLU subjects and GSM8K, formats to the project JSONL schema,
and writes to prompts/. Run with:

    python -m src.build_corpus [--output-dir prompts] [--seed 42]

Produces:
    benchmarks_mmlu.jsonl     (~1500-2000 MMLU questions)
    benchmarks_gsm8k.jsonl    (~300 GSM8K questions, stratified by step count)
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

# --- MMLU subject selection (D11 note: shared with slicer) ---
# ~27 subjects across all four MMLU clusters, redundant pairs dropped.

MMLU_SUBJECTS = {
    # STEM
    "abstract_algebra",
    "college_physics",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "electrical_engineering",
    "astronomy",
    "machine_learning",
    # Humanities
    "philosophy",
    "formal_logic",
    "world_religions",
    "moral_scenarios",
    "prehistory",
    "jurisprudence",
    # Social science
    "econometrics",
    "high_school_geography",
    "sociology",
    "public_relations",
    "us_foreign_policy",
    "human_sexuality",
    # Professional
    "clinical_knowledge",
    "professional_medicine",
    "professional_accounting",
    "professional_law",
    "management",
    "nutrition",
    "computer_security",
}

MMLU_CHOICES = ["A", "B", "C", "D"]

GSM8K_TARGET = 300


def count_gsm8k_steps(answer: str) -> int:
    """Count reasoning steps in a GSM8K answer by counting calculator annotations."""
    return len(re.findall(r"<<[^>]+>>", answer))


def format_mmlu_prompt(row: dict, subject: str) -> str:
    """Format an MMLU row as an open-ended prompt with choices."""
    choices_str = "\n".join(
        f"{letter}. {row['choices'][i]}"
        for i, letter in enumerate(MMLU_CHOICES)
    )
    subject_display = subject.replace("_", " ").title()
    return f"[{subject_display}] {row['question']}\n{choices_str}"


def format_gsm8k_prompt(row: dict) -> str:
    """Format a GSM8K row as a math reasoning prompt."""
    return row["question"]


def build_mmlu(output_dir: Path, seed: int) -> list[dict]:
    """Download and format MMLU subjects."""
    rng = random.Random(seed)
    prompts = []
    idx = 0

    for subject in sorted(MMLU_SUBJECTS):
        try:
            ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        except Exception:
            # Fallback to the original MMLU source
            try:
                ds = load_dataset("hails/mmlu_no_train", subject, split="test")
            except Exception as e:
                print(f"  Skipping {subject}: {e}")
                continue

        rows = list(ds)
        # Cap at 80 per subject to keep balance
        if len(rows) > 80:
            rows = rng.sample(rows, 80)

        for row in rows:
            prompts.append({
                "prompt_id": f"mmlu_{idx:04d}",
                "text": format_mmlu_prompt(row, subject),
                "category": "benchmark",
                "subcategory": f"mmlu_{subject}",
                "group_id": None,
            })
            idx += 1

    rng.shuffle(prompts)
    # Re-index after shuffle
    for i, p in enumerate(prompts):
        p["prompt_id"] = f"mmlu_{i:04d}"

    out_path = output_dir / "benchmarks_mmlu.jsonl"
    _save_jsonl(prompts, out_path)
    print(f"  MMLU: {len(prompts)} prompts from {len(MMLU_SUBJECTS)} subjects → {out_path}")
    return prompts


def build_gsm8k(output_dir: Path, seed: int) -> list[dict]:
    """Download and format GSM8K, stratified by step count."""
    rng = random.Random(seed)
    ds = load_dataset("openai/gsm8k", "main", split="test")

    # Bin by step count
    bins: dict[int, list] = {}
    for row in ds:
        steps = count_gsm8k_steps(row["answer"])
        bins.setdefault(steps, []).append(row)

    # Stratified sample: proportional to bin size, minimum 5 per bin
    total_available = sum(len(v) for v in bins.values())
    prompts = []
    idx = 0

    for step_count in sorted(bins):
        rows = bins[step_count]
        # Proportional share, but at least 5 (if available)
        target = max(5, round(GSM8K_TARGET * len(rows) / total_available))
        target = min(target, len(rows))
        sampled = rng.sample(rows, target)

        for row in sampled:
            # Extract final answer from #### line
            final_answer = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else ""
            prompts.append({
                "prompt_id": f"gsm8k_{idx:04d}",
                "text": format_gsm8k_prompt(row),
                "category": "benchmark",
                "subcategory": "gsm8k",
                "group_id": None,
                "metadata": {"steps": step_count, "answer": final_answer},
            })
            idx += 1

    # Trim to target if overshot
    if len(prompts) > GSM8K_TARGET:
        prompts = rng.sample(prompts, GSM8K_TARGET)
        for i, p in enumerate(prompts):
            p["prompt_id"] = f"gsm8k_{i:04d}"

    out_path = output_dir / "benchmarks_gsm8k.jsonl"
    _save_jsonl(prompts, out_path)
    print(f"  GSM8K: {len(prompts)} prompts (stratified by step count) → {out_path}")
    return prompts


def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build benchmark corpus (MMLU + GSM8K)")
    parser.add_argument("--output-dir", default="prompts", help="Output directory for JSONL files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print("Building benchmark corpus...")
    mmlu = build_mmlu(output_dir, args.seed)
    gsm8k = build_gsm8k(output_dir, args.seed)
    print(f"\nTotal benchmark prompts: {len(mmlu) + len(gsm8k)}")


if __name__ == "__main__":
    main()
