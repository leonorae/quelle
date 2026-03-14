"""
Phase 0B: Generate reasoning traces from a language model on GSM8K.

Usage:
    python src/phase0b_generate_traces.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --n_problems 1000 \
        --output data/traces_gsm8k_qwen.jsonl

Outputs:
    - JSONL file with one trace per line (schema in CONTEXT.md §3.4)
    - Summary report printed to stdout and written to outputs/phase0b_report.txt

Requirements:
    pip install torch transformers accelerate datasets tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. Show all your work clearly.\n"
    "End your response with \"Final answer: [number]\" on its own line.\n\n"
    "Problem: {question}"
)

# Regex to extract the final numeric answer from generated text.
# Handles integers and simple decimals; strips commas from large numbers.
ANSWER_RE = re.compile(
    r"[Ff]inal\s+answer\s*:\s*([+-]?\s*[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Regex to extract the ground-truth answer from GSM8K's answer field.
# GSM8K answers end with "#### <number>".
GSM8K_GT_RE = re.compile(r"####\s*([\d,]+)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_gsm8k_ground_truth(answer_field: str) -> int | None:
    """Return the integer ground-truth answer from a GSM8K answer string."""
    m = GSM8K_GT_RE.search(answer_field)
    if m is None:
        return None
    return int(m.group(1).replace(",", ""))


def extract_generated_answer(text: str) -> int | None:
    """Return the integer extracted from 'Final answer: N' in generated text."""
    # Use the last match in case the model self-corrects mid-trace.
    matches = list(ANSWER_RE.finditer(text))
    if not matches:
        return None
    raw = matches[-1].group(1).replace(",", "").replace(" ", "")
    try:
        # Accept integers and floats; round to nearest int for GSM8K comparison.
        return round(float(raw))
    except ValueError:
        return None


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer with bfloat16 precision and auto device mapping."""
    print(f"Loading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name} (bfloat16, device_map=auto)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_trace(
    model,
    tokenizer,
    question: str,
    generation_config: dict,
) -> str:
    """Generate a reasoning trace for a single question."""
    prompt = PROMPT_TEMPLATE.format(question=question)

    # Use chat template if available (Qwen, Llama-Instruct models).
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            input_text = prompt
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            max_new_tokens=generation_config["max_new_tokens"],
            do_sample=generation_config["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens.
    generated_ids = output_ids[0, input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_dir = Path("outputs")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "phase0b_report.txt"

    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
    }

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    print("Loading GSM8K dataset ...", flush=True)
    dataset = load_dataset("gsm8k", "main", split="train")
    problems = list(dataset)
    if args.n_problems > 0:
        problems = problems[: args.n_problems]
    print(f"  Using {len(problems)} problems.", flush=True)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(args.model)

    # -----------------------------------------------------------------------
    # Resume support: count already-written lines.
    # -----------------------------------------------------------------------
    already_done = 0
    if output_path.exists():
        with open(output_path) as f:
            already_done = sum(1 for _ in f)
        if already_done > 0:
            print(
                f"Resuming from example {already_done} "
                f"({len(problems) - already_done} remaining).",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Generation loop
    # -----------------------------------------------------------------------
    n_correct = 0
    total_tokens = 0
    answer_counts: Counter = Counter()

    # First pass: accumulate stats from already-written lines.
    if already_done > 0:
        with open(output_path) as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("is_correct"):
                    n_correct += 1
                trace = obj.get("generated_trace", "")
                total_tokens += len(trace.split())
                ans = obj.get("extracted_answer")
                if ans is not None:
                    answer_counts[ans] += 1

    with open(output_path, "a") as out_f:
        for idx in tqdm(
            range(already_done, len(problems)),
            desc="Generating traces",
            initial=already_done,
            total=len(problems),
        ):
            problem = problems[idx]
            question = problem["question"]
            ground_truth = extract_gsm8k_ground_truth(problem["answer"])

            trace = generate_trace(model, tokenizer, question, generation_config)
            extracted = extract_generated_answer(trace)
            is_correct = (extracted is not None) and (extracted == ground_truth)

            record = {
                "id": f"gsm8k_{idx}",
                "question": question,
                "ground_truth": ground_truth,
                "generated_trace": trace,
                "extracted_answer": extracted,
                "is_correct": is_correct,
                "model_name": args.model,
                "generation_config": generation_config,
            }
            out_f.write(json.dumps(record) + "\n")

            if is_correct:
                n_correct += 1
            total_tokens += len(trace.split())
            if extracted is not None:
                answer_counts[extracted] += 1

            # Flush periodically so crashes don't lose data.
            if (idx + 1) % args.checkpoint_every == 0:
                out_f.flush()

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    n_total = len(problems)
    accuracy = n_correct / n_total if n_total > 0 else 0.0
    mean_tokens = total_tokens / n_total if n_total > 0 else 0.0
    top_answers = answer_counts.most_common(10)

    report_lines = [
        "=" * 60,
        "Phase 0B — Trace Generation Report",
        "=" * 60,
        f"Model         : {args.model}",
        f"Problems      : {n_total}",
        f"Correct (pass@1): {n_correct} / {n_total} = {accuracy:.1%}",
        f"Mean trace length (words): {mean_tokens:.1f}",
        "",
        "Top-10 generated answers:",
    ]
    for ans, cnt in top_answers:
        report_lines.append(f"  {ans:>8} : {cnt}")
    report_lines += [
        "",
        f"Output        : {output_path.resolve()}",
        "=" * 60,
    ]
    report_text = "\n".join(report_lines)

    print(report_text)
    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    print(f"\nReport saved to {report_path.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GSM8K reasoning traces (Phase 0B)."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--n_problems",
        type=int,
        default=1000,
        help="Number of GSM8K problems to use (0 = all).",
    )
    parser.add_argument(
        "--output",
        default="data/traces_gsm8k_qwen.jsonl",
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="Flush output file every N examples.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
