"""
Phase 1: Score reasoning traces with an LLM judge ensemble using the RIS rubric.

NOT YET IMPLEMENTED. See CONTEXT.md §4 and wiki/concepts/ris-scoring.md.

Planned interface:
    python src/phase1_ris_scoring.py \
        --input data/traces_gsm8k_qwen.jsonl \
        --output data/traces_gsm8k_qwen_ris.jsonl \
        --judges deepseek-v3 qwen-72b llama-3.1-70b \
        --api_base https://openrouter.ai/api/v1

Each output record will be the input record augmented with a `ris_scores` field:
    "ris_scores": {
        "deepseek-v3":   {"overall": 4, "steps": [5, 4, 3, 4]},
        "qwen-72b":      {"overall": 3, "steps": [4, 3, 3, 3]},
        "llama-3.1-70b": {"overall": 4, "steps": [4, 4, 4, 4]},
        "ensemble_mean": 3.67
    }
"""

raise NotImplementedError("Phase 1 is not yet implemented.")
