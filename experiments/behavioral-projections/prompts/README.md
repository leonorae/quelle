# Prompt Sets

JSONL format, one JSON object per line:

```json
{"prompt_id": "bench_001", "text": "What is the capital of France?", "category": "benchmark", "group_id": null}
{"prompt_id": "pert_001_base", "text": "Review this code...", "category": "perturbation", "group_id": "pert_001", "perturbation_type": "base"}
{"prompt_id": "pert_001_rephrase", "text": "Look at this code...", "category": "perturbation", "group_id": "pert_001", "perturbation_type": "rephrase"}
```

## Fields

| Field | Required | Description |
|-------|----------|-------------|
| `prompt_id` | yes | Unique identifier |
| `text` | yes | The prompt string |
| `category` | yes | One of: benchmark, semantic, perturbation, agent, diagnostic |
| `group_id` | no | Links perturbation variants to their base prompt |
| `perturbation_type` | no | One of: base, rephrase, context_added, authority_bias, negation, register_shift |

## Bootstrap set (~500 prompts)

- `benchmarks.jsonl` — 200 prompts from MMLU/GSM8K subsets
- `semantic_diversity.jsonl` — 100 concrete/abstract concept prompts
- `perturbation_pairs.jsonl` — 100 prompts (20 base × 5 perturbations)
- `agent_decisions.jsonl` — 100 code review / debugging / planning prompts

## Full set (5–10k prompts) — see D8–D10

| Component | Target | Category tag | Strategy |
|---|---|---|---|
| Benchmark anchors | ~2k | `benchmark` | Curated MMLU/GSM8K subsets, hand-picked for domain breadth |
| KL-spectrum filling | ~2k | `kl_selected` | Greedy facility-location on pairwise KL from Phase 0 candidate pool |
| Perturbation families | ~2k | `perturbation` | ~400 bases × 5 LLM-generated variants (not templates) |
| Semantic diversity | ~2k | `semantic` | Embed candidates, cluster, sample per cluster |
| Sensitivity probes | ~1k | `sensitivity` | Gated on Phase 1 — high-residual prompts from bisimulation probe |

Components 1–4 are pre-model. Component 5 is gated on Phase 1 results.
