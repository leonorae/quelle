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
| `category` | yes | One of: benchmark, semantic, perturbation, periphery, agent, diagnostic |
| `group_id` | no | Links perturbation variants to their base prompt |
| `perturbation_type` | no | One of: base, rephrase, context_added, authority_bias, negation, register_shift |

## Bootstrap set (~500 prompts)

- `benchmarks.jsonl` — 200 prompts from MMLU/GSM8K subsets
- `semantic_diversity.jsonl` — 100 concrete/abstract concept prompts
- `perturbation_pairs.jsonl` — 100 prompts (20 base × 5 perturbations)
- `agent_decisions.jsonl` — 100 code review / debugging / planning prompts

## Full set (7,304 prompts) — see D8–D12

| File | Count | Category | Strategy |
|---|---|---|---|
| `benchmarks_mmlu.jsonl` | 2,160 | `benchmark` | 27 MMLU subjects × 80, curated for domain breadth |
| `benchmarks_gsm8k.jsonl` | 300 | `benchmark` | GSM8K test split, stratified by step count (2–8 steps) |
| `semantic_diversity_full.jsonl` | 2,000 | `semantic` | 14 domains, embed + k-means cluster + uniform sample (D11: 300 visual-grounding) |
| `perturbation_families.jsonl` | 2,400 | `perturbation` | 400 bases × 6 (base + 5 perturbation types). Template-based placeholder — upgrade to LLM-generated per D9 |
| `periphery_probes.jsonl` | 444 | `periphery` | Formally diverse inputs: malformed, mixed-language, contradictory, nonsense, unusual registers, naturalistic prose, domain outliers (D12) |
| `corpus_full.jsonl` | 7,304 | (merged) | All components, deduplicated, source_file tagged |

### Remaining components (gated)

| Component | Target | Category tag | Gate |
|---|---|---|---|
| KL-spectrum filling | ~2k | `kl_selected` | Phase 0 on candidate pool (D10) |
| Sensitivity probes | ~1k | `sensitivity` | Phase 1 results (D8) |

### Build scripts

```bash
python -m src.build_corpus              # MMLU + GSM8K → benchmarks_*.jsonl
python -m src.build_semantic_diversity  # embed/cluster → semantic_diversity_full.jsonl
python -m src.build_perturbations       # base selection + variants → perturbation_families.jsonl
python -m src.build_periphery           # malformed/nonsense/registers → periphery_probes.jsonl
python -m src.merge_corpus              # merge → corpus_full.jsonl
```

### D11 note (slicer coordination)

This corpus is shared with the slicer experiment (CLIP projection). The semantic
diversity component includes 300 visual-grounding prompts for slicer compatibility.
See DECISIONS.md D11.
