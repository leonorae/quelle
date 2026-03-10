"""
Quick check: does karpathy/nanochat-d32 have value embeddings?

Downloads only the meta JSON (~KB) from HuggingFace — no model weights.
Prints the full model config so we can see ve-related fields.

Usage:
    pip install huggingface_hub
    python experiments/VVVVVV/src/check_d32_ve.py
"""

import json
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

REPO = "karpathy/nanochat-d32"
META_FILE = "meta_000650.json"

print(f"Downloading {META_FILE} from {REPO} ...")
path = hf_hub_download(repo_id=REPO, filename=META_FILE)

with open(path) as f:
    meta = json.load(f)

config = meta.get("model_config", meta)
print("\n=== model_config ===")
print(json.dumps(config, indent=2))

# Summarise ve-relevant fields
print("\n=== ve check ===")
ve_fields = {k: v for k, v in config.items() if "ve" in k.lower() or "value" in k.lower()}
if ve_fields:
    print("ve-related fields found:")
    for k, v in ve_fields.items():
        print(f"  {k}: {v}")
else:
    print("No ve-related fields found in model_config.")
    print("(May still be present in architecture — check model weights for 'value_embeds' keys)")
    print("\nTo check weight keys without loading the full model:")
    print("  python -c \"import torch; w=torch.load('<model_000650.pt>', map_location='cpu'); print([k for k in w if 'value' in k or 've' in k.lower()])\"")
