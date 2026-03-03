"""
Phase 2: Extract geometric features from hidden states for each reasoning trace.

NOT YET IMPLEMENTED. See CONTEXT.md §5.

Planned features per reasoning step:
    - angle_concentration : mean off-diagonal cosine similarity of token hidden states
    - velocity            : L2 norm of centroid shift between consecutive steps
    - curvature           : Menger curvature of three consecutive step centroids
    - manifold_dim        : intrinsic dimensionality via PCA/MLE on step hidden states
    - centroid_distance   : distance from step centroid to correct/incorrect manifold

Planned interface:
    python src/phase2_extract_geometry.py \
        --traces data/traces_gsm8k_qwen_ris.jsonl \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output data/traces_gsm8k_qwen_geometry.jsonl \
        --layer -1          # which transformer layer to extract (default: last)
"""

raise NotImplementedError("Phase 2 is not yet implemented.")
