"""
Phase 3: Correlate geometric features with RIS scores and train predictors.

NOT YET IMPLEMENTED. See CONTEXT.md §6.

Planned analyses:
    1. Pearson/Spearman r: each geometric feature vs. overall RIS score.
    2. Logistic regression: predict step-level flaw (RIS < 3) from geometry.
    3. Temporal early-warning: geometry at step t predicting flaw at step t+k.
    4. Visualizations: scatter plots, UMAP of hidden states coloured by RIS.

Planned interface:
    python src/phase3_analysis.py \
        --input data/traces_gsm8k_qwen_geometry.jsonl \
        --output_dir outputs/phase3/
"""

raise NotImplementedError("Phase 3 is not yet implemented.")
