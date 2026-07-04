# evaluation

Method-independent retrieval evaluation for the current wafer-map retrieval plan.

This directory owns the official metrics used to compare `partial_match`, `Wafer-DenseIR`, and later methods. Each method should output a ranking file, while this evaluator reads the shared WM38K split/query manifests and computes label-derived retrieval metrics in one consistent way.

Expected ranking columns:

- `query_id`
- `candidate_id`
- `similarity_score`

Optional ranking column:

- `rank`

Primary label metrics:

- `LabelNDCG@K`
- `MeanJaccard@K`
- `ExactRate@K`
- `HitRate@K`
- `Precision@K`
- `mAP_hit`
- `mAP_exact`

Example:

```bash
python3 evaluation/evaluate_rankings.py \
  --rankings artifacts/proposal_based/test/rankings.csv \
  --split-manifest artifacts/splits/wm38k_seed2026_sig_70_10_20.csv \
  --query-manifest artifacts/splits/wm38k_seed2026_test_queries_2000.csv \
  --candidate-manifest artifacts/splits/wm38k_seed2026_test_candidates_1000.csv \
  --out artifacts/proposal_based/test/label_metrics.json
```

If `--candidate-manifest` is provided, IDCG, Recall, mAP, and Top-K metrics are computed against that fixed per-query candidate pool. If it is omitted, the evaluator uses the full test split from `--split-manifest` as the candidate pool.
