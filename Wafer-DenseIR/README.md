# Wafer-DenseIR

Wafer-DenseIR implements the current main proposal:

```text
WaPIRL-style self-supervised encoder
-> dense feature map
-> proposal-free dense local matching
-> retrieval ranking
-> correspondence heatmap explanation
```

This version does not implement comparison methods. Labels are used only for retrieval evaluation.

## Method

1. Load MixedWM38K wafer maps.
2. Convert wafer bins to WaPIRL-style decoupled input:
   - channel 0: defect bin
   - channel 1: valid wafer mask
3. Extract dense encoder features before global pooling.
4. Select dense tokens from a defect band.
   Dilation is used only for token selection, not for hard grouping or proposal generation.
5. Match query and candidate tokens with cosine similarity plus spatial affinity.
6. Aggregate query-token top-k matches into a wafer-level retrieval score.
7. Save rankings, metrics, and query/candidate correspondence heatmaps.

## Run

Smoke test without a pretrained checkpoint:

```bash
cd /Users/kayw/Documents/trae_projects/match-test/wbm-wdm-matching/Wafer-DenseIR
python3 run_dense_retrieval.py \
  --data_file ../../data/wm38k/Wafer_Map_Datasets.npz \
  --split test \
  --max_samples 32 \
  --input_size 96 \
  --batch_size 16 \
  --token_mode defect_band \
  --token_dilation 1 \
  --topk_tokens 5 \
  --sigma_pos 0.35
```

Real retrieval experiment with a WaPIRL checkpoint:

```bash
python3 run_dense_retrieval.py \
  --data_file ../../data/wm38k/Wafer_Map_Datasets.npz \
  --pretrained_model_file /path/to/wapirl_checkpoint.pt \
  --pretrained_model_key backbone \
  --split test \
  --input_size 96 \
  --token_mode defect_band \
  --token_dilation 1 \
  --topk_tokens 5 \
  --sigma_pos 0.35
```

## Outputs

Outputs are saved under:

```text
artifacts/dense_retrieval/wm38k/denseir/resnet.18/<timestamp>/
```

Files:

- `configs.json`: runtime configuration.
- `rankings.csv`: `query_id, rank, candidate_id, similarity_score`.
- `metrics.json`: multi-label retrieval metrics, including Precision@K, Recall@K, NDCG@K, and mAP.
- `explanations/*.png`: query wafer, top-1 candidate wafer, query correspondence heatmap, and candidate response heatmap.
- `dense_features.npz`: optional dense token dump when `--save_features` is used.

## Notes

The main method is proposal-free. Hard cluster filtering, adhesion separation, and geometry merge should be treated as baselines or ablations, not as the primary path.
