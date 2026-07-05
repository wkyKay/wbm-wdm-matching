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

WaPIRL-style pretraining on MixedWM38K:

```bash
cd /Users/kayw/Documents/trae_projects/match-test/wbm-wdm-matching/Wafer-DenseIR
python3 run_wapirl_pretrain.py \
  --data_file ../../data/wm38k/Wafer_Map_Datasets.npz \
  --split_manifest ../artifacts/splits/wm38k_seed2026_sig_70_10_20.csv \
  --device cuda \
  --split train \
  --input_size 96 \
  --batch_size 64 \
  --num_workers 4 \
  --epochs 100 \
  --augmentation crop_noise_rotate \
  --num_negatives 1024
```

This script saves checkpoints under:

```text
checkpoints/wm38k/wapirl_pretrain/<backbone>/<timestamp>/
```

Use the resulting `best_model.pt` for DenseIR retrieval with `--pretrained_model_key backbone`.

Smoke test without a pretrained checkpoint:

```bash
python3 run_dense_retrieval.py \
  --data_file ../../data/wm38k/Wafer_Map_Datasets.npz \
  --device cuda \
  --split test \
  --max_samples 32 \
  --input_size 96 \
  --batch_size 16 \
  --num_workers 4 \
  --token_mode defect_band \
  --token_dilation 1 \
  --topk_tokens 5 \
  --sigma_pos 0.35
```

Real retrieval experiment with a WaPIRL checkpoint:

```bash
python3 run_dense_retrieval.py \
  --data_file ../../data/wm38k/Wafer_Map_Datasets.npz \
  --pretrained_model_file checkpoints/wm38k/wapirl_pretrain/resnet.18/2026-07-05_01:15:49/best_model.pt \
  --pretrained_model_key backbone \
  --device cuda \
  --split test \
  --input_size 96 \
  --num_workers 4 \
  --token_mode defect_band \
  --token_dilation 1 \
  --topk_tokens 5 \
  --sigma_pos 0.35
```

Formal comparison experiment using the same fixed test query set and candidate pool as `partial_match`:

```bash
cd /Users/kayw/Documents/trae_projects/match-test/wbm-wdm-matching/Wafer-DenseIR
python3 run_dense_retrieval.py \
  --data_file ../../data/wm38k/Wafer_Map_Datasets.npz \
  --split_manifest ../artifacts/splits/wm38k_seed2026_sig_70_10_20.csv \
  --query_manifest ../artifacts/splits/wm38k_seed2026_test_queries_2000.csv \
  --candidate_manifest ../artifacts/splits/wm38k_seed2026_test_candidates_1000.csv \
  --pretrained_model_file checkpoints/wm38k/wapirl_pretrain/resnet.18/2026-07-05_01:15:49/best_model.pt \
  --pretrained_model_key backbone \
  --device cuda \
  --split test \
  --input_size 96 \
  --num_workers 4 \
  --token_mode defect_band \
  --token_dilation 1 \
  --topk_tokens 5 \
  --sigma_pos 0.35 \
  --topk_retrieval 1 5 10
```

In this formal mode:

```text
query set = ../artifacts/splits/wm38k_seed2026_test_queries_2000.csv
candidate pool = ../artifacts/splits/wm38k_seed2026_test_candidates_1000.csv, fixed 1000 candidates per query
official metrics = label_metrics.json / label_metrics_flat.csv
```

Run with Tiny-ViT patch tokens:

```bash
python3 run_dense_retrieval.py \
  --data_file ../../data/wm38k/Wafer_Map_Datasets.npz \
  --backbone_type vit \
  --backbone_config tiny \
  --device cuda \
  --split test \
  --input_size 96 \
  --num_workers 4 \
  --token_mode defect_band \
  --token_dilation 1 \
  --topk_tokens 5 \
  --sigma_pos 0.35
```

`--backbone_type vit` defaults to `--backbone_config tiny` when no explicit ViT config is supplied.

ViT options:

- `--backbone_config tiny`: patch=16, embed=192, depth=12, 6x6 patch tokens for 96x96 input.
- `--backbone_config small`: patch=12, embed=96, depth=8, 8x8 patch tokens.
- `--backbone_config micro`: patch=16, embed=96, depth=6, faster smoke tests.

For this retrieval task the ViT path uses patch tokens, not only the CLS token, so explanations remain local and heatmap-based.

## Outputs

Outputs are saved under:

```text
artifacts/dense_retrieval/wm38k/denseir/resnet.18/<timestamp>/
```

Files:

- `configs.json`: runtime configuration.
- `rankings.csv`: `query_id, rank, candidate_id, similarity_score`.
- `metrics.json`: multi-label retrieval metrics, including Precision@K, Recall@K, NDCG@K, and mAP.
- `label_metrics.json`: official method-independent label metrics from `evaluation/`.
- `label_metrics_flat.csv`: flattened official label metrics for tables.
- `explanations/*.png`: query wafer, top-1 candidate wafer, query correspondence heatmap, and candidate response heatmap.
- `dense_features.npz`: optional dense token dump when `--save_features` is used.

## Notes

The main method is proposal-free. Hard cluster filtering, adhesion separation, and geometry merge should be treated as baselines or ablations, not as the primary path.
