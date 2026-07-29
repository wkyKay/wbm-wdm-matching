# Wafer-DenseIR

Wafer-DenseIR is the proposal-free dense retrieval baseline. It now shares the proposed method's contrastive-learning implementation exactly: the same `simple`/`resnet18` encoder definitions, MLP projection head, dual-view augmentation, memory bank, NCE loss, optimizer defaults, scheduler defaults and checkpoint-selection rule. The only intentional difference is the retrieval representation: proposed pools a proposal-centered patch into one Token embedding, while DenseIR uses the same encoder's final spatial feature map for proposal-free dense token matching.

## Run

Run commands from the repository root.

### 1. Pretraining

```bash
python3 Wafer-DenseIR/run_wapirl_pretrain.py \
  --data_file ../data/wm38k/Wafer_Map_Datasets.npz \
  --split_manifest artifacts/splits/wm38k_seed2026_sig_70_10_20.csv \
  --split train \
  --input_size 64 \
  --encoder resnet18 \
  --embedding_dim 256 \
  --projector_size 256 \
  --batch_size 128 \
  --num_workers 4 \
  --epochs 100 \
  --optimizer adamw \
  --learning_rate 1e-3 \
  --scheduler cosine \
  --num_negatives 1024 \
  --temperature 0.07 \
  --device cuda \
  --checkpoint_root artifacts/denseir_pretrain
```

The best checkpoint is saved as `best_model.pt`; it stores the encoder under the `encoder` key.

### 2. Formal dense retrieval

```bash
python3 Wafer-DenseIR/run_dense_retrieval.py \
  --data_file ../data/wm38k/Wafer_Map_Datasets.npz \
  --split_manifest artifacts/splits/wm38k_seed2026_sig_70_10_20.csv \
  --query_manifest artifacts/splits/wm38k_seed2026_test_queries_2000.csv \
  --candidate_manifest artifacts/splits/wm38k_seed2026_test_candidates_1000.csv \
  --pretrained_model_file artifacts/denseir_pretrain/wm38k/wapirl_pretrain/resnet18/<run-id>/best_model.pt \
  --pretrained_model_key encoder \
  --split test \
  --input_size 64 \
  --encoder resnet18 \
  --embedding_dim 256 \
  --batch_size 128 \
  --token_mode defect_band \
  --token_dilation 1 \
  --topk_tokens 5 \
  --sigma_pos 0.35 \
  --topk_retrieval 1 5 10 \
  --device cuda \
  --output_root artifacts/dense_retrieval
```

## Shared Contrastive Protocol

Whole-map samples are placed on a centered $64\times64$ zero-padded canvas without interpolation. DenseIR uses the proposed method's 3-channel layout adapted to a whole map: all-defect mask, selected-region mask and valid-wafer mask. Since DenseIR has no proposal, the selected-region channel equals the full defect mask.

For every whole-map sample, two independently augmented views are produced with the same implementation as `proposed.core.cluster_patches.augment_patch`:

- random rotation in $\{0^\circ,90^\circ,180^\circ,270^\circ\}$;
- row and column shift in $[-3,3]$ pixels with zero fill;
- dropout probability $0.02$ on the two defect channels;
- Bernoulli noise probability $0.01$ on the all-defect channel only.

The random seed is determined by the global seed and sample index, matching the proposed dataset behavior. Both views are augmented; no crop-and-resize augmentation, arbitrary-angle rotation, or ViT model path remains available.

The shared encoder is selected by `--encoder`:

| Parameter | Default | Meaning |
|---|---:|---|
| `--encoder` | `resnet18` | `simple` or the proposed ResNet-18 |
| `--embedding_dim` | `256` | global encoder embedding width |
| `--encoder_width` | `32` | base width for `simple` only |
| `--projector_size` | `256` | MLP projection and memory-bank width |
| `--temperature` | `0.07` | NCE temperature |
| `--num_negatives` | `1024` | sampled memory-bank negatives |
| `--loss_weight` | `0.5` | second-view loss weight |
| `--memory_momentum` | `0.5` | memory-bank EMA coefficient |

Pretraining uses the same MLP head, memory initialization/update, NCE computation and validation-loss checkpoint selection as `proposed/run_cluster_pretrain.py`. DenseIR checkpoints therefore use the same `encoder` key as proposed checkpoints.

## Dense Retrieval

At retrieval time, DenseIR does not use augmentation, projector or memory bank. It feeds the whole-map tensor through the shared encoder and uses `forward_features()` to obtain the final spatial feature map before global pooling. Dense tokens are selected from the defect band, then query and candidate tokens are matched using the existing DenseIR cosine-plus-position scoring and aggregation. This preserves DenseIR's proposal-free dense matching protocol and explanation heatmaps.

The main retrieval parameters remain:

| Parameter | Default | Meaning |
|---|---:|---|
| `--token_mode` | `defect_band` | dense-token selection region |
| `--token_dilation` | `1` | selection-mask dilation radius |
| `--topk_tokens` | `5` | top dense correspondences per query token |
| `--sigma_pos` | `0.35` | dense-token position tolerance |
| `--max_tokens` | `256` | maximum selected dense tokens per image |

## Outputs

Dense retrieval writes to `artifacts/dense_retrieval/wm38k/denseir/<encoder>/<run-id>/` by default:

```text
configs.json
rankings.csv
metrics.json
label_metrics.json
label_metrics_flat.csv
explanations/
dense_features.npz  # only with --save_features
```

`rankings.csv` uses the shared schema:

```text
query_id,rank,candidate_id,similarity_score
```

Labels are not used during training or ranking. They are consumed only after ranking by the shared `evaluation/` code.
