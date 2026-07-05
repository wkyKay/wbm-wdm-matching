# Experiment B：Transformation-Derived Preference Accuracy

## 目标

Experiment B 用可控变换构造 pairwise preference，用来评估检索方法是否符合预期的相似性偏好。

它和 Experiment A 是独立评估：

```text
Experiment A:
  在真实 test candidate pool 上评价 label-derived retrieval。
  重点是 same-label / label-overlap 的排序表现。

Experiment B:
  在 synthetic variants + real hard negatives 上评价 pairwise preference。
  重点是旋转、平移、噪声、缺失、额外 cluster 等扰动下的鲁棒性和形态偏好。
```

Experiment B 不训练模型，也不改变 train / validation / test 划分。它只在既有 frozen test split 上生成一个固定的 B benchmark。

## 数据流

完整流程如下：

```text
1. shared 从 frozen WM38K test split 中分层采样 query。
2. shared 针对每个 query 生成 synthetic variants。
3. shared 为每个 query 采样 real negative candidates。
4. shared 写出 candidate manifest 和 preference rules。
5. 每个 method 读取同一套 B benchmark，输出 similarity scores。
6. evaluation 根据 b_preferences.csv 比较 preferred candidate 和 less-preferred candidate 的分数。
```

每个方法统一输出：

```text
query_id,candidate_id,similarity_score
```

方法目录不能重新生成 variants 或 negatives。所有变换逻辑都放在 shared 中：

```text
wbm-wdm-matching/shared/wm38k/experiment_b/transforms.py
wbm-wdm-matching/shared/wm38k/experiment_b/preference_benchmark.py
```

## 正式规模

推荐正式配置：

```text
queries: 1000
synthetic candidates/query: 12
real negatives/query: 8
candidate scores: about 20,000
preference tests/query: about 15
preference tests total: about 15,000
```

每个 query 的 synthetic variants：

```text
identity
rot_90
rot_180
shift_mild
shift_strong
scale_mild
noise_mild
noise_strong
dropout_mild
dropout_strong
cluster_extra
cluster_dropout
```

每个 query 的 real negatives：

```text
easy_diff_label_random x2
same_area_wrong_shape x2
same_position_wrong_shape x2
same_label_hard_negative x1
diff_label_similar_morphology x1
```

## Step 1：生成 Experiment B Benchmark

从 repo 根目录运行：

```bash
cd /Users/kayw/Documents/trae_projects/match-test/wbm-wdm-matching

python3 shared/wm38k/experiment_b/cli_build_preference_b.py \
  --data-file ../data/wm38k/Wafer_Map_Datasets.npz \
  --split-manifest artifacts/splits/wm38k_seed2026_sig_70_10_20.csv \
  --out-dir artifacts/preference_b/wm38k_seed2026 \
  --split test \
  --num-queries 1000 \
  --seed 2026
```

输出文件：

```text
artifacts/preference_b/wm38k_seed2026/b_data.npz
artifacts/preference_b/wm38k_seed2026/b_split_manifest.csv
artifacts/preference_b/wm38k_seed2026/b_queries.csv
artifacts/preference_b/wm38k_seed2026/b_candidates.csv
artifacts/preference_b/wm38k_seed2026/b_preferences.csv
artifacts/preference_b/wm38k_seed2026/b_sources.csv
artifacts/preference_b/wm38k_seed2026/b_config.json
```

文件含义：

```text
b_data.npz:
  compact benchmark 数据文件。
  包含 query、synthetic variants、real negatives 的 wafer map 和 labels。
  同时保存 maps/labels 和 arr_0/arr_1 两组 key，用于兼容不同 loader。

b_split_manifest.csv:
  B benchmark 的 split manifest。
  所有 B 样本都写为 test split。
  现有 retrieval 脚本通过它读取 compact benchmark 中的样本。

b_queries.csv:
  B benchmark 的 query manifest。
  注意：这里的 query_id 是 compact benchmark ID，不是原始 WM38K ID。

b_candidates.csv:
  每个 query 对应的 candidate manifest。
  同时记录 candidate_kind、source_id、transform_type、transform_strength、negative_type 等元数据。

b_preferences.csv:
  pairwise preference 规则。
  每一行表示：对某个 query，preferred_candidate_id 应该排在 less_preferred_candidate_id 前面。

b_sources.csv:
  compact benchmark ID 到原始 WM38K source_id 的映射。
  用于追踪 synthetic variant 或 real negative 来自哪个原始样本。

b_config.json:
  本次 B benchmark 的配置记录，包括 seed、query 数量、变换列表、negative 类型和输出文件。
```

## Step 2：运行各方法

公共路径变量：

```bash
B_ROOT=artifacts/preference_b/wm38k_seed2026
B_DATA=$B_ROOT/b_data.npz
B_SPLIT=$B_ROOT/b_split_manifest.csv
B_QUERIES=$B_ROOT/b_queries.csv
B_CANDIDATES=$B_ROOT/b_candidates.csv
B_PREFS=$B_ROOT/b_preferences.csv
```

### 2.1 partial_match：Local Handcrafted Descriptor Retrieval

```bash
python3 partial_match/run_retrieval_B.py \
  --b-data $B_DATA \
  --b-split-manifest $B_SPLIT \
  --b-queries $B_QUERIES \
  --b-candidates $B_CANDIDATES \
  --b-preferences $B_PREFS \
  --out-dir artifacts/preference_b/results/partial_match \
  --method retrieval_compact \
  --min-area 5 \
  --top-k-proposals 6 \
  --topk-match 1 \
  --sigma-pos 0.35 \
  --sigma-area 1.0 \
  --save-token-details
```

输出文件：

```text
artifacts/preference_b/results/partial_match/rankings.csv
artifacts/preference_b/results/partial_match/preference_metrics.json
artifacts/preference_b/results/partial_match/tokens.csv
artifacts/preference_b/results/partial_match/descriptors.npz
```

### 2.2 proposed：Learned Local Cluster Retrieval

需要已经训练好的 cluster encoder checkpoint。

```bash
python3 proposed/run_retrieval_B.py \
  --b-data $B_DATA \
  --b-split-manifest $B_SPLIT \
  --b-queries $B_QUERIES \
  --b-candidates $B_CANDIDATES \
  --b-preferences $B_PREFS \
  --checkpoint artifacts/proposed/checkpoints/cluster_encoder.pt \
  --checkpoint-key encoder \
  --out-dir artifacts/preference_b/results/proposed \
  --proposal-method retrieval_compact \
  --min-area 5 \
  --top-k-proposals 6 \
  --topk-match 1 \
  --sigma-pos 0.35 \
  --sigma-area 1.0 \
  --device cuda
```

输出文件：

```text
artifacts/preference_b/results/proposed/rankings.csv
artifacts/preference_b/results/proposed/preference_metrics.json
artifacts/preference_b/results/proposed/proposal_tokens.csv
artifacts/preference_b/results/proposed/tokens.csv
artifacts/preference_b/results/proposed/embeddings.npz
```

### 2.3 Wafer-DenseIR

使用和 Experiment A 中一致的 checkpoint 和配置。

```bash
python3 Wafer-DenseIR/run_retrieval_B.py \
  --b-data $B_DATA \
  --b-split-manifest $B_SPLIT \
  --b-queries $B_QUERIES \
  --b-candidates $B_CANDIDATES \
  --b-preferences $B_PREFS \
  --out-root artifacts/preference_b/results/denseir \
  --hash experiment_b \
  --backbone-type resnet \
  --backbone-config 18 \
  --input-size 96 \
  --token-mode defect_band \
  --topk-tokens 5 \
  --sigma-pos 0.35 \
  --max-tokens 256 \
  --device cuda
```

输出文件：

```text
artifacts/preference_b/results/denseir/wm38k/denseir/resnet.18/experiment_b/rankings.csv
artifacts/preference_b/results/denseir/wm38k/denseir/resnet.18/experiment_b/preference_metrics.json
artifacts/preference_b/results/denseir/wm38k/denseir/resnet.18/experiment_b/metrics.json
artifacts/preference_b/results/denseir/wm38k/denseir/resnet.18/experiment_b/configs.json
```

### 2.4 WaPIRL Whole-Map Encoder Baseline

该脚本使用 WaPIRL 风格 backbone 提取 whole-map embedding，再用 cosine similarity 对 B candidates 打分。如果有 WaPIRL baseline 的 checkpoint，应通过 `--checkpoint` 传入。

```bash
python3 WaPIRL/run_retrieval_B.py \
  --b-data $B_DATA \
  --b-split-manifest $B_SPLIT \
  --b-queries $B_QUERIES \
  --b-candidates $B_CANDIDATES \
  --b-preferences $B_PREFS \
  --checkpoint artifacts/wapirl/checkpoints/encoder.pt \
  --checkpoint-key encoder \
  --backbone-type resnet \
  --backbone-config 18 \
  --input-size 96 \
  --device cuda \
  --out-dir artifacts/preference_b/results/wapirl
```

输出文件：

```text
artifacts/preference_b/results/wapirl/rankings.csv
artifacts/preference_b/results/wapirl/preference_metrics.json
```

## Step 3：单独评估已有 score 文件

如果某个方法已经输出了 `rankings.csv`，可以单独运行 evaluator：

```bash
python3 evaluation/experiment_b/evaluate_preferences.py \
  --scores artifacts/preference_b/results/partial_match/rankings.csv \
  --preferences artifacts/preference_b/wm38k_seed2026/b_preferences.csv \
  --out artifacts/preference_b/results/partial_match/preference_metrics.json \
  --save-details
```

输出文件：

```text
preference_metrics.json:
  overall preference accuracy
  accuracy by rule_group
  accuracy by rule_type
  tie rate
  missing score count

preference_details.csv:
  每条 preference 的 preferred score、less-preferred score 和 outcome。
```

## 报告指标

主指标：

```text
overall.preference_accuracy
by_group.rotation.preference_accuracy
by_group.shift.preference_accuracy
by_group.scale.preference_accuracy
by_group.noise.preference_accuracy
by_group.dropout.preference_accuracy
by_group.cluster_extra.preference_accuracy
by_group.cluster_dropout.preference_accuracy
by_group.hard_negative.preference_accuracy
```

诊断指标：

```text
counts.num_preferences
counts.num_evaluated_preferences
counts.num_missing_preferences
overall.tie_rate
```

## 注意事项

Experiment B 中的 ID 是 compact benchmark ID，不是原始 WM38K ID。需要追踪原始样本时，使用 `b_sources.csv` 中的 `source_id`。

Experiment B 不应该和 Experiment A 合并成单一分数。A 和 B 回答的问题不同，应分别报告。

如果需要检查变体质量，应从 `b_sources.csv` 中选取 query 和 synthetic candidate，再到 `b_data.npz` 中读取对应 map 可视化。
