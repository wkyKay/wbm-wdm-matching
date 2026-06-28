# Proposal-Free Dense Local Wafer Retrieval Plan

## 1. Motivation

当前 hard cluster proposal 路线表现不稳定：

- `adhesion` 难以稳定分离 `donut + scratch` 等粘连图案。
- `dilated_group` 容易把不同区域错误联通。
- `geometry_merge` 对断裂 ring 的合并仍依赖手工规则，鲁棒性不足。

因此主方法不再依赖显式 cluster proposal。新的目标是保留局部可解释性，但用 dense feature tokens 替代 hard cluster tokens。

核心思路：

```text
WaPIRL self-supervised encoder
-> dense feature map
-> proposal-free patch/token matching
-> map retrieval score
-> local correspondence heatmap for explanation
```

## 2. Problem Definition

给定一张 query WBM，以及候选 WBM/WDM-wafer-grid 数据库，输出空间缺陷分布最相似的 Top-K。

公开数据集阶段：

```text
WBM-to-WBM retrieval on MixedWM38K
```

生产数据阶段：

```text
WBM query
WDM converted to wafer-grid representation
WBM-to-WDM retrieval
```

相似性原则：

- 形状相似是主要依据。
- 局部结构对应应可解释。
- 面积与位置相近时得分更高。
- 不要求先准确分割 donut、scratch、ring 等 pattern。

## 3. Main Method

### 3.1 WaPIRL Pretraining

沿用 Kahng and Kim 2021 WaPIRL 的自监督预训练框架：

```text
input wafer map x
augmentation t(x)
encoder f_theta
projection head g_phi
memory bank M
NCE / contrastive loss
```

训练目标：

```text
same wafer under different augmentation -> close
different wafers -> far
```

主方法不做有监督 fine-tuning。标签只用于 validation 超参数选择和 test retrieval 评价。

推荐增强：

```text
crop
crop + noise
crop + rotate
```

谨慎使用：

```text
large shift
aggressive cutout
```

因为 WBM/WDM 检索中位置可能有语义，过强 shift 可能破坏 root-cause 相关信息。

### 3.2 Dense Feature Tokens

WaPIRL 原论文使用 global pooled embedding 做分类。这里保留 encoder 的 dense feature map：

```text
F = f_theta^dense(x) in R^{h x w x c}
```

每个 spatial cell 是一个 dense token：

```text
t_i = {
  feature: F_i,
  position: p_i,
  optional mask weight: defectness_i
}
```

可选 token filtering：

```text
只保留 defect mask 附近 tokens
或保留 top activation tokens
或使用所有 valid wafer tokens
```

第一版建议：

```text
保留 defect mask dilated band 内的 dense tokens
```

注意：这里 dilation 只用于选择 feature tokens，不用于生成 hard cluster，也不产生 proposal。

### 3.3 Dense Local Matching

对 query map `Q` 和 candidate map `C`：

```text
Q = {q_1, ..., q_m}
C = {c_1, ..., c_n}
```

token pair score：

```text
s(q_i, c_j)
  = cosine(q_i.feature, c_j.feature)
    * position_affinity(q_i, c_j)
    * optional_area_or_defect_weight
```

位置权重：

```text
position_affinity(q_i, c_j)
  = exp(- ||p_i - p_j||^2 / sigma_pos^2)
```

map-level score 第一版：

```text
S(Q, C)
  = mean_i topk_j s(q_i, c_j)
```

候选聚合可比较：

```text
max pooling
top-k mean
mutual nearest neighbor
Sinkhorn / optimal transport
```

建议默认：

```text
top-k local matching + query-token weighted mean
```

### 3.4 Explanation

解释不来自 cluster proposal，而来自 dense token correspondence。

输出：

```text
query heatmap: 每个 query token 的 best-match score
candidate heatmap: 被匹配 candidate tokens 的累计响应
top matched patch pairs
```

可视化：

```text
query wafer + matched local regions
candidate wafer + corresponding regions
similarity heatmap overlay
top patch correspondence lines
```

解释性量化：

```text
heatmap_on_defect_mean / heatmap_on_normal_mean
defect-mask overlap of top activated tokens
human review for explanation usefulness
```

## 4. Training And Evaluation Protocol

### 4.1 Data Split

```text
train:
  unlabeled wafer maps for self-supervised WaPIRL pretraining

validation:
  labels only used for hyperparameter selection

test:
  labels only used for final retrieval evaluation
```

No label is used in the main encoder training or similarity computation.

### 4.2 Retrieval Evaluation

For every query in test set:

```text
candidate pool = test set - query
rank candidates by similarity score
evaluate Top-K
```

MixedWM38K is multi-label, so relevance should support label sets.

Relevance options:

```text
binary hit: label_q ∩ label_c != empty
Jaccard relevance: |label_q ∩ label_c| / |label_q ∪ label_c|
exact-set match as stricter secondary metric
```

Metrics:

```text
Recall@K
Precision@K
mAP
NDCG@K
Macro Recall@K per class
Multi-label hit rate
```

## 5. Baselines

### 5.1 Global Handcrafted Similarity

Purpose: check whether simple whole-map shape similarity is already enough.

```text
Global-IoU
Global-Chamfer
Global-Hu
Global-radial profile
projection profile
```

### 5.2 WaPIRL Global Retrieval

Purpose: isolate the value of dense local matching.

```text
WaPIRL encoder
global average pooling
cosine similarity
```

This is the most important baseline.

### 5.3 Supervised / Fine-Tuned Global Embedding

Purpose: reference comparison, not main method.

```text
WaPIRL pretrained encoder
supervised fine-tuning for classification
use penultimate embedding for retrieval
```

This tests whether classification-oriented fine-tuning improves or hurts retrieval.

### 5.4 Proposal-Based Local Retrieval

Purpose: show hard proposal instability.

```text
filtered proposal + token matching
adhesion proposal + token matching
geometry_merge proposal + token matching
```

These are ablations, not the main method.

### 5.5 Dense Local Retrieval Ablations

Purpose: verify each design choice.

```text
random encoder + dense local matching
WaPIRL encoder + global embedding
WaPIRL encoder + dense tokens without position weight
WaPIRL encoder + dense tokens with position weight
max aggregation vs top-k aggregation vs Sinkhorn
all valid tokens vs defect-band tokens
```

## 6. Expected Contributions

The paper should not claim WaPIRL itself as the contribution. WaPIRL is the pretraining backbone.

Main contributions:

```text
1. A proposal-free dense local retrieval framework for wafer maps.
2. Adaptation of WaPIRL representations from classification to retrieval.
3. Dense local matching that avoids fragile hard cluster proposal.
4. Interpretable patch-level correspondence heatmaps for retrieval.
5. A retrieval protocol and baselines for multi-label MixedWM38K.
```

## 7. Implementation Roadmap

### Stage 1: WaPIRL Feature Extraction

```text
reuse WaPIRL encoder
load self-supervised checkpoint
export global embedding and dense feature map
save features for train / val / test
```

Output:

```text
artifacts/dense_retrieval/features/*.npz
```

### Stage 2: Dense Matching Baseline

```text
implement cosine token similarity
implement position affinity
implement top-k aggregation
write retrieval ranking file
```

Output:

```text
query_id,candidate_id,similarity_score
```

### Stage 3: Evaluation

```text
multi-label relevance
Recall@K / mAP / NDCG@K
per-class metrics
```

### Stage 4: Explanation Visualization

```text
top matched query/candidate token pairs
query heatmap
candidate heatmap
side-by-side retrieval panel
```

### Stage 5: Baselines And Ablations

```text
global handcrafted
WaPIRL global
supervised fine-tuned global
proposal local
dense local ablations
```

## 8. Decision

Current hard proposal method is not reliable enough to be the main research path.

Final main method:

```text
WaPIRL-based proposal-free dense local wafer retrieval
```

Hard proposal methods remain useful only as:

```text
baseline
ablation
failure analysis
```
