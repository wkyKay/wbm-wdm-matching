# Cluster Proposal Design

本文档记录 `partial_match` 中 cluster proposal 的设计思路，以及针对粘连、断续环形结构和 token 数量控制等问题的处理方案。

## 1. 目标

本项目的检索任务不是做精确语义分割，而是从 52x52 wafer map 中提取少量可解释的局部结构 token，用于后续 local matching。

最终目标：

```text
每张 map 输出 1-5 个主要区域 token，最多可配置到 10 个。
```

因此 proposal 分两层：

```text
debug proposal:
  raw / filtered / adhesion / closing / DBSCAN / spectral / ...
  可以产生较多碎片，用于分析方法行为。

retrieval proposal:
  topk / compact
  只保留最大的 K 个主要区域，用于检索。
```

## 2. 遇到的问题

### 2.1 连通域欠分割

`raw`、`filtered` 都基于 8-连通域。只要两个 pattern 有轻微接触，就会被当成一个 component。

典型问题：

```text
圆形 blob + 线段轻微粘连
两个不同斜率的线段交叉或接触
ring fragment 和 scratch 接触
```

### 2.2 closing 会加重粘连

`closing` 可以补断裂，但它是全图形态学操作，容易把本来只轻微接触或接近的结构进一步连在一起。

因此：

```text
closing 不作为 mixed-pattern retrieval 的默认 proposal。
closing 只保留为 debug / ablation 方法。
```

### 2.3 adhesion split 会过分割断续 ring

`adhesion_split` 对“圆形 + 线段”或“不同斜率线段”有帮助，但对于沿 wafer 边缘断续分布的 edge-ring，它可能把 ring 切成很多段。

这个问题不适合单靠 adhesion split 解决，后续应增加 ring-aware proposal。

### 2.4 多路 proposal 会产生太多 token

如果直接合并：

```text
filtered + adhesion + ring-aware
```

每张 map 可能产生几十个 token。对于 52x52 map，这会让 local matching 变慢且噪声过多。

因此最终检索阶段需要 compact selection，即 `topk`。

## 3. 当前实现的方法

### 3.1 filtered

基础稳定方案：

```text
connected components
 -> area >= min_area
 -> cluster stats
```

特点：

```text
优点: 快、稳定、可解释
缺点: 粘连时欠分割
用途: debug / fallback / ablation
```

### 3.2 adhesion

处理轻微粘连的候选生成器。

流程：

```text
connected components
 -> 过滤 area < min_area 的小噪声
 -> 判断 component 是否可疑粘连
 -> 对可疑 component 调用 adhesion_split
 -> 拆分失败则回退原 component
```

可疑粘连判断使用简单形状规则：

```text
area >= suspicious_area
and at least min_suspicious_cues signals:
  bbox aspect ratio extreme
  PCA elongation high
  compactness high
  fill ratio low
```

当前默认：

```python
cluster(
    defect_mask,
    valid_mask,
    method="adhesion",
    min_area=3,
    split_method="tv_hybrid",
    suspicious_area=40,
    min_suspicious_cues=1,
    max_split_count=12,
    min_split_coverage=0.5,
)
```

适合处理：

```text
圆形区域 + 线段粘连
两个斜率不同的线段
T / X / Y 形接触
```

不稳定场景：

```text
断续 edge-ring
ring fragment 与 scratch 方向连续地粘在一起
大量噪声桥接
```

### 3.3 topk / compact

最终检索推荐使用的 compact proposal。

流程：

```text
base_method candidates, default group_then_adhesion
 -> area >= min_area
 -> sort by area desc
 -> keep top_k
```

默认：

```python
cluster(
    defect_mask,
    valid_mask,
    method="topk",
    top_k=5,
    base_method="geometry_merge",
    min_area=5,
)
```

别名：

```text
compact
dilated_group_topk
```

设计目的：

```text
把 debug proposal 的多个碎片压缩成少量主要区域。
使最终 retrieval token 数可控。
```

建议配置：

```text
top_k = 5 作为默认
top_k = 8 或 10 用于更复杂 mixed pattern
不建议超过 10
```

## 4. 后续 ring-aware proposal 设计

edge-ring 的问题是几何结构特殊：它沿 wafer 边缘分布，可能断断续续，但整体仍应视作一个 ring pattern。

不建议对全图做 closing。更合理的是只在边缘环带内做 constrained merge。

建议流程：

```text
compute wafer center and radius
 -> convert defect pixels to (r_norm, theta)
 -> select edge-band pixels, e.g. r_norm > 0.70
 -> group by radius band
 -> merge angular fragments with small theta gaps
 -> output 1-2 ring tokens
 -> remove high-confidence ring pixels from remaining mask
 -> remaining mask uses adhesion/topk
```

这样可以解决：

```text
断续 edge-ring 被切成很多块
edge-ring fragment 与线段粘连
```

ring-aware token 应该带额外字段：

```python
{
    "proposal_type": "ring_aware",
    "geometry_type": "edge_ring",
    "confidence": ...,
    "angular_coverage": ...,
    "radial_consistency": ...,
}
```

## 5. 多方法如何协作

推荐关系：

```text
filtered:
  稳定 fallback，主要用于 debug 和 ablation。

adhesion:
  更强的候选生成器，用于拆明显粘连；由于可能过度切分，不作为默认 retrieval base。

dilated_group:
  更保守的候选生成器，用于合并断续 ring / scratch，作为对照方法。

ring-aware:
  后续新增，专门处理 edge-ring。

topk:
  最终 retrieval proposal，负责控制 token 数量。
```

当前实际推荐：

```text
retrieval:
  geometry_merge
  topk(base_method="geometry_merge", top_k=5)

debug / visualization:
  raw / filtered / adhesion / closing / simi_paper / DBSCAN / spectral / tensor_voting
```

未来加入 ring-aware 后，推荐变成：

```text
ring candidates
 + adhesion candidates on remaining mask
 -> compact top-k selection
```

## 6. Dilated-group proposal 设计

针对断续 ring / scratch 被切成很多小块的问题，已增加一种更温和的 grouping 方法：

```text
dilated proposal for grouping, original pixels for token
```

核心思想：

```text
膨胀或 closing 只用于判断哪些原始碎片属于同一个候选区域。
最终 token 仍然只使用原始 defect pixels，不使用膨胀出来的虚假像素。
```

### 6.1 为什么不能直接用全图 closing

直接对原图做 closing，然后把 closing 后的像素作为 token，会引入两个问题：

```text
1. 虚假像素进入 token，影响面积、形状和后续 descriptor。
2. 原本相近但不同的 pattern 会被更强地粘连。
```

`dilated_group` 只把 closing/dilation 当作 grouping 辅助，不改变最终 token 的像素集合。

### 6.2 基本流程

```text
1. original_mask = defect_mask & valid_mask

2. grouping_mask = dilate(original_mask, radius=1)
   或 grouping_mask = closing(original_mask, radius=1)

3. groups = connected_components(grouping_mask)

4. 对每个 group:
     original_pixels = original_mask & group

5. 如果 original_pixels 为空:
     skip

6. 用 original_pixels 计算 token stats
```

也就是说：

```text
grouping region 决定哪些碎片合并。
original pixels 决定最终 token 的真实形状、面积和坐标。
```

### 6.3 适合解决的问题

`dilated_group` 更适合：

```text
断续 edge-ring
断续 scratch
同一图案中间有小 gap
AC 清洗后被打断的局部结构
```

它不适合：

```text
需要拆开的圆形 + 线段粘连
两个真实不同 pattern 非常接近
复杂 mixed pattern 的强粘连
```

这些情况仍应交给 `adhesion` 或后续 `ring-aware`。

### 6.4 与 adhesion / topk 的关系

`dilated_group` 不替代 `adhesion`。两者倾向相反：

```text
adhesion:
  把粘连的大 component 拆开。

dilated_group:
  把断续的小 fragments 合并成一个候选区域。
```

因此可以作为 `topk` 的另一个 base method；当前默认 base 是 `geometry_merge`：

```python
cluster(
    defect_mask,
    valid_mask,
    method="topk",
    base_method="geometry_merge",
    top_k=5,
)
```

建议对比：

```text
topk(base_method="geometry_merge")
topk(base_method="group_then_adhesion")
topk(base_method="filtered")
topk(base_method="adhesion")
```

预期：

```text
adhesion_topk:
  更适合圆形和线段粘连、不同斜率线段接触。

dilated_group_topk:
  更适合断续 ring / scratch，不容易把同一图案切碎。
```

### 6.5 风险控制

为了避免膨胀把不同 pattern 合并过度，建议第一版参数保守：

```text
dilation_radius = 1
use_closing = False 或只使用 3x3 cross closing
min_area = 3
top_k = 5
```

如果 group 过大或形状非常可疑，可以有两种策略：

```text
策略 A:
  保留为一个大 group token，让 topk 选择。

策略 B:
  对该 group 的 original_pixels 再跑 adhesion split。
```

第一版建议使用策略 A，保持方法简单，先观察可视化结果。

### 6.6 当前实现

当前已实现三个入口：

```python
cluster(defect_mask, valid_mask, method="dilated_group")
cluster(defect_mask, valid_mask, method="dilated_adhesion")
cluster(defect_mask, valid_mask, method="topk_dilated", top_k=5)
```

含义：

```text
dilated_group:
  只做膨胀 grouping，最终 token 使用原始 pixels。

dilated_adhesion:
  先做 dilated_group。
  对可疑大 group 的原始 pixels 再做 adhesion split。

topk_dilated:
  base_method = dilated_adhesion。
  再按面积选择最大的 K 个主要区域。
```

建议第一轮比较：

```text
topk(base_method="geometry_merge", top_k=5)
topk(base_method="group_then_adhesion", top_k=5)
topk_dilated(top_k=5)
topk(base_method="adhesion", top_k=5)
```

### 6.7 后续 hybrid 方案

后续可以实现：

```text
hybrid_topk:
  ring/dilated grouping candidates
  + adhesion candidates
  -> 去重
  -> 按面积或 coverage gain 选择 top_k
```

但当前阶段优先保持简单：

```text
先比较 dilated_group_topk vs adhesion_topk vs topk_dilated
最后再考虑 hybrid_topk
```

## 7. 去重和 token 数量控制

目前第一版使用简单可靠的面积 top-k：

```text
sort by area desc
keep top_k
```

这是有意选择的简单 baseline，原因：

```text
52x52 map 尺度很小
主要 pattern 往往由少数大区域决定
过多小碎片会污染 matching
```

如果后续改成多路 ensemble，可增加以下去重策略：

### 7.1 同类去重

```text
same proposal_type 内 IoU > 0.90
保留 confidence 更高或 area 更大的 token
```

### 7.2 父子替代

如果一个 `filtered` 大 component 被多个 `adhesion` child 覆盖：

```text
coverage(filtered, union(adhesion_children)) > 0.75
and number of children >= 2
```

则删除 filtered parent，保留 adhesion children。

### 7.3 ring 保护

ring-aware token 不应被普通 edge fragments 删除。

反过来，应删除被 ring token 覆盖的边缘碎片：

```text
overlap(fragment, ring_token) > 0.70
and fragment centroid in edge band
```

### 7.4 token budget

最终仍应加硬上限：

```text
max_tokens_per_map = 5 by default
max_tokens_per_map <= 10 for complex cases
```

## 8. Matching 阶段的配合

proposal 只负责生成 token，最终相似度由 matching 决定。

token 应保留来源：

```python
{
    "proposal_type": "topk",
    "proposal_source": "group_then_adhesion",
    "topk_rank": 0,
    "topk_base_method": "group_then_adhesion",
    "area": ...,
    "centroid_row": ...,
    "centroid_col": ...,
    "bbox": ...,
}
```

后续 pair score 可以加入：

```text
shape_sim
area_affinity
position_affinity
proposal_compatibility
confidence_weight
```

对于第一版，可先只使用 topk token，避免 matching 阶段过度复杂。

## 9. 当前推荐使用方式

### 9.1 Python API

```python
from partial_match.core.clustering import cluster

clusters = cluster(
    defect_mask,
    valid_mask,
    method="topk",
    top_k=5,
    base_method="group_then_adhesion",
    min_area=3,
)
```

### 9.2 Historical token extraction

早期 `week1_*` token extraction 脚本已废弃并删除。当前 retrieval baseline 使用：

```bash
python3 partial_match/run_proposal_retrieval_pipeline.py \
  --data-file ../data/wm38k/Wafer_Map_Datasets.npz \
  --max-samples 512 \
  --sample-strategy stratified \
  --out-dir ../artifacts/proposal_based/system_test_512_stratified \
  --review-max-queries 64 \
  --review-top-k 3 \
  --metric-k 1 3 5 10
```

### 9.3 Visualization

```bash
python3 partial_match/cluster_test/compare_all_methods.py
```

输出：

```text
artifacts/week1/figures/all_proposal_methods_comparison.png
artifacts/week1/figures/all_proposal_methods_cleaned_comparison.png
artifacts/week1/figures/before_vs_after_cleaning.png
```

## 10. 当前结论

当前最可行的 proposal 方案：

```text
Filtered -> Radial Ring Split / Adhesion -> Geometry Merge Proposal
  method = geometry_merge
  topk base_method = geometry_merge
  top_k = 5
  min_area = 5
```

原因：

```text
1. 先过滤小连通域，减少噪声 fragments。
2. radial ring split 先尝试把 donut / edge-ring 半径带从 scratch 粘连 component 中拆出来。
3. adhesion 继续拆明显粘连 component，避免后续 merge 建在错误大块上。
4. geometry merge 只在 ring / line / local blob 几何一致时合并 fragments，不做全图像素膨胀。
5. topk 能把 token 数稳定控制在 5 个以内。
6. filtered / adhesion / dilated_group / group_then_adhesion / topk_dilated 继续保留为对照方法。
```

## 11. 当前方案：Geometry Merge

`geometry_merge` 的目标是合并被截断但几何上属于同一 pattern 的 fragments，同时避免 dilation 把邻近但不同的 pattern 粘在一起。

流程：

```text
defect_mask
 -> filtered components, default min_area=5
 -> radial ring split / adhesion candidates for suspicious merged components
 -> build component graph
 -> ring-aware merge: similar radial band and small pixel angular gap
 -> line-aware merge: similar PCA orientation and small projection gap
 -> local blob merge: very close fragments with bounded bbox
 -> topk(base_method="geometry_merge", top_k=5)
```

默认调用：

```python
clusters = cluster(
    defect_mask,
    valid_mask,
    method="geometry_merge",
    min_area=5,
)
```

逐步可视化由 root pipeline 可选生成：

```bash
python3 partial_match/run_proposal_retrieval_pipeline.py \
  --data-file ../data/wm38k/Wafer_Map_Datasets.npz \
  --out-dir ../artifacts/proposal_based/system_test_512_stratified \
  --save-step-figures \
  --step-samples 24
```

## 12. Ablation：Group Then Adhesion

`group_then_adhesion` 仍保留为 ablation，但不再作为默认方案。它使用 dilation grouping，容易把相近但不同的 pattern 连成一个 group。

```text
group_then_adhesion
```

### 11.1 目标

```text
先用保守 dilation 合并局部断裂结构；
再只对真正可疑的 group 做 adhesion split；
最终仍用 top_k=5 控制 retrieval token 数。
```

关键原则：

```text
dilation 只用于 grouping；
最终 token 只使用原始 defect pixels；
adhesion 只做 selective split；
split 不好就回退到 unsplit group。
```

### 11.2 流程

```text
defect_mask
 -> filtered mask, remove connected components with area < min_area
 -> conservative dilation grouping, default dilation_radius=1
 -> connected components on dilated mask
 -> recover original defect pixels inside each group
 -> build base group token
 -> skip ring-like / edge-like groups
 -> split suspicious non-ring groups with adhesion tv_hybrid
 -> reject poor split and fallback to base group
 -> collect candidates
 -> topk(base_method="group_then_adhesion", top_k=5)
```

### 11.3 默认参数

```python
cluster(
    defect_mask,
    valid_mask,
    method="group_then_adhesion",
    dilation_radius=1,
    min_area=3,
    suspicious_area=40,
    min_suspicious_cues=1,
    max_split_count=12,
    min_split_coverage=0.5,
    skip_ring_like=True,
)
```

TopK retrieval:

```python
clusters = cluster(
    defect_mask,
    valid_mask,
    method="topk",
    base_method="group_then_adhesion",
    top_k=5,
    dilation_radius=1,
)
```

等价快捷方法：

```python
clusters = cluster(
    defect_mask,
    valid_mask,
    method="topk_group_then_adhesion",
    top_k=5,
)
```

### 11.4 Historical commands

早期 `week1_*` 命令已废弃并删除。当前 proposal-based local retrieval 的运行、评估和 Top3 可视化统一使用 root pipeline：

```bash
python3 partial_match/run_proposal_retrieval_pipeline.py
```

`partial_match/scripts/` 下的 retrieval、evaluation 和 visualization 文件是 pipeline 调用的 helper，仍保留为可复用方法。

### 11.5 与已有方法的关系

```text
filtered:
  不额外切分，但碎片较多，topk 可能只选大碎片。

adhesion:
  能拆粘连，但作为 topk base 容易过切，coverage 下降。

dilated_group:
  能恢复断续 ring / scratch，但可能把整图合成 1 个大 group。

group_then_adhesion:
  先 group 后 selective split，目标是在 pattern-level group 和粘连拆分之间折中。
```

### 11.6 风险

```text
dilation_radius 太大:
  不相关 pattern 被合并。52x52 map 默认使用 1；只有断裂特别严重时再手动尝试 2。

ring-like guard 太强:
  一些 edge scratch 可能不会被拆。

adhesion split 太弱:
  group 仍然过大。

adhesion split 太强:
  又回到 ring / scratch 过切问题。
```
