# Task Plan: Combinatorial Overlay Similarity Search for WBM-WDM Matching

## 研究定位

当前研究方向不再是传统的 WBM-to-WBM single image similarity ranking，也不再依赖公开数据集、WaPIRL 或固定 pattern 分类。

新的研究问题是：

```text
Given one target WBM A and a sequence of WDMs B_1...B_N,
find a sparse subset of WDMs whose overlay best explains A.
```

形式化定义：

```text
x* = argmax_x S(A, C(x))
x in {0,1}^N
C(x) = Normalize(sum_i x_i * T(B_i))
```

其中：
- `A` 是目标 WBM。
- `B_i` 是候选 WDM。
- `T(B_i)` 是 WDM 到 WBM grid 的配准、栅格化或密度映射。
- `C(x)` 是选中 WDM 的叠图。
- `S(A, C)` 是解释型相似度。

文章定位建议：

```text
A combinatorial overlay similarity search framework for explaining wafer bin maps
with a sparse subset of wafer defect maps.
```

中文表述：

```text
一种面向 WBM-WDM 跨数据源匹配的组合式叠图相似性搜索方法。
该方法通过覆盖-泄漏相似度、局部组件选择和预算约束 Beam Search，
从一组候选 WDM 中搜索能够解释目标 WBM 的最小充分组合。
```

---

## 核心创新点

### 1. 问题定义创新

现有论文主要研究：

```text
S(A, B): query one WBM, retrieve similar WBM
```

本研究扩展为：

```text
S(A, C(x)): query one WBM, retrieve a subset of WDMs whose overlay explains it
```

这是从单图检索到组合式解释匹配的转变。

### 2. Coverage-Leakage Similarity

传统 Dice、IoU、NCC、Hausdorff、BBS、WMHD 都不完全适合当前任务，因为当前任务不是判断两张图整体是否相似，而是判断：

```text
组合 C 是否覆盖 A 的失效区域
组合 C 是否在 A 外部产生过多无关泄漏
```

推荐基础得分：

```text
Score(A, C) = alpha * Coverage(A, C)
            - beta  * Leakage(C, A)
            - gamma * selected_count
```

其中：

```text
Coverage(A, C) = sum(min(A, C)) / (sum(A) + eps)
Leakage(C, A)  = sum(C outside Dilate(A)) / (sum(C) + eps)
```

解释：
- `Coverage` 奖励 B 组合解释 A 的部分。
- `Leakage` 惩罚 B 组合中落在 A 不需要区域的部分。
- `selected_count` 防止选中过多 WDM。

### 3. Region-Level Candidate Selection

整张 `B_i` 可能只有局部区域能解释 `A`，其他区域是无关缺陷。因此搜索对象不应只限于整张 WDM。

扩展建模：

```text
B_i = {B_i,1, B_i,2, ..., B_i,K_i}
C(x) = sum_{i,k} x_{i,k} * B_i,k
```

其中 `B_i,k` 可以是：
- connected component
- density cluster
- local region
- tensor-voting salient component
- DPGMM local cluster

约束：

```text
total selected components <= K
same original B_i can contribute limited number of components
```

### 4. Budgeted Beam Search

组合搜索空间巨大，不能直接穷举。推荐使用：

```text
Top-M prefilter + Beam Search + Marginal Gain Pruning
```

流程：

```text
1. 将所有 B_i 或 B_i,k 转成 chip-level density map
2. 用单体 coverage-leakage score 做 Top-M 预筛
3. 在 Top-M 内用 Beam Search 搜索组合
4. 每次加入候选时计算 marginal gain
5. 若 gain < delta，停止扩展
```

### 5. Constrained Geometric Registration

叠图仍可考虑旋转、缩放、平移，但应作为受限配准变量，而不是任意几何不变性。

扩展目标：

```text
x*, G* = argmax_{x,G} S(A, C(x, G))
```

推荐变换范围：

```text
rotation: [-5, 0, 5] degrees
scale: [0.95, 1.0, 1.05]
shift_x/y: [-1, 0, 1] chip
```

原则：
- 优先整体变换，而不是每张 B 独立变换。
- 不默认任意旋转不变，因为 wafer notch、die grid 和工艺方向有物理意义。
- 加入 transform penalty，防止大变换强行贴合。

---

## 推荐主方法框架

方法名称候选：

```text
COSS: Combinatorial Overlay Similarity Search
SEOM: Sparse Explanatory Overlay Matching
```

推荐整体流程：

```text
Input:
  target WBM A
  candidate WDM sequence B_1...B_N

Step 1: WDM-to-WBM Representation
  coordinate alignment
  chip-grid mapping
  density map generation
  optional morphology / mountain smoothing

Step 2: Candidate Decomposition
  whole-image candidate B_i
  region/component candidate B_i,k

Step 3: Similarity Scoring
  coverage reward
  leakage penalty
  shape similarity
  location similarity
  size/mass similarity
  sparsity penalty

Step 4: Search
  Top-M prefilter
  Beam Search
  marginal gain pruning

Step 5: Optional Constrained Geometry
  small global rotation / scale / shift search

Output:
  selected WDM combination
  selected regions/components
  overlay visualization
  score decomposition
```

---

## 推荐得分函数

完整版本：

```text
Score(A, C) = w_cov   * Coverage(A, C)
            - w_leak  * Leakage(C, A)
            + w_shape * ShapeSim(A, C)
            + w_loc   * LocationSim(A, C)
            + w_size  * SizeSim(A, C)
            - w_num   * SelectedCountPenalty
            - w_trans * TransformPenalty
```

各项来源：
- `Coverage/Leakage`：当前任务的解释性匹配核心。
- `ShapeSim`：可用 soft Dice、Tensor Voting、Mountain Function、WBBS。
- `LocationSim`：可用 centroid distance、region overlap、MoD-style location penalty。
- `SizeSim`：可用 defect mass、area、average radius。
- `SelectedCountPenalty`：稀疏组合约束。
- `TransformPenalty`：约束几何配准。

第一版建议简化为：

```text
Score(A, C) = Coverage(A, C) - beta * Leakage(C, A) - gamma * selected_count
```

第二版加入：

```text
LocationSim + SizeSim
```

第三版加入：

```text
Tensor Voting / Mountain / WBBS / WMHD reranking
```

---

## 可借鉴现有论文

### SIMI Ratio / Morphology

可借鉴点：
- `Closing + spatial filter` 快速预处理。
- `0 / 0.5 / 1` 三值图适合表达弱缺陷和强缺陷。
- 可作为第一版快速 baseline。

用途：

```text
WDM -> chip-level soft map
Top-M prefilter by SIMI-like score
```

### Hsu 2020: Mountain Function + WMHD

可借鉴点：
- Mountain Function 将离散 defect points 转成连续 density surface。
- WMHD 可以处理点集匹配。
- Outlier penalty 对应当前 leakage penalty。
- 邻域权重比单点权重更稳。

用途：

```text
Density transform
WMHD reranking
unmatched/outlier penalty design
```

### Lee 2021: DPGMM + HCM + JSD

可借鉴点：
- 将 WBM 分解为 local clusters。
- 用 global cluster weight vector 压缩表示。
- JSD 可用于快速比较分布。

用途：

```text
B_i component decomposition
region-level candidates
large-scale candidate indexing
```

### Wang 2023: Tensor Voting + WBBS

可借鉴点：
- Tensor Voting 提取 curve/region structural saliency。
- WBBS 对 outlier 和 partial matching 更鲁棒。
- `alpha` 惩罚点数差异，`mu` 控制位置权重。

用途：

```text
structure-aware reranking
partial matching score
noise removal
```

### Kang 2024: Shape/Location/Size + Entropy Weighting

可借鉴点：
- 将 similarity 拆成 shape、location、size 三个可解释分量。
- 用 entropy 自动给区分度高的分量更高权重。

用途：

```text
multi-component score design
adaptive weighting without labels
size/location consistency
```

---

## 实验计划

### Experiment 1: 组合是否优于单张

目的：证明 combinatorial overlay search 的必要性。

Baseline：

```text
Best single B
All B overlay
Random combination
Greedy combination
Beam search combination
Beam search + pruning
```

指标：

```text
top-1 / top-3 人工通过率
coverage score
leakage score
overall score
average selected count
runtime
```

### Experiment 2: Coverage-Leakage 是否优于传统相似度

比较：

```text
Dice
IoU
NCC
Hausdorff
BBS / WBBS
WMHD
MoD-like score
Coverage-Leakage score
```

目标：证明 coverage-leakage 更符合“解释 A”的工程判断。

### Experiment 3: 局部组件选择是否有效

Ablation：

```text
whole-B selection
component-level selection
component-level + leakage penalty
component-level + size/location consistency
```

目标：证明 component-level 能保留 B 的有用部分，并降低无关区域泄漏。

### Experiment 4: 搜索算法质量与效率

比较：

```text
Exhaustive search on small N
Greedy
Beam search
Beam search + marginal gain pruning
Genetic algorithm optional
```

指标：

```text
score gap to exhaustive optimum
runtime
selected count
stability under shuffled candidate order
```

### Experiment 5: 几何变换消融

比较：

```text
no transform
global shift only
global shift + rotation
global shift + rotation + scale
local transform per B optional
```

目标：证明受限整体变换足以处理配准误差，且不会引入过拟合。

---

## 无标签评估方案

### 1. Synthetic Recovery

从真实 WDM 中随机选若干个生成伪 A：

```text
x_gt = random sparse selection
A_fake = overlay(selected B) + blur + downsample + noise
run search to get x_pred
```

指标：

```text
precision / recall / F1 of selected B
IoU between C(x_pred) and A_fake
coverage / leakage
```

### 2. Engineer Audit

对生产数据输出 top-k 组合，让工程师判断：

```text
是否解释 WBM
是否包含明显无关 WDM
是否优于最佳单张
是否选择数量合理
```

指标：

```text
top-1 pass rate
top-3 pass rate
mean expert score
inter-rater consistency if multiple engineers
```

### 3. Historical Consistency

如果有 lot/process/time/equipment 信息，验证被选中的 WDM 是否在合理工艺窗口内更集中。

指标：

```text
selected WDM time-window concentration
process-layer consistency
equipment-path consistency
same-lot / nearby-lot enrichment
```

---

## 实现里程碑

### Milestone 1: 最小可行 baseline

目标：完成无学习、无局部组件、无几何变换的第一版。

任务：

```text
1. WDM -> WBM grid density map
2. overlay selected B
3. coverage-leakage score
4. top-M prefilter
5. beam search
6. visualization: A, best single B, best combination C
```

产出：

```text
baseline code
case visualization
runtime report
first ablation: single vs combination
```

### Milestone 2: 局部组件候选

任务：

```text
1. connected component extraction
2. component-level candidate generation
3. constraints for same original B_i
4. component-level beam search
```

产出：

```text
whole-B vs component-level comparison
leakage reduction analysis
```

### Milestone 3: 多分量相似度

任务：

```text
1. add location similarity
2. add size/mass similarity
3. add optional shape similarity
4. entropy-based adaptive weighting
```

产出：

```text
score decomposition visualization
ablation of each score component
```

### Milestone 4: 结构相似度增强

任务：

```text
1. implement mountain map or tensor voting transform
2. implement WBBS or WMHD reranking
3. compare with pixel-level score
```

产出：

```text
structure-aware reranking results
partial matching examples
```

### Milestone 5: 几何配准扩展

任务：

```text
1. global shift search
2. add small rotation and scale
3. transform penalty
4. geometry ablation
```

产出：

```text
registration robustness analysis
overfitting check under large transforms
```

### Milestone 6: 论文实验与写作

任务：

```text
1. synthetic recovery benchmark
2. production case study
3. expert audit
4. runtime and complexity analysis
5. method comparison table
6. paper figures and algorithm pseudocode
```

产出：

```text
full experimental results
paper draft
```

---

## 推荐论文结构

```text
1. Introduction
   - WBM-WDM matching motivation
   - limitation of single-image WBM similarity search
   - need for combinatorial explanatory matching

2. Related Work
   - WBM similarity search
   - image matching and partial matching
   - combinatorial optimization / maximum coverage

3. Problem Formulation
   - A, B sequence, overlay C(x)
   - sparse subset selection
   - optional geometry variable G

4. Proposed Method
   - WDM-to-WBM representation
   - candidate decomposition
   - coverage-leakage similarity
   - beam search
   - optional adaptive weighting and geometry search

5. Experiments
   - synthetic recovery
   - production data case study
   - baseline comparison
   - ablation studies
   - runtime analysis

6. Discussion
   - interpretability
   - limitations
   - extension to learned similarity

7. Conclusion
```

---

## 当前不建议作为主创新的方向

不建议主打：

```text
再设计一个单图 WBM similarity metric
```

原因：已有论文已覆盖很多单图 metric，例如 WMHD、BBS/WBBS、Tensor Voting、DPGMM/HCM、shape-location-size、SuperPoint/SuperGlue。

不建议第一篇主打：

```text
deep learning similarity / SuperPoint / SuperGlue
```

原因：
- 需要训练或伪标签。
- 解释性较弱。
- 与组合优化主线不如规则方法贴合。

深度模型适合后续作为 reranking 或 learned similarity。

---

## 一句话总结

最有价值的研究方向不是继续优化 `S(A, B)`，而是提出并验证：

```text
Find a sparse subset of WDMs whose overlay best explains the target WBM.
```

围绕该目标，优先完成：

```text
coverage-leakage score
region-level candidate decomposition
budgeted beam search
constrained geometric registration
```
