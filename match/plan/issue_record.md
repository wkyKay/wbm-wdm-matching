# Issue Record: Density Map Match vs Visual Pattern Mismatch

## 问题背景

在生产数据测试中，当前 WBM-WDM 匹配方法使用 `density map` 作为主要表达。实验观察到：

- 从高亮和 overlay 结果看，选中的 WDM density map 与目标 WBM 在空间位置上是匹配的。
- 但从工程师肉眼观察原始图或常规 wafer pattern 时，二者的 pattern 分布并不明显相似。
- 因此很难判断当前结果到底是“准”还是“不准”。

该问题说明：当前 `density map` 的得分结果与人工视觉判断之间存在解释差异。

---

## 当前现象

具体表现为：

```text
Density / overlay 视角：
  WDM 的高密度区域能够覆盖 WBM 的主要失效区域。

Engineer visual pattern 视角：
  原始 WDM 散点图与 WBM fail pattern 的整体形状、轮廓、走向或分布形态并不相似。
```

因此，density-based score 可能给出较高排名，但工程师会觉得该 WDM 与 WBM “不像”。

---

## 初步判断

这不是简单的算法错误，而是评价目标不一致：

```text
Density map 匹配的是：
  缺陷质量 / 密度是否落在 WBM 失效区域。

工程师肉眼判断的是：
  wafer pattern 的形状、轮廓、走向、局部结构是否相似。
```

也就是说，当前方法主要回答：

```text
这个 WDM 的缺陷密度能否解释 WBM 的失效位置？
```

但工程师同时在判断：

```text
这个 WDM 看起来是否具有与 WBM 类似的 pattern morphology？
```

二者不完全等价。

---

## 可能原因

### 1. Density map 强调位置和质量覆盖

`density map` 会保留 defect count / defect mass 信息，因此高密度区域只要落在 WBM failure 区域附近，就可能获得较高得分。

这适合衡量：

- coverage
- leakage
- defect mass concentration
- spatial explanation

但它不一定能表达：

- pattern 轮廓
- 缺陷走向
- ring / scratch / edge / cluster 结构
- 局部连通形态

### 2. 平滑或归一化可能弱化结构差异

如果 density map 使用归一化、Gaussian smoothing、neighbor spreading 或 mountain transform，局部散点结构会被平滑成连续热区。

这会提升配准鲁棒性，但也可能导致：

- 不同形状的 pattern 被映射成相似的 density blob。
- 结构差异被弱化。
- 肉眼可见的不匹配在 density score 中不明显。

### 3. 跨源数据本身存在表达差异

WBM 是 chip-level fail map，WDM 是 process-level defect scatter map。二者不是同一种图像来源。

因此，WDM 不一定需要在视觉上完全像 WBM，才可能对 WBM 有解释意义。

但如果完全忽略视觉 pattern，相似度又会与工程判断脱节。

---

## 风险

如果继续只使用 density-based score，可能产生以下问题：

- Top-K 结果在数值上合理，但人工审核通过率不稳定。
- 工程师难以信任结果，因为视觉 pattern 不直观。
- 论文实验中难以解释“为什么高分结果看起来不像”。
- 方法可能偏向选择高密度覆盖候选，而不是结构上真正相似的候选。

---

## 处理方向

### 1. 将 density map 定位为 prefilter

不建议否定 density map，而是调整其角色：

```text
Stage 1: density / coverage-leakage score 做 Top-K prefilter
Stage 2: shape / structure-aware score 对 Top-K rerank
```

这样可以保留 density map 对跨源空间解释的优势，同时避免它单独决定最终排序。

### 2. 增加多分量可解释评分

最终得分建议拆成：

```text
Score(A, B) = w_cov   * Coverage(A, B)
            - w_leak  * Leakage(B, A)
            + w_shape * ShapeSimilarity(A, B)
            + w_loc   * LocationSimilarity(A, B)
            + w_size  * SizeSimilarity(A, B)
```

其中：

- `Coverage / Leakage` 解释 density 是否合理。
- `ShapeSimilarity` 解释肉眼 pattern 是否相似。
- `LocationSimilarity` 解释位置是否一致。
- `SizeSimilarity` 解释缺陷规模是否接近。

当出现 density 高但肉眼不像时，可以通过 score decomposition 解释为：

```text
Coverage 高，但 ShapeSimilarity 低。
```

### 3. 增加结构感知 reranking

可考虑的结构特征包括：

- connected component 形状
- contour / boundary similarity
- skeleton / principal direction
- region-level IoU
- tensor voting saliency
- WBBS / MBBS partial matching
- WMHD / Chamfer distance
- shape-location-size similarity

第一版可优先实现简单结构指标，例如：

```text
binary / three-value map + connected component + centroid + area + bounding box + orientation
```

不必一开始就实现复杂深度模型。

### 4. 改进可视化输出

人工审核时不应只展示 density heatmap。建议每个 Top-K case 输出：

```text
1. target WBM binary/fail map
2. mapped WDM density map
3. mapped WDM binary or three-value map
4. overlay visualization
5. coverage region highlight
6. leakage region highlight
7. residual map: WBM missed by WDM
8. score decomposition table
```

这样工程师可以区分：

- 位置是否解释得通
- 密度是否集中在失效区
- 是否存在明显无关散点
- 形状是否真的相似

### 5. 调整人工审核标准

人工审核不建议只问“像不像”，而应拆成多个问题：

```text
1. 主要失效区域是否被 WDM 覆盖？
2. WDM 是否在 WBM 外部产生明显无关缺陷？
3. 缺陷位置是否合理？
4. pattern 形状是否相似？
5. 缺陷规模是否接近？
6. 工程上是否认为该 WDM 可解释该 WBM？
```

这样可以避免 density match 与 visual pattern match 被混成一个主观判断。

---

## 推荐实验补充

为验证该问题，建议增加一个 failure / disagreement analysis：

```text
Group A: density score 高，工程师认为 pattern 相似
Group B: density score 高，工程师认为 pattern 不相似
Group C: density score 低，但工程师认为 pattern 相似
Group D: density score 低，工程师认为 pattern 不相似
```

重点分析：

- Group B 是否由 shape mismatch 导致。
- Group C 是否由轻微错位、低密度但结构相似导致。
- shape/location/size 分量能否提升人工审核一致性。

---

## 当前结论

当前现象不应解释为 `density map` 完全无效，而应解释为：

```text
Density map 对 WDM-WBM 跨源匹配中的空间密度解释是合理的，
但它偏向 defect mass / coverage matching，
不能充分表达工程师肉眼关注的 pattern morphology。
```

因此，下一步建议采用：

```text
density prefilter + structure-aware reranking + score decomposition + improved visualization
```

这既能保留当前 density 方法的逻辑合理性，也能解释为什么它与人工视觉判断存在差异。

---

## 对论文叙事的影响

该问题可以作为方法设计动机之一：

```text
Single density-based matching is insufficient for production WBM-WDM retrieval,
because it captures spatial mass consistency but ignores morphology consistency.
Therefore, an interpretable multi-component similarity framework is needed.
```

中文表述：

```text
单一 density 匹配只能刻画缺陷密度在空间上的覆盖关系，
无法充分表达工程师关注的 pattern 形态一致性。
因此，需要将 coverage-leakage 与 shape、location、size 等可解释分量结合。
```
