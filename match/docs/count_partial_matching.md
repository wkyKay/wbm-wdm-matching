# Count-Partial Matching 技术说明

本文档记录当前 `count-partial` 匹配策略，从 proposal 生成、token 统计、descriptor 构造，到最终相似度计算与 Top-K 图片排序的完整流程。

## 1. 方法定位

`count-partial` 是 WBM-WDM 匹配中的局部解释型分数。它不直接比较整张图的逐像素重叠，而是比较：

> WBM 上的局部失效区域，是否能被 WDM count map 中的局部缺陷 cluster 解释。

输入包括：

- `reference`: WBM `GridMaps`
- `candidate`: WDM 映射后的 `GridMaps`
- WBM 使用 `status_map`
- WDM 使用 `count_map`

输出为 `LocalMatchResult`：

- `score`: 最终 count-partial 分数
- `mean_shape`: shape descriptor 平均相似度
- `mean_position`: 位置平均亲和度
- `mean_scale`: 尺度平均亲和度
- `mean_type`: 几何类型平均亲和度
- `matched_tokens`: 匹配到的 WBM token 数
- `wbm_tokens`: WBM proposal token 数
- `wdm_tokens`: WDM proposal token 数

## 2. 有效区域定义

所有 proposal 和匹配都以 WBM 的 `status_map` 为准定义有效区域：

```text
VALID_NO_DEFECT  = 晶圆内无失效 die
VALID_HAS_DEFECT = 晶圆内失效 die
```

有效区域为：

```text
valid_mask = VALID_NO_DEFECT or VALID_HAS_DEFECT
```

背景、未检测区域不参与 token 提取和 WDM count 统计。

WBM token 来源：

```text
wbm_mask = reference.status_map == VALID_HAS_DEFECT
```

WDM token 来源：

```text
wdm_count = candidate.count_map inside valid_mask
wdm_mask  = wdm_count > 0
```

如果使用 classnumber binary matching，则 WDM token 来源改为：

```text
candidate.binary_map > 0
```

其中 `binary_map` 由 `count_map >= --die-defect-threshold` 生成。默认阈值为 `1`，等价于旧的 `count_map > 0`；调高阈值可以过滤单个 die/cell 上缺陷数过少的噪声点。

## 3. 自适应 Proposal 配置

当前 proposal 不再使用固定配置，而是根据 WBM map 尺寸自动选择参数。该策略用于解决生产数据中 WBM 尺寸较小，例如 `10x10` 左右时，固定 `min_area=5` 和 8 邻域连通域过粗的问题。

配置由 `_proposal_config(shape, valid_area, min_area, top_k)` 生成。

### 3.1 小图配置

适用条件：

```text
min(height, width) <= 12
```

配置：

```text
adaptive_min_area = 2
adaptive_top_k = 4
connectivity = 4
descriptor_mode = coarse
```

设计原因：

- 小图中 5 个 cell 已经占很大比例，容易过滤掉真实小失效区域。
- 4 邻域可以减少对角接触导致的误合并。
- 小图上的方向角和细粒度 histogram 不稳定，因此使用 coarse descriptor。

### 3.2 中图配置

适用条件：

```text
13 <= min(height, width) <= 25
```

配置：

```text
adaptive_min_area = max(3, round(valid_area * 0.01))
adaptive_top_k = 6
connectivity = 8
descriptor_mode = normal
```

### 3.3 大图配置

适用条件：

```text
min(height, width) >= 26
```

配置：

```text
adaptive_min_area = max(5, round(valid_area * 0.005))
adaptive_top_k = 8
connectivity = 8
descriptor_mode = normal
```

### 3.4 与命令行参数的关系

用户传入的 `min_area` 和 `top_k` 仍然生效，但会与自适应配置合并：

```text
effective_min_area = min(user_min_area, adaptive_min_area)
effective_top_k    = min(user_top_k, adaptive_top_k)
```

因此自适应策略会降低小图的面积门槛，但不会突破用户指定的 proposal 数量上限。

当前版本没有加入 global token。

## 4. Proposal 生成

proposal 生成使用 connected components：

- WBM: 对 `wbm_mask & valid_mask` 提取连通域。
- WDM: 对 `(count_map > 0) & valid_mask` 提取连通域。

连通域支持两种模式：

```text
4-neighbor: up, down, left, right
8-neighbor: includes diagonal neighbors
```

流程：

1. 遍历 mask 中所有 true 像素。
2. 对未访问像素启动 BFS。
3. 根据 connectivity 扩展邻居。
4. 得到一个 component。
5. 过滤面积小于 `effective_min_area` 的 component。
6. 将 component 转成 token。
7. 按 importance 排序，保留前 `effective_top_k` 个。

## 5. Token 统计特征

每个 proposal component 会被转成一个 token。token 保存原始统计特征、几何类型、descriptor 和 proposal 配置。

主要统计特征包括：

```text
area
support_area
support_area_ratio
mass
mass_ratio
peak
mean_weight
centroid_row
centroid_col
pos
bbox_row_min
bbox_col_min
bbox_row_max
bbox_col_max
bbox_height
bbox_width
pca_lambda1
pca_lambda2
orientation
perimeter
compactness
radial_distance_norm
angular_coverage
radial_std
pixels
geometry_type
descriptor
proposal_config
```

### 5.1 权重

WBM token 的权重图来自二值 mask：

```text
weight = 1 for failed die
```

WDM token 的权重图来自 count map：

```text
weight = defect count in cell
```

因此 WDM token 的质心、mass、peak、mean weight 都会保留 defect count 信息。

### 5.2 PCA 与方向

token 内部会基于加权像素坐标计算协方差矩阵：

```text
cov = [[cov_rr, cov_rc],
       [cov_rc, cov_cc]]
```

然后对协方差矩阵做 eigen decomposition：

```text
pca_lambda1 = major eigenvalue
pca_lambda2 = minor eigenvalue
orientation = major eigenvector angle
```

这些特征用于判断线状形态、构造 descriptor。

### 5.3 Wafer 相对位置特征

以 map 中心作为 wafer 中心近似，计算：

```text
radial_distance_norm = token centroid 到中心距离 / valid region 最大半径
angular_coverage = component 像素覆盖的角度范围比例，5 度一个 bin
radial_std = component 像素径向距离标准差
```

这些特征用于识别 edge-ring、central 等形态。

## 6. 几何类型分类

`geometry_type` 是规则分类结果，不是学习模型。类别包括：

```text
blob
line
edge_ring
central
irregular
```

分类规则：

### 6.1 Edge Ring

满足以下条件时分类为 `edge_ring`：

```text
radial_distance_norm >= 0.65
angular_coverage >= 0.16
radial_std <= 0.14
```

含义：token 位于晶圆边缘附近，沿角向覆盖一定范围，并且径向厚度较薄。

### 6.2 Line

满足以下任一条件时分类为 `line`：

```text
elongation >= 6.0
aspect >= 4.0
```

其中：

```text
elongation = pca_lambda1 / pca_lambda2
aspect = max(bbox_height / bbox_width, bbox_width / bbox_height)
```

### 6.3 Blob

满足以下条件时分类为 `blob`：

```text
fill_ratio >= 0.45
compactness <= 1.6
```

其中：

```text
fill_ratio = area / bbox_area
compactness = perimeter / area
```

### 6.4 Central

满足以下条件时分类为 `central`：

```text
radial_distance_norm <= 0.35
```

### 6.5 Irregular

不满足上述规则时，归为 `irregular`。

## 7. Token Importance 排序

proposal 生成后，会按 importance 排序，保留前 `effective_top_k` 个。

公式：

```text
importance = sqrt(mass) + 0.25 * sqrt(area) + type_bonus
```

`type_bonus`：

```text
edge_ring = 2.0
central   = 1.5
blob      = 1.2
line      = 1.0
irregular = 0.8
```

该排序倾向保留：

- WDM 中 count mass 较大的 cluster
- WBM 中面积较大的 failure component
- edge-ring、central、blob 等更有解释价值的形态

## 8. Shape Descriptor

descriptor 是最终 `shape_sim` 的输入。当前有两种模式：

```text
normal
coarse
```

### 8.1 Normal Descriptor

normal 模式用于中图和大图。

基础形态特征 8 维：

```text
fill_ratio
log1p(aspect) / log(16)
log1p(elongation) / log(64)
min(compactness / 4, 1)
cos(orientation)
sin(orientation)
angular_coverage
radial_std
```

profile 特征 16 维：

```text
8-bin radial histogram
8-bin angular histogram
```

总维度：

```text
8 + 8 + 8 = 24
```

最后做 L2 normalization。

### 8.2 Coarse Descriptor

coarse 模式用于小图。

基础形态特征 7 维：

```text
fill_ratio
log1p(aspect) / log(16)
log1p(elongation) / log(64)
min(compactness / 4, 1)
radial_distance_norm
angular_coverage
radial_std
```

profile 特征 8 维：

```text
4-bin radial histogram
4-bin angular histogram
```

总维度：

```text
7 + 4 + 4 = 15
```

设计原因：

- 小图上 orientation 容易受 1-2 个 cell 波动影响，因此 coarse descriptor 去掉 `cos/sin(orientation)`。
- histogram bin 数减少，避免 10x10 左右图中 descriptor 过稀疏。
- 保留 fill、aspect、elongation、radial、angular 等粗形状信号。

## 9. Token Pair 相似度

每个 WBM token 会和每个 WDM token 计算 pairwise score。

分量包括：

```text
shape_sim
position_affinity
scale_affinity
type_affinity
```

### 9.1 Shape Similarity

使用 descriptor 点积：

```text
shape_sim = max(dot(query.descriptor, candidate.descriptor), 0)
```

因为 descriptor 已经 L2 normalization，该值等价于截断后的 cosine similarity。

### 9.2 Position Affinity

使用归一化质心坐标计算 RBF：

```text
pos_dist2 = ||query.pos - candidate.pos||^2
position_affinity = exp(-pos_dist2 / sigma_pos^2)
```

默认：

```text
sigma_pos = 0.35
```

### 9.3 Scale Affinity

使用 token 面积占有效区域比例：

```text
q_scale = query.support_area_ratio
c_scale = candidate.support_area_ratio
scale_affinity = exp(-abs(log(q_scale / c_scale)) / sigma_scale)
```

默认：

```text
sigma_scale = 1.5
```

### 9.4 Type Affinity

几何类型亲和度：

```text
same type = 1.0
compatible type = 0.6
otherwise = 0.25
```

兼容类型包括：

```text
line <-> irregular
blob <-> irregular
central <-> blob
```

## 10. 形状强约束

生产数据中发现小尺寸 WBM 下，位置和尺度容易掩盖形状差异。因此当前版本增加了形状强约束。

阈值：

```text
MIN_SHAPE_SIM_FOR_MATCH = 0.45
MIN_TYPE_AFFINITY_FOR_MATCH = 0.6
SHAPE_SCORE_POWER = 2.0
```

规则：

```text
if shape_sim < 0.45:
    pair_score = 0
elif type_affinity < 0.6:
    pair_score = 0
else:
    pair_score = shape_sim^2 * position_affinity * scale_affinity * type_affinity
```

含义：

- descriptor 不相似时，不允许靠位置接近拿高分。
- 几何类型完全不兼容时，直接置 0。
- shape 使用平方项，进一步放大形状差异。

这使 `count-partial` 更接近“形状先验约束下的位置/尺度匹配”。

## 11. WBM Token 到 WDM Token 的匹配

当前匹配不是 Hungarian matching，也不是一对一匹配。

流程：

1. 对每个 WBM token，遍历所有 WDM token。
2. 计算所有 pairwise token score。
3. 选择得分最高的 WDM token 作为该 WBM token 的解释。
4. 多个 WBM token 可以匹配到同一个 WDM token。

该策略的优点是简单、稳定、可解释。缺点是缺少一对一约束，可能出现一个强 WDM cluster 同时解释多个 WBM token 的情况。

## 12. 最终 Count-Partial Score

每个 WBM token 的最佳匹配分数按 WBM token 面积加权平均。

token 权重：

```text
weight = sqrt(wbm_token.area)
```

最终分数：

```text
score = sum(best_pair_score_i * weight_i) / sum(weight_i)
```

四个解释分量也使用同一组权重做加权平均：

```text
mean_shape
mean_position
mean_scale
mean_type
```

如果 WBM token 或 WDM token 为空，则：

```text
score = 0
matched_tokens = 0
```

## 13. Top-K 图片排序

`--save-count-partial-figures` 生成的图片只按 `count-partial` 分数排序。

相关参数：

```text
--count-partial-review-top-k
--count-partial-step-max
```

含义：

- `topN_count_partial.png`: 取 `count-partial` 分数最高的前 N 个候选。
- `proposal_steps/rankXX_*.png`: 取 `count-partial` 分数最高的前 M 个候选生成步骤图。

不会使用 `dice`、`iou`、`coverage-leakage`、`chamfer`，也不会融合多个指标。

## 14. 可视化颜色

当前 count-partial 可视化中：

- WBM failure 使用红色系。
- WDM count heatmap 也使用红色系。

这样 WBM/WDM 的视觉语义保持一致，避免红蓝对比造成“不同类别”的误解。

## 15. 当前限制

当前方案仍有以下限制：

- proposal 仍基于 connected components，没有 global token。
- 小图的形状 descriptor 信息量有限，coarse descriptor 只能缓解，不能完全解决。
- 类型分类是启发式规则，不是训练模型。
- token 匹配不是一对一约束。
- 对旋转不具备不变性。
- WDM 与 WBM 若存在明显错位，仍可能影响分数。

## 16. 后续可选改进

后续如果生产数据需要，可以考虑：

- 加入 global token 表示整图失效模式。
- 对 small map 使用模板级 shape category。
- 引入 Hungarian matching 避免重复解释。
- 对 edge-ring、line 单独设计 proposal 生成器。
- 加入轻量平移搜索或局部 offset tolerance。
- 将 shape/type 阈值暴露为 CLI 参数，便于按产品调优。
