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

`count-partial` 当前支持两种 proposal mode：

```text
cc       legacy 默认模式，仅使用 connected components
compact 试验模式，在 cc 基础上增加保守 compact 后处理
```

CLI 参数：

```text
--count-partial-proposal-mode {cc,compact}
```

默认值为 `cc`，因此不传该参数时保持旧逻辑。

另有一个独立的 shape descriptor 开关：

```text
--count-partial-rotation-tolerance
```

该开关不会改变 proposal 生成流程，也不会做全局位置旋转搜索。它只改变 shape descriptor：去掉绝对方向相关的 orientation 和 angular histogram，保留更稳定的形态统计与 radial profile，使 token shape 相似度对旋转更不敏感。

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

默认 `cc` proposal 生成使用 connected components：

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

### 4.1 Compact Proposal Mode

`compact` 模式保留 connected components 作为基础候选，但在进入 descriptor 和匹配前增加保守后处理：

```text
base connected components
 -> gap-aware grouping
 -> size-adaptive ring-aware token extraction
 -> geometry fragment merge
 -> type-diverse top-k selection
```

该模式不会改变 descriptor 计算，也不会改变 token pair 匹配公式。

#### Gap-aware Grouping

gap-aware grouping 用来处理小图中常见的 1-cell 断点。它只在 `compact` 模式下启用，且仅用于小图/中图：

```text
short_side <= 25
```

流程：

```text
original defect mask
 -> 3x3 square structuring element closing
 -> connected components on closed mask
 -> 每个 closed group 回取其中的 original defect pixels
 -> 用 original pixels 重新计算 token stats
```

关键点：

- closing 只用于 grouping。
- token 的 `area`、`mass`、PCA、descriptor 仍然只来自原始 defect pixels。
- closing 产生的虚拟填补像素不会进入 token pixels。
- 如果 closed group 中虚拟 gap 面积占比过高，则该 group 会被丢弃，回退到原始 token。

保护阈值：

```text
short_side <= 12:
  max_virtual_gap_ratio = 0.75

13 <= short_side <= 25:
  max_virtual_gap_ratio = 0.45

short_side > 25:
  disabled
```

其中：

```text
virtual_gap_ratio = virtual_gap_area / original_pixels_in_group
```

这样可以允许小图中一个像素的断点，同时避免 closing 把大量无关区域粘成一个 token。

#### Ring-aware

ring-aware 在 `compact` 模式下会对所有尺寸尝试启用，但参数会随 map 尺寸自适应：

```text
short_side = min(height, width)
```

小图不会跳过 ring extraction，而是使用更粗的 radial/angular bins、更低的最小 ring 支持点数和更宽松的 radial std 阈值。这样可以在 10x10 左右的小图上保留 edge-ring 的主动提取能力，同时避免照搬大图参数导致过度抖动。

ring-aware 会在 wafer edge band 中寻找稳定半径带，生成 `edge_ring` token，并从 residual components 中移除这些 ring pixels。保守判据包括：

```text
edge fraction
radial band support area
angular coverage
radial std
```

当前小图与中大图的核心差异：

```text
small map:
  edge_r_min 更低
  radial/angular bins 更少
  min_ring_points 更低
  band_width / max_radial_std 更宽松

medium/large map:
  使用更细的 radial/angular bins
  对 radial consistency 要求更严格
```

#### Geometry Fragment Merge

fragment merge 会合并相近且几何兼容的 token：

```text
edge_ring <-> edge_ring
line      <-> line      且方向接近
blob/central/irregular 之间的近邻 fragment
```

小图允许的 bbox gap 更小，大图允许更宽松的 gap。合并后重新计算 token stats 和 descriptor。

#### Type-diverse Top-K

`compact` 不只按 importance 截断，而是先尽量保留不同 geometry type 的强 token，再用 importance 填满剩余名额。这样可以降低单一高 mass WDM cluster 占满 top-k 的风险。

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

### 8.3 Rotation-Tolerant Descriptor

启用：

```text
--count-partial-rotation-tolerance
```

后，descriptor 会进入 rotation-tolerant 版本。该版本仍是手工 descriptor，不使用训练模型。

它保留：

```text
fill_ratio
log1p(aspect) / log(16)
log1p(elongation) / log(64)
min(compactness / 4, 1)
angular_coverage
radial_std
radial histogram
```

它去掉：

```text
cos(orientation)
sin(orientation)
angular histogram
radial_distance_norm
```

设计目的：

- 降低 token shape descriptor 对绝对方向的敏感度。
- 保留面积、长宽、细长度、紧密度和径向分布等旋转更稳定的形态信息。
- 不改变 `position_affinity`，因此整张图匹配仍然不是严格旋转不变；它只是 token shape 层面的旋转容忍。

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

## 10. Shape Gate 与 Type Soft Penalty

生产数据中发现小尺寸 WBM 下，位置和尺度容易掩盖形状差异。因此当前版本保留 shape gate：descriptor 明显不相似时，不允许靠位置接近拿高分。

`geometry_type` 仍由 proposal 阶段的启发式规则生成，并用于 proposal 排序、compact 多样性选择和匹配解释。但在 token matching 中，type 不再作为硬门槛；它只作为 soft penalty 乘到最终 pair score 上。

阈值：

```text
MIN_SHAPE_SIM_FOR_MATCH = 0.45
SHAPE_SCORE_POWER = 2.0
```

规则：

```text
if shape_sim < 0.45:
    pair_score = 0
else:
    pair_score = shape_sim^2 * position_affinity * scale_affinity * type_affinity
```

含义：

- descriptor 不相似时，不允许靠位置接近拿高分。
- shape 使用平方项，进一步放大形状差异。
- 几何类型不兼容时不会直接清零，但会通过较低的 `type_affinity` 降权。

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

当前 count-partial 和 classnumber review 可视化中：

- WBM 面板使用 WM811K 风格的状态色：背景黑色、晶圆内正常 die 灰色、失效 die 白色。
- WDM count / binary heatmap 使用晶圆外黑色、晶圆内由浅到深的红色热力图。count 模式的 colorbar 显示原始整数 defect count，不做 log 变换；binary 模式显示 0/1。

这样 WBM 保持与原始参考图一致，WDM 热力图保持统一红色系，避免 WBM/WDM 红蓝对比造成“不同类别”的误解。

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
