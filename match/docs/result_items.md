# Count-Partial / Classnumber Result Items

本文档说明运行 `match.scripts.main` 后，`--count-partial` 相关列、`--classnumber` 相关列、TopK TSV 和 review 图片上方分数的含义。

更完整的算法细节见 `count_partial_matching.md`。本文档只面向“结果解读”。

## 1. 输出文件

主命令通常会生成：

```text
batch_results.tsv
batch_topk.tsv
count_partial_review/
classnumber_review_*/
```

其中：

- `batch_results.tsv`: 每个 KLARF 文件一行，记录整图指标、count-partial 指标、classnumber 指标和映射成功数。
- `batch_topk.tsv`: 对每个可排序指标分别列出 Top-K 文件。
- `count_partial_review/topN_count_partial.png`: count-partial 排名前 N 的候选可视化。
- `count_partial_review/proposal_steps/rankXX_*.png`: 单个候选的 token 生成和匹配过程图。
- `classnumber_review_*/classnumber_topk.tsv`: 所有 classnumber 分图的全局排序明细。
- `classnumber_review_*/classnumber_topN.png`: classnumber 分图 Top-N 总览。
- `classnumber_review_*/topk_steps/rankXX_*.png`: classnumber 分图的 token 匹配过程图。

## 2. batch_results.tsv 通用列

| 列名 | 含义 |
|---|---|
| `file` | 输入 KLARF 文件名。 |
| `dice` | WBM 与 WDM representation map 的 Dice 相似度。越高越相似。 |
| `iou` | WBM 与 WDM representation map 的 Intersection over Union。越高越相似。 |
| `ncc` | Normalized Cross-Correlation。越高表示空间分布相关性越强。 |
| `cosine` | 展平 map 后的 cosine similarity。越高越相似。 |
| `coverage` | WDM 对 WBM 失效区域的覆盖程度。越高表示 WBM 失效区域更容易被 WDM 解释。 |
| `leakage` | WDM 落在 WBM 失效区域之外的泄漏程度。越高通常越差。 |
| `coverage-leakage` | 覆盖收益减去泄漏惩罚后的综合分。越高越好。 |
| `chamfer` | WBM/WDM 缺陷点集之间的倒角距离型相似度结果。日志中写入的是该方法返回的 score。 |
| `mapped_defects` | `mapped/input`，表示成功映射到 grid 的 defect 数 / KLARF 输入 defect 总数。 |

这些整图指标使用 `representation_map`，会受 `--representation` 影响；但 count-partial token 提取固定使用 WDM `count_map`，binary classnumber 使用 WDM `binary_map`。

## 3. Count-Partial 列

`count-partial` 是局部解释型分数：它衡量 WBM 上的 failure token 是否能被 WDM count-map 中的局部 defect token 解释。

| 列名 | 含义 |
|---|---|
| `count-partial` | 最终局部匹配分数，范围通常在 0 到 1 附近。越高表示 WDM 局部结构越能解释 WBM 失效结构。 |
| `count-partial-shape` | 所有 WBM token 最佳匹配的 shape similarity 加权平均。 |
| `count-partial-position` | 所有 WBM token 最佳匹配的位置亲和度加权平均。 |
| `count-partial-scale` | 所有 WBM token 最佳匹配的尺度亲和度加权平均。 |
| `count-partial-type` | 所有 WBM token 最佳匹配的几何类型亲和度加权平均。 |
| `count-partial-tokens` | `matched/wbm/wdm`，例如 `4/4/6` 表示 4 个 WBM token 都参与匹配，WBM 共 4 个 token，WDM 共 6 个 token。 |

计算路径：

```text
WBM status_map == VALID_HAS_DEFECT
 -> WBM proposal tokens

WDM count_map > 0 inside WBM valid_mask
 -> WDM proposal tokens

每个 WBM token 和所有 WDM token 计算 pair score
 -> 每个 WBM token 选择最高分 WDM token
 -> 按 sqrt(WBM token area) 加权平均
 -> count-partial
```

每个 token pair 的分数为：

```text
if shape_sim < 0.45:
    pair_score = 0
elif type_affinity < 0.6:
    pair_score = 0
else:
    pair_score = shape_sim^2 * position_affinity * scale_affinity * type_affinity
```

解释：

- `shape_sim`: token shape descriptor 的 cosine similarity。
- `position_affinity`: 质心位置越接近越高。
- `scale_affinity`: token 面积比例越接近越高。
- `type_affinity`: 几何类型越兼容越高，同类型为 1.0，兼容类型为 0.6，不兼容为 0.25。

最终四个分量列不是独立排序分数，而是最佳 token-pair 分量的加权平均，用于解释 `count-partial` 为什么高或低。

## 4. Classnumber 列

启用 `--use-classnumber` 后，每个 KLARF 的 DefectList 会按 `classnumber` 拆成多个 WDM 分图。每个分图分别和同一个 WBM reference 做局部匹配。

| 列名 | 含义 |
|---|---|
| `classnumber-count` | 当前 KLARF 中成功拆出的 classnumber 分图数量。 |
| `best-classnumber` | 在所有 classnumber 分图中，按指定 rank mode 得分最高的 classnumber。 |
| `best-classnumber-partial` | 最佳 classnumber 分图的 count-partial 分数。只在 `--classnumber-match-mode count/both` 时有值。 |
| `best-classnumber-tokens` | 最佳 classnumber 分图 count 模式下的 `matched/wbm/wdm` token 数。 |
| `best-classnumber-binary` | 最佳 classnumber 分图的 binary partial 分数。只在 `--classnumber-match-mode binary/both` 时有值。 |
| `best-classnumber-binary-shape` | binary partial 的 shape 分量加权平均。 |
| `best-classnumber-binary-position` | binary partial 的 position 分量加权平均。 |
| `best-classnumber-binary-scale` | binary partial 的 scale 分量加权平均。 |
| `best-classnumber-binary-type` | binary partial 的 type 分量加权平均。 |
| `best-classnumber-binary-tokens` | binary 模式下的 `matched/wbm/wdm` token 数。 |
| `best-classnumber-binary-coverage` | 兼容旧输出的预留列；当前 binary matching 不使用 coverage-leakage 计算，通常为空。 |
| `best-classnumber-binary-leakage` | 兼容旧输出的预留列；当前 binary matching 不使用 leakage 计算，通常为空。 |
| `best-classnumber-rank-mode` | 选择最佳 classnumber 时使用的排序依据，值为 `count` 或 `binary`。 |
| `best-classnumber-rank-score` | 最佳 classnumber 用于排序的实际分数。若 `rank-mode=count`，等于 `best-classnumber-partial`；若 `rank-mode=binary`，等于 `best-classnumber-binary`。 |

`count` 和 `binary` 的区别：

- `count`: WDM token 来自 `count_map > 0`，并使用 count 作为 token 权重。
- `binary`: WDM token 来自 `binary_map > 0`，token 权重统一为 1。

两者共享同一套 proposal、descriptor、shape/type gate 和最终加权平均公式。

## 5. classnumber_topk.tsv

`classnumber_topk.tsv` 汇总所有文件、所有 classnumber 分图的排序结果。

| 列名 | 含义 |
|---|---|
| `rank` | 在全局 classnumber 分图排序中的名次。 |
| `file` | 来源 KLARF 文件。 |
| `classnumber` | 当前分图对应的 classnumber。 |
| `rank_by` | 当前排序依据，`count` 或 `binary`。 |
| `rank_score` | 当前分图用于排序的分数。 |
| `count_partial` | 当前分图的 count-partial 分数。 |
| `shape` | count-partial 的 shape 分量。 |
| `position` | count-partial 的 position 分量。 |
| `scale` | count-partial 的 scale 分量。 |
| `type` | count-partial 的 type 分量。 |
| `tokens` | count 模式 token 数，格式 `matched/wbm/wdm`。 |
| `binary` | 当前分图的 binary partial 分数。 |
| `binary_shape` | binary partial 的 shape 分量。 |
| `binary_position` | binary partial 的 position 分量。 |
| `binary_scale` | binary partial 的 scale 分量。 |
| `binary_type` | binary partial 的 type 分量。 |
| `binary_tokens` | binary 模式 token 数，格式 `matched/wbm/wdm`。 |

## 6. count_partial_review 图片上方分数

### 6.1 topN_count_partial.png

每一行展示：

```text
Reference WBM tokens | Candidate WDM heatmap + tokens + matches
```

候选图下方会显示：

```text
score=... shape=... pos=... scale=... type=... tokens=matched/wbm/wdm
```

含义：

- `score`: `count-partial` 最终分数。
- `shape`: 最佳 token-pair 的 `shape_sim` 加权平均。
- `pos`: 最佳 token-pair 的 `position_affinity` 加权平均。
- `scale`: 最佳 token-pair 的 `scale_affinity` 加权平均。
- `type`: 最佳 token-pair 的 `type_affinity` 加权平均。
- `tokens`: 匹配 token 数，格式 `matched/wbm/wdm`。

这些值和 `batch_results.tsv` 中同名 `count-partial*` 列一致。

### 6.2 proposal_steps/rankXX_*.png

单张 step 图包含 5 个面板：

```text
WBM defects
WBM tokens
WDM count/binary
WDM tokens
Local matches
```

图片标题第二行显示：

```text
count-partial=... shape=... pos=... scale=... type=... tokens=...
```

即同一候选的完整 `LocalMatchResult`。即使 `map_mode=binary`，标题字段名仍沿用 `count-partial` 字样，但其数值来自 binary partial 的同一套 token matching 公式。

图中 token 轮廓颜色表示不同 token。`Local matches` 面板中：

- 虚线轮廓通常表示 WBM token。
- 实线轮廓表示 WDM token。
- 连线表示每个 WBM token 选择到的最佳 WDM token。

注意：当前不是 Hungarian 一对一匹配。多个 WBM token 可以连到同一个 WDM token。

## 7. classnumber 图片上方分数

### 7.1 `<file>_classnumber_splits.png`

该图展示：

```text
WBM
WDM all
class <id>
class <id>
...
```

每个 class 面板标题格式：

```text
class <classnumber>: <score> (<rank_by>)
```

含义：

- `<score>` 是该 classnumber 分图按 `rank_by` 选择的分数。
- 如果 `rank_by=count`，该值是该分图的 count-partial 分数。
- 如果 `rank_by=binary`，该值是该分图的 binary partial 分数。
- 标记 `BEST` 的面板是该 KLARF 内得分最高的 classnumber 分图。

### 7.2 classnumber_topN.png

该图是跨所有文件、所有 classnumber 分图的全局 Top-N。

每个面板标题格式：

```text
<file> / class <classnumber>
<score_mode>: <score>
```

含义：

- `<score_mode>` 是当前全局排序依据，`count` 或 `binary`。
- `<score>` 是当前分图在该模式下的 partial matching 分数。

### 7.3 topk_steps/rankXX_*_steps.png

这些图和 count-partial step 图结构相同，但 candidate 是某一个 classnumber 分图，而不是完整 WDM。

标题中的：

```text
count-partial=... shape=... pos=... scale=... type=... tokens=...
```

表示该 classnumber 分图与 WBM reference 的 token matching 结果。若当前按 binary 排序，底图和 token 来源使用 `binary_map > 0`；若按 count 排序，使用 `count_map > 0` 和 count 权重。

## 8. 如何解读高低分

常见情况：

- `score` 高、`shape` 高、`pos` 高：局部形状和位置都吻合，是较可信匹配。
- `shape` 高但 `pos` 低：形状类似，但位置偏移明显。
- `pos` 高但 `shape` 低：位置接近但形状不匹配；由于 shape gate，最终 `score` 往往不会高。
- `scale` 低：candidate token 面积和 WBM token 面积比例差异大。
- `type` 低：几何类型不兼容，pair score 会被 gate 置 0。
- `tokens` 中 WDM token 很少：candidate 结构过少，可能解释不了全部 WBM 局部区域。
- `tokens` 中 WDM token 很多：candidate 结构碎片多，可能噪声多或 proposal 过碎。

如果启用：

```text
--count-partial-proposal-mode compact
```

token 数和图中轮廓可能和默认 `cc` 不同，因为 compact 会尝试 ring-aware、fragment merge 和多样性 Top-K；但图片上方分数的含义和计算公式不变。
