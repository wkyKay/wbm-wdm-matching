# 512x512 生产 WDM 数据清洗方案

本文档说明当生产 WDM 已经被读入或栅格化为 `512x512` 图像后，进入 `MixedWa` 第二阶段 domain adaptation 前应进行的数据清洗流程。

清洗目标不是只保留分类器最容易识别的样本，而是筛出具有稳定空间 pattern、经过 `WDM -> pseudo-WBM` 退化后仍保留主要拓扑结构的样本。

## 总体流程

推荐 pipeline：

```text
512x512 WDM（全量加载）
  -> stage1 classifier 批量打分（可选，若提供 checkpoint）
  -> 逐张循环：
        -> 解析 defect map / wafer mask
        -> 基础有效性检查
        -> 缺陷密度过滤
        -> 轻量形态学处理（仅用于连通域统计）
        -> 连通域统计
        -> 空间结构指标计算
        -> pseudo-WBM 生成
        -> pseudo-WBM 退化检查
        -> 结合 stage1 分数进行 high / medium / low 分组
```

> stage1 classifier 放在逐张循环之前执行是工程上的批量推理优化——GPU 对全量数据做 batch inference 效率远高于逐张推理。stage1 分数仅作为辅助信号，分组主依据仍然是几何结构指标和 pseudo-WBM 质量。

最终建议进入 domain adaptation 的数据比例：

```text
70%~80% high-confidence pattern WDM
20%~30% medium-confidence / weak pattern WDM
low-confidence 不进入训练
```

## 1. 基础有效性清洗

第一步排除明显不能训练的图，包括无缺陷、异常全满、坐标错位或 wafer mask 异常样本。

建议检查：

| 指标 | 目的 |
|------|------|
| 是否全 0 | 去掉无缺陷图 |
| 是否几乎全满 | 去掉异常扫描或全片污染图 |
| 缺陷点数量 | 去掉极稀疏或异常密集样本 |
| 缺陷密度 | 控制有效 pattern 范围 |
| wafer mask 是否存在 | 避免把背景误当晶圆区域 |
| wafer 外缺陷比例 | 发现坐标错位或无效样本 |

核心指标：

```text
defect_density = defect_pixels / wafer_area
outside_ratio = defect_pixels_outside_wafer / total_defect_pixels
```

初始过滤建议：

```text
defect_density < 0.00005  -> 过滤或降权
defect_density > 0.30     -> 过滤或人工检查
outside_ratio 过高        -> 过滤
wafer_area 太小           -> 过滤
```

实际阈值不应一次定死。建议先统计全量生产 WDM 的分布，再使用分位数过滤，例如过滤最低 `1%~5%` 和最高 `1%~5%` 的异常样本。

## 2. 连通域与碎片清洗

该步骤用于区分结构性 pattern 和纯随机散点/扫描噪声。

建议先做轻量形态学处理，再计算连通域：

```text
binary defect map
  -> morphology closing / dilation 小核
  -> connected components
```

推荐统计：

| 指标 | 含义 |
|------|------|
| `num_components` | 连通域数量 |
| `max_component_area` | 最大连通域面积 |
| `max_component_ratio` | 最大连通域占全部缺陷面积比例 |
| `small_component_ratio` | 小碎片占全部缺陷面积比例 |
| `component_area_entropy` | 连通域面积是否极度碎片化 |

过滤或降权倾向：

```text
num_components 极多 且 max_component_ratio 很低
small_component_ratio 很高
max_component_area 太小
```

注意不要把所有碎片型样本一刀切删除。真实 `random` 类或点状缺陷也可能有工程意义，因此更建议将此类样本放入 `medium-confidence` 或 `weak/uncertain` 组，而不是全部丢弃。

## 3. 空间分布与 Pattern 结构清洗

该步骤判断 WDM 是否具有可用于匹配的 wafer-level 空间结构。

建议计算：

| 指标 | 用途 |
|------|------|
| 缺陷质心 | 判断 center / edge-loc / loc |
| 距离中心的径向分布 | 判断 edge-ring / donut / center |
| 径向直方图峰值 | 判断环状 pattern |
| 主方向 PCA | 判断 scratch-like 线状结构 |
| bounding box 长宽比 | 判断 scratch 或局部 cluster |
| eccentricity / elongation | 判断形状方向性 |
| edge defect ratio | 判断边缘型缺陷 |
| 局部窗口最大密度 | 判断 localized cluster |

推荐保留具有以下任一结构证据的样本：

```text
中心聚集
或边缘聚集
或环状分布
或局部大连通域
或明显线状结构
或高密度局部 cluster
```

如果所有结构指标都弱，且缺陷分布接近均匀随机散点，则应降为 `low-confidence` 或只保留极少量作为噪声鲁棒性样本。

## 4. Pseudo-WBM 生成与退化检查

第二阶段 domain adaptation 的正样本来自：

```text
WDM -> pseudo-WBM
```

因此必须检查生成的 pseudo-WBM 是否仍保留有效 pattern。若 pseudo-WBM 退化为全 0、全 1 或无结构碎片，则该样本不适合作为 NCE 正样本对。

推荐生成流程：

```text
512x512 WDM
  -> morphology closing
  -> gaussian blur
  -> downsample 到 11x11
  -> Otsu / adaptive threshold
  -> upsample 到 96x96
```

需要检查的指标：

| 指标 | 风险 |
|------|------|
| pseudo-WBM 全 0 | pattern 在退化后消失 |
| pseudo-WBM 全 1 | 阈值或密度异常 |
| fail die 数太少 | 训练信号太弱 |
| fail die 数太多 | 接近 near-full 或异常全片污染 |
| pseudo-WBM 与原 WDM 质心偏差过大 | 退化过程不稳定 |
| pseudo-WBM 结构完全碎裂 | 正样本不可靠 |

初始规则建议：

```text
fail_die_count < 2      -> 过滤或降权
fail_die_count > 100    -> 检查是否 near-full，否则过滤
pseudo_density < 0.02   -> 过滤或降权
pseudo_density > 0.85   -> 过滤或降权
```

其中 `11x11` pseudo-WBM 总共 121 个 die，实际阈值应结合真实生产 WBM 的 fail die 分布调整。

## 5. Stage1 Encoder / Classifier 辅助筛选

将 `512x512` WDM 生成模型输入，例如 `96x96` 或 `224x224`，送入 stage1 encoder/classifier 得到 pattern 预测分数：

```text
pattern probabilities:
  p_center, p_donut, p_edge-loc, p_edge-ring,
  p_loc, p_random, p_scratch, p_near-full

derived scores:
  max_prob
  top-k labels
  prediction entropy
```

推荐使用方式：

```text
max_prob 高
或 top-2 / top-3 有明确 pattern
且 prediction entropy 较低
```

但不要只因为 stage1 classifier 低置信度就直接删除样本。stage1 模型主要在 WM38K WBM 上训练，真实 WDM 与 WBM 存在 domain gap。低置信度可能表示无 pattern，也可能只是跨域不适应。

stage1 分数应与几何规则、pseudo-WBM 退化检查联合使用。

## 6. High / Medium / Low 分组规则

推荐将清洗后的 WDM 分为三组：

| 分组 | 条件 | 用途 |
|------|------|------|
| high-confidence | 几何结构清楚，pseudo-WBM 正常，stage1 置信度高 | 主训练数据 |
| medium-confidence | 几何结构清楚，但 stage1 置信度一般；或结构较弱但 pseudo-WBM 正常 | 少量保留，用于真实噪声鲁棒性 |
| low-confidence | 几何无结构，pseudo-WBM 异常，stage1 置信度低 | 不进入训练 |

推荐采样策略：

```text
high-confidence: 全部保留
medium-confidence: 采样 20%~50%
low-confidence: 排除
```

如果 high-confidence 样本过少，可以适当增加 medium-confidence 比例，但不建议降低 low-confidence 的门槛来硬凑数量。

## 7. 不建议的做法

不建议只做：

```text
512x512 -> resize 到 96x96 -> stage1 classifier 低置信度就删除
```

原因是强下采样会损失真实 WDM 的细节，stage1 classifier 又存在 WBM-WDM domain gap，容易误删真实有效 pattern。

也不建议只靠连通域规则删除所有随机散点型图。真实 `random` 或点状缺陷可能仍有工程价值，更稳妥的方式是分层降权，而不是一刀切过滤。

## 8. 推荐输出

清洗脚本建议输出以下文件，方便复现实验和后续消融：

```text
cleaned_wdm.npz             # high + sampled medium，用于 domain adaptation
high_confidence_wdm.npz     # 仅 high-confidence
medium_confidence_wdm.npz   # medium-confidence 候选
rejected_wdm.npz            # 被过滤样本，供抽样检查
cleaning_metadata.json      # 每张图的指标、分组、过滤原因
preview/                    # 每组样本预览图
```

`cleaning_metadata.json` 至少应包含：

```text
sample_id
defect_density
num_components
max_component_area
max_component_ratio
small_component_ratio
centroid
radial_features
linearity_score
pseudo_fail_die_count
pseudo_density
stage1_top_labels
stage1_max_prob
stage1_entropy
confidence_group
reject_reason
```
