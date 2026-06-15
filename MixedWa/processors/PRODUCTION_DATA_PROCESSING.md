# 原始生产 WDM 数据处理建议

本文档整理生产 WDM 原始数据进入 `MixedWa` 训练/域适应流程前的处理原则，重点回答三个问题：是否需要清洗、是否需要用 stage1 encoder 筛选有 pattern 的样本、是否需要混合合成数据与真实数据，以及超高分辨率 WDM 是否应直接缩放到 `96x96`。

## 核心结论

1. 生产 WDM 数据需要清洗，且清洗质量会直接影响第二阶段 domain adaptation 的效果。
2. 可以用 stage1 encoder/classifier 辅助筛选有 pattern 的真实 WDM，但不应把它作为唯一硬筛标准。
3. 可以混合 synthetic WDM 和 real WDM 训练，但真实生产数据应作为最终适应目标，合成数据主要用于补足 pattern 覆盖和鲁棒性。
4. 不建议把 `30000x30000` 原始 WDM 直接一步缩放到 `96x96` 后再做所有处理；应先在原始坐标或较高分辨率栅格上完成清洗，再生成模型输入。

## 为什么生产数据需要清洗

第二阶段 `run_domain_adapt.py` 使用无标签 WDM 做 WaPIRL/NCE 域适应，其正样本对来自：

```text
WDM -> pseudo-WBM
```

该阶段的目标不是重新学习类别，而是缓解真实 WDM 与 WBM-like 表示之间的 domain gap。它依赖一个关键假设：输入 WDM 本身具有明确、稳定的空间 pattern，并且经过 pseudo-WBM 生成流程后仍保留主要拓扑结构。

如果大量无结构样本进入第二阶段，例如纯随机散点、极稀疏孤立缺陷、扫描噪声、几乎全空或全满图，NCE 会把这些样本也当作有效正样本对，导致 encoder 更关注点密度或噪声纹理，而不是 pattern 语义。

## 推荐保留的数据

建议作为 domain adaptation 主要数据来源：

- center cluster
- edge-ring
- edge-loc
- scratch-like
- localized cluster
- donut-like
- near-full
- 结构清晰的 mixed-pattern WDM

可以少量保留：

- 弱 pattern WDM
- 边界不清但仍有空间聚集趋势的 WDM
- 分类器置信度一般但几何上有结构的 WDM

推荐比例：

```text
70%~80% 明显 pattern WDM
20%~30% 弱 pattern / 不确定 pattern WDM
```

## 不建议作为主要训练数据

以下样本应过滤或显著降权：

- 纯随机散点
- 极稀疏孤立缺陷
- 缺陷密度极高导致接近全满的图
- 几乎全空或几乎全满的图
- 扫描噪声或无结构背景缺陷
- 最大连通域过小、全是小碎片的图
- pseudo-WBM 后几乎全 0 或全 1 的样本
- stage1 classifier 对所有类别置信度都很低，且几何规则也显示无结构的样本

## 是否用 Stage1 Encoder 筛选 Pattern

可以使用 stage1 encoder/classifier 辅助筛选，但不建议只靠它硬筛。

stage1 模型是在 WM38K WBM 上做多标签监督训练得到的，学到的是 WBM pattern 语义。真实 WDM 和 WBM 之间存在 domain gap，因此 stage1 classifier 在真实 WDM 上置信度低，并不一定表示该 WDM 没有 pattern，也可能只是跨域差异导致识别失败。

推荐使用双重筛选：

```text
真实 WDM
  -> 无监督几何规则筛选
  -> stage1 classifier/encoder 打分
  -> 分成 high-confidence / medium-confidence / low-confidence
```

分组建议：

| 分组 | 条件 | 处理方式 |
|------|------|----------|
| high-confidence | 几何结构清晰，stage1 分类置信度较高 | 作为主要 domain adaptation 数据 |
| medium-confidence | 几何上有结构，但分类置信度一般 | 少量保留，降低权重或控制比例 |
| low-confidence | 几何无结构，pseudo-WBM 异常，分类器全低置信度 | 排除 |

## 原始 WDM 是否直接 Resize 到 96x96

不建议把 `30000x30000` 原始 WDM 直接一步缩放到 `96x96` 作为唯一表示。

`96x96` 与当前 `MixedWa` 的默认训练流程一致，适合快速 baseline，也便于复用 WM38K 上训练好的 stage1 encoder。但真实 WDM 原始分辨率很高，直接强下采样可能抹掉很多重要结构，尤其是：

- 细长 scratch
- 局部 cluster
- 稀疏但有空间趋势的缺陷点
- 小面积 edge-loc
- 多 pattern 组合中的次级 pattern

更推荐的流程是：

```text
原始 WDM 30000x30000
  -> 坐标/缺陷点解析
  -> 对齐 wafer 有效区域
  -> 栅格化到中间分辨率，例如 512x512 或 1024x1024
  -> 清洗、连通域、密度、形态学分析
  -> 生成 96x96 或 224x224 模型输入
```

## 清洗应该在 Resize 前还是 Resize 后

应分两层做。

### 第一层：原始坐标或高分辨率栅格清洗

这一步应在生成 `96x96` 模型输入之前完成。推荐在原始坐标、`512x512` 或 `1024x1024` 栅格上做。

适合检查：

- 缺陷点数量
- 缺陷密度
- wafer 有效区域覆盖情况
- 明显扫描异常
- 连通域数量
- 最大连通域面积
- 小碎片比例
- 缺陷是否集中在中心、边缘或局部区域
- scratch 是否具有线状结构
- 是否接近纯随机散点

### 第二层：模型输入尺度清洗

生成 `96x96` 或 `224x224` 后，需要再检查一次输入是否退化。

适合检查：

- resize 后是否几乎全 0
- resize 后是否几乎全 1
- pseudo-WBM 后是否几乎全 0 或全 1
- pattern 是否仍然可见
- stage1 classifier 是否所有类别置信度都极低

推荐完整流程：

```text
原始 WDM
  -> 高分辨率栅格化 512/1024
  -> 第一层清洗：几何 / 密度 / 连通域
  -> 生成模型输入 96 或 224
  -> 生成 pseudo-WBM
  -> 第二层清洗：退化检测 + stage1 置信度
  -> 用于 domain adaptation
```

## 散点型 WDM 的 Pattern 面提取

生产 WDM 往往是离散缺陷点云，缺陷点之间并不连通。即使肉眼能看出中心聚集、边缘环、局部带状或 scratch-like 趋势，直接做 connected components 仍会把它们判成大量小碎片。因此，不建议直接在原始散点图上判断 pattern 强弱。

更合理的做法是先把散点 WDM 转成连续的弱 pattern 面，再基于该 pattern mask 做结构指标、pseudo-WBM 和预览检查。

推荐流程：

```text
WDM defect points
  -> Gaussian blur / KDE density map
  -> percentile threshold 或 Otsu threshold
  -> morphology closing
  -> remove small components
  -> pattern mask
  -> 结构指标 / pseudo-WBM / preview
```

### 为什么不用单纯 connected components

原始 WDM 中每个缺陷点可能相隔较远，导致：

- `num_components` 虚高
- `small_component_ratio` 虚高
- `max_component_area` 偏低
- 明显弱 pattern 被误判为随机散点
- pseudo-grid 最近邻下采样时容易 aliasing，漏掉真实趋势或采到局部噪声

### 首选方法：Gaussian Density / KDE

把每个缺陷点看作一个小高斯核，对整张 wafer 生成连续密度场：

```text
defect_map
  -> gaussian_filter(sigma)
  -> density_map
```

然后从密度场中提取 pattern mask：

```text
threshold = percentile(density_map[wafer_mask], 90~95)
pattern_mask = density_map >= threshold
pattern_mask = morphology_closing(pattern_mask)
pattern_mask = remove_small_components(pattern_mask)
```

推荐初始参数：

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `density_sigma` | 6 / 10 / 16 | 512x512 WDM 上的高斯平滑尺度，建议多尺度对比 |
| `threshold_method` | percentile | 稀疏 WDM 优先用 percentile，Otsu 可作为备选 |
| `threshold_percentile` | 90~95 | 越高越保守，保留更核心的 pattern 区域 |
| `closing_kernel` | 3 或 5 | 连接相邻高密度区域，避免过度扩张 |
| `min_component_area` | 20~100 | 删除小噪声区域，按 512x512 分辨率调参 |

对于非常稀疏的 WDM，可以尝试：

```text
density_sigma = 12~24
threshold_percentile = 85~95
```

### 多尺度策略

单一平滑尺度可能不稳定。建议同时计算多个尺度的 pattern mask：

```text
sigma = [6, 10, 16]
```

如果某个空间趋势在多个尺度下都稳定出现，例如质心位置、径向分布或主方向一致，则说明该 pattern 更可信。多尺度结果可以用于：

- 提高 high-confidence 的可靠性
- 区分真实弱 pattern 和随机噪声
- 为 pseudo-WBM 选择更稳定的输入 mask

### Dilation / Closing 的定位

可以使用膨胀，但不建议只依赖单次 dilation：

- `dilation` 会无差别扩大所有点，容易把 random 散点也连成假 pattern
- `closing = dilation + erosion` 更适合填补近邻点之间的小空隙，同时不会无限扩张
- dilation/closing 应该作为 density mask 后处理，而不是唯一的 pattern 生成方法

推荐使用：

```text
density_map -> threshold -> closing -> pattern_mask
```

而不是：

```text
raw_points -> large dilation -> pattern_mask
```

### Preview 建议

清洗脚本应至少保存三类可视化，方便人工判断规则是否合理：

```text
原始 WDM 散点图
density / pattern mask 图
pseudo_grid_size x pseudo_grid_size die-level wafer 图
```

如果 high-confidence 中大量样本在 density/pattern mask 图上没有稳定结构，说明阈值或 high 分组规则过宽；如果原始散点肉眼可见 pattern，但 density mask 没有提取出来，则需要增大 `density_sigma` 或降低 percentile 阈值。

## 模型输入尺寸建议

第一版建议保持模型输入为 `96x96`，但清洗不要只在 `96x96` 上完成。

推荐第一版配置：

```text
清洗分辨率：512x512
训练输入：96x96
```

理由：

- 当前 `run_train.py` 默认 `img_size=96`
- WM38K 原始图只有 `52x52`，stage1 监督信号本身不支持特别高分辨率语义学习
- `96x96` 可以最大程度复用已有 supervised checkpoint、matching 逻辑和 CAM 参数
- 清洗阶段使用 `512x512` 能减少强下采样导致的信息丢失

可以做扩展实验：

| 实验 | 清洗分辨率 | 训练输入 | 目的 |
|------|------------|----------|------|
| A | 直接 96x96 | 96x96 | 最快 baseline |
| B | 512x512 | 96x96 | 验证高分辨率清洗收益 |
| C | 512x512 | 224x224 | 验证更大模型输入是否保留更多 WDM 结构 |

如果实验 C 明显优于 B，再考虑把 stage1 训练也迁移到 `224x224`。否则保持 `96x96` 更稳。

## Synthetic WDM 与 Real WDM 是否混合

可以混合，但建议分阶段或控制比例。

推荐顺序：

```text
阶段一：WM38K 监督训练
  学习 pattern 类型语义

阶段二-A：synthetic WDM 域适应
  让模型先适应 WDM-like 形态与 mixed-pattern 组合

阶段二-B：真实生产 WDM 域适应
  用清洗后的真实 WDM 做最终 domain adaptation
```

如果直接混合训练，推荐比例：

```text
真实 WDM：60%~80%
合成 WDM：20%~40%
```

`run_domain_adapt.py` 支持直接传入两路 npz：清洗后的真实 WDM 和 synthetic WDM。真实 WDM 会全量使用，synthetic WDM 按 `--synthetic_to_real_ratio` 从 synthetic pool 中随机采样后拼接进入 domain adaptation。该比例定义为：

```text
synthetic_count = round(real_count * synthetic_to_real_ratio)
```

例如 `--synthetic_to_real_ratio 0.25` 表示每 100 张真实清洗 WDM 搭配约 25 张 synthetic WDM。若 synthetic pool 不足，会有放回采样；若比例为 `0`，则只使用真实 WDM。

混合输入前，真实清洗 WDM 和 synthetic 清洗 WDM 的 npz 必须都是 `arr_0: (N,H,W)`，值域 `{0,1,2}`。两路数据的单样本尺寸 `H,W` 必须一致；若不一致，应先用 `process_wdm_512_cleaning.py --expected_size` 统一清洗输出尺寸。

示例：

```bash
python run_domain_adapt.py \
  --real_wdm_npz ../../data/production/wdm_512_cleaned/cleaned_wdm.npz \
  --synthetic_wdm_npz ../../data/synthetic_wdm_mixed_wm811k_512_cleaned/cleaned_wdm.npz \
  --synthetic_to_real_ratio 0.25 \
  --synthetic_seed 0 \
  --wdm_format wbm_values \
  --pseudo_grid_size 26 \
  --supervised_ckpt ./checkpoints/train/best_model.pt \
  --checkpoint_dir ./checkpoints/domain_adapt_real_synth
```

不要让合成数据长期占主导。最终目标是真实生产 WDM，合成数据主要用于补足 mixed-pattern、多 pattern 组合、稀有 pattern 和噪声鲁棒性。

当真实数据质量较差时，可以先使用：

```text
synthetic 70% + real 30%
```

当真实数据完成清洗后，再切换为：

```text
real 80% + synthetic 20%
```

## 推荐消融实验

为了验证真实数据清洗和合成数据混合是否有效，建议至少比较以下实验组：

| 实验组 | 数据 | 目的 |
|--------|------|------|
| baseline | 不做 domain adaptation | 判断 stage1 直接迁移能力 |
| synthetic only | 只用 synthetic WDM | 判断合成 WDM 是否提供有效跨域桥接 |
| real only raw | 未充分清洗的真实 WDM | 观察噪声数据是否损害 encoder |
| real only clean | 清洗后的真实 WDM | 判断真实域适应收益 |
| synthetic + clean real | 合成 WDM + 清洗真实 WDM | 判断混合训练是否优于单独真实数据 |

若 `synthetic + clean real` 优于 `real only clean`，说明合成数据补足了真实数据 pattern 覆盖。若 `real only clean` 最好，则应降低 synthetic 比例，把真实 WDM 作为主训练源。
