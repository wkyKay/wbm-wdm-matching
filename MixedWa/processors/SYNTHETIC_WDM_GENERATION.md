# Synthetic WDM 生成方案

本文档说明当使用 WM811K / WM38K 的 WBM 作为原料生成 synthetic WDM 时，推荐采用的生成逻辑。重点回答：小尺寸 WBM 放大到 `512x512` 后，缺陷点应该在哪里出现。

## 核心结论

小尺寸 WBM 放大到 `512x512` 后，不应把整张 wafer 视为均匀采样空间。

更合理的做法是：

```text
WBM 小图
  -> 放大到 512x512
  -> 转换为 defect probability map
  -> 主要在原 WBM defect pattern 对应区域附近采样缺陷点
  -> 少量允许扩散到 pattern 边界外
  -> 极少量加入全 wafer 背景噪声
```

也就是说，synthetic WDM 的缺陷点主要来自 WBM 中原本存在缺陷的位置及其邻域，而不是每个 wafer 位置都有同等概率出现缺陷。

## 为什么不能全 Wafer 均匀采样

如果在 `512x512` 全 wafer 上均匀随机采样缺陷点，会破坏原始 WBM 的 pattern 拓扑。生成结果会更像 random noise，而不是与原 WBM pattern 对应的 WDM。

不推荐：

```text
resize WBM -> 512x512
  -> 在整张 wafer 内均匀随机撒点
```

这种方式会导致：

- center / edge-ring / scratch / loc 等结构被稀释；
- systematic pattern 变成 random pattern；
- 后续 `WDM -> pseudo-WBM` 难以恢复原始拓扑；
- domain adaptation 学到噪声和点密度，而不是 pattern 结构。

## 为什么也不能只在 Defect Mask 内采样

如果只在放大后的 defect mask 内采样，完全不允许外扩，生成结果会太像“放大 WBM 的像素子采样”，不够像真实 WDM。

也不推荐：

```text
resize WBM defect mask -> 512x512
  -> 只在 mask 内采样
  -> mask 外永远没有缺陷
```

这种方式的问题是：

- 缺陷边界过硬；
- 缺少真实 WDM 中常见的坐标扰动和空间扩散；
- 稀疏点云仍然保留 WBM 格子痕迹；
- 对真实 WDM 的 domain gap 仍然较大。

## 推荐：基于 Probability Map 的采样

推荐把 WBM defect mask 放大后作为一个低分辨率 pattern prior，而不是直接作为最终缺陷点。

推荐概率场：

```text
p(x, y) = α * resized_defect_mask
        + β * gaussian_blur(resized_defect_mask)
        + γ * background_noise_prior
```

其中：

- `resized_defect_mask`：原 WBM defect 区域放大后的核心 pattern 区；
- `gaussian_blur(resized_defect_mask)`：pattern 周围的扩散邻域；
- `background_noise_prior`：wafer 内极低强度的背景噪声先验。

推荐权重：

```text
α = 0.60
β = 0.35
γ = 0.05
```

含义：

| 区域 | 缺陷出现概率 | 作用 |
|------|--------------|------|
| 原 WBM defect 区域放大后的核心区域 | 高 | 保留原 pattern 拓扑 |
| defect 区域周围的模糊/扩散邻域 | 中 | 模拟 WDM 相对 WBM 的空间偏移和扩散 |
| wafer 其他区域 | 很低 | 模拟背景噪声、随机散点、次级弱缺陷 |

## 推荐采样比例

总体建议：

```text
80%~90% 缺陷点来自 pattern 核心区 + 扩散邻域
10%~20% 缺陷点来自背景噪声 / 次级扰动
```

对于 systematic pattern，例如：

- center
- edge-ring
- edge-loc
- scratch
- donut
- loc

背景噪声比例应更低：

```text
2%~8% background noise
```

如果生成 `systematic + random`，才提高背景散点比例。

## 推荐生成流程

完整流程：

```text
WM811K / WM38K WBM
  -> 提取 defect mask
  -> resize 到 512x512
  -> random affine：旋转 / 平移 / 缩放
  -> Gaussian blur 生成扩散邻域
  -> 构造 probability map
  -> 按 probability map 采样 defect points
  -> coordinate jitter
  -> local cluster diffusion
  -> random dropout
  -> background noise
  -> density control
  -> circular wafer mask 裁剪
  -> synthetic WDM 512x512
  -> 使用 process_wdm_512_cleaning.py 再清洗
```

注意：`512x512` 是 synthetic WDM 的生成与清洗尺度，不代表最终 encoder 一定用 `512x512` 输入。domain adaptation 仍可将清洗后的 synthetic WDM resize 到 `96x96` 进入现有模型。

## 与当前 sparse-mask 方案的关系

当前 `process_mixed_synthetic_wdm.py` 的主要逻辑是：

```text
resize source mask
  -> random affine
  -> sparse sample
  -> coordinate jitter
  -> local diffusion
  -> dropout
  -> background noise
  -> density control
```

该方案可以作为 baseline，但如果原料是小尺寸 WM811K WBM，建议升级为 probability-map 方案。

建议保留两种模式：

```text
--generation_mode sparse_mask      # 当前逻辑，作为 baseline
--generation_mode probability_map  # 推荐逻辑，作为主方案
```

## 推荐命令

从 WM811K `LSWMD.pkl` 生成 `512x512` synthetic WDM：

```bash
python processors/process_mixed_synthetic_wdm.py \
  --wm811k_pkl ../../data/wm811k/LSWMD.pkl \
  --output_dir ../../data/synthetic_wdm_mixed_wm811k_512 \
  --num_samples 5000 \
  --out_size 512 \
  --generation_mode probability_map \
  --save_preview
```

随后用统一清洗流程过滤 synthetic WDM：

```bash
python processors/process_wdm_512_cleaning.py \
  --wdm_npz ../../data/synthetic_wdm_mixed_wm811k_512/synthetic_wdm.npz \
  --wdm_format wbm_values \
  --output_dir ../../data/synthetic_wdm_mixed_wm811k_512_cleaned \
  --pseudo_grid_size 26 \
  --save_preview
```

## 推荐消融实验

为了证明生成方式有效，建议比较：

| 实验组 | 说明 |
|--------|------|
| no synthetic | 不使用 synthetic WDM |
| sparse_mask synthetic | 当前 sparse sample 方案 |
| probability_map synthetic | 推荐 probability map 方案 |
| clean real WDM | 只使用清洗后的真实 WDM |
| probability_map synthetic + clean real WDM | 合成 + 真实混合 |

如果 `probability_map synthetic` 优于 `sparse_mask synthetic`，说明基于概率场的生成方式更适合从小尺寸 WBM 构造 WDM-like 点云。
