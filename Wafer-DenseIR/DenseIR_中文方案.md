# Wafer-DenseIR 中文方案

## 1. 目标

本方案面向 MixedWM38K 晶圆图检索任务。给定一张 query wafer bin map，在候选 wafer map 库中找出空间缺陷分布最相似的 Top-K 样本。

核心目标：

- 不依赖人工 proposal 或 hard cluster 分割。
- 保留局部缺陷结构的可解释性。
- 通过 dense feature token matching 衡量 map 之间的局部对应关系。
- 标签只用于最终检索评价，不参与自监督训练和相似度计算。

整体流程：

```text
MixedWM38K wafer maps
-> WaPIRL-style 自监督预训练 encoder
-> Dense feature map 提取
-> defect-band dense token 选择
-> proposal-free dense local matching
-> retrieval ranking
-> label-based retrieval metrics
-> correspondence heatmap explanation
```

## 2. 数据与输入表示

数据集使用 MixedWM38K，原始数据来自 `.npz` 文件：

```text
data/wm38k/Wafer_Map_Datasets.npz
```

每张 wafer map 是离散 bin map。当前代码会统一处理为：

```text
0: background
1: valid wafer die / normal bin
2: defect bin
```

默认输入尺寸：

```text
96 x 96
```

默认输入通道为 WaPIRL-style decoupled input：

```text
channel 0: defect mask
channel 1: valid wafer mask
```

这样做的原因是避免把 bin 值当成连续强度，同时让模型显式区分“缺陷位置”和“有效晶圆区域”。

## 3. WaPIRL-Style 自监督预训练

### 3.1 训练目标

预训练阶段不使用 label。训练目标是让同一张 wafer map 的两个不同增强视图在 embedding 空间中接近，让不同 wafer map 的表示远离。

形式上：

```text
x: 原始 wafer map
t(x): 增强后的同一 wafer map
f_theta: backbone encoder
g_phi: projection head
M: memory bank
L: NCE / contrastive loss
```

正样本：

```text
x 与 t(x)
```

负样本：

```text
memory bank 中其他 wafer map 的表示
```

当前新增训练入口：

```text
run_wapirl_pretrain.py
```

训练输出：

```text
checkpoints/wm38k/wapirl_pretrain/<backbone>/<timestamp>/best_model.pt
checkpoints/wm38k/wapirl_pretrain/<backbone>/<timestamp>/last_model.pt
```

DenseIR 检索阶段只加载 checkpoint 中的：

```text
backbone
```

projection head 只用于自监督训练，不用于后续 dense retrieval。

### 3.2 与原 WaPIRL 代码的差异

原 WaPIRL 代码主要面向 WM811K 文件夹式数据和分类预训练流程。当前 DenseIR 版本做了 MixedWM38K 适配：

```text
数据源:
  WM811K image folders -> MixedWM38K .npz

输入:
  原始 bin image -> defect/valid 两通道 decoupled input

增强:
  轻量 crop / sparse noise / 90度旋转

数据质量:
  支持过滤空图、valid mask 异常、极端 defect ratio、重复图

模型:
  复用 Wafer-DenseIR 内的 ResNet / ViT backbone

checkpoint:
  保存 backbone key，直接供 run_dense_retrieval.py 加载
```

因此当前实现不是逐行复刻原 WaPIRL，而是保留 WaPIRL 的自监督 NCE 思路，并针对 MixedWM38K 和 DenseIR retrieval 做了工程调整。

### 3.3 数据增强策略

当前默认增强：

```text
--augmentation crop_noise_rotate
```

包含三类轻量增强。

#### Crop

默认参数：

```text
--crop_min_scale 0.85
```

含义：

```text
随机裁剪 85% 到 100% 边长范围内的区域，再 resize 回原始输入尺寸
```

这样可以增强模型对轻微边界变化的鲁棒性，但不会像 aggressive crop 那样破坏主要缺陷结构。

#### Sparse Noise

默认参数：

```text
--noise_prob 0.002
```

只在 valid wafer 区域内做很低概率的 defect/normal 翻转，用于模拟少量 bin noise。

#### Rotate

当前不是任意角度旋转，而是：

```text
0 / 90 / 180 / 270 度
```

默认参数：

```text
--rotate_prob 0.5
```

使用 90 度倍数旋转的原因是 wafer bin map 是离散 map，任意角度旋转会引入插值伪影和新的非离散 bin 值。

如果工艺方位具有明确语义，可以关闭 rotate：

```bash
--augmentation crop_noise
```

更保守时只使用 crop：

```bash
--augmentation crop
```

### 3.4 数据质量过滤

MixedWM38K 数据量相对更小，组合 pattern 更复杂，因此噪声样本对自监督训练影响更明显。当前训练数据包装支持轻量过滤：

```text
min_defect_pixels
min_defect_ratio
max_defect_ratio
min_valid_ratio
max_valid_ratio
deduplicate
```

默认过滤目标：

- 去掉几乎没有缺陷的空图。
- 去掉 valid wafer mask 异常的样本。
- 去掉 defect ratio 极端异常的样本。
- 去掉完全重复的 wafer map。

该过滤只用于提升自监督表征质量，不改变 test retrieval 的 label 评价定义。

## 4. Backbone 与 Dense Feature Map

当前支持两类 backbone：

```text
ResNet
ViT
```

DenseIR 不使用分类 logits，也不使用 projection head。检索阶段使用 backbone 输出的 dense feature map。

### 4.1 ResNet Dense Feature Map

当前 ResNet forward：

```python
return self.layers(x)
```

默认 `resnet.18` 包含：

```text
block0
block1
block2
...
block8
```

当前 dense feature map 来自：

```text
最后一个 residual block，也就是 block8 输出
```

默认输入：

```text
2 x 96 x 96
```

默认输出：

```text
512 x 12 x 12
```

含义是每张 wafer map 被表示成 `12 x 12` 个 dense tokens，每个 token 是 512 维。

### 4.2 ViT Dense Feature Map

当前 ViT forward 逻辑：

```text
patch embedding
-> 加 CLS token
-> Transformer blocks
-> final LayerNorm
-> 去掉 CLS token
-> patch tokens reshape 成 dense feature map
```

因此 ViT dense feature map 来自：

```text
最后一个 Transformer block 之后，final LayerNorm 之后的 patch tokens
```

默认 `vit tiny`：

```text
patch_size = 16
embed_dim = 192
depth = 12
```

输入为 `96 x 96` 时：

```text
patch grid = 6 x 6
输出 = 192 x 6 x 6
```

## 5. DenseIR 检索流程

检索入口：

```text
run_dense_retrieval.py
```

输入：

```text
MixedWM38K test split
WaPIRL-pretrained backbone checkpoint
```

主要步骤：

```text
1. 加载 MixedWM38K 数据
2. 构建 ResNet 或 ViT backbone
3. 加载 WaPIRL 预训练 checkpoint 中的 backbone
4. 对每张 wafer map 提取 dense feature map
5. 根据 defect mask 选择 dense tokens
6. 对 query 和 candidate 做 dense local matching
7. 聚合 token-level score 得到 wafer-level similarity score
8. 对每个 query 排序候选样本
9. 保存 rankings、metrics 和 heatmap explanation
```

### 5.1 Token Selection

当前默认 token 选择模式：

```text
--token_mode defect_band
```

含义：

```text
先找到 defect mask
再做轻量 dilation
保留 defect 周边 band 内的 dense tokens
```

注意：

```text
dilation 只用于 token selection
不生成 hard cluster
不生成 proposal
不做 adhesion split
```

其他可选模式：

```text
defect:      只保留 defect 区域 token
defect_band: 保留 defect dilated band token
valid:       保留 valid wafer 区域 token
all:         保留全部 dense grid token
```

### 5.2 Dense Local Matching

对 query token `q_i` 和 candidate token `c_j`，pair score 为：

```text
s(q_i, c_j)
  = cosine(q_i.feature, c_j.feature)
    * exp(- ||pos_i - pos_j||^2 / sigma_pos^2)
```

其中：

```text
cosine similarity: 衡量局部特征相似
position affinity: 鼓励空间位置相近的局部结构对应
```

当前 wafer-level score：

```text
对每个 query token，取 candidate tokens 中 top-k 匹配
再对 query tokens 做加权平均
```

默认参数：

```text
--topk_tokens 5
--sigma_pos 0.35
```

## 6. Label-Based Evaluation

检索排序过程不使用 label。

Label 只在最后用于评价 ranking 质量。MixedWM38K 是 multi-label 数据，因此当前评价支持：

```text
binary relevance:
  query label 与 candidate label 至少有一个交集

Jaccard relevance:
  |label_q ∩ label_c| / |label_q ∪ label_c|
```

当前指标：

```text
Precision@K
Recall@K
NDCG@K
mAP
```

输出文件：

```text
rankings.csv
metrics.json
explanations/*.png
```

## 7. Heatmap Explanation

解释图来自 dense token correspondence，不来自 cluster proposal。

每个 query 的 top-1 candidate 会保存一张 2x2 解释图：

```text
query wafer map
top-1 candidate wafer map
query correspondence heatmap
candidate response heatmap
```

Query heatmap：

```text
每个 query token 的 best/top-k matching score
```

Candidate heatmap：

```text
被 query token 匹配到的 candidate token 响应强度
```

该解释方式可以直观看到模型认为 query 和 candidate 的哪些局部区域相互对应。

## 8. 中间层特征改进方案

当前第一版只使用 backbone 最后一层 dense feature map。这样做实现简单，语义最强，但空间分辨率较低。

对于 wafer dense retrieval，中间层可能有价值，因为任务更关注：

```text
局部缺陷结构
细小 scratch / loc
断裂 ring
局部空间对应
热力图可解释性
```

### 8.1 ResNet 中间层候选

默认 ResNet-18 输入 `96 x 96` 时可比较：

```text
block4 输出:
  128 x 48 x 48
  局部细节最多，但噪声也更多

block6 输出:
  256 x 24 x 24
  局部细节和语义强度较均衡

block8 输出:
  512 x 12 x 12
  当前默认，语义最强，但热力图较粗
```

优先建议做 ablation：

```text
block6 dense feature map
```

原因：

```text
相比 block8，空间分辨率提升一倍
相比 block4，语义抽象更强、噪声更少
```

### 8.2 ViT 中间层候选

当前 ViT 使用最后一个 Transformer block 后的 patch tokens。

可做 ablation：

```text
layer 4 patch tokens
layer 8 patch tokens
layer 12 patch tokens
multi-layer patch token fusion
```

ViT 的空间分辨率主要由 patch size 决定：

```text
patch_size = 16 -> 6 x 6 tokens
patch_size = 12 -> 8 x 8 tokens
```

因此 ViT 中间层实验需要同时考虑：

```text
Transformer depth
patch size
token grid resolution
```

### 8.3 多层融合

后续可以比较：

```text
last layer only
middle layer only
middle + last fusion
```

融合方式可以从简单方法开始：

```text
1. 分别计算 middle-layer score 和 last-layer score
2. 对两个 wafer-level scores 加权平均
```

示例：

```text
S = alpha * S_middle + (1 - alpha) * S_last
```

这样比直接 concat feature 更容易解释，也便于做 ablation。

## 9. 推荐实验顺序

第一阶段：跑通完整流程

```text
1. WaPIRL pretrain on MixedWM38K train split
2. DenseIR retrieval on test split
3. 输出 rankings / metrics / heatmaps
```

第二阶段：增强策略 ablation

```text
crop
crop_noise
crop_noise_rotate
no_quality_filter vs quality_filter
```

第三阶段：dense feature layer ablation

```text
ResNet block8 当前默认
ResNet block6
ResNet block4
block6 + block8 score fusion
```

第四阶段：matching 参数 ablation

```text
token_mode: defect / defect_band / valid
token_dilation: 0 / 1 / 2
topk_tokens: 1 / 5 / 10
sigma_pos: 0.2 / 0.35 / 0.5
```

第五阶段：ViT 对比

```text
vit micro
vit tiny
vit small
不同 transformer layer patch tokens
```

## 10. 当前结论

当前方案的核心假设是：

```text
WaPIRL-style 自监督训练可以让 encoder 学到 wafer map 的缺陷结构表征。
Dense feature map 保留空间局部信息。
Proposal-free token matching 可以替代不稳定的 hard cluster proposal。
Label 只用于评价，避免把任务退化成监督分类。
```

当前实现已经具备：

```text
MixedWM38K WaPIRL-style pretraining
Dense feature extraction
Dense local matching
Retrieval metrics
Correspondence heatmap explanation
```

下一步最值得做的改进是：

```text
ResNet block6 中间层 dense feature ablation
最后层与中间层 score-level fusion
增强策略和质量过滤 ablation
```
