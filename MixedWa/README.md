# MixedWa — WBM/WDM 晶圆图匹配框架

基于多标签分类 + 位置感知对比学习的三阶段训练流程，实现晶圆图（WBM）与缺陷图（WDM）的 pattern 匹配。

## 设计思路

**为什么直接用 WM38K，不用 WM811K 预训练？**

WM38K 包含单类、两类、三类组合的多标签晶圆图（如 `center_edge-ring_loc`），标签空间与匹配任务完全一致。WM811K 是单标签数据集（9 类互斥），其 softmax 训练目标与多标签任务的 sigmoid 目标在 embedding 空间的组织方式上存在冲突，迁移收益有限。因此直接从 ImageNet 预训练权重出发，在 WM38K 上端到端训练。

## 目录结构

```
MixedWa/
├── configs/
├── datasets/
│   ├── datasets.py      # WM38KRaw / WM38KFromDir
│   ├── transforms.py    # WaferTransform（含 shift 模式用于位置感知）
│   └── loaders.py       # standard_loader
├── models/
│   ├── factory.py       # build_backbone() 工厂函数
│   ├── head.py          # LinearClassifier / MLPProjector
│   ├── resnet/backbone.py
│   └── vit/backbone.py
├── tasks/
│   ├── base.py
│   ├── train.py         # WM38KTrainer（主训练）
│   └── stage3.py        # Stage3DomainAdaptation（生产数据域适应）
├── utils/
│   ├── loss.py          # PositionAwareLoss / WaPIRLLoss
│   ├── metrics.py       # mAP / classification_metrics
│   └── logging.py
├── matching/
│   └── matcher.py       # WaferMatcher
├── run_train.py         # 主训练入口
├── run_stage3.py        # 域适应（可选）
└── run_matching.py      # 匹配推理
```

## 数据准备

```
data/
└── wm38k/
    └── Wafer_Map_Datasets.npz   # arr_0: (N,52,52) 晶圆图，arr_1: (N,8) 多热标签
```

WM38K 包含 38,015 张晶圆图，8 个缺陷类别：

| 索引 | 类别 |
|------|------|
| 0 | center |
| 1 | donut |
| 2 | edge-loc |
| 3 | edge-ring |
| 4 | loc |
| 5 | random |
| 6 | scratch |
| 7 | near-full |

目录结构示例（`--data_dir` 模式）：
```
data/wm38k/images/
├── center/
├── center_edge-loc/
├── center_edge-ring_loc/
├── donut/
├── donut_scratch/
└── ...
```

---

## 日志与 Checkpoint

每次训练后，`checkpoint_dir` 下产出：

```
checkpoints/train/
├── best_model.pt        # 验证集 mAP 最优的模型权重
├── last_model.pt        # 最后一个 epoch 的权重
├── config.json          # 完整训练参数
├── train_history.json   # 每 epoch 的 loss / mAP / f1 / exact_match / hamming_acc
├── test_history.json    # 测试集评估结果（含位置敏感性指标）
├── train.log            # 完整训练日志
└── tensorboard/         # TensorBoard 事件文件
```

**读取训练历史：**

```python
import torch, json

ckpt = torch.load('checkpoints/train/best_model.pt', map_location='cpu')
for ep in ckpt['history']:
    print(f"Epoch {ep['epoch']}: "
          f"valid_mAP={ep['mAP']['valid']:.4f} "
          f"valid_f1_macro={ep['f1_macro']['valid']:.4f} "
          f"valid_exact_match={ep['exact_match']['valid']:.4f}")

# 或直接读 JSON
with open('checkpoints/train/train_history.json') as f:
    history = json.load(f)
```

**TensorBoard：**
```bash
tensorboard --logdir ./checkpoints/train/tensorboard
```

---

## Backbone 选择

| backbone | 参数量 | out_dim | 适用场景 |
|----------|--------|---------|----------|
| `resnet18` | 11M | 512 | 默认，均衡选择 |
| `mobilenet_v3` | 2.5M | 576 | 显存受限 / 快速推理 |
| `efficientnet_b0` | 5.3M | 1280 | 精度优先 |
| `vit_tiny` | 5.7M | 192 | 全局 pattern 感知，推荐尝试 |
| `vit_small` | 1.2M | 96 | 轻量 ViT |
| `vit_timm` | 5.7M | 192 | 需 timm，ImageNet 预训练 |

---

## 训练流程

### 阶段一：WM38K 多标签分类（主训练）

从 ImageNet 预训练权重出发，在 WM38K 上训练多标签分类器。同时加入平移负样本，迫使 encoder 学习位置敏感特征。

```bash
# 推荐配置
python run_train.py \
  --npz_file ../../data/wm38k/Wafer_Map_Datasets.npz \
  --epochs 100 \
  --lr 1e-3 \
  --lr_scheduler plateau \
  --patience 15 \
  --checkpoint_dir ./checkpoints/train

# 使用 ViT-Tiny backbone
python run_train.py \
  --npz_file ../../data/wm38k/Wafer_Map_Datasets.npz \
  --backbone vit_tiny \
  --batch_size 128 \
  --checkpoint_dir ./checkpoints/train_vit

# 从图像目录读取
python run_train.py \
  --data_dir ../../data/wm38k/images \
  --checkpoint_dir ./checkpoints/train
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--npz_file` | — | Wafer_Map_Datasets.npz 路径（与 `--data_dir` 二选一） |
| `--data_dir` | — | 已处理图像目录 |
| `--backbone` | `resnet18` | backbone 类型，见上方表格 |
| `--pretrained` | `True` | 使用 ImageNet 预训练权重 |
| `--epochs` | 100 | 最大训练轮数（early stopping 可提前终止） |
| `--batch_size` | 128 | ViT 建议 64-128 |
| `--lr` | 1e-3 | 初始学习率 |
| `--lr_scheduler` | `plateau` | `plateau`=ReduceLROnPlateau，`cosine`=CosineAnnealingLR |
| `--patience` | 15 | early stopping 容忍轮数 |
| `--freeze_layers` | `[]` | 冻结的 backbone 层（默认全量微调） |
| `--pos_margin` | 0.5 | 位置感知 margin |
| `--pos_lambda` | 0.1 | 位置感知损失权重 λ |
| `--shift` | 0.3 | 平移幅度（图宽比例，正样本无空间增强时建议 ≥0.3） |
| `--in_channels` | 2 | 输入通道（2=解耦双通道，1=原始单通道） |
| `--img_size` | 96 | 输入图像尺寸（正方形） |
| `--dropout` | 0.3 | 分类头 dropout |
| `--checkpoint_dir` | `./checkpoints/train` | 权重保存目录 |

#### 输入解耦（双通道）

WBM 原始值域为 `{0, 1, 2}`（0=背景，1=正常，2=缺陷）。`--in_channels 2` 时将其解耦为：

- **channel 0（缺陷图）**：`clamp(x-1, 0, 1)`，缺陷格→1，其余→0
- **channel 1（存在掩码）**：`x>0`，非背景格→1

这使 encoder 能同时感知缺陷位置和晶圆有效区域边界。

#### 位置感知训练

训练时正样本**不做任何空间变换**（只 resize），对每张图额外生成一个大幅平移版本（`shift=0.3`，即平移 30% 图宽）作为负样本，通过 margin loss 推远平移版本的 embedding：

```
L = BCE(logits, label) + λ * max(0, margin - cosine_distance(z_orig, z_shift))
```

正样本不做 crop/shift 等空间变换，是为了避免与位置感知 loss 的信号冲突：若正样本也经过 crop（隐含位置变化），分类 loss 会要求 crop 后的 embedding 与原图相似，而位置感知 loss 要求位置变化后 embedding 不同，两者直接矛盾。

训练后 embedding 的余弦相似度隐含位置信息，可直接用于推理阶段的位置相似度计算。

#### 评估指标

训练过程中每个 epoch 记录以下指标（train/valid 各一份），测试集额外计算位置敏感性指标：

| 指标 | 含义 | 说明 |
|------|------|------|
| `mAP` | macro Average Precision | 与阈值无关，基于排序质量，主要监控指标 |
| `f1_macro` | macro F1 | 每类单独算 F1 后取均值，对稀有组合类敏感 |
| `f1_micro` | micro F1 | 全局 TP/FP/FN 汇总后计算，反映整体表现 |
| `exact_match` | Exact Match Ratio | 所有 8 个标签完全预测正确的样本比例，最严格 |
| `hamming_acc` | 1 - Hamming Loss | 单个标签位的平均正确率 |
| `shift_dist` | 平移 embedding 距离 | **仅测试集**，原图与平移 30% 版本的平均余弦距离，越大越位置敏感 |
| `shift_false_accept` | 位置误接受率 | **仅测试集**，平移版本相似度高于同类正确位置图的比例，越低越好 |

early stopping 和 checkpoint 保存均以 `valid_mAP` 为准（越高越好）。ReduceLROnPlateau 也监控 `valid_mAP`。

测试日志示例：
```
[Test] loss=0.4821 mAP=0.9038 f1_macro=0.8712 f1_micro=0.9103 exact_match=0.7654 hamming_acc=0.9521
[Test] shift_dist=0.3421 shift_false_accept=0.1823
```

---

### 阶段二：生产数据自监督域适应（可选）

使用无标签生产 WDM 数据进行域适应。自动生成伪 WBM 作为正样本对，用 WaPIRL NCE Loss 微调 backbone 后两层。

> 若生产数据与 WM38K 分布接近，可跳过此阶段直接进行匹配推理。

```bash
python run_stage3.py \
  --wdm_npz ../../data/production/wdm.npz \
  --stage2_ckpt ./checkpoints/train/best_model.pt \
  --epochs 30 \
  --batch_size 64 \
  --num_negatives 2000 \
  --checkpoint_dir ./checkpoints/stage3
```

**伪 WBM 生成流程（模拟缺陷→芯片失效的空间平均过程）：**

```
WDM（96×96）
  → 形态学闭运算（填充稀疏缺陷点）
  → 高斯模糊（模拟芯片级空间平均效应）
  → 下采样到 11×11（模拟 WBM 芯片级分辨率）
  → 二值化（Otsu 阈值）
  → 上采样回 96×96
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--wdm_npz` | — | 生产 WDM npz（arr_0 为 (N,H,W) 数组） |
| `--stage2_ckpt` | — | 主训练阶段 checkpoint（run_train.py 产出） |
| `--epochs` | 30 | |
| `--batch_size` | 64 | |
| `--num_negatives` | 2000 | 记忆库负采样数 |
| `--memory_weight` | 0.5 | 记忆库 EMA 更新系数 |
| `--temperature` | 0.07 | NCE Loss 温度 |
| `--img_size` | 96 | 输入图像尺寸 |
| `--checkpoint_dir` | `./checkpoints/stage3` | |

---

## 匹配推理

### 单次推理

```bash
python run_matching.py \
  --wbm_path ../../data/production/wbm_sample.png \
  --wdm_dir  ../../data/production/wdm_images/ \
  --stage2_ckpt ./checkpoints/train/best_model.pt \
  --top_k 3
```

### 批量评估（WM38K 测试集 top-3 准确率）

```bash
python run_matching.py \
  --eval_npz ../../data/wm38k/Wafer_Map_Datasets.npz \
  --stage2_ckpt ./checkpoints/train/best_model.pt
```

### 匹配得分

```
最终得分 = α × 重叠率 + β × 位置相似度 + γ × 面积相似度

重叠率     = |S_wdm ∩ S_wbm| / |S_wdm|        # pattern 类型一致性
位置相似度 = cosine(z_wbm, z_wdm)              # embedding 空间距离
面积相似度 = 1 - |area_wbm - area_wdm| / max   # pattern 大小一致性

过滤条件：重叠率 ≥ θ（默认 0.6），低于阈值直接排除
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage2_ckpt` | 必填 | 主训练 checkpoint |
| `--stage3_ckpt` | — | 域适应 checkpoint（可选，覆盖 backbone） |
| `--alpha` | 0.6 | 重叠率权重 |
| `--beta` | 0.2 | 位置相似度权重 |
| `--gamma` | 0.2 | 面积相似度权重 |
| `--theta` | 0.6 | 重叠率过滤阈值 |
| `--cls_threshold` | 0.5 | 多标签分类阈值 |
| `--top_k` | 3 | 返回前 k 个匹配结果 |
| `--img_size` | 96 | 输入图像尺寸 |

---

## 依赖

```
torch >= 1.10
torchvision >= 0.11
tensorboard
albumentations >= 1.3
opencv-python
scikit-image
scikit-learn
scipy
numpy
pandas
tqdm

# ViT-Timm backbone 额外依赖（可选）
timm >= 0.9.0
```
