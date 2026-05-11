# MixedWa — WBM/WDM 匹配训练框架

基于 WaPIRL 的三阶段训练流程，实现晶圆图（WBM）与缺陷图（WDM）的 pattern 匹配。

## 目录结构

```
MixedWa/
├── configs/
├── datasets/
│   ├── datasets.py      # WM811KRaw / WM38KRaw（直接读 pkl/npz）+ FromDir 版本
│   ├── transforms.py    # WaferTransform（含 shift 模式用于位置感知）
│   └── loaders.py       # balanced_loader / standard_loader
├── models/
│   ├── base.py          # BackboneBase / HeadBase
│   ├── factory.py       # build_backbone() 工厂函数，统一创建各类 backbone
│   ├── head.py          # LinearClassifier / MLPProjector
│   ├── resnet/
│   │   └── backbone.py  # ResNet18Backbone
│   └── vit/
│       └── backbone.py  # ViTTinyBackbone（纯 PyTorch）/ ViTTimmBackbone（需 timm）
├── tasks/
│   ├── base.py
│   ├── stage1.py        # 单标签分类
│   ├── stage2.py        # 多标签微调 + 位置感知损失
│   └── stage3.py        # WaPIRL 自监督域适应 + 伪 WBM 生成
├── utils/
│   ├── loss.py          # BCEWithLogitsLoss / PositionAwareLoss / WaPIRLLoss
│   ├── metrics.py       # mAP / subset_match_recall / top_k_accuracy
│   └── logging.py       # get_logger()（文件+控制台双输出）/ tqdm 配置
├── matching/
│   └── matcher.py       # WaferMatcher（重叠率 + 位置相似度 + 面积相似度）
├── run_stage1.py
├── run_stage2.py
├── run_stage3.py
└── run_matching.py
```

## 数据准备

原始数据放置于 `data/` 目录（相对于项目根目录）：

```
data/
├── wm811k/
│   └── LSWMD.pkl
└── wm38k/
    └── Wafer_Map_Datasets.npz
```

所有 run 脚本支持直接读取原始文件（`--pkl_file` / `--npz_file`），**无需预先运行 process 脚本**。若已用 WaPIRL 的 process 脚本生成图像目录，也可通过 `--data_dir` 指定。

---

## 日志与 Checkpoint

每个训练阶段结束后，`checkpoint_dir` 下会产出以下文件：

```
checkpoints/stage1/          # stage2/、stage3/ 结构相同
├── best_model.pt            # 验证集 loss 最优的模型权重
│                            #   包含字段：backbone、classifier/projector、
│                            #             optimizer、scheduler、epoch、history
├── last_model.pt            # 最后一个 epoch 的模型权重（同上）
├── train_history.json       # 所有 epoch 的完整训练指标
│                            #   格式：[{epoch, lr, loss:{train,valid}, acc/mAP:{train,valid}}, ...]
├── test_history.json        # 测试集评估结果（若提供了 test_set）
└── train.log                # 完整训练日志（追加模式，重启训练不覆盖）
```

**从 checkpoint 读取训练历史：**

```python
import torch, json

# 方式一：从 .pt 文件读取（无需重新训练）
ckpt = torch.load('checkpoints/stage1/best_model.pt', map_location='cpu')
history = ckpt['history']  # list of dicts
for ep in history:
    print(f"Epoch {ep['epoch']}: train_loss={ep['loss']['train']:.4f} "
          f"valid_loss={ep['loss']['valid']:.4f} lr={ep['lr']:.6f}")

# 方式二：直接读取 JSON（更轻量）
with open('checkpoints/stage1/train_history.json') as f:
    history = json.load(f)
```

**日志格式示例（train.log）：**

```
2026-05-09 10:23:01 [INFO] Stage1 started | epochs=100 batch_size=256 device=cuda train=118595 valid=14824
2026-05-09 10:25:43 [INFO] Epoch [   1/ 100] (best:    1): train_loss: 1.2341 | valid_loss: 1.1023 | train_acc: 0.5821 | valid_acc: 0.6134
2026-05-09 10:28:11 [INFO] Epoch [   2/ 100] (best:    2): train_loss: 0.9876 | valid_loss: 0.9234 | ...
...
2026-05-09 14:52:33 [INFO] Stage1 finished | best_epoch=87 best_valid_loss=0.1823
2026-05-09 14:53:01 [INFO] [Test] loss=0.1901 acc=0.9412
```

---

## Backbone 选择

所有 run 脚本均支持 `--backbone` 参数切换模型，**无需修改代码**。切换后 `in_dim` 自动从 `backbone.out_dim` 读取，分类头随之适配。

| backbone | 参数量 | out_dim | 适用场景 |
|----------|--------|---------|----------|
| `resnet18` | 11M | 512 | 默认，均衡选择 |
| `mobilenet_v3` | 2.5M | 576 | 显存受限 / 快速推理 |
| `efficientnet_b0` | 5.3M | 1280 | 精度优先 |
| `vit_tiny` | 5.7M | 192 | 全局 pattern 感知，推荐尝试 |
| `vit_small` | 1.2M | 96 | 轻量 ViT |
| `vit_micro` | ~0.8M | 96 | 极轻量，快速验证 |
| `vit_timm` | 5.7M | 192 | 需 timm，ImageNet 预训练 |

### ResNet-18（默认）

WaPIRL 原论文（Kahng & Kim 2021）使用的 backbone，在 WM811K 上验证充分。

- 架构：4 个残差层（layer1-4），每层 2 个 BasicBlock，输出 512 维特征
- stem：`small_input=True` 时使用 3×3 conv（适配 96×96），避免 7×7 conv 过度下采样
- 阶段二冻结 layer1/layer2，微调 layer3/layer4
- 优点：训练稳定，ImageNet 预训练权重质量高，对比学习收敛快
- 缺点：感受野局限于局部，对跨区域 pattern（如 Edge-Ring）的全局建模能力弱于 ViT

### MobileNetV3-Small

- 架构：基于 depthwise separable conv + SE 模块，输出 576 维特征（经 avgpool 后）
- 参数量仅 2.5M，推理速度比 ResNet-18 快 3-4 倍
- 适合显存 ≤ 8GB 或需要实时推理的场景
- 注意：depthwise conv 对位置信息的保留略弱，位置感知训练效果可能稍差于 ResNet-18
- 使用 torchvision 实现，可选 ImageNet 预训练权重（`--pretrained` 暂未暴露为参数，可在 factory.py 中修改）

### EfficientNet-B0

- 架构：复合缩放 CNN，输出 1280 维特征（经 avgpool 后）
- 参数量 5.3M，在 ImageNet 上精度优于同参数量的 ResNet
- out_dim=1280 较大，分类头参数量相应增加，小数据集（WM38K）上需注意过拟合
- 适合追求分类精度、显存充足（≥ 10GB）的场景

### ViT-Tiny（纯 PyTorch，推荐尝试）

基于两个参考仓库的最优配置，针对 96×96 晶圆图适配：

- **参考来源**：
  - [Fmohammadsofi/ViTTinyMixed-Defect-Wafer-Maps](https://github.com/Fmohammadsofi/ViTTinyMixed-Defect-Wafer-Maps)：消融研究最优配置（patch=16, embed=192, heads=3, layers=12），验证精度 98.41%
  - [PanithanS/Wafers-Defect-Recognition-using-Visual-Transformer](https://github.com/PanithanS/Wafers-Defect-Recognition-using-Visual-Transformer)：patch=13, embed=96, heads=4, layers=16，验证精度 98.98%

- **本项目适配**（相比原仓库的改动）：

  | 维度 | 原仓库 | 本项目 |
  |------|--------|--------|
  | 输入尺寸 | 52×52 | 96×96 |
  | patch_size | 16（Fmohammadsofi）/ 13（PanithanS） | 16 |
  | patch 数量 | 16 | **36**（6×6，覆盖更充分） |
  | 输入通道 | 1（灰度）| **2**（解耦双通道，通过 patch embedding Conv2d 直接支持） |
  | 框架 | PyTorch+timm / TensorFlow | 纯 PyTorch，无额外依赖 |

- **架构细节**（`vit_tiny` 预设）：
  - patch_size=16 → 36 个 patch token + 1 个 [CLS] token = 37 个 token
  - embed_dim=192，num_heads=3，depth=12，mlp_ratio=4（MLP 隐层 768 维）
  - 可学习位置编码（比固定 sin/cos 编码在小数据集上更灵活）
  - 输出 [CLS] token 的 192 维特征向量

- **优点**：全局自注意力天然捕捉跨区域 pattern（如 Edge-Ring 覆盖整圈），对 pattern 的空间关系建模能力强于 CNN
- **缺点**：对比学习阶段收敛比 ResNet 慢（需更多 epoch），batch_size 建议降至 128

### ViT-Small（轻量变体）

- 参考 PanithanS 仓库配置（patch=12, embed=96, heads=4, depth=8）
- patch_size=12 → 8×8=64 个 patch，比 vit_tiny 的 36 个更密集，适合细粒度 pattern
- 参数量仅 1.2M，显存占用极低
- 适合快速验证 ViT 方向是否有效

### ViT-Micro（极轻量）

- patch_size=16, embed_dim=96, num_heads=3, depth=6
- 参数量约 0.8M，主要用于调试和快速实验
- 不建议用于最终评估

### ViT-Timm（ImageNet 预训练）

- 基于 [timm](https://github.com/huggingface/pytorch-image-models) 库的 `vit_tiny_patch16_224`
- 支持加载 ImageNet-21k 预训练权重，迁移学习效果通常优于随机初始化
- 需要额外安装：`pip install timm>=0.9.0`
- 通道适配：通过 1×1 Conv2d 将 2 通道输入投影到 3 通道（timm 模型期望 3 通道）
- 注意：timm 的 ViT 支持动态 `img_size`，96×96 可直接传入，无需 resize 到 224×224

### 各模型显存估算（batch_size=128，96×96 输入）

| backbone | 训练显存 | 推荐 batch_size |
|----------|----------|-----------------|
| `resnet18` | ~6 GB | 256 |
| `mobilenet_v3` | ~4 GB | 256 |
| `efficientnet_b0` | ~8 GB | 128 |
| `vit_tiny` | ~8 GB | 128 |
| `vit_small` | ~5 GB | 256 |
| `vit_timm` | ~8 GB | 128 |

---

## 训练流程

### 阶段一：WM811K 有监督单标签分类

使用 WM811K 有标签数据（172K 张，9 类）训练 ResNet-18，为阶段二提供预训练权重。

```bash
# 从原始 pkl 直接读取（推荐）
python run_stage1.py \
  --pkl_file ../../data/wm811k/LSWMD.pkl \
  --epochs 100 \
  --batch_size 256 \
  --lr 1e-3 \
  --checkpoint_dir ./checkpoints/stage1

# 使用 ViT-Tiny backbone
python run_stage1.py \
  --pkl_file ../../data/wm811k/LSWMD.pkl \
  --backbone vit_tiny \
  --batch_size 128 \
  --checkpoint_dir ./checkpoints/stage1_vit

# 从已处理图像目录读取
python run_stage1.py \
  --data_dir ../../data/wm811k/labeled \
  --epochs 100 \
  --batch_size 256 \
  --checkpoint_dir ./checkpoints/stage1
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pkl_file` | — | LSWMD.pkl 路径（与 `--data_dir` 二选一） |
| `--data_dir` | — | 已处理图像目录 |
| `--backbone` | `resnet18` | backbone 类型，见上方 Backbone 选择表 |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 256 | ViT 建议降至 128 |
| `--lr` | 1e-3 | 学习率 |
| `--dropout` | 0.3 | 分类头 dropout |
| `--proportion` | 1.0 | 使用有标签数据的比例 |
| `--max_per_class` | None | 每类最多采样数，见下方说明 |
| `--in_channels` | 2 | 输入通道（2=解耦双通道，1=原始单通道） |
| `--img_size` | 96 | 输入图像尺寸（正方形） |
| `--checkpoint_dir` | `./checkpoints/stage1` | 权重保存目录 |

#### 类别平衡采样策略

WM811K 存在严重的类别不平衡：`none` 类约 8 万张，`donut`/`scratch` 等稀有类仅数百张。直接训练会导致 accuracy 虚高而 macro recall/F1 偏低（模型偏向多数类）。

`balanced_loader` 采用**少数类过采样 + 多数类下采样**的混合策略：

```
每类目标采样数 = min(该类实际数量, max_per_class)
采样权重       = 目标采样数 / 实际数量
总采样数       = Σ 各类目标采样数
```

- `max_per_class=None`（默认）：取各类数量的**中位数**作为上限，自动平衡
- `max_per_class=N`：手动指定上限，值越大多数类下采样越少

WM811K 中位数约 2000-3000，修改后每 epoch 有效样本数从 17 万降至约 2 万，但各类分布均匀，macro recall/F1 显著提升。

```bash
# 使用默认中位数平衡（推荐）
python run_stage1.py --pkl_file ../../data/wm811k/LSWMD.pkl

# 手动指定每类上限为 3000
python run_stage1.py --pkl_file ../../data/wm811k/LSWMD.pkl --max_per_class 3000
```

---

### 阶段二：WM38K 多标签微调 + 位置感知训练

在阶段一权重基础上，用 WM38K（含组合 pattern）进行多标签微调。默认只冻结 layer1，同时加入平移负样本迫使 encoder 学习位置敏感特征。支持 early stopping 和 ReduceLROnPlateau 调度器以缓解过拟合。

```bash
# 推荐配置（ReduceLROnPlateau + early stopping）
python run_stage2.py \
  --npz_file ../../data/wm38k/Wafer_Map_Datasets.npz \
  --stage1_ckpt ./checkpoints/stage1/best_model.pt \
  --freeze_layers layer1 \
  --lr_scheduler plateau \
  --patience 15 \
  --epochs 100 \
  --checkpoint_dir ./checkpoints/stage2

# 使用 ViT-Tiny backbone（需与阶段一保持一致）
python run_stage2.py \
  --npz_file ../../data/wm38k/Wafer_Map_Datasets.npz \
  --backbone vit_tiny \
  --stage1_ckpt ./checkpoints/stage1_vit/best_model.pt \
  --batch_size 128 \
  --lr 1e-4 \
  --checkpoint_dir ./checkpoints/stage2_vit

# 从已处理图像目录读取
python run_stage2.py \
  --data_dir ../../data/wm38k/images \
  --stage1_ckpt ./checkpoints/stage1/best_model.pt \
  --epochs 100 \
  --checkpoint_dir ./checkpoints/stage2
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--npz_file` | — | Wafer_Map_Datasets.npz 路径 |
| `--data_dir` | — | 已处理图像目录 |
| `--backbone` | `resnet18` | backbone 类型，需与阶段一保持一致 |
| `--stage1_ckpt` | — | 阶段一 checkpoint（不指定则随机初始化） |
| `--epochs` | 50 | 最大训练轮数（early stopping 可提前终止） |
| `--batch_size` | 128 | ViT 建议 64-128 |
| `--lr` | 1e-4 | 初始学习率 |
| `--freeze_layers` | `layer1` | 冻结的 backbone 层，可传多个（如 `layer1 layer2`）或空列表全解冻 |
| `--lr_scheduler` | `plateau` | `plateau`=ReduceLROnPlateau，`cosine`=CosineAnnealingLR |
| `--patience` | 15 | early stopping 容忍轮数 |
| `--pos_margin` | 0.5 | 位置感知 margin |
| `--pos_lambda` | 0.1 | 位置感知损失权重 λ |
| `--shift` | 0.25 | 平移幅度（图宽比例，25%） |
| `--img_size` | 96 | 输入图像尺寸（正方形） |
| `--checkpoint_dir` | `./checkpoints/stage2` | |

#### 过拟合问题与解决方案

WM38K 数据量（3.8 万张）远小于 WM811K，直接微调容易出现训练 mAP 接近 1.0 而验证 mAP 停滞在 0.75 左右的过拟合现象。主要原因及对应修复：

| 原因 | 修复 |
|------|------|
| 冻结 layer1+layer2，可训练参数过少，在小数据集上反而过拟合 | 默认只冻 layer1，`--freeze_layers` 可调 |
| CosineAnnealingLR 在 epoch 10 后 lr 已极小，模型停止有效学习 | 改为 ReduceLROnPlateau，验证 loss 不降才减半 lr |
| 无 early stopping，后续 epoch 白跑且可能加剧过拟合 | `--patience=15`，连续 15 epoch 无改善则停止 |
| optimizer 在冻结前构建，冻结层参数混入更新 | 先冻结再收集 `requires_grad=True` 参数构建 optimizer |

---

### 阶段三：生产数据自监督域适应（可选）

使用无标签生产 WDM 数据进行域适应。自动生成伪 WBM 作为正样本对，用 WaPIRL NCE Loss 微调 backbone 后两层。

> 若生产数据与 WM38K 分布接近，可跳过此阶段直接进行匹配推理。

```bash
python run_stage3.py \
  --wdm_npz ../../data/production/wdm.npz \
  --stage2_ckpt ./checkpoints/stage2/best_model.pt \
  --epochs 30 \
  --batch_size 64 \
  --num_negatives 2000 \
  --checkpoint_dir ./checkpoints/stage3

# 使用 ViT-Tiny backbone
python run_stage3.py \
  --wdm_npz ../../data/production/wdm.npz \
  --backbone vit_tiny \
  --stage2_ckpt ./checkpoints/stage2_vit/best_model.pt \
  --batch_size 64 \
  --num_negatives 1000 \
  --checkpoint_dir ./checkpoints/stage3_vit
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--wdm_npz` | — | 生产 WDM npz（arr_0 为 (N,H,W) 数组） |
| `--wdm_dir` | — | 生产 WDM 图像目录（与 `--wdm_npz` 二选一） |
| `--backbone` | `resnet18` | backbone 类型，需与阶段二保持一致 |
| `--stage2_ckpt` | — | 阶段二 checkpoint |
| `--epochs` | 30 | |
| `--batch_size` | 64 | 显存不足时降低 |
| `--num_negatives` | 2000 | 记忆库负采样数（ViT 建议降至 1000） |
| `--memory_weight` | 0.5 | 记忆库 EMA 更新系数 |
| `--temperature` | 0.07 | NCE Loss 温度 |
| `--checkpoint_dir` | `./checkpoints/stage3` | |

---

## 匹配推理

### 单次推理

给定一张 WBM，从 WDM 库中找出 top-k 匹配。

```bash
python run_matching.py \
  --wbm_path ../../data/production/wbm_sample.png \
  --wdm_dir  ../../data/production/wdm_images/ \
  --stage2_ckpt ./checkpoints/stage2/best_model.pt \
  --top_k 3

# 使用阶段三权重（域适应后）
python run_matching.py \
  --wbm_path ../../data/production/wbm_sample.png \
  --wdm_npz  ../../data/production/wdm.npz \
  --stage2_ckpt ./checkpoints/stage2/best_model.pt \
  --stage3_ckpt ./checkpoints/stage3/best_model.pt \
  --top_k 3
```

### 批量评估（WM38K 测试集 top-3 准确率）

```bash
python run_matching.py \
  --eval_npz ../../data/wm38k/Wafer_Map_Datasets.npz \
  --stage2_ckpt ./checkpoints/stage2/best_model.pt
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage2_ckpt` | 必填 | 阶段二 checkpoint |
| `--stage3_ckpt` | — | 阶段三 checkpoint（可选，覆盖 backbone） |
| `--backbone` | `resnet18` | 需与训练时保持一致 |
| `--alpha` | 0.6 | 重叠率权重 |
| `--beta` | 0.2 | 位置相似度权重 |
| `--gamma` | 0.2 | 面积相似度权重 |
| `--theta` | 0.6 | 重叠率过滤阈值（低于此值直接过滤） |
| `--cls_threshold` | 0.5 | 多标签分类阈值 |
| `--top_k` | 3 | 返回前 k 个匹配结果 |

---

## 匹配得分说明

```
最终得分 = α × 重叠率 + β × 位置相似度 + γ × 面积相似度

重叠率     = |S_wdm ∩ S_wbm| / |S_wdm|        # pattern 类型一致性
位置相似度 = cosine(z_wbm, z_wdm)              # embedding 空间距离（隐含位置信息）
面积相似度 = 1 - |area_wbm - area_wdm| / max   # pattern 大小一致性

过滤条件：重叠率 ≥ θ（默认 0.6），低于阈值直接排除
```

θ=1.0 时退化为严格子集匹配（S_wdm ⊆ S_wbm）。

---

## 依赖

```
# 核心依赖
torch >= 1.10
torchvision >= 0.11
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
