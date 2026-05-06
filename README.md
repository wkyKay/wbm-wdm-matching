# Wafer Bin Map Matching based on Self-Supervised Learning

基于自监督学习的晶圆缺陷图（WDM）与晶圆失效芯片图（WBM）匹配

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
## 📌 项目简介

本项目研究晶圆制造过程中的缺陷匹配问题。通过自监督对比学习（Contrastive Learning）方法，分别基于 **CNN** 和 **Vision Transformer (ViT)** 两种架构，实现晶圆缺陷图（Wafer Defect Map, WDM）与晶圆失效芯片图（Wafer Bin Map, WBM）之间的跨模态匹配。

### 核心任务

```
给定 WBM (晶圆失效芯片图)  →  检索匹配的 WDM (晶圆缺陷图)
```

### 技术路线

| 方法 | 架构 | 特点 |
|------|------|------|
| WaPIRL | CNN (ResNet/VGG/AlexNet) | 基于 Memory Bank 的对比学习 |
| ViT | Vision Transformer | 预训练Transformer直接微调 |

---

## 🏗️ 项目结构

```
wbm_wdm_matching/
├── WaPIRL/                    # CNN-based 自监督学习方法
│   ├── configs/                # 配置文件
│   ├── datasets/               # 数据集加载
│   ├── models/                 # 模型定义 (ResNet/VGG/AlexNet)
│   ├── tasks/                  # 训练任务 (WaPIRL/分类)
│   ├── utils/                  # 工具函数
│   ├── test/                   # 测试代码
│   │   ├── run_wbm.py         # WBM 预训练
│   │   ├── run_wdm.py         # WDM 预训练
│   │   ├── run_triplet.py     # 跨模态匹配训练
│   │   ├── wbm_classification.py  # WBM 分类
│   │   ├── triplet_loss/       # Triplet/InfoNCE 损失
│   │   └── processors/         # 数据预处理
│   ├── run_wapirl.py          # WaPIRL 预训练入口
│   └── run_classification.py  # 分类微调入口
│
├── ViT/                       # Vision Transformer 方法
│   └── analysis.py            # ViT 分类代码
│
├── requirements.txt            # Python 依赖
└── .gitignore                 # Git 忽略配置
```

---

## 📊 数据集

### WM-811K 数据集

- 来源：[Kaggle WM-811K](https://www.kaggle.com/qingyi/wm811k-wafer-map)
- 包含：172,950 张晶圆图
- 缺陷类别：9 类

### 数据格式

| 类型 | 说明 | 尺寸 |
|------|------|------|
| WBM (Wafer Bin Map) | 晶圆失效芯片图（Pass/Fail） | 10×10 / 20×20 / 40×40 |
| WDM (Wafer Defect Map) | 晶圆缺陷图 | 96×96 |

### 数据预处理

```bash
# 1. 下载 WM-811K 数据集
# 2. 将 LSWMD.pkl 放入 ./data/wm811k/
# 3. 运行预处理脚本
python WaPIRL/process_wm811k.py

# 4. 生成配对数据（用于跨模态匹配）
python WaPIRL/test/processors/process_paired_data.py
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/wkyKay/wbm-wdm-matching.git
cd wbm_wdm_matching

# 创建 conda 环境
conda create -n wbm python=3.10
conda activate wbm

# 安装依赖
pip install -r requirements.txt
```

### 2. WaPIRL 预训练 (CNN)

```bash
# WBM 预训练
cd WaPIRL
python run_wapirl.py \
    --input_size 10 \
    --augmentation flip \
    --backbone_type resnet \
    --backbone_config 18 \
    --epochs 100 \
    --batch_size 256 \
    --learning_rate 1e-2 \
    --checkpoint_root ./checkpoints

# WDM 预训练
python run_wdm.py \
    --input_size 96 \
    --augmentation rotate \
    --backbone_type resnet \
    --backbone_config 18 \
    --epochs 100
```

### 3. 跨模态匹配训练

```bash
cd WaPIRL/test
python run_triplet.py \
    --pretrained_wbm_file /path/to/wbm_checkpoint \
    --pretrained_wdm_file /path/to/wdm_checkpoint \
    --loss_type multi_similarity \
    --epochs 100 \
    --batch_size 32
```

### 4. 分类微调

```bash
cd WaPIRL
python run_classification.py \
    --pretrained_model_file /path/to/pretrained_model \
    --input_size 10 \
    --augmentation crop \
    --epochs 100
```

### 5. ViT 方法

```bash
cd ViT
# 修改数据路径后运行
python analysis.py
```

---

## 📈 模型架构

### WaPIRL (CNN)

```
输入图像
    ↓
Backbone (ResNet-18 / VGG-16 / AlexNet)
    ↓
投影头 (Linear: 512 → 128)
    ↓
对比学习 / 分类
```

### 跨模态匹配模型

```
WBM ─→ Encoder_WBM ─→ 投影头 ─┐
                                ├─→ 对比学习 → 检索
WDM ─→ Encoder_WDM ─→ 投影头 ─┘
```

---

## 📝 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_size` | 输入图像尺寸 | 10 / 96 |
| `--backbone_type` | 骨干网络 | resnet |
| `--backbone_config` | 网络配置 | 18 |
| `--augmentation` | 数据增强方式 | flip / rotate |
| `--temperature` | 对比学习温度参数 | 0.07 |
| `--projector_size` | 投影头维度 | 128 |
| `--batch_size` | 批大小 | 256 |
| `--learning_rate` | 学习率 | 1e-2 |

---

## 🔧 损失函数

项目实现了多种对比学习损失：

- **WaPIRL Loss**: 基于 Memory Bank 的 NT-Xent 变体
- **Triplet Loss**: 基本三元组损失
- **Online Triplet Loss**: 在线难例挖掘
- **InfoNCE Loss**: 温度缩放对比损失
- **Multi-Similarity Loss**: 多相似度损失

---

## 📂 输出目录

训练产生的模型和日志保存在：

```
WaPIRL/checkpoints/wm811k/
├── wapirl/              # WaPIRL 预训练模型
│   └── resnet.18/
│       ├── flip/
│       └── rotate/
├── finetune_wapirl/     # 分类微调模型
│   └── resnet.18/
└── triplet/             # 跨模态匹配模型
```

---

## 📚 参考

- 论文: "Self-Supervised Representation Learning for Wafer Bin Map Defect Pattern Classification"
- WaPIRL: https://github.com/xxx/wapirl (原始实现)
- WM-811K Dataset: https://www.kaggle.com/qingyi/wm811k-wafer-map

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 License

MIT License
