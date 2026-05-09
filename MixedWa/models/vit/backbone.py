# -*- coding: utf-8 -*-
"""
ViT-Tiny Backbone，适配晶圆图匹配任务。

设计参考：
  - Fmohammadsofi/ViTTinyMixed-Defect-Wafer-Maps（最优配置：patch=16, embed=192, heads=3, layers=12）
  - PanithanS/Wafers-Defect-Recognition-using-Visual-Transformer（patch=13, embed=96, heads=4, layers=16）

适配当前项目的改动：
  1. 输入尺寸 96×96（两个参考仓库均为 52×52，patch 数量从 16 增至 36）
  2. 支持 2 通道解耦输入（[defect_map, existence_mask]），通过 1×1 conv 投影到 embed_dim
  3. 纯 PyTorch 实现，无需 timm 依赖（可选）；也提供 timm 版本
  4. 输出 (B, embed_dim) 的 [CLS] token 特征，与 ResNet18Backbone 接口一致
"""

import math
import torch
import torch.nn as nn
from models.base import BackboneBase


# ---------------------------------------------------------------------------
# 位置编码
# ---------------------------------------------------------------------------

class LearnablePositionEmbedding(nn.Module):
    """可学习位置编码，比固定 sin/cos 编码在小数据集上更灵活。"""
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        # +1 for [CLS] token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# Transformer Encoder Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch Embedding（支持多通道输入）
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    将 (B, C, H, W) 图像切分为 patch 并投影到 embed_dim。
    使用 Conv2d 实现，stride=patch_size 等价于无重叠切分。
    """
    def __init__(self, img_size: int, patch_size: int,
                 in_channels: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size {img_size} must be divisible by patch_size {patch_size}"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, N, embed_dim)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# ViT-Tiny Backbone（纯 PyTorch，无 timm 依赖）
# ---------------------------------------------------------------------------

class ViTTinyBackbone(BackboneBase):
    """
    ViT-Tiny，针对 96×96 晶圆图优化。

    默认配置（来自 Fmohammadsofi 仓库消融研究最优结果，适配 96×96）：
      patch_size=16 → 6×6=36 个 patch（52×52 时为 16 个）
      embed_dim=192, num_heads=3, depth=12, mlp_ratio=4
      参数量约 5.7M

    备选轻量配置（参考 PanithanS 仓库，patch_size=12 → 8×8=64 个 patch）：
      patch_size=12, embed_dim=96, num_heads=4, depth=8
      参数量约 1.2M，适合显存极度受限场景
    """
    PRESETS = {
        # name: (patch_size, embed_dim, num_heads, depth)
        'tiny':   (16, 192, 3,  12),  # 参考 Fmohammadsofi，最优配置
        'small':  (12, 96,  4,  8),   # 参考 PanithanS，更轻量
        'micro':  (16, 96,  3,  6),   # 极轻量，快速验证用
    }

    def __init__(self,
                 img_size: int = 96,
                 patch_size: int = 16,
                 in_channels: int = 2,
                 embed_dim: int = 192,
                 num_heads: int = 3,
                 depth: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 preset: str = None):
        """
        Args:
            img_size:    输入图像尺寸（正方形），默认 96
            patch_size:  patch 尺寸，需能整除 img_size
                         96×96 时推荐 16（36 patches）或 12（64 patches）
            in_channels: 输入通道数（解耦输入=2，原始单通道=1）
            embed_dim:   token 嵌入维度
            num_heads:   注意力头数，需整除 embed_dim
            depth:       Transformer 层数
            mlp_ratio:   MLP 隐层维度倍数
            dropout:     attention 和 MLP 的 dropout
            preset:      预设配置名 'tiny'|'small'|'micro'，覆盖上述参数
        """
        super().__init__()

        if preset is not None:
            assert preset in self.PRESETS, f"Unknown preset: {preset}. Choose from {list(self.PRESETS)}"
            patch_size, embed_dim, num_heads, depth = self.PRESETS[preset]

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.out_dim = embed_dim  # 与 ResNet18Backbone.out_dim 接口一致

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # [CLS] token + 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = LearnablePositionEmbedding(num_patches, embed_dim)
        self.pos_drop  = nn.Dropout(dropout)

        # Transformer encoder
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            (B, embed_dim) — [CLS] token 特征，与 ResNet18Backbone 输出接口一致
        """
        B = x.size(0)

        # Patch embedding: (B, N, embed_dim)
        x = self.patch_embed(x)

        # 拼接 [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, embed_dim)

        # 位置编码 + dropout
        x = self.pos_drop(self.pos_embed(x))

        # Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # 返回 [CLS] token
        return x[:, 0]  # (B, embed_dim)


# ---------------------------------------------------------------------------
# timm 版本（需要 pip install timm，支持 ImageNet 预训练权重）
# ---------------------------------------------------------------------------

class ViTTimmBackbone(BackboneBase):
    """
    基于 timm 的 ViT backbone，支持加载 ImageNet 预训练权重。
    需要 pip install timm>=0.9.0。

    推荐模型名：
      'vit_tiny_patch16_224'  — ViT-Tiny，patch=16，需 resize 到 224×224
      'vit_small_patch16_224' — ViT-Small，更强但更慢
    """
    def __init__(self, model_name: str = 'vit_tiny_patch16_224',
                 in_channels: int = 2,
                 pretrained: bool = True,
                 img_size: int = 96):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("请先安装 timm：pip install timm>=0.9.0")

        self.in_channels = in_channels

        # 通道适配：将 in_channels 通道投影到 3 通道（timm 模型期望 3 通道输入）
        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.channel_adapter = nn.Identity()

        # 创建 timm 模型，num_classes=0 表示只要特征，不要分类头
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # 返回特征向量
            img_size=img_size,   # 适配 96×96（timm 支持动态 img_size）
        )
        self.out_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_adapter(x)
        return self.backbone(x)
