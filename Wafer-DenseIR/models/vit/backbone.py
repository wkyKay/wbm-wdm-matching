# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn

from models.base import BackboneBase


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super(PatchEmbedding, self).__init__()
        if img_size % patch_size != 0:
            raise ValueError(f'img_size {img_size} must be divisible by patch_size {patch_size}')
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super(MultiHeadSelfAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f'embed_dim {embed_dim} must be divisible by num_heads {num_heads}')
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTinyBackbone(BackboneBase):
    """
    Tiny ViT backbone for proposal-free dense wafer retrieval.

    Unlike classification ViT, forward returns patch tokens reshaped as a dense
    feature map: (B, embed_dim, H_patch, W_patch). The current retrieval task can
    then reuse the same dense-token matching code used by ResNet.
    """
    def __init__(self, layer_config: dict, in_channels: int = 2, img_size: int = 96):
        super(ViTTinyBackbone, self).__init__(layer_config, in_channels)
        self.layer_config = layer_config
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = layer_config.get('patch_size', 16)
        self.embed_dim = layer_config.get('embed_dim', 192)
        self.num_heads = layer_config.get('num_heads', 3)
        self.depth = layer_config.get('depth', 12)
        self.mlp_ratio = layer_config.get('mlp_ratio', 4.0)
        self.dropout = layer_config.get('dropout', 0.1)

        self.patch_embed = PatchEmbedding(img_size, self.patch_size, in_channels, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(self.dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(self.embed_dim, self.num_heads, self.mlp_ratio, self.dropout)
            for _ in range(self.depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                fan_out //= module.groups
                module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.norm(self.blocks(x))
        patch_tokens = x[:, 1:, :]
        h = w = self.patch_embed.grid_size
        return patch_tokens.transpose(1, 2).reshape(b, self.embed_dim, h, w)

    @property
    def out_channels(self):
        return self.embed_dim
