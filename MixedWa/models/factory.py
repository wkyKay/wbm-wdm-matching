# -*- coding: utf-8 -*-
"""
Backbone 工厂函数，统一创建各类 backbone。

支持的 backbone：
  resnet18        — ResNet-18，11M 参数，当前默认
  mobilenet_v3    — MobileNetV3-Small，2.5M 参数，轻量快速
  efficientnet_b0 — EfficientNet-B0，5.3M 参数，精度/速度平衡
  vit_tiny        — ViT-Tiny（纯 PyTorch），5.7M 参数，patch=16
  vit_small       — ViT-Tiny 轻量变体（patch=12，embed=96），1.2M 参数
  vit_micro       — ViT 极轻量（patch=16，embed=96，depth=6），快速验证
  vit_timm        — 基于 timm 的 ViT-Tiny，支持 ImageNet 预训练（需 pip install timm）
"""

import torch
import torch.nn as nn

from models.resnet.backbone import ResNet18Backbone
from models.vit.backbone import ViTTinyBackbone, ViTTimmBackbone


def build_backbone(name: str, in_channels: int = 2, **kwargs) -> nn.Module:
    """
    创建 backbone。

    Args:
        name:        backbone 名称，见模块文档
        in_channels: 输入通道数（解耦输入=2，原始单通道=1）
        **kwargs:    传递给具体 backbone 的额外参数

    Returns:
        backbone 实例，具有 .out_dim 属性表示输出特征维度
    """
    name = name.lower()

    if name == 'resnet18':
        return ResNet18Backbone(
            in_channels=in_channels,
            small_input=kwargs.get('small_input', True),
        )

    elif name == 'mobilenet_v3':
        return _build_mobilenet_v3(in_channels, **kwargs)

    elif name == 'efficientnet_b0':
        return _build_efficientnet_b0(in_channels, **kwargs)

    elif name in ('vit_tiny', 'vit_small', 'vit_micro'):
        preset_map = {'vit_tiny': 'tiny', 'vit_small': 'small', 'vit_micro': 'micro'}
        return ViTTinyBackbone(
            img_size=kwargs.get('img_size', 96),
            in_channels=in_channels,
            dropout=kwargs.get('dropout', 0.1),
            preset=preset_map[name],
        )

    elif name == 'vit_timm':
        return ViTTimmBackbone(
            model_name=kwargs.get('timm_model', 'vit_tiny_patch16_224'),
            in_channels=in_channels,
            pretrained=kwargs.get('pretrained', True),
            img_size=kwargs.get('img_size', 96),
        )

    else:
        raise ValueError(
            f"Unknown backbone: '{name}'. "
            f"Choose from: resnet18, mobilenet_v3, efficientnet_b0, "
            f"vit_tiny, vit_small, vit_micro, vit_timm"
        )


# ---------------------------------------------------------------------------
# torchvision 轻量 backbone 封装
# ---------------------------------------------------------------------------

class _TorchvisionBackbone(nn.Module):
    """将 torchvision 模型封装为与 ResNet18Backbone 接口一致的 backbone。"""
    def __init__(self, tv_model: nn.Module, in_channels: int, out_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # 通道适配：将 in_channels 投影到模型期望的 3 通道
        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.channel_adapter = nn.Identity()

        self.backbone = tv_model

    def freeze_layers(self, layer_names: list):
        """兼容 BackboneBase 接口，按名称冻结层。"""
        for name, param in self.named_parameters():
            if any(name.startswith(ln) for ln in layer_names):
                param.requires_grad = False

    def load_weights_from_checkpoint(self, path: str, strict: bool = False):
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('backbone', ckpt)
        self.load_state_dict(state, strict=strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.channel_adapter(x))


def _build_mobilenet_v3(in_channels: int, **kwargs) -> _TorchvisionBackbone:
    try:
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if kwargs.get('pretrained', False) else None
        model = mobilenet_v3_small(weights=weights)
    except ImportError:
        from torchvision.models import mobilenet_v3_small
        model = mobilenet_v3_small(pretrained=kwargs.get('pretrained', False))

    # 移除分类头，保留特征提取部分
    out_dim = model.classifier[0].in_features  # 576
    model.classifier = nn.Identity()
    # avgpool 已在 features 末尾，输出 (B, 576)
    return _TorchvisionBackbone(model, in_channels, out_dim)


def _build_efficientnet_b0(in_channels: int, **kwargs) -> _TorchvisionBackbone:
    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT if kwargs.get('pretrained', False) else None
        model = efficientnet_b0(weights=weights)
    except ImportError:
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=kwargs.get('pretrained', False))

    out_dim = model.classifier[1].in_features  # 1280
    model.classifier = nn.Identity()
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    # 需要 flatten
    original_forward = model.forward

    def new_forward(x):
        x = model.features(x)
        x = model.avgpool(x)
        return x.flatten(1)

    model.forward = new_forward
    return _TorchvisionBackbone(model, in_channels, out_dim)


# ---------------------------------------------------------------------------
# 便捷信息查询
# ---------------------------------------------------------------------------

BACKBONE_INFO = {
    'resnet18':        {'params': '11M',  'out_dim': 512,  'note': '默认，WaPIRL 原论文使用'},
    'mobilenet_v3':    {'params': '2.5M', 'out_dim': 576,  'note': '最轻量，推理快 3-4x'},
    'efficientnet_b0': {'params': '5.3M', 'out_dim': 1280, 'note': '精度/速度平衡'},
    'vit_tiny':        {'params': '5.7M', 'out_dim': 192,  'note': 'patch=16, 36 patches，参考 Fmohammadsofi 最优配置'},
    'vit_small':       {'params': '1.2M', 'out_dim': 96,   'note': 'patch=12, 64 patches，参考 PanithanS 配置'},
    'vit_micro':       {'params': '~0.8M','out_dim': 96,   'note': '极轻量，快速验证'},
    'vit_timm':        {'params': '5.7M', 'out_dim': 192,  'note': '需 timm，支持 ImageNet 预训练权重'},
}
