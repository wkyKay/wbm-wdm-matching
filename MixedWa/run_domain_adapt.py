# -*- coding: utf-8 -*-
"""
生产数据自监督域适应运行脚本。

用法（从 MixedWa/ 目录运行）：
  python run_domain_adapt.py \
    --wdm_npz ../../data/production/wdm.npz \
    --supervised_ckpt ./checkpoints/train/best_model.pt \
    --epochs 30
"""

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from datasets.transforms import WaferTransform
from models.factory import build_backbone, BACKBONE_INFO
from models.head import MLPProjector
from tasks.domain_adapt import DomainAdaptation, ProductionWDMDataset, MemoryBank
from utils.loss import WaPIRLLoss


def parse_args():
    p = argparse.ArgumentParser('Domain adaptation: production data self-supervised adaptation')
    # 数据
    p.add_argument('--wdm_npz',  type=str, default=None,
                   help='生产 WDM 数据 npz 文件（arr_0 为 (N,H,W) 数组）')
    p.add_argument('--wdm_dir',  type=str, default=None,
                   help='生产 WDM 图像目录（与 --wdm_npz 二选一）')
    p.add_argument('--wdm_format', type=str, default='auto',
                   choices=['auto', 'binary', 'wm811k_png', 'wbm_values'],
                   help='WDM 输入值域格式：auto 自动判断；binary=非零即缺陷；'
                         'wm811k_png=process_wm811k.py 导出的 0/127/255 PNG；'
                         'wbm_values=0/1/2 wafer map，仅 2 视为缺陷')
    p.add_argument('--wafer_mask_mode', type=str, default='auto',
                   choices=['auto', 'circular', 'nonzero', 'full'],
                   help='生成 WBM-like 存在掩码的方式；binary WDM 默认使用 circular')
    # 预训练权重
    p.add_argument('--supervised_ckpt', type=str, default=None,
                   help='监督训练 checkpoint 路径（run_train.py 产出的 best_model.pt）')
    # 训练参数
    p.add_argument('--epochs',        type=int,   default=30)
    p.add_argument('--batch_size',    type=int,   default=64)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--num_negatives', type=int,   default=2000,
                   help='记忆库负采样数（显存不足时降低）')
    p.add_argument('--memory_weight', type=float, default=0.5,
                   help='记忆库 EMA 更新系数')
    p.add_argument('--loss_weight',   type=float, default=0.5,
                   help='WaPIRL loss_weight λ')
    p.add_argument('--temperature',   type=float, default=0.07)
    # 模型 / 输出
    p.add_argument('--backbone',       type=str, default='resnet18',
                   help='backbone 类型：resnet18 | mobilenet_v3 | efficientnet_b0 | '
                        'vit_tiny | vit_small | vit_micro | vit_timm')
    p.add_argument('--in_channels',    type=int, default=2)
    p.add_argument('--img_size',        type=int, default=96,
                    help='输入图像尺寸（正方形，默认 96）')
    p.add_argument('--pseudo_out_size', type=int, default=None,
                   help='pseudo-WBM 最终输出尺寸；默认跟随 --img_size')
    p.add_argument('--pseudo_grid_size', type=int, default=11,
                   help='pseudo-WBM 中间 die-level 网格尺寸，需按产品实际 WBM 尺寸设置')
    p.add_argument('--proj_dim',       type=int, default=128)
    p.add_argument('--checkpoint_dir', type=str, default='./checkpoints/domain_adapt')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def make_circular_mask(shape: tuple) -> np.ndarray:
    """生成与 WM38K/WBM 输入一致的晶圆有效区域掩码。"""
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = min(h, w) * 0.48
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2


def normalize_wdm_array(arr: np.ndarray,
                        data_format: str = 'auto',
                        wafer_mask_mode: str = 'auto') -> np.ndarray:
    """
    将不同来源的 WDM/WBM-like 数组统一为 run_train 使用的 WBM 值域：
      0=背景，1=晶圆有效区域正常，2=缺陷。

    这样 decouple_mask(x) 后：
      channel 0 = defect map，channel 1 = wafer existence mask。
    """
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D WDM array, got shape: {arr.shape}")

    if data_format == 'auto':
        unique = np.unique(arr)
        unique_set = set(unique.tolist())
        if unique_set.issubset({0, 1}):
            data_format = 'binary'
        elif unique_set.issubset({0, 1, 2}):
            data_format = 'wbm_values'
        elif unique.max() > 2 and np.any(unique >= 200):
            data_format = 'wm811k_png'
        else:
            data_format = 'binary'

    if data_format == 'binary':
        defect = arr > 0
        default_mask_mode = 'circular'
    elif data_format == 'wm811k_png':
        defect = arr >= 200
        default_mask_mode = 'nonzero'
    elif data_format == 'wbm_values':
        defect = arr == 2
        default_mask_mode = 'nonzero'
    else:
        raise ValueError(f"Unknown wdm_format: {data_format}")

    mask_mode = default_mask_mode if wafer_mask_mode == 'auto' else wafer_mask_mode
    if mask_mode == 'circular':
        valid = make_circular_mask(arr.shape)
    elif mask_mode == 'nonzero':
        valid = arr > 0
        if valid.sum() <= defect.sum():
            valid = make_circular_mask(arr.shape)
    elif mask_mode == 'full':
        valid = np.ones(arr.shape, dtype=bool)
    else:
        raise ValueError(f"Unknown wafer_mask_mode: {wafer_mask_mode}")

    wbm_like = valid.astype(np.uint8)
    wbm_like[defect] = 2
    return wbm_like


def load_wdm_arrays(args) -> np.ndarray:
    """从 npz 或图像目录加载 WDM 数组，并统一为 {0,1,2} WBM-like 图。"""
    if args.wdm_npz:
        data = np.load(args.wdm_npz)
        key = 'arr_0' if 'arr_0' in data else list(data.keys())[0]
        arrays = data[key]
    elif args.wdm_dir:
        import glob
        import cv2
        paths = sorted(glob.glob(os.path.join(args.wdm_dir, '**/*.png'), recursive=True))
        if len(paths) == 0:
            raise ValueError(f"No PNG files found under --wdm_dir: {args.wdm_dir}")
        arrays = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
        arrays = [arr for arr in arrays if arr is not None]
        if len(arrays) == 0:
            raise ValueError(f"Failed to read PNG files under --wdm_dir: {args.wdm_dir}")
        arrays = np.stack(arrays)
    else:
        raise ValueError("必须指定 --wdm_npz 或 --wdm_dir")

    arrays = np.asarray(arrays)
    normalized = np.stack([
        normalize_wdm_array(arr, args.wdm_format, args.wafer_mask_mode)
        for arr in arrays
    ])
    return normalized


def main():
    args = parse_args()
    if args.pseudo_out_size is None:
        args.pseudo_out_size = args.img_size

    wdm_arrays = load_wdm_arrays(args)
    print(f"Loaded {len(wdm_arrays)} WDM samples, shape: {wdm_arrays[0].shape}")

    decouple = (args.in_channels == 2)
    transform = WaferTransform(size=(args.img_size, args.img_size), mode='test')
    if args.pseudo_out_size != args.img_size:
        print(f"pseudo_out_size={args.pseudo_out_size} will be resized to img_size={args.img_size} by WaferTransform")

    dataset = ProductionWDMDataset(
        wdm_arrays=wdm_arrays,
        transform=transform,
        decouple_input=decouple,
        img_size=args.pseudo_out_size,
        pseudo_grid_size=args.pseudo_grid_size,
    )

    # 模型
    backbone  = build_backbone(args.backbone, in_channels=args.in_channels, img_size=args.img_size)
    projector = MLPProjector(in_dim=backbone.out_dim, hidden_dim=256, out_dim=args.proj_dim)
    info = BACKBONE_INFO.get(args.backbone, {})
    print(f"Backbone: {args.backbone}  params≈{info.get('params','?')}  out_dim={backbone.out_dim}"
          f"  [{info.get('note','')}]")

    # 加载监督训练 backbone 权重
    if args.supervised_ckpt and os.path.exists(args.supervised_ckpt):
        backbone.load_weights_from_checkpoint(args.supervised_ckpt, strict=False)
        print(f"Loaded supervised backbone from: {args.supervised_ckpt}")
    else:
        print("No supervised checkpoint found, using random initialization.")

    # 记忆库
    memory = MemoryBank(
        size=(len(dataset), args.proj_dim),
        device=args.device,
        weight=args.memory_weight,
    )

    # 优化器（只优化 layer3/layer4 + projector）
    params = (
        [p for n, p in backbone.named_parameters()
         if any(n.startswith(ln) for ln in ['layer3', 'layer4'])]
        + list(projector.parameters())
    )
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn   = WaPIRLLoss(temperature=args.temperature)

    task = DomainAdaptation(
        backbone=backbone,
        projector=projector,
        memory=memory,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_fn,
        num_negatives=args.num_negatives,
        loss_weight=args.loss_weight,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
    task.save_config(vars(args))
    task.run(dataset, epochs=args.epochs,
             batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"\n[DomainAdapt] Done. Checkpoint saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
