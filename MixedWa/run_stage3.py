# -*- coding: utf-8 -*-
"""
阶段三运行脚本：生产数据自监督域适应。

用法（从 MixedWa/ 目录运行）：
  python run_stage3.py \
    --wdm_npz ../../data/production/wdm.npz \
    --stage2_ckpt ./checkpoints/stage2/best_model.pt \
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
from tasks.stage3 import Stage3DomainAdaptation, ProductionWDMDataset, MemoryBank
from utils.loss import WaPIRLLoss


def parse_args():
    p = argparse.ArgumentParser('Stage3: Production data self-supervised domain adaptation')
    # 数据
    p.add_argument('--wdm_npz',  type=str, default=None,
                   help='生产 WDM 数据 npz 文件（arr_0 为 (N,H,W) 数组）')
    p.add_argument('--wdm_dir',  type=str, default=None,
                   help='生产 WDM 图像目录（与 --wdm_npz 二选一）')
    # 预训练权重
    p.add_argument('--stage2_ckpt', type=str, default=None,
                   help='主训练阶段 checkpoint 路径（run_train.py 产出的 best_model.pt）')
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
    p.add_argument('--proj_dim',       type=int, default=128)
    p.add_argument('--checkpoint_dir', type=str, default='./checkpoints/stage3')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def load_wdm_arrays(args) -> np.ndarray:
    """从 npz 或图像目录加载 WDM 数组。"""
    if args.wdm_npz:
        data = np.load(args.wdm_npz)
        key = 'arr_0' if 'arr_0' in data else list(data.keys())[0]
        return data[key]
    elif args.wdm_dir:
        import glob
        import cv2
        paths = sorted(glob.glob(os.path.join(args.wdm_dir, '**/*.png'), recursive=True))
        arrays = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
        return np.stack(arrays)
    else:
        raise ValueError("必须指定 --wdm_npz 或 --wdm_dir")


def main():
    args = parse_args()

    wdm_arrays = load_wdm_arrays(args)
    print(f"Loaded {len(wdm_arrays)} WDM samples, shape: {wdm_arrays[0].shape}")

    decouple = (args.in_channels == 2)
    transform = WaferTransform(size=(args.img_size, args.img_size), mode='test')

    dataset = ProductionWDMDataset(
        wdm_arrays=wdm_arrays,
        transform=transform,
        decouple_input=decouple,
        img_size=args.img_size,
    )

    # 模型
    backbone  = build_backbone(args.backbone, in_channels=args.in_channels, img_size=args.img_size)
    projector = MLPProjector(in_dim=backbone.out_dim, hidden_dim=256, out_dim=args.proj_dim)
    info = BACKBONE_INFO.get(args.backbone, {})
    print(f"Backbone: {args.backbone}  params≈{info.get('params','?')}  out_dim={backbone.out_dim}"
          f"  [{info.get('note','')}]")

    # 加载阶段二 backbone 权重
    if args.stage2_ckpt and os.path.exists(args.stage2_ckpt):
        backbone.load_weights_from_checkpoint(args.stage2_ckpt, strict=False)
        print(f"Loaded stage2 backbone from: {args.stage2_ckpt}")
    else:
        print("No stage2 checkpoint found, using random initialization.")

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

    task = Stage3DomainAdaptation(
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

    print(f"\n[Stage3] Done. Checkpoint saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
