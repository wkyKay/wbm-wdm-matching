# -*- coding: utf-8 -*-
"""
阶段二运行脚本：WM38K 多标签微调 + 位置感知训练。

用法（从 MixedWa/ 目录运行）：
  python run_stage2.py \
    --npz_file ../../data/wm38k/Wafer_Map_Datasets.npz \
    --stage1_ckpt ./checkpoints/stage1/best_model.pt \
    --epochs 50
"""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(__file__))

from datasets.datasets import WM38KRaw, WM38KFromDir
from datasets.transforms import WaferTransform
from models.factory import build_backbone, BACKBONE_INFO
from models.head import LinearClassifier
from tasks.stage2 import Stage2MultiLabel
from utils.loss import PositionAwareLoss


def parse_args():
    p = argparse.ArgumentParser('Stage2: WM38K multi-label fine-tuning')
    # 数据源（二选一）
    p.add_argument('--npz_file', type=str, default=None,
                   help='直接从 Wafer_Map_Datasets.npz 读取')
    p.add_argument('--data_dir', type=str, default=None,
                   help='已处理图像目录（如 data/wm38k/images）')
    # 预训练权重
    p.add_argument('--stage1_ckpt', type=str, default=None,
                   help='阶段一 checkpoint 路径（不指定则随机初始化）')
    # 训练参数
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--batch_size',   type=int,   default=128)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--dropout',      type=float, default=0.3)
    # 位置感知损失参数
    p.add_argument('--pos_margin', type=float, default=0.5,
                   help='平移负样本 margin')
    p.add_argument('--pos_lambda', type=float, default=0.1,
                   help='位置感知损失权重 λ')
    p.add_argument('--shift',      type=float, default=0.25,
                   help='平移幅度（图宽比例）')
    # 模型 / 输出
    p.add_argument('--backbone',       type=str, default='resnet18',
                   help='backbone 类型：resnet18 | mobilenet_v3 | efficientnet_b0 | '
                        'vit_tiny | vit_small | vit_micro | vit_timm')
    p.add_argument('--in_channels',    type=int, default=2)
    p.add_argument('--img_size',        type=int, default=96,
                   help='输入图像尺寸（正方形，默认 96）')
    p.add_argument('--checkpoint_dir', type=str, default='./checkpoints/stage2')
    p.add_argument('--device', type=str, default='cuda:3')
    return p.parse_args()


def main():
    args = parse_args()
    assert args.npz_file or args.data_dir, "必须指定 --npz_file 或 --data_dir"

    if args.device.startswith('cuda:'):
        gpu_id = args.device.split(':')[1]
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', gpu_id)
        args.device = 'cuda'

    decouple = (args.in_channels == 2)
    size = (args.img_size, args.img_size)
    train_transform = WaferTransform(size=size, mode='crop')
    shift_transform = WaferTransform(size=size, mode='shift', shift=args.shift)
    test_transform  = WaferTransform(size=size, mode='test')

    if args.npz_file:
        train_set = WM38KRaw(args.npz_file, split='train',
                             transform=train_transform,
                             shift_transform=shift_transform,
                             decouple_input=decouple,
                             img_size=args.img_size)
        valid_set = WM38KRaw(args.npz_file, split='valid',
                             transform=test_transform, decouple_input=decouple,
                             img_size=args.img_size)
        test_set  = WM38KRaw(args.npz_file, split='test',
                             transform=test_transform, decouple_input=decouple,
                             img_size=args.img_size)
    else:
        train_set = WM38KFromDir(os.path.join(args.data_dir, 'train'),
                                 transform=train_transform,
                                 shift_transform=shift_transform,
                                 decouple_input=decouple)
        valid_set = WM38KFromDir(os.path.join(args.data_dir, 'valid'),
                                 transform=test_transform, decouple_input=decouple)
        test_set  = WM38KFromDir(os.path.join(args.data_dir, 'test'),
                                 transform=test_transform, decouple_input=decouple)

    print(f"Train: {len(train_set)} | Valid: {len(valid_set)} | Test: {len(test_set)}")

    # 模型
    backbone   = build_backbone(args.backbone, in_channels=args.in_channels, img_size=args.img_size)
    classifier = LinearClassifier(in_dim=backbone.out_dim, num_classes=8, dropout=args.dropout)
    info = BACKBONE_INFO.get(args.backbone, {})
    print(f"Backbone: {args.backbone}  params≈{info.get('params','?')}  out_dim={backbone.out_dim}"
          f"  [{info.get('note','')}]")

    # 加载阶段一权重（backbone 部分）
    if args.stage1_ckpt and os.path.exists(args.stage1_ckpt):
        backbone.load_weights_from_checkpoint(args.stage1_ckpt, strict=False)
        print(f"Loaded stage1 backbone from: {args.stage1_ckpt}")
    else:
        print("No stage1 checkpoint found, using random initialization.")

    # 优化器（只优化未冻结参数，冻结在 Task.run() 中执行）
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad,
               list(backbone.parameters()) + list(classifier.parameters())),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn   = PositionAwareLoss(margin=args.pos_margin, lam=args.pos_lambda)

    task = Stage2MultiLabel(
        backbone=backbone,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_fn,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
    task.run(train_set, valid_set, epochs=args.epochs,
             batch_size=args.batch_size, num_workers=args.num_workers,
             test_set=test_set)

    print(f"\n[Stage2] Done. Checkpoint saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
