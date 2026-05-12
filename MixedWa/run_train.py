# -*- coding: utf-8 -*-
"""
主训练脚本：WM38K 多标签分类。

直接从 ImageNet 预训练权重初始化 backbone，在 WM38K（含单类/两类/三类组合 pattern）
上训练多标签分类器，无需 WM811K 预训练阶段。

用法（从 MixedWa/ 目录运行）：
  python run_train.py --npz_file ../../data/wm38k/Wafer_Map_Datasets.npz
  python run_train.py --data_dir ../../data/wm38k/images
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser('WM38K multi-label training')
    # 数据源（二选一）
    p.add_argument('--npz_file', type=str, default=None,
                   help='直接从 Wafer_Map_Datasets.npz 读取（推荐）')
    p.add_argument('--data_dir', type=str, default=None,
                   help='已处理图像目录（如 data/wm38k/images）')
    # 训练参数
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--batch_size',   type=int,   default=128)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--dropout',      type=float, default=0.3)
    # 冻结 / early stopping / 调度器
    p.add_argument('--freeze_layers', type=str, nargs='*', default=[],
                   help='冻结的 backbone 层名（默认不冻结，全量微调）')
    p.add_argument('--patience',      type=int, default=15,
                   help='early stopping 容忍轮数')
    p.add_argument('--lr_scheduler',  type=str, default='plateau',
                   choices=['cosine', 'plateau'],
                   help='plateau=ReduceLROnPlateau，cosine=CosineAnnealingLR')
    # 位置感知损失
    p.add_argument('--pos_margin', type=float, default=0.5)
    p.add_argument('--pos_lambda', type=float, default=0.1)
    p.add_argument('--shift',      type=float, default=0.3,
                   help='平移幅度（图宽比例，正样本无空间增强时建议 ≥0.3）')
    # 模型
    p.add_argument('--backbone',    type=str, default='resnet18',
                   help='resnet18 | mobilenet_v3 | efficientnet_b0 | '
                        'vit_tiny | vit_small | vit_micro | vit_timm')
    p.add_argument('--in_channels', type=int, default=2,
                   help='输入通道（2=解耦双通道，1=原始单通道）')
    p.add_argument('--img_size',    type=int, default=96,
                   help='输入图像尺寸（正方形，默认 96）')
    p.add_argument('--pretrained',  action='store_true', default=True,
                   help='使用 ImageNet 预训练权重初始化（默认开启）')
    # 输出
    p.add_argument('--checkpoint_dir', type=str, default='./checkpoints/train')
    p.add_argument('--device',         type=str, default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    assert args.npz_file or args.data_dir, "必须指定 --npz_file 或 --data_dir"

    if args.device.startswith('cuda:'):
        gpu_id = args.device.split(':')[1]
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', gpu_id)
        args.device = 'cuda'

    import torch
    from datasets.datasets import WM38KRaw, WM38KFromDir
    from datasets.transforms import WaferTransform
    from models.factory import build_backbone, BACKBONE_INFO
    from models.head import LinearClassifier
    from tasks.train import WM38KTrainer
    from utils.loss import PositionAwareLoss

    size = (args.img_size, args.img_size)
    decouple = (args.in_channels == 2)
    train_transform = WaferTransform(size=size, mode='test')   # 不做空间变换，避免与位置感知 loss 冲突
    shift_transform = WaferTransform(size=size, mode='shift', shift=args.shift)
    test_transform  = WaferTransform(size=size, mode='test')

    if args.npz_file:
        train_set = WM38KRaw(args.npz_file, split='train',
                             transform=train_transform,
                             shift_transform=shift_transform,
                             decouple_input=decouple, img_size=args.img_size)
        valid_set = WM38KRaw(args.npz_file, split='valid',
                             transform=test_transform,
                             decouple_input=decouple, img_size=args.img_size)
        test_set  = WM38KRaw(args.npz_file, split='test',
                             transform=test_transform,
                             decouple_input=decouple, img_size=args.img_size)
    else:
        train_set = WM38KFromDir(os.path.join(args.data_dir, 'train'),
                                 transform=train_transform,
                                 shift_transform=shift_transform,
                                 decouple_input=decouple, img_size=args.img_size)
        valid_set = WM38KFromDir(os.path.join(args.data_dir, 'valid'),
                                 transform=test_transform,
                                 decouple_input=decouple, img_size=args.img_size)
        test_set  = WM38KFromDir(os.path.join(args.data_dir, 'test'),
                                 transform=test_transform,
                                 decouple_input=decouple, img_size=args.img_size)

    print(f"Train: {len(train_set)} | Valid: {len(valid_set)} | Test: {len(test_set)}")

    # 模型（ImageNet 预训练权重）
    backbone   = build_backbone(args.backbone, in_channels=args.in_channels,
                                img_size=args.img_size, pretrained=args.pretrained)
    classifier = LinearClassifier(in_dim=backbone.out_dim, num_classes=8, dropout=args.dropout)
    info = BACKBONE_INFO.get(args.backbone, {})
    print(f"Backbone: {args.backbone}  params≈{info.get('params','?')}  out_dim={backbone.out_dim}"
          f"  pretrained={args.pretrained}  [{info.get('note','')}]")

    # 先冻结，再收集可训练参数
    if args.freeze_layers:
        backbone.freeze_layers(args.freeze_layers)
        print(f"Frozen layers: {args.freeze_layers}")

    trainable = list(filter(lambda p: p.requires_grad,
                            list(backbone.parameters()) + list(classifier.parameters())))
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_fn = PositionAwareLoss(margin=args.pos_margin, lam=args.pos_lambda)

    task = WM38KTrainer(
        backbone=backbone,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_fn,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        freeze_layers=[],   # 已在此处冻结
        patience=args.patience,
    )
    task.save_config(vars(args))
    task.run(train_set, valid_set, epochs=args.epochs,
             batch_size=args.batch_size, num_workers=args.num_workers,
             test_set=test_set)

    print(f"\n[Train] Done. Checkpoint saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
