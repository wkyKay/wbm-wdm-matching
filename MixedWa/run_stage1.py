# -*- coding: utf-8 -*-
"""
阶段一运行脚本：WM811K 有监督单标签分类训练。

用法（从 MixedWa/ 目录运行）：
  python run_stage1.py --pkl_file ../../data/wm811k/LSWMD.pkl --epochs 100
  python run_stage1.py --data_dir ../../data/wm811k/labeled   --epochs 100  # 已处理图像目录
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser('Stage1: WM811K supervised classification')
    # 数据源（二选一）
    p.add_argument('--pkl_file', type=str, default=None,
                   help='直接从 LSWMD.pkl 读取（无需预处理）')
    p.add_argument('--data_dir', type=str, default=None,
                   help='已处理图像目录（如 data/wm811k/labeled')
    # 训练参数
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=256)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--dropout',      type=float, default=0.3)
    p.add_argument('--proportion',   type=float, default=1.0,
                   help='使用有标签数据的比例')
    # 模型 / 输出
    p.add_argument('--backbone',      type=str,  default='resnet18',
                   help='backbone 类型：resnet18 | mobilenet_v3 | efficientnet_b0 | '
                        'vit_tiny | vit_small | vit_micro | vit_timm')
    p.add_argument('--in_channels',    type=int,  default=2,
                   help='输入通道数（解耦输入=2，原始单通道=1）')
    p.add_argument('--img_size',        type=int,  default=96,
                   help='输入图像尺寸（正方形，默认 96）')
    p.add_argument('--checkpoint_dir', type=str,  default='./checkpoints/stage1')
    p.add_argument('--device',         type=str,  default='cuda:3')
    return p.parse_args()


def main():
    args = parse_args()
    assert args.pkl_file or args.data_dir, "必须指定 --pkl_file 或 --data_dir"

    # 限制只使用指定 GPU（必须在 import torch 之前设置
    if args.device.startswith('cuda:'):
        print(f"Setting visible GPU to {args.device}")
        gpu_id = args.device.split(':')[1]
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', gpu_id)
        args.device = 'cuda'  # CUDA_VISIBLE_DEVICES 生效后，cuda:0 即为物理 GPU 3

    # 检查 GPU 设置是否生效
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")

    # 现在才导入 torch 和其他依赖
    import torch
    import torch.nn as nn
    from datasets.datasets import WM811KRaw, WM811KFromDir
    from datasets.transforms import WaferTransform
    from models.factory import build_backbone, BACKBONE_INFO
    from models.head import LinearClassifier
    from tasks.stage1 import Stage1Classification

    print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 数据变换
    size = (args.img_size, args.img_size)
    train_transform = WaferTransform(size=size, mode='crop')
    test_transform  = WaferTransform(size=size, mode='test')
    decouple = (args.in_channels == 2)

    # 数据集
    if args.pkl_file:
        train_set = WM811KRaw(args.pkl_file, split='train',
                              transform=train_transform, decouple_input=decouple,
                              proportion=args.proportion, img_size=args.img_size)
        valid_set = WM811KRaw(args.pkl_file, split='valid',
                              transform=test_transform, decouple_input=decouple,
                              img_size=args.img_size)
        test_set  = WM811KRaw(args.pkl_file, split='test',
                              transform=test_transform, decouple_input=decouple,
                              img_size=args.img_size)
    else:
        train_set = WM811KFromDir(os.path.join(args.data_dir, 'train'),
                                  transform=train_transform, decouple_input=decouple,
                                  proportion=args.proportion)
        valid_set = WM811KFromDir(os.path.join(args.data_dir, 'valid'),
                                  transform=test_transform, decouple_input=decouple)
        test_set  = WM811KFromDir(os.path.join(args.data_dir, 'test'),
                                  transform=test_transform, decouple_input=decouple)

    print(f"Train: {len(train_set)} | Valid: {len(valid_set)} | Test: {len(test_set)}")

    # 模型
    backbone   = build_backbone(args.backbone, in_channels=args.in_channels, img_size=args.img_size)
    classifier = LinearClassifier(in_dim=backbone.out_dim, num_classes=9, dropout=args.dropout)
    info = BACKBONE_INFO.get(args.backbone, {})
    print(f"Backbone: {args.backbone}  params≈{info.get('params','?')}  out_dim={backbone.out_dim}"
          f"  [{info.get('note','')}]")

    # 优化器 &amp; 调度器
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn   = nn.CrossEntropyLoss()

    # 训练
    task = Stage1Classification(
        backbone=backbone,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_fn,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
    task.save_config(vars(args))
    task.run(train_set, valid_set, epochs=args.epochs,
             batch_size=args.batch_size, num_workers=args.num_workers,
             test_set=test_set)

    print(f"\n[Stage1] Done. Checkpoint saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
