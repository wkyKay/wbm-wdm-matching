# -*- coding: utf-8 -*-
"""
匹配推理脚本：给定一张 WBM，从 WDM 库中找出 top-k 匹配。

用法（从 MixedWa/ 目录运行）：
  python run_matching.py \
    --wbm_path ../../data/production/wbm_sample.png \
    --wdm_dir  ../../data/production/wdm_images/ \
    --stage2_ckpt ./checkpoints/stage2/best_model.pt \
    --top_k 3

  # 批量评估（WM38K 测试集）
  python run_matching.py \
    --eval_npz ../../data/wm38k/Wafer_Map_Datasets.npz \
    --stage2_ckpt ./checkpoints/stage2/best_model.pt
"""

import os
import sys
import argparse

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from datasets.datasets import decouple_mask, DEFECT_CLASSES
from datasets.transforms import WaferTransform
from models.factory import build_backbone
from models.head import LinearClassifier
from matching.matcher import WaferMatcher


def parse_args():
    p = argparse.ArgumentParser('WBM-WDM Matching Inference')
    # 单次推理
    p.add_argument('--wbm_path',  type=str, default=None, help='WBM 图像路径')
    p.add_argument('--wdm_dir',   type=str, default=None, help='WDM 图像目录')
    p.add_argument('--wdm_npz',   type=str, default=None, help='WDM npz 文件')
    # 批量评估
    p.add_argument('--eval_npz',  type=str, default=None,
                   help='WM38K npz，用于批量 top-3 准确率评估')
    # 模型
    p.add_argument('--stage2_ckpt', type=str, required=True)
    p.add_argument('--stage3_ckpt', type=str, default=None,
                   help='阶段三 checkpoint（可选，覆盖 backbone）')
    p.add_argument('--backbone',    type=str, default='resnet18',
                   help='backbone 类型，需与训练时一致：resnet18 | mobilenet_v3 | '
                        'efficientnet_b0 | vit_tiny | vit_small | vit_micro | vit_timm')
    p.add_argument('--in_channels', type=int, default=2)
    p.add_argument('--img_size',    type=int, default=96,
                   help='输入图像尺寸（正方形，默认 96）')
    # 匹配参数
    p.add_argument('--alpha',         type=float, default=0.6)
    p.add_argument('--beta',          type=float, default=0.2)
    p.add_argument('--gamma',         type=float, default=0.2)
    p.add_argument('--theta',         type=float, default=0.6,
                   help='重叠率过滤阈值')
    p.add_argument('--cls_threshold', type=float, default=0.5)
    p.add_argument('--top_k',         type=int,   default=3)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def load_image_tensor(path: str, transform, decouple: bool, device: str) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x_np = np.expand_dims(img, axis=2)
    x = transform(x_np)
    if decouple:
        x = decouple_mask(x)
    return x.unsqueeze(0).to(device)


def load_wdm_tensors(args, transform, decouple: bool, device: str):
    tensors, raw_maps = [], []
    if args.wdm_npz:
        data = np.load(args.wdm_npz)
        key = 'arr_0' if 'arr_0' in data else list(data.keys())[0]
        arrays = data[key]
        for arr in arrays:
            x_np = np.expand_dims(cv2.resize(arr.astype(np.uint8), (args.img_size, args.img_size),
                                             interpolation=cv2.INTER_NEAREST), axis=2)
            x = transform(x_np)
            raw_maps.append(x)
            if decouple:
                x = decouple_mask(x)
            tensors.append(x)
    elif args.wdm_dir:
        import glob
        paths = sorted(glob.glob(os.path.join(args.wdm_dir, '**/*.png'), recursive=True))
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            x_np = np.expand_dims(img, axis=2)
            x = transform(x_np)
            raw_maps.append(x)
            if decouple:
                x = decouple_mask(x)
            tensors.append(x)
    return torch.stack(tensors).to(device), torch.stack(raw_maps).to(device)


def evaluate_top3(matcher: WaferMatcher, npz_file: str, device: str, img_size: int = 96):
    """在 WM38K 测试集上评估 top-3 准确率。"""
    from datasets.datasets import WM38KRaw
    from datasets.transforms import WaferTransform

    transform = WaferTransform(size=(img_size, img_size), mode='test')
    test_set = WM38KRaw(npz_file, split='test',
                        transform=transform,
                        decouple_input=(matcher.backbone.in_channels == 2),
                        img_size=img_size)

    correct, total = 0, 0
    for i in range(len(test_set)):
        sample = test_set[i]
        wbm_x = sample['x'].unsqueeze(0).to(device)
        wbm_label = sample['y']  # (8,) 多热

        # 从测试集中随机抽取 50 个 WDM 候选（含真实匹配）
        import random
        candidate_indices = random.sample(range(len(test_set)), min(50, len(test_set)))
        if i not in candidate_indices:
            candidate_indices[0] = i
        gt_idx_in_candidates = candidate_indices.index(i)

        wdm_tensors = torch.stack([test_set[j]['x'] for j in candidate_indices]).to(device)

        results = matcher.match(wbm_x, wdm_tensors, top_k=3)
        top3_indices = [r['wdm_idx'] for r in results]

        if gt_idx_in_candidates in top3_indices:
            correct += 1
        total += 1

        if total % 500 == 0:
            print(f"  [{total}/{len(test_set)}] top-3 acc: {correct/total:.4f}")

    print(f"\n[Eval] Top-3 Accuracy: {correct}/{total} = {correct/total:.4f}")


def main():
    args = parse_args()
    decouple = (args.in_channels == 2)
    transform = WaferTransform(size=(args.img_size, args.img_size), mode='test')

    # 加载模型
    backbone   = build_backbone(args.backbone, in_channels=args.in_channels, img_size=args.img_size)
    classifier = LinearClassifier(in_dim=backbone.out_dim, num_classes=8)

    backbone.load_weights_from_checkpoint(args.stage2_ckpt, strict=False)
    classifier.load_weights_from_checkpoint(args.stage2_ckpt, key='classifier', strict=False)

    if args.stage3_ckpt and os.path.exists(args.stage3_ckpt):
        backbone.load_weights_from_checkpoint(args.stage3_ckpt, strict=False)
        print(f"Loaded stage3 backbone from: {args.stage3_ckpt}")

    matcher = WaferMatcher(
        backbone=backbone,
        classifier=classifier,
        device=args.device,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        theta=args.theta,
        cls_threshold=args.cls_threshold,
    )

    # 批量评估模式
    if args.eval_npz:
        evaluate_top3(matcher, args.eval_npz, args.device, img_size=args.img_size)
        return

    # 单次推理模式
    assert args.wbm_path, "单次推理需要指定 --wbm_path"
    assert args.wdm_dir or args.wdm_npz, "单次推理需要指定 --wdm_dir 或 --wdm_npz"

    wbm_tensor = load_image_tensor(args.wbm_path, transform, decouple, args.device)
    wdm_tensors, wdm_maps = load_wdm_tensors(args, transform, decouple, args.device)

    print(f"WBM: {args.wbm_path}")
    print(f"WDM candidates: {len(wdm_tensors)}")

    results = matcher.match(wbm_tensor, wdm_tensors,
                            wdm_maps=wdm_maps, top_k=args.top_k)

    print(f"\nTop-{args.top_k} matches:")
    for rank, r in enumerate(results, 1):
        s_names = [DEFECT_CLASSES[c] for c in sorted(r['s_wdm'])]
        area_str = f"{r['area_sim']:.3f}" if r['area_sim'] is not None else 'N/A'
        print(f"  #{rank}: WDM[{r['wdm_idx']:>4d}] "
              f"score={r['score']:.4f} "
              f"overlap={r['overlap']:.3f} "
              f"pos={r['pos_sim']:.3f} "
              f"area={area_str} "
              f"patterns={s_names}")


if __name__ == '__main__':
    main()
