
# -*- coding: utf-8 -*-
"""测试 local_feature_matcher 模块"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from models.factory import build_backbone
from models.head import LinearClassifier
from matching.local_feature_matcher import LocalFeatureMatcher, LocalFeatureExtractor


def test_basic_functionality():
    """测试基本功能是否正常"""
    print("="*60)
    print("测试 1: LocalFeatureExtractor")
    print("="*60)
    
    device = 'cpu'
    backbone = build_backbone('resnet18', in_channels=2, img_size=96)
    
    extractor = LocalFeatureExtractor(backbone, device=device)
    
    # 创建假输入
    dummy_input = torch.randn(1, 2, 96, 96)
    features, global_feat = extractor.extract(dummy_input)
    
    print(f"✓ 提取了 {len(features)} 层特征")
    for i, feat in enumerate(features):
        print(f"  - Layer {i}: {feat.shape}")
    print(f"✓ 全局特征: {global_feat.shape}")
    
    print("\n" + "="*60)
    print("测试 2: LocalFeatureMatcher")
    print("="*60)
    
    matcher = LocalFeatureMatcher(
        backbone=backbone,
        device=device,
        max_shift=5
    )
    
    # 创建假的map
    dummy_wbm_map = np.zeros((96, 96), dtype=np.float32)
    dummy_wbm_map[30:60, 30:60] = 1.0  # 中心方块
    
    dummy_wdm_map = np.zeros((96, 96), dtype=np.float32)
    dummy_wdm_map[35:65, 35:65] = 1.0  # 稍微偏移的方块
    
    # 创建假的tensor（格式要对）
    from datasets.datasets import decouple_mask
    from datasets.transforms import WaferTransform
    transform = WaferTransform(size=(96, 96), mode='test')
    
    wbm_np = (dummy_wbm_map * 2).astype(np.uint8)  # 0/2
    wbm_np = np.expand_dims(wbm_np, axis=2)
    wbm_tensor = transform(wbm_np)
    wbm_tensor = decouple_mask(wbm_tensor).unsqueeze(0)
    
    wdm_np = (dummy_wdm_map * 2).astype(np.uint8)
    wdm_np = np.expand_dims(wdm_np, axis=2)
    wdm_tensor = transform(wdm_np)
    wdm_tensor = decouple_mask(wdm_tensor).unsqueeze(0)
    
    # 测试匹配
    result = matcher.match_single(
        wbm_tensor,
        wdm_tensor,
        dummy_wbm_map,
        dummy_wdm_map
    )
    
    print(f"✓ 匹配完成！")
    print(f"  - Score: {result.score:.4f}")
    print(f"  - Shape sim: {result.shape_sim:.4f}")
    print(f"  - Size sim: {result.size_sim:.4f}")
    print(f"  - Position sim: {result.position_sim:.4f}")
    print(f"  - Local feature sim: {result.local_feature_sim:.4f}")
    print(f"  - Best offset: {result.best_offset}")
    
    print("\n" + "="*60)
    print("✓ 所有测试通过！")
    print("="*60)


if __name__ == '__main__':
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"\n✗ 错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
