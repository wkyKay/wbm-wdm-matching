# -*- coding: utf-8 -*-
"""
Week 1: Full End-to-End Pipeline
完整的端到端 pipeline，串联 Week 1 的所有任务
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import numpy as np
import pandas as pd
import json
import time

def main():
    parser = argparse.ArgumentParser(description="Week 1 Full End-to-End Pipeline")
    parser.add_argument("--npz", type=str, 
                        default="/Users/kayw/Documents/trae_projects/match-test/data/wm38k/Wafer_Map_Datasets.npz",
                        help="Path to Wafer_Map_Datasets.npz")
    parser.add_argument("--out-dir", type=str, 
                        default="/Users/kayw/Documents/trae_projects/match-test/artifacts/week1",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--proposal-types", type=str, nargs='+',
                        default=['topk'],
                        help="Cluster proposal types")
    parser.add_argument("--min-area", type=int, default=5, help="Minimum area for filtered clusters")
    parser.add_argument("--proposal-top-k", type=int, default=5,
                        help="Number of regions to keep for topk proposal")
    parser.add_argument("--topk-base-method", type=str, default='geometry_merge',
                        help="Candidate generator used by topk proposal")
    parser.add_argument("--dilation-radius", type=int, default=1,
                        help="Dilation radius for dilated grouping proposals")
    parser.add_argument("--use-closing-for-grouping", action="store_true",
                        help="Use closing instead of dilation for dilated grouping")
    parser.add_argument("--suspicious-area", type=int, default=40,
                        help="Minimum area before adhesion suspicious checks")
    parser.add_argument("--min-suspicious-cues", type=int, default=1,
                        help="Number of shape cues required before adhesion split")
    parser.add_argument("--max-split-count", type=int, default=12,
                        help="Reject adhesion split results with too many fragments")
    parser.add_argument("--min-split-coverage", type=float, default=0.5,
                        help="Reject adhesion split results that keep too little original area")
    parser.add_argument("--disable-ring-guard", action="store_true",
                        help="Allow adhesion split on ring-like dilated groups")
    parser.add_argument("--method", type=str, choices=['iou', 'coverage_leakage'], 
                        default='coverage_leakage',
                        help="Smoke baseline method")
    parser.add_argument("--top-k", type=int, default=100, help="Top-K candidates")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta for coverage-leakage")
    parser.add_argument("--num-samples", type=int, default=None, 
                        help="Number of samples to use (None = full dataset)")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Week 1: MixedWM38K Retrieval Protocol and Cluster Proposal")
    print("=" * 70)
    
    # Import modules here to avoid scipy warnings at top
    from partial_match.data.data_io import load_wm38k, filter_valid_samples, get_label_info
    from partial_match.data.preprocessing import preprocess_batch
    from partial_match.data.split import split_by_signature, get_split_info
    from partial_match.data.metadata import generate_metadata, analyze_metadata
    from partial_match.core.cluster_proposal import (
        generate_cluster_tokens, 
        save_cluster_tokens,
        compute_cluster_statistics
    )
    from partial_match.evaluation.smoke_baseline import generate_smoke_rankings
    from partial_match.evaluation.metrics_fast import evaluate_rankings_fast
    try:
        from partial_match.utils.visualization import visualize_cluster_analysis
        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False
    
    start_time = time.time()
    
    # Step 1: Load and filter data
    print("\n[Step 1] Loading and filtering data...")
    t0 = time.time()
    maps, labels = load_wm38k(args.npz)
    valid_maps, valid_labels, original_indices = filter_valid_samples(maps, labels)
    
    if args.num_samples is not None:
        print(f"Using subset of {args.num_samples} samples")
        valid_maps = valid_maps[:args.num_samples]
        valid_labels = valid_labels[:args.num_samples]
        original_indices = original_indices[:args.num_samples]
    
    print(f"Total samples: {len(maps)}")
    print(f"Valid samples: {len(valid_maps)}")
    print(f"Time: {time.time() - t0:.2f}s")
    
    # Step 2: Split data
    print("\n[Step 2] Splitting dataset...")
    t0 = time.time()
    split_indices = split_by_signature(
        valid_labels,
        args.train_ratio,
        args.valid_ratio,
        args.test_ratio,
        args.seed
    )
    
    split_info = get_split_info(split_indices, valid_labels)
    print(f"Train samples: {split_info['train']['n_samples']}")
    print(f"Validation samples: {split_info['validation']['n_samples']}")
    print(f"Test samples: {split_info['test']['n_samples']}")
    print(f"Time: {time.time() - t0:.2f}s")
    
    # Save splits
    splits_data = {
        'seed': args.seed,
        'source': args.npz,
        'class_names': ['center', 'donut', 'edge-loc', 'edge-ring', 
                        'loc', 'random', 'scratch', 'near-full'],
        'train': [int(x) for x in split_indices['train']],
        'validation': [int(x) for x in split_indices['validation']],
        'test': [int(x) for x in split_indices['test']],
    }
    
    with open(out_dir / 'wm38k_splits.json', 'w') as f:
        json.dump(splits_data, f, ensure_ascii=False, indent=2)
    
    # Step 3: Preprocess maps
    print("\n[Step 3] Preprocessing maps...")
    t0 = time.time()
    preprocessed = preprocess_batch(valid_maps)
    print("Generated maps: status_maps, valid_masks, defect_masks, binary_maps, "
          "density_maps, soft_maps, three_value_maps")
    print(f"Time: {time.time() - t0:.2f}s")
    
    # Save preprocessed maps
    np.savez_compressed(
        out_dir / 'wm38k_maps.npz',
        status_maps=preprocessed['status_maps'],
        binary_maps=preprocessed['binary_maps'],
        count_maps=preprocessed['count_maps'],
        density_maps=preprocessed['density_maps'],
        soft_maps=preprocessed['soft_maps'],
        three_value_maps=preprocessed['three_value_maps'],
        valid_maps=valid_maps,
        valid_labels=valid_labels,
        original_indices=original_indices
    )
    
    # Step 4: Generate metadata
    print("\n[Step 4] Generating metadata...")
    t0 = time.time()
    metadata_df = generate_metadata(
        valid_maps,
        valid_labels,
        original_indices,
        split_indices
    )
    metadata_df.to_csv(out_dir / 'wm38k_metadata.csv', index=True, index_label='sample_id')
    print(f"Metadata saved with {len(metadata_df)} samples")
    print(f"Time: {time.time() - t0:.2f}s")
    
    # Step 5: Analyze and save report
    print("\n[Step 5] Analyzing data and generating report...")
    t0 = time.time()
    analysis = analyze_metadata(metadata_df)
    
    def convert_numpy(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        return obj
    
    analysis_converted = convert_numpy(analysis)
    
    with open(out_dir / 'week1_data_report.json', 'w') as f:
        json.dump(analysis_converted, f, ensure_ascii=False, indent=2)
    print(f"Time: {time.time() - t0:.2f}s")
    
    # Step 6: Generate cluster tokens
    print("\n[Step 6] Extracting cluster tokens...")
    t0 = time.time()
    tokens = generate_cluster_tokens(
        valid_maps,
        original_indices,
        args.proposal_types,
        args.min_area,
        top_k=args.proposal_top_k,
        topk_base_method=args.topk_base_method,
        dilation_radius=args.dilation_radius,
        use_closing=args.use_closing_for_grouping,
        suspicious_area=args.suspicious_area,
        min_suspicious_cues=args.min_suspicious_cues,
        max_split_count=args.max_split_count,
        min_split_coverage=args.min_split_coverage,
        skip_ring_like=not args.disable_ring_guard,
    )
    
    token_file = 'wm38k_cluster_tokens.jsonl' if args.num_samples is None else 'wm38k_cluster_tokens_small.jsonl'
    save_cluster_tokens(
        tokens,
        json_path=out_dir / token_file,
        npz_path=out_dir / token_file.replace('.jsonl', '.npz')
    )
    print(f"Generated {len(tokens)} cluster tokens")
    print(f"Time: {time.time() - t0:.2f}s")
    
    # Cluster statistics
    cluster_stats = compute_cluster_statistics(tokens)
    with open(out_dir / 'cluster_statistics.json', 'w') as f:
        json.dump(cluster_stats, f, ensure_ascii=False, indent=2)
    
    print("\nCluster statistics:")
    for pt, stats in cluster_stats.items():
        print(f"\n{pt}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Step 7: Run smoke baseline retrieval
    print("\n[Step 7] Running smoke baseline retrieval...")
    t0 = time.time()
    
    sample_ids = split_indices['validation']
    binary_maps = preprocessed['binary_maps']
    
    print(f"Method: {args.method}")
    print(f"Running on validation split ({len(sample_ids)} samples)...")
    
    rankings_df = generate_smoke_rankings(
        binary_maps[sample_ids],
        sample_ids,
        method=args.method,
        top_k=args.top_k,
        beta=args.beta
    )
    
    ranking_file = 'eval_smoke_rankings.csv' if args.num_samples is None else 'eval_smoke_rankings_small.csv'
    rankings_df.to_csv(out_dir / ranking_file, index=False)
    print(f"Generated {len(rankings_df)} ranking entries")
    print(f"Time: {time.time() - t0:.2f}s")
    
    # Step 8: Evaluate rankings
    print("\n[Step 8] Evaluating retrieval performance...")
    t0 = time.time()
    metrics = evaluate_rankings_fast(
        rankings_df,
        metadata_df,
        k_values=[1, 5, 10]
    )
    
    metrics_converted = convert_numpy(metrics)
    metric_file = 'eval_smoke_metrics.json' if args.num_samples is None else 'eval_smoke_metrics_small.json'
    with open(out_dir / metric_file, 'w') as f:
        json.dump(metrics_converted, f, ensure_ascii=False, indent=2)
    
    print(f"Time: {time.time() - t0:.2f}s")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print("\nMicro Average:")
    for key, value in sorted(metrics_converted['micro_average'].items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nTotal queries: {metrics_converted['total_queries']}")
    print(f"Skipped exact queries: {metrics_converted['skipped_exact_queries']}")
    
    # Generate Visualizations
    print("\n[Step 9] Generating cluster visualizations...")
    t0 = time.time()
    if VISUALIZATION_AVAILABLE:
        try:
            visualize_dir = os.path.join(out_dir, 'figures', 'cluster_visualization')
            visualize_cluster_analysis(
                valid_maps, 
                original_indices, 
                visualize_dir, 
                num_samples=min(10, len(valid_maps))
            )
            print(f"Visualizations saved to: {visualize_dir}")
        except Exception as e:
            print(f"Visualization skipped due to error: {e}")
    else:
        print("Visualization skipped: matplotlib not available")
    print(f"Time: {time.time() - t0:.2f}s")
    
    # Generate Markdown report
    print("\n[Step 10] Generating final report...")
    t0 = time.time()
    generate_markdown_report(
        out_dir, analysis_converted, split_info, cluster_stats, metrics_converted, args
    )
    print(f"Time: {time.time() - t0:.2f}s")
    
    # Complete
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Week 1 Pipeline Complete! Total time: {total_time:.2f}s")
    print(f"Output saved to: {out_dir}")
    print("=" * 70)


def generate_markdown_report(out_dir, analysis, split_info, cluster_stats, metrics, args):
    """Generate comprehensive Markdown report"""
    report = f"""# Week 1: MixedWM38K Retrieval Protocol and Cluster Proposal

## 1. Overview

Week 1 完整 pipeline 完成！
- ✅ 数据加载和预处理
- ✅ 数据集 split (训练/验证/测试)
- ✅ 地图表达生成
- ✅ 聚类提议和 token 生成
- ✅ 检索协议和评估指标
- ✅ Smoke Baseline 检索和评估

## 2. 数据集统计

### 2.1 基本信息
- 原始数据集: {split_info['train']['n_samples'] + split_info['validation']['n_samples'] + split_info['test']['n_samples']} 张晶圆图
- 标签类别数: 8 个

### 2.2 标签基数分布
| 标签数量 | 样本数 |
|---------|-------|
| 1       | {analysis['cardinality_distribution'].get(1, 0)} |
| 2       | {analysis['cardinality_distribution'].get(2, 0)} |
| 3       | {analysis['cardinality_distribution'].get(3, 0)} |
| 4       | {analysis['cardinality_distribution'].get(4, 0)} |

### 2.3 类别分布
| 类别 | 样本数 |
|------|--------|
| center | {analysis['class_distribution']['center']} |
| donut | {analysis['class_distribution']['donut']} |
| edge-loc | {analysis['class_distribution']['edge-loc']} |
| edge-ring | {analysis['class_distribution']['edge-ring']} |
| loc | {analysis['class_distribution']['loc']} |
| random | {analysis['class_distribution']['random']} |
| scratch | {analysis['class_distribution']['scratch']} |
| near-full | {analysis['class_distribution']['near-full']} |

## 3. 数据集 Split

### 3.1 Split 策略
- 按标签 signature 分层 split
- Seed: {args.seed} (可复现)
- 比例: {args.train_ratio:.0%} / {args.valid_ratio:.0%} / {args.test_ratio:.0%}

### 3.2 Split 结果
| Split | 样本数 |
|------|-------|
| 训练 (train) | {split_info['train']['n_samples']} |
| 验证 (validation) | {split_info['validation']['n_samples']} |
| 测试 (test) | {split_info['test']['n_samples']} |

## 4. 地图表达

### 4.1 生成的 Map 类型
- `status_map`: 状态图
- `valid_mask`: 有效区域 mask
- `defect_mask`: 缺陷区域 mask
- `binary_map`: 二值缺陷图
- `density_map`: 密度归一化图
- `soft_map`: 高斯平滑图
- `three_value_map`: 三值图 (1.0=强缺陷, 0.5=弱缺陷邻域)

### 4.2 缺陷区域统计
- 平均缺陷面积: {analysis['defect_area_stats']['mean']:.2f} 像素
- 最小缺陷面积: {analysis['defect_area_stats']['min']} 像素
- 最大缺陷面积: {analysis['defect_area_stats']['max']} 像素

## 5. 聚类提议

### 5.1 提议类型
1. **TopK**: 基于 `{args.topk_base_method}` 候选，按面积选择前 {args.proposal_top_k} 个主要区域
2. **Filtered**: 原始连通域过滤小聚类 (面积≥{args.min_area})，用于 debug / ablation
3. **Adhesion**: 对可疑粘连连通域做二次拆分，作为 TopK 默认候选来源

### 5.2 聚类统计
"""
    
    for pt, stats in cluster_stats.items():
        report += f"\n**{pt}**:\n"
        for key, value in stats.items():
            report += f"  - {key}: {value:.4f}\n" if isinstance(value, float) else f"  - {key}: {value}\n"
    
    report += f"""
## 6. Smoke Baseline 检索结果

### 6.1 实验设置
- 方法: {args.method}
- 数据集: 验证集 (validation)
- Top-K: {args.top_k}
- Beta: {args.beta}

### 6.2 评估指标
| 指标 | 值 |
|------|----|
"""
    
    for key, value in sorted(metrics['micro_average'].items()):
        if isinstance(value, float):
            report += f"| {key} | {value:.4f} |\n"
        else:
            report += f"| {key} | {value} |\n"
    
    report += f"""
### 6.3 结果说明
- Coverage-Leakage 作为简单的 baseline，已经表现出一定的检索能力
- 局部匹配方法可以进一步提高 exact consistency
- Week 2 将实现传统局部描述符方法进行对比

## 7. 文件结构

```
artifacts/week1/
├── wm38k_metadata.csv        # 样本元数据
├── wm38k_splits.json         # 数据集 split
├── wm38k_maps.npz            # 预处理后的地图
├── wm38k_cluster_tokens.jsonl  # 聚类 token
├── eval_smoke_rankings.csv    # Smoke Baseline 排序结果
├── eval_smoke_metrics.json    # Smoke Baseline 评估结果
├── cluster_statistics.json   # 聚类统计
├── week1_data_report.json    # 数据统计
└── week1_data_report.md      # 本报告
```

## 8. 后续计划

### Week 2
- 实现传统局部描述符方法
- 提取 Hu/Zernike 矩等形状特征
- 使用相同的局部匹配协议进行检索评估

### Week 3
- 实现全图自监督学习 baseline
- 对比全局 vs 局部方法

### Week 4
- 实现 proposed 方法：局部学习的 cluster 检索
- 完成主实验对比

---
*Generated by Week 1 full pipeline*
"""
    
    with open(out_dir / 'week1_data_report.md', 'w') as f:
        f.write(report)


if __name__ == "__main__":
    main()
