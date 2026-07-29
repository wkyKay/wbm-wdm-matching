# partial_match — WBM Cluster Proposal & Retrieval

基于 WM38K 数据集的晶圆图缺陷图案聚类与部分匹配框架。

详细 proposal 设计见 [docs/proposal_design.md](docs/proposal_design.md)。

---

## 文件结构

```
partial_match/
├── __init__.py              # 包入口
├── core/                    # 核心算法
│   ├── clustering.py        # ★ 统一聚类接口（核心模块）
│   ├── cluster_proposal.py  # 基础聚类方法与 token 生成
│   ├── adhesion_split.py    # 粘连区域拆分方法
│   └── tensor_voting_paper.py # Tensor Voting + MBBS 论文完整实现（独立）
│
├── data/                    # 数据处理
│   ├── data_io.py           # 数据加载（WM38K .npz 读取、类别名映射）
│   ├── preprocessing.py     # 预处理（binary / density / soft / three-value maps）
│   ├── split.py             # 数据集划分（train / val / test）
│   └── metadata.py          # 元数据统计（缺陷面积、标签基数等）
│
├── evaluation/              # 检索评价与 baseline
│   ├── metrics.py           # 完整评估指标（含 per-class）
│   ├── metrics_fast.py      # 快速评估（仅 Micro/Macro）
│   └── smoke_baseline.py    # Coverage-Leakage 和 IoU baseline
│
├── utils/
│   └── visualization.py     # 可视化辅助函数
│
├── cluster_test/            # 论文方法实现 + 对比验证
│   ├── __init__.py
│   ├── adjacency_iwmm.py    # Adjacency-Clustering + iWMM（独立实现）
│   ├── dbscan_clustering.py # DBSCAN 聚类（独立实现）
│   ├── spectral_clustering.py # Spectral Clustering（独立实现）
│   ├── compare_all_methods.py # ★ 多 proposal 方法 × 6 样本对比脚本
│
├── run_proposal_retrieval_pipeline.py # 一键运行 proposal retrieval baseline
│
└── scripts/                 # pipeline 调用的可复用 helper
    ├── __init__.py
    ├── run_proposal_local_retrieval.py
    ├── evaluate_proposal_retrieval.py
    └── visualize_topk_retrieval.py
```

---

## 快速开始

### 1. 统一聚类接口

所有 proposal 方法通过同一函数调用：

```python
from partial_match.core.clustering import cluster

# defect_mask: (H, W) bool，True = 缺陷点
# valid_mask:  (H, W) bool，True = 有效芯片位置（die + defect）
clusters = cluster(defect_mask, valid_mask, method='dbscan')
```

返回 `List[Dict]`，每个 cluster 包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `area` | int | 像素面积 |
| `centroid_row` / `centroid_col` | float | 质心坐标 |
| `bbox_row_min/row_max/col_min/col_max` | int | 包围盒 |
| `pca_lambda1` / `pca_lambda2` | float | PCA 主轴长度，可判断形状 |
| `orientation` | float | PCA 主方向（度） |
| `compactness` | float | 紧密性 = perimeter / area |
| `radial_distance_norm` | float | 质心到晶圆中心的归一化距离 |
| `pixels` | List[Tuple] | `[(row, col), ...]` |
| `pixel_coords` | List[Dict] | `[{'row': r, 'col': c}, ...]` |

### 2. 方法列表与参数

| method | 论文 | 说明 | 额外参数 |
|--------|------|------|----------|
| `'raw'` | — | 8-连通域，不做任何处理 | — |
| `'filtered'` | — | 8-连通域 + 面积 ≥ min_area | `min_area=3` |
| `'adhesion'` | — | 8-连通域 + 对可疑粘连 component 二次拆分 | `split_method='tv_hybrid'`, `suspicious_area=40`, `min_suspicious_cues=2`, `max_split_count=6` |
| `'dilated_group'` | — | 膨胀仅用于 grouping，token 仍使用原始像素 | `dilation_radius=1`, `use_closing=False` |
| `'dilated_adhesion'` | — | 先 dilated grouping，再对可疑 group 做 adhesion split | `dilation_radius=1`, `split_method='tv_hybrid'` |
| `'group_then_adhesion'` | — | filtered -> dilated grouping -> adhesion split（ablation） | `min_area=3`, `dilation_radius=1`, `skip_ring_like=True` |
| `'geometry_merge'` | — | filtered -> radial ring split / adhesion -> component-level geometry merge | `min_area=5`, `ring_theta_gap=55`, `line_gap=11`, `blob_gap=3` |
| `'topk'` / `'compact'` | — | 基于候选 proposal 选择面积最大的 K 个主要区域 | `top_k=5`, `base_method='geometry_merge'` |
| `'topk_dilated'` | — | 基于 `dilated_adhesion` 选择面积最大的 K 个主要区域 | `top_k=5` |
| `'closing'` | — | 3×3 cross 闭运算后连通域 | — |
| `'simi_paper'` | Wang '22 | Closing → Spatial Filter → 强缺陷聚类 | — |
| `'dbscan'` | Koo & Hwang '21 | k-distance 自动 eps + DBSCAN | `eps`, `min_samples=5`, `auto_eps=True` |
| `'adjacency_iwmm'` | Ezzat et al. '20 | AC 空间过滤 → DP-GMM | `min_degree=2`, `max_components=8` |
| `'spectral'` | Wang et al. | RBF 相似度 → 谱聚类 | `sigma=3.0`, `n_clusters`, `auto_k=True` |
| `'tensor_voting'` / `'tv'` | Wang et al. '22 | Tensor Voting → Saliency 过滤 → 连通域 | `sigma=5.0`, `noise_ratio=0.3` |

**公共参数**：所有方法均支持 `use_clean=True`，启用 AC 预清洗（详见下方清洗章节）。

**推荐默认方法**：retrieval 使用 `geometry_merge`，即 `filtered -> adhesion -> component-level geometry merge`，再用 `topk(base_method='geometry_merge')` 控制 token 数。`dilated_group` / `group_then_adhesion` 不再作为默认方案，只保留为对照 / ablation 方法，因为像素膨胀容易把多个相近但不同的 pattern 联通。`closing` 会补断裂，但也容易把圆形、线段等轻微接触区域进一步粘在一起，因此不作为 mixed-pattern retrieval 的默认方法。

### 2.1 Adhesion proposal

`adhesion` 的设计目标不是完美恢复真实 pattern，而是在连通域欠分割时做适度 over-segmentation：

1. 先用 8-连通域提取 component，并过滤小噪声。
2. 对面积较大且同时满足多个形状异常信号的 component 判定为可疑粘连，默认需要 `min_suspicious_cues=2`。
3. 仅对可疑 component 调用 `adhesion_split.py` 中的粘连拆分方法，默认 `tv_hybrid`。
4. 若拆分失败、只得到一个区域、碎片数超过 `max_split_count=6`，或保留面积低于 `min_split_coverage=0.75`，则回退到原始 component。

示例：

```python
clusters = cluster(
    defect_mask,
    valid_mask,
    method='adhesion',
    min_area=3,
    split_method='tv_hybrid',
    suspicious_area=40,
    min_suspicious_cues=2,
    max_split_count=6,
    min_split_coverage=0.75,
)
```

### 2.2 TopK compact proposal

`topk` 是最终检索推荐使用的 compact proposal。它不保留所有碎片，而是先生成候选区域，再只取面积最大的 K 个主要区域：

```text
defect mask
 -> base_method candidates, default geometry_merge
 -> area >= min_area
 -> sort by area desc
 -> keep top_k regions
```

示例：

```python
clusters = cluster(
    defect_mask,
    valid_mask,
    method='topk',
    top_k=5,
    base_method='geometry_merge',
    min_area=5,
)
```

对于 52×52 map，建议先用 `top_k=5`。默认优先使用 `geometry_merge` 作为 compact retrieval base；如果需要更保守的对照，可再比较 `filtered`、`adhesion` 和 `dilated_group`。

### 3. 示例

```python
import numpy as np
from partial_match.core.clustering import cluster

# 加载 WM38K 数据
data = np.load("data/wm38k/Wafer_Map_Datasets.npz")['arr_0']
raw_map = data[100]                          # 第 100 张 wafer

# 构建 mask
defect_mask = (raw_map == 2)                  # 缺陷芯片
valid_mask  = (raw_map == 1) | (raw_map == 2) # 有效区域

# 任意方法（无清洗）
for method in ['topk', 'filtered', 'adhesion', 'simi_paper', 'dbscan', 'adjacency_iwmm', 'spectral', 'tensor_voting']:
    clusters = cluster(defect_mask, valid_mask, method=method)
    print(f"{method:20s} → {len(clusters):>3} clusters")

# 带 AC 清洗
clusters_clean = cluster(defect_mask, valid_mask, method='dbscan', use_clean=True)

# 带参数调用
clusters = cluster(defect_mask, valid_mask,
                   method='dbscan', eps=3.0, min_samples=4)

clusters = cluster(defect_mask, valid_mask,
                   method='adhesion', split_method='tv_hybrid')

clusters = cluster(defect_mask, valid_mask,
                   method='topk', top_k=5, base_method='geometry_merge', min_area=5)

clusters = cluster(defect_mask, valid_mask,
                   method='topk_dilated', top_k=5, dilation_radius=1)

clusters = cluster(defect_mask, valid_mask,
                   method='tensor_voting', sigma=6.0, noise_ratio=0.25)
```

---

## 运行对比可视化

生成多种 proposal 方法 × 6 个样本的综合对比图：

```bash
cd wbm-wdm-matching
python3 partial_match/cluster_test/compare_all_methods.py
```

输出（保存到 `artifacts/week1/figures/`）：
- `all_proposal_methods_comparison.png` — 多方法 × 6 样本（**无清洗**）
- `all_proposal_methods_cleaned_comparison.png` — 多方法 × 6 样本（**AC 清洗后**）
- `before_vs_after_cleaning.png` — 每种方法 × 2 样本的清洗前/后并排对比

---

## Proposal-based Local Retrieval 系统测试

当前 proposal retrieval baseline 使用 `arc-ring-residual` proposal、Zernike/几何描述子、硬门槛和贪心一对一局部匹配。推荐使用一键入口完成检索和 TopK review 图生成：

```bash
cd wbm-wdm-matching
python3 partial_match/run_proposal_retrieval_pipeline.py \
  --data-file ../data/wm38k/Wafer_Map_Datasets.npz \
  --max-samples 512 \
  --sample-strategy stratified \
  --seed 42 \
  --out-dir ../artifacts/proposal_based/system_test_512_stratified \
  --review-max-queries 64 \
  --review-top-k 3 \
  --metric-k 1 3 5 10
```

正式对比实验必须使用 `shared` 生成的固定 split/query manifest，以保证它和 `Wafer-DenseIR` 使用完全一致的 test query set 和 candidate pool：

```bash
cd /Users/kayw/Documents/trae_projects/match-test/wbm-wdm-matching
python3 partial_match/run_proposal_retrieval_pipeline.py \
  --data-file ../data/wm38k/Wafer_Map_Datasets.npz \
  --split-manifest artifacts/splits/wm38k_seed2026_sig_70_10_20.csv \
  --query-manifest artifacts/splits/wm38k_seed2026_test_queries_2000.csv \
  --candidate-manifest artifacts/splits/wm38k_seed2026_test_candidates_1000.csv \
  --split test \
  --out-dir artifacts/partial_match/pretrain/resnet18 \
  --review-max-queries 64 \
  --review-top-k 3 \
  --metric-k 1 5 10
```

该正式模式中：

```text
query set = artifacts/splits/wm38k_seed2026_test_queries_2000.csv
candidate pool = artifacts/splits/wm38k_seed2026_test_candidates_1000.csv 中每个 query 固定 1000 个 candidates
official metrics = label_metrics.json / label_metrics_flat.csv
```

上述 pipeline 会依次调用 `scripts/` 下的 helper：

```text
run_proposal_local_retrieval.py
evaluate_proposal_retrieval.py
visualize_topk_retrieval.py
```

输出目录示例：

```text
artifacts/proposal_based/system_test_512_stratified/
├── rankings.csv
├── tokens.csv
├── descriptors.npz
├── metrics_summary.json
├── metrics_summary_flat.csv
├── label_metrics.json
├── label_metrics_flat.csv
└── top3_review/
```

早期 `week1_*` 探索脚本已废弃并删除，旧的 `artifacts/week1/` 仅作为历史结果保留。

---

## Tensor Voting Paper（独立模块）

`tensor_voting_paper.py` 实现了完整的论文流程（Tensor Voting + MBBS 相似度），与 `clustering.py` 中的 `tensor_voting` 方法一致，但额外提供：

```python
from partial_match.core.tensor_voting_paper import paper_pipeline, mbbs_score

# 完整 TV pipeline（返回 saliency map + 保留点）
result = paper_pipeline(defect_mask, sigma=5.0, noise_ratio=0.3)
# result['kept_points'], result['stick_sal_map'], result['ball_sal_map'], ...

# MBBS 相似度
score = mbbs_score(query_pts, query_sal, cand_pts, cand_sal, w=1.0)
```

---

## 方法对比总结

| 方法 | 不需预设 k | 分离粘连 | 任意形状 | 自动滤噪 | 混合图案 | 速度 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| 8-Connected | ✅ | ❌ | ❌ | ❌ | ❌ | 快 |
| Filtered | ✅ | ❌ | ❌ | ✅ | ❌ | 快 |
| Adhesion Split | ✅ | 部分 | ✅ | 部分 | ✅ | 中 |
| Dilated Group | ✅ | ❌ | ✅ | 部分 | 部分 | 快 |
| Dilated + Adhesion | ✅ | 部分 | ✅ | 部分 | ✅ | 中 |
| TopK Compact | ✅ | 部分 | ✅ | 部分 | ✅ | 快 |
| SIMI Paper | ✅ | 部分 | ❌ | ✅ | ❌ | 快 |
| DBSCAN | ✅ | ✅ | ✅ | ✅ | ✅ | 中 |
| Adj-Cluster + iWMM | ✅ | ✅ | ✅ | ✅ | ✅ | 慢 |
| Spectral Clustering | ✅ | ✅ | ✅ | 部分 | ✅ | 慢 |
| Tensor Voting | ✅ | ✅ | ✅ | ✅ | ❌ | 慢 |

**推荐**：检索 token 默认使用 `topk(base_method='geometry_merge')`。`geometry_merge` 不做像素膨胀，而是在 `filtered -> radial ring split / adhesion` 候选上按 ring / line / local blob 几何一致性合并截断 fragments。radial ring split 会优先尝试把 donut / edge-ring 半径带从 scratch 粘连 component 中拆出来；ring merge 使用像素角度区间而不是质心角度，以便合并断裂 ring。`dilated_group` / `topk_dilated` 仅保留为 ablation 方法；`adjacency_iwmm`、`spectral` 和 `tensor_voting` 保留为 proposal 对比方法。

---

## 清洗方法详解

每种聚类方法的**前处理 / 空间过滤**策略：

### 1. Closing（形态学闭运算）

| 属性 | 内容 |
|------|------|
| **论文来源** | Xu et al. 2021 (CSTIC) — SIMI Paper |
| **核心思想** | 先膨胀再腐蚀（3×3 cross kernel），填补缺陷簇中的小缝隙，消除孤立的单点噪声 |
| **速度** | 极快（100 万张晶圆仅 30.98 s，比 OPTICS 快 25 倍） |

### 2. Spatial Filter（SIMI Paper 三值过滤器）

| 属性 | 内容 |
|------|------|
| **论文来源** | Xu et al. 2021 (CSTIC) — SIMI Paper |
| **核心思想** | 在 CLOSING 之后使用 5×5 窗口统计邻域缺陷密度：近邻 ≥ 3 个记为 `1`（强缺陷），轮廓 ≥ 5 个记为 `0.5`（边缘），其余归零 |
| **输出** | 三值图 {0, 0.5, 1}，其中 `1` 的连通域作为最终 cluster proposal |

### 3. Adjacency-Clustering（图论空间过滤）

| 属性 | 内容 |
|------|------|
| **论文来源** | Ezzat et al. 2020 — AC-iWMM |
| **核心思想** | 将晶圆建模为 8-邻接图，利用**偏差代价**（标签偏离观测值）和**分离代价**（相邻芯片标签不同）构建最小割问题。孤立噪声因分离代价远大于偏差代价被洗白；成片缺陷因偏差代价远大于分离代价被保留 |
| **算法** | 最小 s-t 割，多项式时间，数千芯片秒级求解 |
| **本实现** | 简化版：迭代移除邻接缺陷数 < `min_degree` 的孤立点，收敛后送入 DP-GMM 聚类 |

> **论文详细原理**：AC 在二值标签下退化为 minimum s-excess 问题，构造图 \( G_{st} = (V \cup \{s, t\}, A_{st}) \)，源点侧节点标为 1（系统缺陷），汇点侧标为 0（随机噪声）。详见 [Ezzat et al. 2020](file:///Users/kayw/Documents/trae_projects/match-test/related_works/2006.13824v2.pdf) Section 2.1 和 Page 9。

### 4. Tensor Voting（张量投票显著性过滤）

| 属性 | 内容 |
|------|------|
| **论文来源** | Wang et al. 2022 (DSIT) |
| **核心思想** | 基于 Gestalt 感知原则（邻近性 + 良好连续性），每个缺陷点向邻域投票传播结构信息。成簇点相互加强获得高显著性，孤立噪声因无投票支持获得低显著性 |
| **方法** | Guy-Medioni Ball Voting：累积邻域投票张量 \( T = \sum e^{-d^2/\sigma^2} (I - vv^T) \)，分解为 stick saliency (\( \lambda_1 - \lambda_2 \)，曲线结构) 和 ball saliency (\( \lambda_2 \)，区域结构)，低于阈值的点作为噪声过滤 |
| **参数** | `sigma`（投票尺度，默认 5.0）、`noise_ratio`（噪声阈值比例，默认 0.3） |

> **论文详细原理**：Tensor Voting 将每个点编码为二阶对称张量，通过投票传播后分解为 stick/ball 分量，曲线类缺陷（scratch、ring）具有高 stick 显著性，区域类缺陷（zone、center）具有高 ball 显著性，孤立噪声两者皆低。详见 [Wang et al. 2022](file:///Users/kayw/Documents/trae_projects/match-test/related_works/Tensor_Voting_Based_Similarity_Matching_of_Wafer_Bin_Maps_in_Semiconductor_Manufacturing.pdf) Section II-A。

### 5. ESRN — 邻域多数投票（Hsu et al. 2020，未实现）

| 属性 | 内容 |
|------|------|
| **论文来源** | Hsu et al. 2020 — WMHD Similarity Matching |
| **核心思想** | 对每个 die 检查其 King-Move 8 邻域，若好 die 比例达到阈值则翻转为好，若坏 die 比例达到阈值则翻转为坏。本质是空间平滑投票 |
| **后续** | 论文还使用 Modified Mountain Function 做特征变换，再用 Weighted MHD 做相似度匹配 |
| **状态** | 本仓库未实现，作为参考方法 |

### 6. DBSCAN（密度聚类）

| 属性 | 内容 |
|------|------|
| **论文来源** | Koo & Hwang 2021 |
| **核心思想** | 基于密度的空间聚类，自动识别任意形状的簇并分离噪声（label = -1） |
| **清洗机制** | DBSCAN 本身包含噪声识别：密度不足的孤立点被标记为 `-1`，天然具备滤噪能力，无需前置清洗步骤 |
| **参数** | `eps`（邻域半径，支持 k-distance 自动估计）、`min_samples`（最小邻域点数，默认 5） |

### 清洗方法对比

| 方法 | 清洗方式 | 理论基础 | 输出形式 | 速度 |
|------|---------|----------|---------|:--:|
| Closing | 形态学闭运算 | 数学形态学 | 填补后二值图 | 极快 |
| SIMI Spatial Filter | 5×5 窗口密度投票 | 空间统计 | 三值图 {0, 0.5, 1} | 快 |
| Adjacency-Clustering | 图最小割 / 迭代度过滤 | 图论 / MRF | 二值 mask（系统缺陷/噪声） | 秒级 |
| Tensor Voting | 张量投票 + 显著性阈值 | Gestalt 感知 / 微分几何 | 保留高显著性点 | 慢 |
| ESRN | 8-邻域多数投票 | 空间平滑 | 二值图 | 快 |
| DBSCAN | 密度阈值（内建） | 密度聚类 | 标签（含 -1 噪声类） | 中 |

---

## 清洗前后聚类对比

运行对比脚本生成三张可视化图（含 Tensor Voting）：

```bash
cd wbm-wdm-matching
python3 partial_match/cluster_test/compare_all_methods.py
```

### 清洗效果

| 样本 | 清洗前缺陷点 | 清洗后缺陷点 | 移除比例 |
|:--:|:--:|:--:|:--:|
| 100 | 759 | 576 | 24.1% |
| 500 | 656 | 432 | 34.1% |
| 800 | 499 | 244 | **51.1%** |
| 1200 | 549 | 322 | 41.3% |
| 2000 | 658 | 481 | 26.9% |
| 3500 | 699 | 519 | 25.8% |

### 各方法聚类数对比（清洗前 → 清洗后）

| 方法 | Sample 100 | Sample 500 | Sample 800 | Sample 1200 | Sample 2000 | Sample 3500 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| 8-Connected | 83→11 | 103→12 | 141→22 | 138→22 | 108→22 | 102→18 |
| Filtered | 24→11 | 33→11 | 38→21 | 38→21 | 33→22 | 29→18 |
| Adhesion Split | 32→18 | 35→13 | 40→24 | 38→21 | 34→24 | 32→20 |
| Dilated Group | 1→5 | 1→8 | 1→13 | 2→10 | 1→15 | 1→6 |
| Dilated + Adhesion | 7→13 | 14→16 | 24→18 | 18→14 | 9→17 | 14→10 |
| **TopK Compact (k=5)** | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 |
| **TopK Dilated (k=5)** | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 | 5→5 |
| Closing | 31→9 | 45→13 | 64→22 | 60→15 | 38→18 | 29→13 |
| **SIMI Paper** | 8→8 | 21→12 | 18→21 | 19→14 | 17→18 | 9→13 |
| DBSCAN | 2→2 | 10→2 | 1→4 | 1→3 | 1→4 | 1→1 |
| Adj-Cluster+iWMM | 6→6 | 6→6 | 7→7 | 6→6 | 7→7 | 6→6 |
| Spectral | 10→10 | 9→8 | 10→8 | 8→10 | 9→9 | 8→6 |
| Tensor Voting | 35→10 | 27→6 | 112→11 | 123→13 | 42→18 | 72→16 |

> **关键发现**：
> - AC 清洗对简单方法（8-Connected、Filtered、Closing）提升最大，聚类数大幅下降
> - Adhesion Split 会对可疑大连通域做二次拆分，比 Filtered 更能处理圆形、线段轻微粘连，但会引入适度过分割
> - Dilated Group 会先把断续碎片合并成 group，但 token 面积仍基于原始 defect pixels，不包含膨胀像素
> - Dilated + Adhesion 在合并断续结构后再拆可疑 group，可作为 ring/scratch 断裂场景的候选方案
> - TopK Compact 将最终 retrieval token 数稳定控制在 `k` 个以内，默认 `k=5`
> - SIMI Paper 自带 Spatial Filter，清洗后变化适中但在高噪声样本（800）反而从 18→21（清洗后点更干净，8-连通域不再被噪声桥接打断）
> - DBSCAN、Adj-Cluster+iWMM、Spectral 已内置噪声处理，受清洗影响较小
> - **Tensor Voting 未清洗时碎片化严重**（112、123 个 cluster），清洗后降至合理水平（11、13），说明 TV 对输入噪声非常敏感
