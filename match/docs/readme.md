# WBM-WDM Matching

## 1. 运行命令

### 1.1 两种模式

`--mode` 参数控制匹配流程，所有日志与图表自动保存到 `<output-dir>/<identifier>/<mode>/`。以下命令均从仓库根目录执行；`--klarf-dir`、`--reference` 为必填参数：

| 模式 | 计算内容 | 自动输出 |
|---|---|---|
| `count-partial`（默认） | 完整 WDM 的 IoU 基线和局部 token 匹配 | `results.tsv`、`topk.tsv`、`token_match.tsv`、`map_match.tsv`、参数快照及基线、局部匹配、原始 WDM 和汇总图表 |
| `classnumber` | 按 KLARF `CLASSNUMBER` 拆分 WDM，并选择得分最高的子图 | 与 `count-partial` 相同的日志结构，另含最佳 classnumber、count/binary 局部匹配及对应图表 |

```bash
# count-partial — 默认模式，完整 WDM 的 IoU 基线 + 局部匹配
python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --representation density \
  --mode count-partial \
  --identifier AF00138

# classnumber — 按 CLASSNUMBER 拆分；count/binary 匹配路径由 --representation 推导
python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --representation count \
  --mode classnumber \
  --identifier AF00138

# 也可用自定义 JSON 文件提供默认参数，命令行参数优先
python3 -m match.scripts.main \
  --config /path/to/match_config.json
```

CLI 参数优先级高于配置文件。

### 1.1.1 稀疏缺陷 proposal

当 WBM 和 WDM 都由分散点构成、普通连通域无法恢复肉眼可见的环、带状或线状模式时，可使用 `sparse-density` proposal。WDM 仍先映射到与 WBM 完全一致的网格；两侧随后都把缺陷格作为 impulse，在同一组 grid-cell 尺度上生成高斯密度场、提取阈值 support，并进行跨尺度 token 去重。support 仅用于将邻近真实缺陷归为同一候选簇及绘制轮廓；token 的像素、面积、几何统计、描述子和匹配分数均只使用 support 内的原始缺陷格。token 会保留 `raw_pixels`、`raw_mass`、`raw_point_count`、`proposal_scale` 和 `kde_support_pixels` 作为证据与追溯信息。该功能同时适用于 `--mode count-partial` 和 `--mode classnumber`。

```bash
# count-partial 模式（sparse-density proposal）
python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --representation density \
  --mode count-partial \
  --proposal-mode sparse-density \
  --density-sigmas 0.8 1.6 3.2 \
  --density-threshold 0.20 \
  --identifier AF00138_sparse

# classnumber 模式（sparse-density proposal，按 classnumber 拆分后每子 WDM 独立提取 token）
python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --representation count \
  --mode classnumber \
  --proposal-mode sparse-density \
  --density-sigmas 0.8 1.6 3.2 \
  --density-threshold 0.20 \
  --identifier AF00138_sparse_class
```

`--density-sigmas` 的单位是对齐后 WBM 网格的 cell（die）尺度，而不是 PNG 像素。`--density-threshold` 是每个尺度相对于该密度图峰值的 support 阈值；`--density-min-raw-points` 和 `--density-min-raw-mass` 用于拒绝没有足够原始缺陷证据的平滑候选。WDM 默认使用 `sqrt(count)` 作为 KDE 权重，可通过 `--density-weight-transform count|sqrt|log1p` 调整。使用 `--proposal-mode auto` 时，系统会根据碎片化程度和原始证据在两侧共同选择 `cc` 或 `sparse-density`；`cc` 仍是默认模式。

### 1.1.2 切向断裂环 proposal

`tangential-ring` 是独立于 `compact` 的高召回环状 proposal。它先在原始外圈缺陷点上估计主环半径，并将环宽限制在最多两个 die；随后仅在极坐标的角度轴上桥接不超过两个 die 的短缺口。桥接结果只记录为 `ring_contour_bins` 和覆盖率证据，绝不生成 token 像素；因此 token 的面积、几何统计、描述子和匹配始终只使用原始缺陷格。检测到的 ring 从 residual 中按原始像素扣除，其余区域仍按连通域提取。

```bash
python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --representation count \
  --mode count-partial \
  --proposal-mode tangential-ring \
  --identifier AF00138_tangential_ring
```

该模式不要求 ring 贴到晶圆最外边界，而是在 `ring_edge_r_min` 之外寻找主径向带。它适合小图中只断开一两个 die 的环；较大截断、宽外侧区域或与环半径不一致的散点不会被切向桥接补成真实 token 像素。

### 1.2 CP CSV 参考图预处理

另一类 CP test 输入可先从 CSV 拆分为逐 wafer、逐 hardbin 的 WBM 参考图：

```bash
python3 -m match.scripts.prepare_cp_refs \
  --cp-csv /path/to/cp_result.csv \
  --out-dir match/output/cp_refs
```

默认按 `Lot_Id/Wafer_Number` 隔离 wafer，并使用 `Die_X/Die_Y` 生成 PNG。输出结构：

```text
cp_refs/
  Lot_Id/
    Wafer_Number/
      metadata.json
      die_results.csv
      pf.png
      hardbin/
        1_PASS.png
        42_FAIL_A.png
        hardbin_index.tsv
```

PNG 仍使用三值编码：黑色=无 die/晶圆外，灰色=有效 die 但非当前类别，白色=当前 PF fail 或 hardbin 命中。`hardbin_index.tsv` 记录每张 hardbin PNG 对应的 `hardbin_number`、`hardbin_name`、命中 die 数和文件名，便于批量调用 `main.py --reference <hardbin_png>` 并追溯结果。

### 1.3 批量对比实验

通过 JSON 定义多条实验，调用 `batch_run` 依次执行：

```bash
python3 -m match.scripts.batch_run \
  --experiments batch.json
```

JSON 格式 (`batch.json`)：

```json
{
  "common": {
    "mode": "count-partial",
    "mapper": "physical-coordinate",
    "representation": "density",
    "die_x_range": [-20, 20],
    "die_y_range": [-20, 20],
    "topk": 10,
    "proposal_mode": "cc"
  },
  "experiments": [
    {
      "klarf_dir": "/data/klarf_batch1/",
      "reference": "/data/wm811k/000604.png",
      "identifier": "batch1_density"
    },
    {
      "klarf_dir": "/data/klarf_batch1/",
      "reference": "/data/wm811k/000610.png",
      "identifier": "batch1_count",
      "representation": "count"
    }
  ]
}
```

`common` 定义所有实验共享的默认参数（可选），每个实验**必须**提供 `klarf_dir`、`reference`、`identifier`，可按需覆盖 `common` 中任意参数。`identifier` 必须唯一，结果输出到 `<output-dir>/<identifier>/`。

---

## 2. 项目定位

给定一张目标 WBM（Wafer Bin Map，芯片测试分 bin 图）和一组候选 WDM（Wafer Defect Map，KLARF 缺陷扫描数据），找到能最佳「解释」该 WBM 的 WDM 稀疏子集。

核心挑战是**跨数据源匹配**——WBM 来自电性测试（黑白灰三值，粒度为 die），WDM 来自光学扫描（散点，粒度为亚 die 坐标），两者粒度、来源、语义均不同。

输出包括：全局相似度排名、count-map 局部 token 匹配分数、可选按 defect classnumber 拆分的分图匹配、以及匹配过程的可视化图表（token 提取、聚类着色、匹配证据表）。

---

## 3. 核心流水线与代码结构

### 3.1 数据流

入口 `scripts/main.py` 先加载 WBM 参考图（`scripts/reference_loader.py`），再对 KLARF 目录下每个文件执行 `scripts/processing.py → process_one`，最后汇总写入 TSV、参数快照和图表。缺陷数低于 `--defect-threshold` 的文件会标记为跳过，不进入结果表。

```
                         scripts/main.py
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
           reference_loader  processing  batch_io / batch_viz
              (加载WBM)     (逐文件处理)  (TSV + 图表输出)
```

**`process_one` 对每个 KLARF 依次执行五步：**

```
KLARF 文件
  │
  ├─ ① Defect 阈值检查 ── data/fileio.py → load_defect_tables
  │    缺陷数 < threshold → SKIPPED
  │
  ├─ ② Map: 坐标映射 + 网格表达 ── core/pipeline.py → map_klarf_to_grid
  │    ├── core/mappers.py       坐标映射 (die-index / relative-coordinate / physical-coordinate)
  │    └── core/representations.py  网格表达 (count / binary / density / soft / three-value / mountain)
  │    输出: GridMaps (与 WBM 同尺寸 H×W)
  │
  ├─ ③ Global Similarity ── core/similarity.py → compute_similarity (method="iou")
  │    IoU (|A∩B|/|A∪B|) 作为当前批处理基线指标
  │
  ├─ ④ Count-Partial Match ── core/local_matching/scoring.py → explain_count_partial_match
  │    ├── 4a. Token 提案 ── proposal.py
  │    │      WBM: _tokens_from_mask (status_map == VALID_HAS_DEFECT)
  │    │      WDM: _tokens_from_weighted_mask (count_map > 0, count 作为权重)
  │    │      模式: cc、compact、tangential-ring、sparse-density 或 auto
  │    ├── 4b. 形状描述符 ── descriptors.py
  │    │      每个 token → Zernike 矩 (48×48, 8阶) + 几何特征 → 拼接归一化
  │    ├── 4c. Token 对打分 ── scoring.py → _token_match_components
  │    │      shape_sim + position_aff + scale_aff → token_score
  │    ├── 4d. 贪心一对一匹配 ── _greedy_one_to_one_matches
  │    └── 4e. √area 加权聚合 → result + result_matched_only
  │         同时输出 token_topk_matches + map_topk_matches (供图表使用)
  │
  └─ ⑤ Classnumber Match (可选, --mode classnumber)
        ├── data/fileio.py → split_defect_table_by_classnumber
        │     按 defect classnumber 拆分为 N 个子 WDM
        └── core/classnumber_matching.py → compute_classnumber_matches
              对每个子 WDM 重新执行 ② Map + ④ Count-Partial Match
              (+ 可选 binary-partial via explain_binary_partial_match)
              选 rank_score 最高的 classnumber → best-classnumber
```

**后处理（`main.py` 汇总后）：**

```
rows (所有文件的结果)
  │
  ├── scripts/batch_io.py
  │     write_result_log    → results.tsv (每文件一行, 含 iou + partial + classnumber)
  │     write_topk_log      → topk.tsv     (各指标 top-K 排名)
  │     write_token_match_log → token_match.tsv  (每 WBM token 的 top-K WDM 匹配)
  │     write_map_match_log   → map_match.tsv    (每 map 的最高分 token 对)
  │
  └── scripts/batch_viz.py（按 mode 分支）
        ├── count-partial 模式:
        │     ├── save_baseline_figures → baseline_review/
        │     │     (sum map IoU top-K, 左 WBM 右 WDM 对比图)
        │     ├── save_count_partial_figures → count_partial_review/
        │     │     viz/count_partial_visualization.py → plot_count_partial_topk + plot_count_partial_steps
        │     │     (result 与 result_matched_only 各出一套，并输出候选摘要图)
        │     └── save_wdm_raw_figures → wdm_raw_review/
        │           (top-K KLARF 物理坐标 wafer 原图, CLASSNUMBER=0 红色, !=0 蓝色)
        └── classnumber 模式:
              ├── save_classnumber_baseline_figures → baseline_review/
              │     (best split map rank_score top-K, 左 WBM 右 WDM 对比图)
              ├── save_classnumber_figures → classnumber_review/
              │     viz/classnumber_visualization.py → plot_classnumber_splits + plot_classnumber_topk_splits + plot_classnumber_step
              └── save_classnumber_wdm_raw_figures → wdm_raw_classnumber_review/
                    (top-K classnumber split 物理坐标 wafer 原图, 仅该 classnumber 的缺陷, 蓝色)

### 3.2 代码结构

```text
match/
  core/
    models.py              # GridMaps、DefectTable、状态常量
    mappers.py             # 3 种坐标映射器（die-index / relative-coordinate / physical-coordinate）
    representations.py     # 6 种网格表达（binary / count / density / soft / three-value / mountain）
    pipeline.py            # map_klarf_to_grid 端到端入口
    similarity.py          # 全图相似度实现（当前批处理默认记录 IoU）
    local_matching/        # count-partial 局部匹配
      models.py            # LocalMatchResult、ProposalConfig
      morphology.py        # 连通域提取、形态学操作
      proposal.py          # token 提案生成（cc / compact / tangential-ring / sparse-density / auto）
      descriptors.py       # shape descriptor（Zernike 矩 + 几何特征）
      scoring.py           # token 配对打分、贪心匹配、分数聚合
    classnumber_matching.py # classnumber 分图匹配
  data/
    fileio.py              # KLARF 解析、WBM PNG 解码
  viz/
    visualization.py              # 通用对比图
    count_partial_visualization.py # count-partial 步骤图 / TopK 图
    classnumber_visualization.py  # classnumber 分图可视化
    klarfkit.py                   # KLARF 物理坐标 wafer 原图绘制（来自 MichaelHotaling/klarfkit）
  scripts/
    cli_args.py    # CLI 参数定义与列名常量
    main.py        # 批量处理入口
    prepare_cp_refs.py # CP CSV → per-wafer/per-hardbin PNG 参考图
    processing.py  # 单文件处理编排
    batch_io.py    # TSV / TopK / token-match 日志写入
    batch_viz.py   # 批处理图表生成调度
  test/
    test_proposal.py       # WM38K 局部提案与匹配 smoke test
    test_sparse_density.py # sparse-density proposal 回归测试
  requirements.txt         # NumPy、Matplotlib、Pandas、klarfio 依赖
```

---

## 4. 技术思路

### 4.1 Map：坐标映射与网格表达

KLARF 中的每个缺陷带有 die 索引（XINDEX/YINDEX）和 die 内相对坐标（XREL/YREL）。Mapper 将缺陷散点映射到与 WBM 同尺寸的 H×W 网格中：`physical-coordinate` 合并 die 索引与 die 内坐标，计算归一化位置后按比例映射到网格格点；`die-index` 和 `relative-coordinate` 分别只用 die 索引或 die 内坐标做线性缩放。

映射后统计每个格点的缺陷数得到 `count_map`，再派生 `binary_map`（count ≥ `--die-defect-threshold`）、`density_map`（总质量归一化）、`soft`、`three-value` 和 `mountain` 等表示。网格始终以 WBM 尺寸为准，确保两张图可逐格点比对。

WBM 使用三值语义：白色=有缺陷 die（`VALID_HAS_DEFECT=2`）、灰色=无缺陷但晶圆内（`VALID_NO_DEFECT=1`）、黑色=晶圆外背景（`BACKGROUND=0`）。所有后续计算只在有效区域（status ∈ {1,2}）内进行。

### 4.2 Proposal：Token 提取

从 WBM 和 WDM 的 mask 中提取若干局部区域作为 token，每个 token 代表一个有意义的缺陷聚集区。CLI 支持五种模式：

- **CC（Connected Component）**：BFS 连通域提取（4-连通或 8-连通），过滤面积 < `min_area` 的小碎片，按重要性（√mass + √area + 类型加分）排序后取 top-k
- **Compact**：先提取环状候选，再从残差中提取 component token（分类为 blob / line / central / irregular），按重要性与类型多样性保留候选
  - 对短边 ≤12 的小图，Compact 先对原始 mask 做一次受有效区域约束的 `3×3` closing，仅作为 ring 连通性与轮廓证据；ring token 的像素、面积、几何统计和描述子仍只使用原始缺陷格，残余 component 也由去噪后的原始 mask 提取
  - 小图 ring-aware 默认使用最多 24 个角度扇区、最少 6 个 ring-band cell、最少 0.10 的角度覆盖率、最多 0.18 的径向标准差和最多 0.60 的缺陷覆盖率；较大图维持原有的 72 扇区与更严格阈值。可通过 `--ring-min-area`、`--ring-edge-r-min`、`--ring-band-width`、`--ring-min-angular-coverage`、`--ring-angular-bins`、`--ring-max-radial-std`、`--ring-max-defect-ratio`、`--ring-min-edge-defect-fraction` 显式调节
- **Tangential-ring**：仅以原始外圈缺陷点构造受限径向带，并在极坐标角度轴上桥接最多两个 die 的短缺口。桥接只记录 contour 连续性，不新增 token 像素；ring 和 residual 按真实原始像素互斥。
- **Sparse-density**：用于 WBM/WDM 同时稀疏的情况。两侧在同一网格上以多尺度截断高斯核生成连续 density map，再从相对峰值阈值 support 中提取 token。每个尺度的候选按 IoU 去重，token 必须包含足够的原始点数和原始质量；因此 KDE 只改变 proposal 的派生 support，不会覆盖原始 WBM/WDM 数据。
- **Auto**：根据两侧 mask 的碎片化特征和原始证据，在 `cc` 与 `sparse-density` 间自动选择，并使一对 WBM/WDM 使用同一种 proposal 表示。

每个 token 记录以下几何属性：加权质心 (centroid_row, centroid_col)、bbox、PCA 特征值与方向、面积 (area)、质量 (mass)、周长、紧致度 (compactness)、归一化径向距离、角度覆盖度 (angular_coverage)、径向标准差 (radial_std)、几何类型 (geometry_type)。

### 4.3 Descriptor：形状描述符

每个 token 的形状描述符由两部分拼接并归一化到单位长度：

**Zernike 矩（默认权重 0.75）**：将 token 的 bbox 裁剪区域最近邻缩放到 48×48，映射到单位圆盘，计算 8 阶 Zernike 矩的幅值向量。

**几何特征（权重 0.25）**：`[fill_ratio, log(aspect), log(elongation), compactness, angular_coverage, radial_std, orientation_cos, orientation_sin]`。

若启用 `rotation_tolerance`，描述符退化为径向轮廓直方图（旧版方式）。

### 4.4 Score：Token 匹配与分数聚合

**Token 对打分**：对每对 (WBM token, WDM token)，计算三个维度的亲和度，组合为 token pair score：

- **Shape similarity**：两个描述符的余弦相似度；描述子内部默认按 Zernike 0.75、几何特征 0.25 融合。低于内部形状门限或未通过尺度比例门限的 token 对直接置零
- **Position affinity**：`exp(-d² / σ_pos²)`，d 为归一化质心间的欧氏距离
- **Scale affinity**：`w_area × area_aff + w_pca × pca_aff`
  - `area_aff = exp(-|log(area_q / area_c)| / σ_scale)`
  - `pca_aff = exp(-(0.75·|log(long_q/long_c)| + 0.25·|log(short_q/short_c)|) / σ_scale)`

```
token_score = w_shape × shape_sim + w_position × position_aff + w_scale × scale_aff
```

默认权重：`w_shape=0.60`、`w_position=0.25`、`w_scale=0.15`；可通过 `--token-score-*-weight` 调整。尺度亲和度内部默认等权融合支持面积与 PCA 长短轴范围。

**贪心一对一匹配**：所有 token 对按 score 降序排列，贪心选取不重复使用 token 的配对。

**分数聚合**：以每个 WBM token 的 √area 为权重，对匹配到的 token pair score 做加权平均：

- **result**：所有 scored WBM token 参与加权（未匹配到 WDM 的贡献 0）
- **result_matched_only**：仅实际匹配到的 WBM token 参与加权

```
final_score = Σ(matched_score_i × √area_i) / Σ(√area_i)
```

同时对 shape_sim、position_aff、scale_aff、type_aff 分别做同样的加权平均，输出 `mean_shape`、`mean_position`、`mean_scale`、`mean_type`。

**Classnumber 扩展**：将 KLARF 按 defect classnumber 拆分为多个子 WDM，每个子 WDM 独立做上述 count-partial（及可选 binary-partial）匹配，选出 rank_score 最高的 classnumber 作为 best-classnumber 输出。
