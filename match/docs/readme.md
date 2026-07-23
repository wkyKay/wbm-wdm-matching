# WBM-WDM Matching

## 1. 运行命令

### 1.1 两种模式

`--mode` 参数控制匹配深度，所有输出（TSV + 图表）自动保存到 `<output-dir>/<identifier>/<mode>/`：

| 模式 | 计算内容 | 自动输出 |
|---|---|---|
| `count-partial`（默认） | IoU（sum map） + count-partial token 匹配 | results.tsv, topk.tsv, token_match.tsv, map_match.tsv + baseline_review / count_partial_review / wdm_raw_review 图表 |
| `classnumber` | IoU（每个文件的最佳 classnumber split） + classnumber 拆分匹配 | results.tsv, topk.tsv, token_match.tsv, map_match.tsv + classnumber 列 + baseline_review / classnumber_review / wdm_raw_classnumber_review 图表 |

```bash
# count-partial — 默认模式，sum map baseline + count-partial 图表
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --representation density \
  --mode count-partial \
  --identifier AF00138

# classnumber — split map baseline + classnumber 拆分图表，match mode 自动由 --representation 推导
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --representation count \
  --mode classnumber \
  --identifier AF00138

# 也可用 --config 加载 JSON 配置
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.main \
  --config match/run_configs/partial_count.json
```

CLI 参数优先级高于配置文件。

### 1.1.1 稀疏缺陷 proposal

当 WBM 和 WDM 都由分散点构成、普通连通域无法恢复肉眼可见的环、带状或线状模式时，可使用 `sparse-density` proposal。WDM 仍先映射到与 WBM 完全一致的网格；两侧随后都把缺陷格作为 impulse，在同一组 grid-cell 尺度上生成高斯密度场、提取阈值 support，并进行跨尺度 token 去重。support 仅用于将邻近真实缺陷归为同一候选簇及绘制轮廓；token 的像素、面积、几何统计、描述子和匹配分数均只使用 support 内的原始缺陷格。token 会保留 `raw_pixels`、`raw_mass`、`raw_point_count`、`proposal_scale` 和 `kde_support_pixels` 作为证据与追溯信息。该功能同时适用于 `--mode count-partial` 和 `--mode classnumber`。

```bash
# count-partial 模式（sparse-density proposal）
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.main \
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
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.main \
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

`--density-sigmas` 的单位是对齐后 WBM 网格的 cell（die）尺度，而不是 PNG 像素。`threshold` 是每个尺度相对于该密度图峰值的 support 阈值；`--density-min-raw-points` 和 `--density-min-raw-mass` 用于拒绝仅由平滑产生、没有足够原始缺陷证据的候选。WDM 默认使用 `sqrt(count)` 作为 KDE 权重，可通过 `--density-weight-transform count|sqrt|log1p` 调整。

`sparse-density` 还会在每个 sigma 内做轻量 ring-aware merge：先把每个 density support component 当作候选 arc，使用原始缺陷点计算半径、径向标准差和角向覆盖率；半径相近的 arc 会被合并成一个候选 ring token。合并只改变 proposal 分组，不新增 token 像素；merged token 的 `pixels`、`area`、`mass`、描述子和匹配分数仍只来自原始缺陷格。角向覆盖率达到约 0.65 时标为 `edge_ring`，覆盖率约 0.25~0.65 时标为 `ring_arc`，用于记录“像环的一段”但不把半环当完整环处理。token 会记录 `proposal_source=sparse_density_ring_merge`、`ring_radius_norm`、`ring_radial_std`、`ring_angular_coverage`、`ring_occupied_bins` 和 `ring_arc_count` 作为调试证据。

若使用 `--proposal-mode auto`，系统会在任一侧由多个小连通域组成且满足最小原始证据时，让 WBM 与该 WDM 同时切换到 sparse-density；它不以图像尺寸作为判断条件。`cc` 仍是默认模式，以保持既有实验可复现。

### 1.1.2 切向断裂环 proposal

`tangential-ring` 是独立于 `compact` 的高召回环状 proposal。它先在原始外圈缺陷点上估计主环半径，并将环宽限制在最多两个 die；随后仅在极坐标的角度轴上桥接不超过两个 die 的短缺口。桥接结果只记录为 `ring_contour_bins` 和覆盖率证据，绝不生成 token 像素；因此 token 的面积、几何统计、描述子和匹配始终只使用原始缺陷格。检测到的 ring 从 residual 中按原始像素扣除，其余区域仍按连通域提取。

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --representation count \
  --mode count-partial \
  --proposal-mode tangential-ring \
  --identifier AF00138_tangential_ring
```

该模式不要求 ring 贴到晶圆最外边界，而是在 `ring_edge_r_min` 之外寻找主径向带。它适合小图中只断开一两个 die 的环；较大截断、宽外侧区域或与环半径不一致的散点不会被切向桥接补成真实 token 像素。

### 1.1.3 小图刚性叠合匹配

当 WBM 与 WDM 都是小网格，且目标是比较整体图案的实际重合程度时，可使用 `rigid-overlay`。这是独立于 `--proposal-mode` 的匹配打分模式：它不提取 token、不计算形状描述子，也不改变原始缺陷分布；而是直接对 WDM 的原始二值缺陷 mask 搜索离散旋转与小范围平移，并以最大重合分数作为匹配结果。默认仍为 `token`，因此既有命令和实验结果不受影响。

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --representation count \
  --mode count-partial \
  --small-map-match-mode rigid-overlay \
  --rigid-overlay-score dice \
  --rigid-overlay-max-shift 1 \
  --identifier AF00138_overlay
```

搜索角度固定为 `0, 45, 90, 135, 180, 225, 270, 315` 度；`--rigid-overlay-max-shift s` 表示行、列各自搜索 `[-s, s]` 个 die 的整数平移。`--rigid-overlay-score dice|iou` 选择最终排序分数：Dice 更适合比较总体重合，IoU 对额外缺陷更严格。每个候选变换均以晶圆网格中心为旋转中心、最近邻落到网格 cell；不插值、不膨胀、不补点。若旋转或平移使保留的原始 WDM 缺陷点少于 95%，该变换会被拒绝，避免因裁剪或离散化改变缺陷结构。

该模式仅允许网格短边不超过 12 个 cell；较大的 WBM/WDM 会直接报错，以避免将小图全图叠合误用于大图或局部 proposal 场景。它可用于 `count-partial` 与 `classnumber`：后者会对每个 classnumber split 独立执行叠合并按该分数选取最佳 split。由于没有 token 配对，token/map match 日志不会包含配对记录，`shape`、`position`、`scale` 等 token 指标不参与分数；结果中的主 score 即所选 Dice 或 IoU 最大值。

### 1.1.4 大图 proposal 刚性叠合匹配

对于短边大于 12 个 cell 的大图，可使用 `proposal-rigid-overlay`。它继续使用现有 `--proposal-mode` 提取 WBM/WDM proposal；随后将每个 proposal 的原始像素平移到各自的局部锚点，在局部坐标中搜索同一组离散旋转和小范围平移，按 Dice 或 IoU 取每个 proposal 对的最高分。proposal 对仍按现有的一对一贪心规则接受，并按 WBM proposal 面积加权汇总为文件分数。因此，它解决的是“局部 proposal 形状在旋转、轻微位移后是否重合”，并不会将整张大图旋转或扩大缺陷区域。

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --representation count \
  --mode count-partial \
  --proposal-mode tangential-ring \
  --small-map-match-mode proposal-rigid-overlay \
  --rigid-overlay-score dice \
  --rigid-overlay-max-shift 1 \
  --identifier AF00138_proposal_overlay
```

`--proposal-mode` 仍决定 proposal 如何提取，例如 `cc`、`compact`、`tangential-ring` 或 `sparse-density`；`proposal-rigid-overlay` 只替换 proposal 对的打分方式，不会改变提取结果。若 token 带有 `raw_pixels`，叠合必定使用该原始像素；否则使用既有 token 像素。旋转落格造成像素合并时同样受 95% 保留率约束，且局部画布预留了平移空间，因此不会因 proposal bbox 裁剪而丢失缺陷点。该模式不允许用于小图；小图应使用 `rigid-overlay`，默认实验仍使用 `token`。

### 1.2 CP CSV 参考图预处理

另一类 CP test 输入可先从 CSV 拆分为逐 wafer、逐 hardbin 的 WBM 参考图：

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.prepare_cp_refs \
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
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.batch_run \
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
    "topk": 10
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

入口 `scripts/main.py` 先加载 WBM 参考图（`scripts/reference_loader.py`），再对 KLARF 目录下每个文件执行 `scripts/processing.py → process_one`，最后汇总写入 TSV 和图表。

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
  │    ├── core/mappers.py       坐标映射 (die-index / relative / physical)
  │    └── core/representations.py  网格表达 (count / binary / density / soft / three-value)
  │    输出: GridMaps (与 WBM 同尺寸 H×W)
  │
  ├─ ③ Global Similarity ── core/similarity.py → compute_similarity (method="iou")
  │    IoU (|A∩B|/|A∪B|) 作为 baseline 指标
  │
  ├─ ④ Count-Partial Match ── core/local_matching/scoring.py → explain_count_partial_match
  │    ├── 4a. Token 提案 ── proposal.py
  │    │      WBM: _tokens_from_mask (status_map == VALID_HAS_DEFECT)
  │    │      WDM: _tokens_from_weighted_mask (count_map > 0, count 作为权重)
  │    │      模式: CC (连通域) 或 Compact (环状提取 + 残差分类) 或 Sparse-density (多尺度 KDE)
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
        │     │     (result + result_matched_only 各出一套, proposal_steps/ 下 4 图 1 表)
        │     └── save_wdm_raw_figures → wdm_raw_review/
        │           (top-K KLARF 物理坐标 wafer 原图, CLASSNUMBER=0 红色, !=0 蓝色)
        └── classnumber 模式:
              ├── save_classnumber_baseline_figures → baseline_review/
              │     (每个文件取 IoU 最高的 classnumber split，再取 top-K，左 WBM 右 WDM 对比图)
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
    similarity.py          # 8 种全图相似度计算
    local_matching/        # count-partial 局部匹配
      models.py            # LocalMatchResult、ProposalConfig
      morphology.py        # 连通域提取、形态学操作
      proposal.py          # token 提案生成（cc / compact / tangential-ring / sparse-density 四种模式）
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
```

---

## 4. 技术思路

### 4.1 Map：坐标映射与网格表达

KLARF 中的每个缺陷带有 die 索引（XINDEX/YINDEX）和 die 内相对坐标（XREL/YREL）。Mapper 将缺陷散点映射到与 WBM 同尺寸的 H×W 网格中：`physical-coordinate` 合并 die 索引与 die 内坐标，计算归一化位置后按比例映射到网格格点；`die-index` 和 `relative-coordinate` 分别只用 die 索引或 die 内坐标做线性缩放。

映射后统计每个格点的缺陷数得到 `count_map`，再派生 `binary_map`（count ≥ threshold）、`density_map`（count 归一化为概率分布）等表示。网格始终以 WBM 尺寸为准，确保两张图可逐格点比对。

WBM 使用三值语义：白色=有缺陷 die（`VALID_HAS_DEFECT=2`）、灰色=无缺陷但晶圆内（`VALID_NO_DEFECT=1`）、黑色=晶圆外背景（`BACKGROUND=0`）。所有后续计算只在有效区域（status ∈ {1,2}）内进行。

### 4.2 Proposal：Token 提取

从 WBM 和 WDM 的 mask 中提取若干局部区域作为 token，每个 token 代表一个有意义的缺陷聚集区。支持四种模式：

- **CC（Connected Component）**：BFS 连通域提取（4-连通或 8-连通），过滤面积 < `min_area` 的小碎片，按重要性（√mass + √area + 类型加分）排序后取 top-k
- **Compact**：先提取边缘环状 token（radial histogram 检测环形密集带），再从残差中提取 component token（分类为 blob / line / central / irregular），按类型多样性选取
  - 对短边 ≤12 的小图，Compact 先对原始 mask 做一次受有效区域约束的 `3×3` closing，仅作为 ring 连通性与轮廓证据；ring token 的像素、面积、几何统计和描述子仍只使用原始缺陷格，残余 component 也由去噪后的原始 mask 提取
  - 小图 ring-aware 默认使用最多 24 个角度扇区、最少 6 个 ring-band cell、最少 0.10 的角度覆盖率、最多 0.18 的径向标准差和最多 0.60 的缺陷覆盖率；较大图维持原有的 72 扇区与更严格阈值。可通过 `--ring-min-area`、`--ring-edge-r-min`、`--ring-band-width`、`--ring-min-angular-coverage`、`--ring-angular-bins`、`--ring-max-radial-std`、`--ring-max-defect-ratio`、`--ring-min-edge-defect-fraction` 显式调节
- **Tangential-ring**：仅以原始外圈缺陷点构造受限径向带，并在极坐标角度轴上桥接最多两个 die 的短缺口。桥接只记录 contour 连续性，不新增 token 像素；ring 和 residual 按真实原始像素互斥。
- **Sparse-density**：用于 WBM/WDM 同时稀疏的情况。两侧在同一网格上以多尺度截断高斯核生成连续 density map，再从相对峰值阈值 support 中提取 token。每个尺度内会额外识别半径相近、径向窄的 ring arc，并在角向覆盖足够时合并成 `edge_ring`；覆盖不足的片段只标为 `ring_arc`。每个尺度的候选按 IoU 去重，token 必须包含足够的原始点数和原始质量；因此 KDE 和 ring-aware merge 只改变 proposal 的分组与证据，不会覆盖原始 WBM/WDM 数据。

每个 token 记录以下几何属性：加权质心 (centroid_row, centroid_col)、bbox、PCA 特征值与方向、面积 (area)、质量 (mass)、周长、紧致度 (compactness)、归一化径向距离、角度覆盖度 (angular_coverage)、径向标准差 (radial_std)、几何类型 (geometry_type)。

`--proposal-min-area` 与 `--proposal-top-k` 使用默认值时会按图尺寸做保守自适应：短边 ≤12 的小图默认降到 `min_area=2, top_k=4`，中等图最多保留 6 个 token，大图最多保留 8 个 token。若命令行显式传入其他值，则按用户值执行，不再被内部 adaptive 默认值截断，便于在噪声过多时提高 `min_area`，或在复杂图案中提高 `top_k` 保留更多候选。

### 4.3 Descriptor：形状描述符

每个 token 的形状描述符由两部分组成：

**Zernike 矩（权重 0.75）**：将 token 的 bbox 裁剪区域最近邻缩放到 48×48，映射到单位圆盘，计算 8 阶 Zernike 矩的幅值向量。Zernike 矩天然旋转不变，对平移和缩放也有良好鲁棒性。

**几何特征（权重 0.25）**：`[fill_ratio, log(aspect), log(elongation), compactness, angular_coverage, radial_std, orientation_cos, orientation_sin]`。

两部分会分别计算相似度，不再依赖拼接后的单一余弦分数作为最终形状分。Zernike 使用余弦相似度得到 `moment_sim_raw`，几何特征使用平均绝对差的指数衰减得到 `geometry_sim_raw`。

若启用 `rotation_tolerance`，描述符退化为径向轮廓直方图（旧版方式）。

### 4.4 Score：Token 匹配与分数聚合

**Token 对打分**：对每对 (WBM token, WDM token)，计算三个维度的亲和度，组合为 token pair score：

- **Shape similarity**：先分别计算原始分数 `moment_sim_raw` 与 `geometry_sim_raw`，再用对比度校准函数压低中等相似度：

```
contrast(x, floor, gamma) =
  0                                      if x <= floor
  ((x - floor) / (1 - floor)) ^ gamma    otherwise
```

默认校准参数：

```
moment_sim   = contrast(moment_sim_raw,   floor=0.70, gamma=2.0)
geometry_sim = contrast(geometry_sim_raw, floor=0.55, gamma=1.5)
shape_sim    = 0.75 × moment_sim + 0.25 × geometry_sim
```

`moment_sim_raw` 和 `geometry_sim_raw` 会写入 token/map match 日志，便于调参时对比原始分布和校准后分布。校准后的 `shape_sim` 低于 `MIN_SHAPE_SIM_FOR_MATCH=0.45` 时，该 token pair 直接置零。
- **Position affinity**：`exp(-d² / σ_pos²)`，d 为归一化质心间的欧氏距离
- **Scale affinity**：`w_area × area_aff + w_pca × pca_aff`
  - `area_aff = exp(-|log(area_q / area_c)| / σ_scale)`
  - `pca_aff = exp(-(0.75·|log(long_q/long_c)| + 0.25·|log(short_q/short_c)|) / σ_scale)`

另外还有一个硬尺度门控：`--token-scale-ratio-min` 默认 `0.35`，要求面积比、PCA 长轴比、PCA 短轴比都不低于该阈值，否则该 token pair 直接记为 0 分。设为 `0` 可关闭该门控。

```
token_score = w_shape × shape_sim + w_position × position_aff + w_scale × scale_aff
```

默认权重：`w_shape=0.60, w_position=0.25, w_scale=0.15`。

**贪心一对一匹配**：所有 token 对按 score 降序排列，贪心选取不重复使用 token 的配对。

**分数聚合**：以每个 WBM token 的 √area 为权重，对匹配到的 token pair score 做加权平均：

- **result**：所有 scored WBM token 参与加权（未匹配到 WDM 的贡献 0）
- **result_matched_only**：仅实际匹配到的 WBM token 参与加权

```
final_score = Σ(matched_score_i × √area_i) / Σ(√area_i)
```

同时对 shape_sim、position_aff、scale_aff、type_aff 分别做同样的加权平均，输出 `mean_shape`、`mean_position`、`mean_scale`、`mean_type`。

**Classnumber 扩展**：将 KLARF 按 defect classnumber 拆分为多个子 WDM，每个子 WDM 独立做上述 count-partial（及可选 binary-partial）匹配，选出 rank_score 最高的 classnumber 作为 best-classnumber 输出。
