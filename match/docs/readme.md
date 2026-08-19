# WBM-WDM Matching

## 1. 项目定位

给定一张目标 WBM（Wafer Bin Map，芯片测试分 bin 图）和一组候选 WDM（Wafer Defect Map，KLARF 缺陷扫描数据），找到能最佳解释该 WBM 的 WDM 稀疏子集。

核心挑战是跨数据源匹配：WBM 来自电性测试，粒度为 die；WDM 来自光学扫描，粒度为亚 die 缺陷散点，两者的来源、粒度和语义均不同。项目通过坐标映射把 WDM 转换到与 WBM 一致的网格，再结合全图 IoU、局部 token 匹配和可选的 classnumber 分图匹配完成候选排序。

输出包括全局相似度排名、count-map 局部 token 匹配分数、可选的 defect classnumber 分图匹配结果，以及 token 提取、匹配证据和 WDM 原始坐标等可视化文件。

---

## 2. 运行命令

### 2.1 主程序

`--mode` 参数控制匹配流程，所有日志与图表自动保存到 `<output-dir>/<identifier>/<mode>/`。以下命令均从仓库根目录执行；`--klarf-dir`、`--reference` 为必填参数：

| 模式 | 计算内容 | 自动输出 |
|---|---|---|
| `count-partial`（默认） | 完整 WDM 的 IoU 基线和局部 token 匹配 | `results.tsv`、`topk.tsv`、`token_match.tsv`、`map_match.tsv`、参数快照及基线、局部匹配和原始 WDM 图表 |
| `classnumber` | 按 KLARF `CLASSNUMBER` 拆分 WDM，并选择得分最高的子图 | 与 `count-partial` 相同的日志结构，另含最佳 classnumber、count/binary 局部匹配及对应图表 |

#### 2.1.1 Proposal mode

| `--proposal-mode` | 适用情况 | 作用 |
|---|---|---|
| `cc` | 非稀疏 | 按 8 连通域提取局部 token，适合缺陷区域本身连通的情况 |
| `compact` | 非稀疏 | 先检测环状候选，再从残差提取 component token，适合环状缺陷与普通局部缺陷共存的情况 |
| `arc-ring-residual` | 非稀疏 | 提取满足径向带和角度覆盖条件的环/弧 token，再从残差中提取其他 component token |
| `sparse-density` | 稀疏 | 用多尺度 KDE support 恢复分散缺陷的连续区域，并保留 raw 缺陷证据 |
| `sparse-density-arc-ring-residual` | 稀疏 | 在 KDE support 上检测 ring/arc，再按 support/raw 所有权提取 residual；`sparse-arc-ring-residual` 是兼容别名 |

下面两组命令分别对应非稀疏和稀疏缺陷情况；两组命令均固定使用 `--representation count`。

```bash
# 非稀疏：cc
python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --representation count \
  --mode count-partial \
  --proposal-mode cc \
  --identifier AF00138

# 非稀疏：compact 或 arc-ring-residual 时，仅替换 proposal-mode
# --proposal-mode compact
# --proposal-mode arc-ring-residual

# 稀疏：sparse-density
python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --representation count \
  --mode count-partial \
  --proposal-mode sparse-density \
  --density-sigmas 0.6 1.2 1.8 \
  --density-threshold 0.20 \
  --identifier AF00138_sparse

# 稀疏：sparse-density-arc-ring-residual 时替换 proposal-mode
# --proposal-mode sparse-density-arc-ring-residual

# classnumber 模式：仍使用 count 表示，可与以上任一 proposal mode 组合
python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --representation count \
  --mode classnumber \
  --proposal-mode cc \
  --identifier AF00138

# 也可用 JSON 文件提供默认参数，命令行参数优先
python3 -m match.scripts.main \
  --config /path/to/match_config.json
```

CLI 参数优先级高于配置文件。每个参数的含义如下：

| 参数 | 含义 |
|---|---|
| `--klarf-dir` | KLARF 文件目录 |
| `--reference` | WBM 参考 PNG |
| `--mapper` | WDM 到 WBM 网格的坐标映射方式，如 `physical-coordinate` |
| `--representation` | WDM 网格表达；本项目命令统一使用 `count` |
| `--mode` | 匹配流程：`count-partial` 或 `classnumber` |
| `--proposal-mode` | token proposal 方式，取值见上表 |
| `--identifier` | 本次实验名称，决定输出子目录 |
| `--density-sigmas` | 稀疏 proposal 使用的 grid-cell 高斯尺度 |
| `--density-threshold` | KDE support 相对峰值阈值 |
| `--density-min-raw-points` | 一个稀疏 token 所需的最少原始占用格点数 |
| `--density-min-raw-mass` | 一个稀疏 token 所需的最少原始权重质量 |

稀疏模式中，`pixels` 主要表示经过 raw 锚点保护的 support 几何区域，`raw_pixels`、`raw_mass` 和 `raw_point_count` 保留真实缺陷证据；`sparse-density-arc-ring-residual` 还会在 support 上提取 ring/arc，并从未占用区域提取 residual。

### 2.2 其他工具

#### 2.2.1 CP CSV 参考图预处理

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

#### 2.2.2 批量对比实验

`scripts/batch_run.py` 用于读取一个 JSON 实验配置，复用同一组公共参数并依次调用主流程，适合批量比较多个参考图或多个 KLARF 目录。通过 JSON 定义多条实验，调用方式如下：

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
    "representation": "count",
    "die_x_range": [-20, 20],
    "die_y_range": [-20, 20],
    "topk": 10,
    "proposal_mode": "cc"
  },
  "experiments": [
    {
      "klarf_dir": "/data/klarf_batch1/",
      "reference": "/data/wm811k/000604.png",
      "identifier": "batch1_count_ref1"
    },
    {
      "klarf_dir": "/data/klarf_batch1/",
      "reference": "/data/wm811k/000610.png",
      "identifier": "batch1_count_ref2",
      "representation": "count"
    }
  ]
}
```

`common` 定义所有实验共享的默认参数（可选），每个实验**必须**提供 `klarf_dir`、`reference`、`identifier`，可按需覆盖 `common` 中任意参数。`identifier` 必须唯一，结果输出到 `<output-dir>/<identifier>/`。

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
  │    │      模式: cc、compact、arc-ring-residual、sparse-density、sparse-density-arc-ring-residual
  │    ├── 4b. 形状描述符 ── descriptors.py
  │    │      每个 token → Zernike 矩 (默认 48×48, 8阶，可配) + 几何特征 → 分量融合
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
        │     │     (IoU top-K, 左 WBM 右 WDM 对比图)
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
```

### 3.2 代码结构

```text
match/
  core/
    models.py              # GridMaps、DefectTable、状态常量
    mappers.py             # 3 种坐标映射器（die-index / relative-coordinate / physical-coordinate）
    representations.py     # 网格表达实现（命令示例统一使用 count；内部仍保留其他表达）
    pipeline.py            # map_klarf_to_grid 端到端入口
    similarity.py          # 全图相似度实现（当前批处理默认记录 IoU）
    local_matching/        # count-partial 局部匹配
      models.py            # LocalMatchResult、ProposalConfig
      morphology.py        # 连通域提取、形态学操作
      proposal.py          # token 提案生成（cc / compact / arc-ring-residual / sparse-density / sparse-density-arc-ring-residual）
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

从 WBM 和 WDM 的 mask 中提取若干局部区域作为 token，每个 token 代表一个有意义的缺陷聚集区。当前 CLI 支持五种 proposal mode：

- **CC（Connected Component）**：8 连通域提取，过滤面积 < `min_area` 的小碎片，按重要性（√mass + √area + 类型加分）排序后取 top-k。
- **Compact**：先提取环状候选，再从残差中提取 component token（分类为 blob / line / central / irregular），按重要性与类型多样性保留候选
  - 对短边 ≤12 的小图，Compact 先对原始 mask 做一次受有效区域约束的 `3×3` closing，仅作为 ring 连通性与轮廓证据；ring token 的像素、面积、几何统计和描述子仍只使用原始缺陷格，残余 component 也由去噪后的原始 mask 提取
  - 小图 ring-aware 默认使用最多 24 个角度扇区、最少 6 个 ring-band cell、最少 0.10 的角度覆盖率、最多 0.18 的径向标准差和最多 0.60 的缺陷覆盖率；较大图维持原有的 72 扇区与更严格阈值。可通过 `--ring-min-area`、`--ring-edge-r-min`、`--ring-band-width`、`--ring-min-angular-coverage`、`--ring-angular-bins`、`--ring-max-radial-std`、`--ring-max-defect-ratio`、`--ring-min-edge-defect-fraction` 显式调节
- **Arc-ring-residual**：先提取满足径向带和角度覆盖条件的环状 token，再从扣除环状区域后的残差中提取其余 component token；适合边缘环与其他局部缺陷共存的情形。
- **Sparse-density**：用于 WBM/WDM 同时稀疏的情况。两侧在同一网格上以多尺度截断高斯核生成连续 density map，再从相对峰值阈值 support 中提取候选。候选必须包含足够 `raw_pixels` 和 `raw_mass`，跨尺度去重同时检查 support IoU 与 raw IoU。最终 token 的 `pixels` 是受 raw 锚点保护合并/腐蚀后的 support 区域，因此形状描述符、面积、位置和 PCA 直接看 support；`raw_pixels` 只作为真实证据与追溯字段。
- **Sparse-density-arc-ring-residual**：在去重 KDE support 上检测边缘 ring/arc，并在接受后按 support/raw 所有权提取 residual。每个最终 token 同时保留 support 几何视图与 raw 证据视图；前者用于 `pixels`、形状、位置、面积、PCA、描述子和 scale affinity，后者用于有效性门槛、raw mass、去重计数和所有权控制。

每个 token 记录以下几何属性：加权质心 (centroid_row, centroid_col)、bbox、PCA 特征值与方向、面积 (area/support_area)、质量 (mass)、周长、紧致度 (compactness)、归一化径向距离、角度覆盖度 (angular_coverage)、最大连续角度覆盖、最大角度缺口、径向标准差 (radial_std)、径向带宽、几何类型 (geometry_type)。sparse 系列 token 额外记录 `raw_pixels`、`raw_area`、`raw_point_count`、`raw_mass`、`kde_support_pixels`、`kde_support_area` 等证据字段。

### 4.3 Descriptor：形状描述符

每个 token 的形状描述符由两部分组成，并在 `descriptor_parts` 中分开保存：

- **Zernike 矩**：将 `token["pixels"]` 投到以 token 质心为中心的固定画布上，再映射到单位圆盘，计算 Zernike 矩幅值。默认阶数为 8，可通过 `--zernike-degree` 调整；短边 ≤12 的小图使用 16×16 画布，其余使用 48×48 画布。
- **几何特征**：包括填充率、长宽比、PCA 伸长率、紧致度、全图径向/角度统计、token 局部径向填充分布、局部角度覆盖；未开启 `--proposal-rotation-tolerance` 且 descriptor mode 不是 `coarse` 时，还会加入方向的 cos/sin 编码。

当前 shape score 不直接使用拼接向量的整体余弦作为主路径，而是分别计算：

```text
moment_sim   = cosine(zernike_q, zernike_c)
geometry_sim = exp(-mean_abs_diff(geometry_q, geometry_c) / 0.25)
shape_sim    = (moment_weight × moment_sim + geometry_weight × geometry_sim)
               / (moment_weight + geometry_weight)
```

CLI 默认 `--moment-weight 0.25`、`--geometry-weight 0.75`，即当前更偏向几何统计而不是 Zernike 矩。若要测试更高维 Zernike 的区分性，建议同时提高 `--moment-weight`，否则更高阶 moment 对最终 `shape_sim` 的影响仍然有限。

### 4.4 Score：Token 匹配与分数聚合

**Token 对打分**：对每对 (WBM token, WDM token)，计算三个维度的亲和度，组合为 token pair score：

- **Shape similarity**：优先使用 `descriptor_parts` 中的 Zernike moment cosine 与 geometry affinity 加权融合。低于内部形状门限或 ring topology 门限的 token 对直接置零。可通过 `--proposal-min-shape-score` 增加额外 shape gate
- **Position affinity**：`exp(-d² / σ_pos²)`，d 为归一化质心间的欧氏距离
- **Scale affinity**：`w_area × area_aff + w_pca × pca_aff`
  - `area_aff = exp(-|log(support_area_ratio_q / support_area_ratio_c)| / σ_scale)`
  - `pca_aff = exp(-(0.75·|log(long_q/long_c)| + 0.25·|log(short_q/short_c)|) / σ_scale)`

```
token_score = w_shape × shape_sim + w_position × position_aff + w_scale × scale_aff
```

默认权重：`w_shape=0.60`、`w_position=0.25`、`w_scale=0.15`；可通过 `--token-score-*-weight` 调整。尺度亲和度内部默认 `--token-scale-area-weight 0.20`、`--token-scale-pca-weight 0.80`，即更偏向 PCA 长短轴范围而不是面积。

**贪心一对一匹配**：所有 token 对按 score 降序排列，贪心选取不重复使用 token 的配对。

**分数聚合**：以每个 WBM token 的 √area 为权重，对匹配到的 token pair score 做加权平均：

- **result**：所有 scored WBM token 参与加权（未匹配到 WDM 的贡献 0）
- **result_matched_only**：仅实际匹配到的 WBM token 参与加权

```
final_score = Σ(matched_score_i × √area_i) / Σ(√area_i)
```

同时对 shape_sim、position_aff、scale_aff、type_aff 分别做同样的加权平均，输出 `mean_shape`、`mean_position`、`mean_scale`、`mean_type`。

**Classnumber 扩展**：将 KLARF 按 defect classnumber 拆分为多个子 WDM，每个子 WDM 独立做上述 count-partial（及可选 binary-partial）匹配，选出 rank_score 最高的 classnumber 作为 best-classnumber 输出。
