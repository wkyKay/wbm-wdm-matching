# WBM-WDM Matching — 需求与设计文档

## 1. 项目定位

研究目标：**给定一张目标 WBM（Wafer Bin Map）和一组候选 WDM（Wafer Defect Map），找到能最佳"解释"该 WBM 的 WDM 稀疏子集。**

传统方法比较两张 WBM 是否相似；本项目是**跨数据源匹配**——WDM 来自 KLARF 缺陷扫描，WBM 来自芯片测试分 bin，粒度、来源、语义均不同。

---

## 2. 核心流水线（批量模式）

```
┌─────────────────┐     ┌──────────────────┐
│ KLARF 目录      │ →   │ 逐文件            │
│ (n 个 .klarf)   │     │ 1. 坐标映射       │
└─────────────────┘     │ 2. 网格表达       │
                        │ 3. 8 种相似度     │
                        │    (vs. WBM 参考) │
                        └────────┬─────────┘
                                 ↓
                        ┌──────────────────┐
                        │ TSV Log 文件      │
                        │ (n × 9 表格)      │
                        └──────────────────┘
                                 ↑
                        ┌────────┴────────┐
                        │ WBM PNG (参考图) │
                        └─────────────────┘
```

- 输入：一个包含 `*.klarf` 文件的目录 + 一张参考 WBM PNG
- 输出：一个 TSV 文件，每行一个 KLARF 文件，每列一种相似度指标

### 2.1 坐标映射（Mappers）

将 KLARF 缺陷散点映射到 WBM 网格的对应格子。

| Mapper | 逻辑 | 适用场景 |
|--------|------|---------|
| `die-index` | XINDEX/YINDEX 线性缩放 | 快速 baseline，WBM 粒度 ≈ die |
| `relative-coordinate` | XREL/YREL 线性缩放 | die 内部精细位置 |
| `physical-coordinate` | XINDEX × DiePitch + XREL → 归一化比例映射 | **推荐**，合并两层信息，可跨分辨率 |

**`physical-coordinate` 设计要求**：
- 必须传入 `--die-x-range` / `--die-y-range`（该产品的 die 网格行列范围）
- DiePitch 从 KLARF 自动读取
- 计算：`x_norm = (XINDEX - x_min + XREL / DiePitchX) / wafer_die_cols`
- (1,1) die 的 wafer 直接报错跳过
- 生成圆形 mask 区分"晶圆内"与"背景"

### 2.2 网格表达（Representations）

对 count_map 做归一化/平滑/二值化，产出 6 种表示：

| 表达 | 含义 | 特点 |
|------|------|------|
| `binary` | 有缺陷=1，无缺陷=0 | 简单，丢密度信息 |
| `count` | 每个格子的缺陷数量 | 密集区值大，不稳定 |
| `density` | count 归一化为概率分布 | 推荐默认，跨样本可比 |
| `soft` | density + 高斯平滑 | 容忍小偏移，σ 可配 |
| `three-value` | 强证据=1，弱证据=0.5，无=0 | 三值化，可解释性强 |
| `mountain` | 已废弃（与 soft 等价，待移除） | — |

### 2.3 WBM 三值语义

WBM PNG 使用三值编码（与 WM-811K 数据集一致）：

| 颜色 | 像素值 | 语义 | 状态码 |
|------|--------|------|--------|
| 白色 | 255 | 有缺陷 die | `VALID_HAS_DEFECT = 2` |
| 灰色 | 127 | 无缺陷但晶圆内 | `VALID_NO_DEFECT = 1` |
| 黑色 | 0 | 背景（晶圆外） | `BACKGROUND = 0` |

**灰色的作用**：
1. 定义晶圆边界（白色+灰色=有效区域，黑色以外全忽略）
2. 作为无缺陷参照——WDM 缺陷落在灰色区 = leakage / false positive

### 2.4 相似度计算（Similarity）

所有方法以 **WBM status_map 定义的有效区域** 为 mask 进行计算。

| 方法 | 公式 | 特点 |
|------|------|------|
| `dice` | 2|A∩B|/(|A|+|B|) | 二值缺陷重叠，灰色进分母但不进分子 |
| `iou` | |A∩B|/|A∪B| | 比 dice 更严格 |
| `ncc` | 归一化互相关 | 只能在有效区域内算，mask 外排除 |
| `cosine` | 余弦相似度 | 同上 |
| `coverage` | WBM 缺陷被 WDM 覆盖的比例 | 核心正向指标 |
| `leakage` | WDM 缺陷在 WBM 无缺陷区的比例 | 核心负向指标 |
| `coverage-leakage` | coverage − β × leakage | 推荐综合得分 |
| `chamfer` | 点集倒角距离 | 距离越近≈匹配越好 |

**关键设计**：
- 维度对齐：WDM 始终以 WBM 尺寸（H×W）生成，逐像素比对
- Status-aware：黑色(0)区域不参与任何相似度计算
- 向后兼容：不传 status 时退化为全图计算

---

## 3. 代码架构

```text
match/
  __init__.py                 # 统一 re-export 公共 API
  core/
    models.py                 # DefectTable、GridMaps、状态常量
    mappers.py                # 3 种坐标映射器 + MAPPERS 注册表
    representations.py        # 6 种网格表达 + REPRESENTATIONS 注册表
    pipeline.py               # map_klarf_to_grid 端到端入口
    similarity.py             # 8 种全图相似度
    local_matching.py         # count-map partial matching
  data/
    fileio.py                 # KLARF 解析、WBM PNG 解码、npz 读写
  viz/
    visualization.py          # 通用绘图对比：并排/叠加/多视图
    count_partial_visualization.py  # partial matching step/topK 图片
  scripts/
    batch_io.py               # 批处理结果写入/TopK 排序辅助
    batch_viz.py              # 批处理可视化辅助
    main.py                   # 批量处理 CLI 入口
    plot_ref_cnd.py           # CLI 快速看图脚本
    plot_count_partial.py     # count-partial 图片生成脚本
```

---

## 4. CLI 使用方式

```bash
# 批量处理：目录中所有 KLARF vs. 一张 WBM 参考图
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --die-x-range -20 20 --die-y-range -20 20 \
  --representation density \
  --die-defect-threshold 1 \
  --identifier AF00138 \
  --log results.tsv
```

`--die-defect-threshold 1` 是默认值，表示只要某个 die/cell 有至少 1 个 defect，就会在 `binary_map` 中置 1。调大该值只影响 `binary_map` 及基于 binary 的匹配/绘图，不改变 `count_map`。

`--identifier` 用于组织多次实验输出。例如 `--count-partial-fig-dir results --identifier AF00138` 会保存到 `results/AF00138/count_partial_review`。classnumber review 会额外按匹配模式分目录，例如 `results/AF00138/classnumber_review_count`、`results/AF00138/classnumber_review_binary`、`results/AF00138/classnumber_review_both_rank_binary`。不传 `--identifier` 时保持旧行为，直接使用对应的 `*-fig-dir`。

### 4.1 生成 count-partial 图片

结果 TSV 和 review 图片上各分数字段的含义见 [`result_items.md`](result_items.md)。

如果你要看 `count-map partial` 的 top3 和 proposal step 图，直接加这组参数：

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --die-x-range -20 20 --die-y-range -20 20 \
  --representation density \
  --die-defect-threshold 1 \
  --identifier AF00138 \
  --log results.tsv \
  --topk-log topk.tsv \
  --save-count-partial-figures \
  --count-partial-fig-dir results \
  --count-partial-review-top-k 3 \
  --count-partial-step-max 3
```

会生成：
- `results/AF00138/count_partial_review/top3_count_partial.png`
- `results/AF00138/count_partial_review/proposal_steps/rankXX_*.png`

所有 8 种相似度指标自动计算，结果写入 TSV：

| 列 | 内容 |
|----|------|
| `file` | KLARF 文件名 |
| `dice`, `iou`, `ncc`, `cosine` | 基础相似度 |
| `coverage`, `leakage`, `coverage-leakage` | Coverage-Leakage 族 |
| `chamfer` | 点集倒角距离 |
| `count-partial*` | 基于 WDM count_map 的局部 token 匹配分数与分量 |
| `classnumber-count`, `best-classnumber*` | 仅 `--use-classnumber` 时输出，表示按 classnumber 拆分后的最佳分图 |
| `mapped_defects` | 映射成功数/总缺陷数 |

### 4.2 classnumber 分图匹配

classnumber 功能是在整图匹配之外，额外把每个 WDM 按 `classnumber` 拆成多个分图，再做分图匹配和可视化。

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --die-x-range -20 20 --die-y-range -20 20 \
  --representation density \
  --defect-threshold 5 \
  --die-defect-threshold 1 \
  --identifier AF00138 \
  --use-classnumber \
  --save-classnumber-figures \
  --classnumber-fig-dir results
```

默认是 `count` 模式，也就是沿用 `count-partial` 的 token 匹配结果。若要用
binary 形态判断，切到：

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --die-x-range -20 20 --die-y-range -20 20 \
  --representation density \
  --defect-threshold 5 \
  --die-defect-threshold 2 \
  --identifier AF00138 \
  --use-classnumber \
  --classnumber-match-mode binary \
  --save-classnumber-figures \
  --classnumber-fig-dir results
```

也可以同时算 `count` 和 `binary`，并指定最终排序依据：

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.main \
  --klarf-dir /path/to/klarf_files/ \
  --reference data/wm811k/000604.png \
  --mapper physical-coordinate \
  --die-x-range -20 20 --die-y-range -20 20 \
  --representation density \
  --defect-threshold 5 \
  --die-defect-threshold 2 \
  --identifier AF00138 \
  --use-classnumber \
  --classnumber-match-mode both \
  --classnumber-rank-by binary \
  --save-classnumber-figures \
  --classnumber-fig-dir results
```

上面 binary 示例使用 `--die-defect-threshold 2`，表示单个 die/cell 至少有 2 个 defect 才进入 binary proposal。生产数据中如果 binary 图仍然过碎，可以试 `3`；如果担心稀疏真实形状被过滤，则回到 `1`。

这组命令会生成：
- `results/AF00138/classnumber_review_both_rank_binary/<file>_classnumber_splits.png`（只为全局 TopK classnumber 分图涉及的文件生成；同一文件只生成一张）
- `results/AF00138/classnumber_review_both_rank_binary/classnumber_topk.tsv`
- `results/AF00138/classnumber_review_both_rank_binary/classnumber_topK.png`
- `results/AF00138/classnumber_review_both_rank_binary/topk_steps/rankXX_*.png`

其中：
- `classnumber_topk.tsv` 记录所有分图的排序结果
- `classnumber_topK.png` 是全局 topK 分图总览
- `topk_steps/` 是 topK 分图的 cluster / step 参考图

新增参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--defect-threshold` | `5` | 文件级过滤：KLARF 总 defect 数低于该值则跳过 |
| `--die-defect-threshold` | `1` | die/cell 级过滤：单个 die/cell 至少包含多少个 defect 才会在 `binary_map` 中置 1 |
| `--identifier` | 空 | 可选运行标识；设置后 review 图片保存到 `<fig-dir>/<identifier>/<review_name>`，classnumber 会按 count/binary/both_rank_* 自动区分目录 |
| `--count-partial-proposal-mode {cc,compact}` | `cc` | count-partial/classnumber token 提取模式；`cc` 保持旧连通域逻辑，`compact` 启用保守 ring-aware、fragment merge 和多样性 topK |
| `--count-partial-rotation-tolerance` | 关闭 | 使用旋转容忍的 shape descriptor；只影响 token shape 相似度，不做全局位置旋转搜索 |
| `--classnumber-match-mode {count,binary,both}` | `count` | classnumber 分图计算 count、binary 或两者 |
| `--classnumber-rank-by {count,binary}` | `count` | `both` 模式下 topK 和最佳分图的排序依据 |
| `--classnumber-binary-dilation` | `1` | 兼容旧命令保留；当前 binary token 匹配不再使用 dilation |
| `--classnumber-binary-beta` | `0.5` | 兼容旧命令保留；当前 binary token 匹配不再使用 leakage beta |

binary classnumber 分数现在与 count-partial 使用同一套 proposal / descriptor / token 相似度流程：
- WBM token 仍来自 `status_map == VALID_HAS_DEFECT`
- WDM token 来自 `binary_map > 0`
- `binary_map` 由 `count_map >= --die-defect-threshold` 生成，默认 `1` 等价于旧的 `count > 0`
- WDM token 权重统一为 1，不使用 count 强度
- 最终分数仍由 shape、position、scale、type 组成；shape 使用硬门槛，type 作为 soft penalty

保存图片时，无论选择 `count`、`binary` 还是 `both`，都会生成上面的三类图。
`binary` 排序时，step 图展示同一 classnumber 分图的局部结构，方便对照解释路径。

count-partial 和 classnumber review 图片的颜色规则：
- WBM 面板保持 WM811K 原图风格：背景黑色、晶圆内正常 die 灰色、失效 die 白色。
- WDM count / binary heatmap 使用晶圆外黑色、晶圆内灰白到红色的热力图；`rank_by=binary` 时底图使用 binary map，`rank_by=count` 时底图使用 count map。

### 4.3 `density` 和 `count` 的区别

这两个选项都不会改变 `count-partial` 的 token 提取逻辑，因为该部分固定使用
`count_map`。它们的区别主要在整图相似度和保存的候选图底图：

| 选项 | 影响 |
|------|------|
| `count` | 用原始缺陷计数做整图比较，强度保留最直接，但不同样本间数值尺度更敏感 |
| `density` | 把 count 归一化为概率分布，整图相似度更关注空间分布而不是绝对缺陷数 |

如果你的目标是做“候选分布形态”的比较，`density` 更稳；如果你更关心“绝对缺陷强度”，`count` 更直接。对 classnumber 分图和 count-partial 的结构输出来说，主要差异仍体现在整图相似度层，不会改变 WDM 被拆分后的 token 生成方式。

---

## 5. 可视化对比（visualization.py）

提供 4 个绘图函数，核心设计原则：**ref 与 cnd 共用一个 (cmap, vmin, vmax)**，并用 status_map 统一遮蔽 wafer 外区域，确保两张图可直接视觉比较。

### 5.1 vmin / vmax 的作用

| 无统一 range | 有统一 range |
|-------------|-------------|
| 各图用自己的 min/max 映射颜色 | 两图共用同一个 min/max |
| 值 0.001 在 ref 是中等亮度，在 cnd 可能是最高亮度 | 0.001 在两图中是同一个颜色 |
| **不可比** | **可直接对比** |

### 5.2 可用函数

| 函数 | 用途 |
|------|------|
| `plot_single(gm, ...)` | 绘单张 map |
| `plot_comparison(ref, cnd, ...)` | 并排对比 ref vs cnd，统一 colormap |
| `plot_overlay(ref, cnd, ...)` | 差值叠加：蓝=ref独有，红=cnd独有，白=匹配 |
| `plot_representation_panel(gm, ...)` | 一行展示 binary / count / density 三视图 |

### 5.3 代码示例

```python
from match.data.fileio import read_wbm_png
from match.core.pipeline import map_klarf_to_grid
from match.viz.visualization import plot_comparison, plot_overlay

ref_gm = read_wbm_png("reference.png")
cnd_gm = map_klarf_to_grid(
    "data.klarf",
    shape=ref_gm.count_map.shape,
    die_defect_threshold=1,
)

# 并排 density 热力图对比
plot_comparison(ref_gm, cnd_gm, representation="density", cmap="hot",
                save_path="cmp_density.png")

# binary 灰度对比
plot_comparison(ref_gm, cnd_gm, representation="binary", cmap="gray",
                save_path="cmp_binary.png")

# 差值叠加：一眼看出漏检（蓝）与多报（红）
plot_overlay(ref_gm, cnd_gm, representation="binary",
             save_path="cmp_overlay.png")
```

### 5.4 CLI 快速看图

```bash
PYTHONPATH=wbm-wdm-matching python3 -m match.scripts.plot_ref_cnd \
    --ref ../data/wm811k/000604.png \
    --klarf ../data/klarf/some_file \
    --mapper die-index \
    --representation density \
    --die-defect-threshold 1 \
    --output comparison.png
```

Shape 始终从 `--ref` 的 WBM PNG 自动读取，无需手动指定。

---

## 6. 待办事项

- [x] `--similarity` 模式下一次输出所有相似度指标 — 已实现
- [ ] 移除或重命名 MountainMap（与 SoftMap 数学等价，功能冗余）
- [ ] SoftMap σ 配置化（`--soft-sigma` 参数）
- [ ] 实现组合搜索（Beam Search + Coverage-Leakage）— 第二期
- [ ] 旋转容错搜索 — 第三期

---

## 7. 参考论文方向

| 方向 | 可借鉴方法 |
|------|-----------|
| SIMI Ratio / Morphology | 三值图、spatial filter |
| Hsu 2020 Mountain Function + WMHD | 密度表面、outlier penalty |
| Lee 2021 DPGMM + HCM + JSD | 局部 cluster 分解 |
| Wang 2023 Tensor Voting + WBBS | 结构显著性、partial matching |
| Kang 2024 Shape/Location/Size | 多分量可解释得分 |
