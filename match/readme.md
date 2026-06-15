# WBM-WDM Matching — 需求与设计文档

## 1. 项目定位

研究目标：**给定一张目标 WBM（Wafer Bin Map）和一组候选 WDM（Wafer Defect Map），找到能最佳"解释"该 WBM 的 WDM 稀疏子集。**

传统方法比较两张 WBM 是否相似；本项目是**跨数据源匹配**——WDM 来自 KLARF 缺陷扫描，WBM 来自芯片测试分 bin，粒度、来源、语义均不同。

---

## 2. 核心流水线

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────────┐
│ KLARF    │ →   │ Coordinate   │ →   │ Represent-   │ →   │ Similarity     │
│ WDM      │     │ Mapping      │     │ ation Build  │     │ Computation    │
└──────────┘     └──────────────┘     └──────────────┘     └────────────────┘
                                                                    ↑
                                                              ┌─────┴─────┐
                                                              │ WBM PNG   │
                                                              │ (参考图)   │
                                                              └───────────┘
```

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

```
match/
  __init__.py          # 统一 re-export 所有公共 API
  models.py            # DefectTable、GridMaps、状态常量
  io.py                # KLARF 解析、WBM PNG 解码、npz 读写
  mappers.py           # 3 种坐标映射器 + MAPPERS 注册表
  representations.py   # 6 种网格表达 + REPRESENTATIONS 注册表
  pipeline.py          # map_klarf_to_grid 端到端入口
  similarity.py        # 8 种相似度 + SIMILARITIES 注册表
  main.py              # CLI 入口
```

---

## 4. CLI 使用方式

```bash
# 基础：KLARF → GridMaps
python3 -m match.main \
  --klarf data.klarf \
  --wbm target.png \
  --mapper physical-coordinate \
  --die-x-range -20 20 --die-y-range -20 20 \
  --representation density \
  --output wdm.npz

# 带相似度计算（直接比较 WBM PNG）
python3 -m match.main \
  --klarf data.klarf \
  --wbm target.png \
  --mapper physical-coordinate \
  --die-x-range -20 20 --die-y-range -20 20 \
  --similarity coverage-leakage \
  --reference reference.png
```

---

## 5. 待办事项

- [ ] 移除或重命名 MountainMap（与 SoftMap 数学等价，功能冗余）
- [ ] `--similarity` 模式下一次输出所有 6 种 representation 的得分
- [ ] PhysicalCoordinateGridMapper 计算方法存疑（todo 文件标记）
- [ ] SoftMap σ 配置化（`--soft-sigma` 参数）
- [ ] 实现组合搜索（Beam Search + Coverage-Leakage）— 第二期
- [ ] 旋转容错搜索 — 第三期

---

## 6. 参考论文方向

| 方向 | 可借鉴方法 |
|------|-----------|
| SIMI Ratio / Morphology | 三值图、spatial filter |
| Hsu 2020 Mountain Function + WMHD | 密度表面、outlier penalty |
| Lee 2021 DPGMM + HCM + JSD | 局部 cluster 分解 |
| Wang 2023 Tensor Voting + WBBS | 结构显著性、partial matching |
| Kang 2024 Shape/Location/Size | 多分量可解释得分 |
