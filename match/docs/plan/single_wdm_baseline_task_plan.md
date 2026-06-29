# Task Plan: Product-Aware Single-WDM Retrieval Baseline for WBM-WDM Matching

## 研究背景与当前调整

企业方当前更关注一个可落地的第一阶段目标：
> 给定一张 WBM，从候选 WDM 中检索最相似、最可能解释该 WBM 的单张 WDM。

因此，当前任务不再把组合搜索作为第一阶段主线，而是先建立扎实的 **单图检索 baseline**。组合搜索暂时作为第二阶段扩展，用于分析当最佳单张 WDM 无法充分解释 WBM 时，是否需要多张 WDM 共同解释。

新的阶段性研究定位为：
> 一种面向 WBM-WDM 跨数据源匹配的产品感知、可解释单张 WDM 检索基线。

英文表述可写为：
> Product-aware and interpretable single-WDM retrieval for cross-source WBM-WDM matching.

---

## 核心问题定义

输入：
> 目标 WBM: A
> 候选 WDM 序列: B_1, B_2, ..., B_N

输出：
> 按相似度排序的 Top-K 单张 WDM

形式化目标：
> B* = argmax_i S(A, T_p(B_i))

其中：

- `A`：目标 WBM，尺寸随产品不同而变化，例如约 `10x10`、`30x30`。
- `B_i`：第 `i` 张候选 WDM，通常是高分辨率散点缺陷图。
- `T_p(·)`：产品感知的 WDM 到 WBM grid 映射，`p` 表示目标 WBM 所属产品或 die layout。
- `S(A, T_p(B_i))`：单张 WDM 与目标 WBM 的相似度。

---

## 为什么这不是简单相似度计算

如果只是把两张图 resize 后计算 Dice、IoU 或 NCC，确实难以支撑毕业和论文。

本任务真正的问题在于：
> WBM 是芯片级失效图，WDM 是制程扫描散点图。
> 二者数据来源、空间分辨率、缺陷表达方式和产品尺寸都不同。

因此，研究重点应放在：
> 1. 如何把 WDM 散点转换成可与 WBM 比较的产品级 grid 表达。
> 2. 如何设计可解释的单图匹配分数。
> 3. 如何公平比较不同产品尺寸下的 WBM-WDM 相似度。
> 4. 如何参考现有 WBM similarity 论文建立系统 baseline。

---

## 关键设计一：产品感知 Grid 映射

WBM 尺寸随产品变化，不应简单统一 resize 到固定尺寸后比较。

推荐原则：
> 目标 WBM 是 10x10，则将 WDM 映射为 10x10。
> 目标 WBM 是 30x30，则将 WDM 映射为 30x30。

流程：
> 原始 WDM 散点
>   -> wafer 坐标归一化
>   -> 根据目标产品 die layout / WBM shape 建立 chip grid
>   -> 将 defect points 聚合到目标 WBM grid
>   -> 生成 binary / count / density / soft map

该模块可命名为：
> Product-aware WDM-to-WBM grid mapping

需要比较的 WDM 表达：
> 1. binary map: cell 内是否存在 defect
> 2. count map: cell 内 defect 数量
> 3. normalized density map: defect count 归一化
> 4. soft map: density + smoothing / morphology
> 5. three-value map: 0 / 0.5 / 1，参考 SIMI Ratio
> 6. mountain map: 基于缺陷点距离衰减生成连续密度地形

### WDM Grid Map 表达的详细定义

设目标 WBM 的 grid 尺寸为 `H x W`，第 `u,v` 个 chip cell 记为 `cell(u,v)`。原始 WDM 是散点集合：
> P = {p_1, p_2, ..., p_M}

其中 `p_j` 是一个缺陷点在 wafer 坐标系中的位置。产品感知映射的第一步，是统计每个 WBM cell 内落入多少个 WDM defect points：
> n(u,v) = number of WDM defect points inside cell(u,v)

后续六种 map 都可以看作是对 `n(u,v)` 或缺陷点集合 `P` 的不同表达。

---

#### 1. Binary Map

定义：
> B_binary(u,v) = 1, if n(u,v) > 0
>               = 0, otherwise

含义：
> 只关心某个 WBM cell 内是否出现过 WDM 缺陷点。
> 不关心缺陷点数量，也不关心缺陷密度强弱。

例子：
> 某个 cell 内有 1 个 defect point -> 1
> 某个 cell 内有 50 个 defect points -> 1
> 某个 cell 内没有 defect point -> 0

优点：
> 简单、稳定、容易解释。
> 适合第一版 baseline。
> 适合 WDM defect count 受扫描条件影响较大、不希望数量主导得分的情况。

缺点：
> 会丢失缺陷密度信息。
> 稀疏点和高密度 cluster 在 binary map 中可能完全相同。
> 对 WDM 局部缺陷强弱的表达能力较弱。

适合使用的相似度：
> IoU
> Dice
> Coverage / Leakage
> binary overlap

---

#### 2. Count Map

定义：
> B_count(u,v) = n(u,v)

含义：
> 保留每个 WBM cell 内的 WDM defect point 数量。
> 缺陷越密集，对应 cell 的值越大。

例子：
> cell A 内有 0 个点 -> 0
> cell B 内有 3 个点 -> 3
> cell C 内有 40 个点 -> 40

优点：
> 保留原始 WDM 的密度强弱。
> 适合判断 WDM 缺陷是否集中在 WBM 失效区域。
> 对 cluster、near-full、局部密集缺陷更敏感。

缺点：
> 容易被极高 defect count 的 cell 主导。
> 不同 WDM 的总点数差异会影响相似度公平性。
> 不同扫描 recipe 或检测阈值可能导致 count 不可直接比较。

适合使用的相似度：
> Cosine similarity
> NCC
> weighted overlap
> mass-based coverage

通常需要进一步归一化，因此 count map 更常作为 normalized density map 的前一步。

---

#### 3. Normalized Density Map

定义方式可以有多种，推荐比较以下三种：

按最大值归一化：
> B_density(u,v) = n(u,v) / (max_{u,v} n(u,v) + eps)

按总点数归一化：
> B_density(u,v) = n(u,v) / (sum_{u,v} n(u,v) + eps)

按 cell 面积或有效 wafer 区域归一化：
> B_density(u,v) = n(u,v) / area(cell(u,v))

含义：
> 将 count map 转换为 0~1 或概率分布形式，减少不同 WDM 总点数差异带来的影响。

优点：
> 比 binary map 保留更多密度信息。
> 比 count map 更适合跨 WDM、跨产品比较。
> 适合 WDM 总 defect count 差异较大的生产数据。

缺点：
> 按最大值归一化可能放大噪声点。
> 按总点数归一化会把总量信息抹掉。
> 不同归一化方式会影响后续相似度结果，需要实验比较。

适合使用的相似度：
> Cosine similarity
> NCC
> soft Dice
> Earth Mover's Distance optional
> Coverage / Leakage with mass normalization

推荐用途：
> 作为第一阶段最重要的 WDM 表达之一。
> 适合与 WBM 的 defect intensity / fail ratio map 比较。

---

#### 4. Soft Map

定义：
> B_soft = Smooth(B_density)

其中 `Smooth(·)` 可以是：
> Gaussian blur
> morphological closing
> distance transform
> neighbor spreading
> kernel density estimation

一种简单形式：
> B_soft(u,v) = sum_{i,j} B_density(i,j) * exp(-d((u,v),(i,j))^2 / (2*sigma^2))

含义：
> 允许 WDM 缺陷点对邻近 chip cell 产生影响，而不是只影响自己所在的 cell。

为什么需要 soft map：
> WDM 是制程扫描散点，WBM 是测试后的 chip-level fail map。
> 二者之间可能存在轻微配准误差、空间扩散效应和分辨率差异。
> 如果直接硬匹配 binary/count map，轻微错位会导致相似度大幅下降。

优点：
> 对轻微平移、边界误差和 grid 分辨率差异更鲁棒。
> 更接近“缺陷影响周围芯片失效概率”的直觉。
> 适合 WDM 很稀疏、WBM 较粗糙的情况。

缺点：
> 平滑过强会让局部结构变模糊。
> 可能扩大缺陷区域，导致 leakage 变高或虚假覆盖。
> sigma / kernel size 需要按 WBM grid resolution 调整。

适合使用的相似度：
> soft Dice
> NCC
> Cosine similarity
> Coverage / Leakage with dilated target mask

推荐参数原则：
> 10x10 左右的小 grid：sigma 不宜过大，避免整图被抹平。
> 30x30 左右的大 grid：可以允许稍大 sigma，增强局部连续性。

---

#### 5. 0 / 0.5 / 1 Three-Value Map

定义：
> B_tri(u,v) = 1.0, if cell has strong defect evidence
>            = 0.5, if cell has weak / neighboring defect evidence
>            = 0.0, otherwise

一种可实现规则：
> B_tri(u,v) = 1.0, if n(u,v) >= t_strong
> B_tri(u,v) = 0.5, if 0 < n(u,v) < t_strong
> B_tri(u,v) = 0.5, if n(u,v) = 0 but neighboring cells have sufficient defects
> B_tri(u,v) = 0.0, otherwise

也可以参考 SIMI Ratio 的思想：
> 1.0 表示强缺陷区域
> 0.5 表示弱缺陷或邻域支持区域
> 0.0 表示无缺陷区域

含义：
> 在 binary map 和 continuous density map 之间折中。
> 既保留强弱差异，又避免 count map 过度受缺陷数量影响。

优点：
> 可解释性强，适合给企业方说明。
> 比 binary map 多一个弱证据状态。
> 比 count / density map 更稳定，不容易被极端数量支配。
> 适合 SIMI-like weighted overlap baseline。

缺点：
> t_strong、邻域规则和 spatial filter 需要设定。
> 三值化会丢失连续强度信息。
> 不同产品尺寸下邻域大小需要随 grid resolution 调整。

适合使用的相似度：
> SIMI-like weighted overlap
> weighted mismatch penalty
> Coverage / Leakage with weak evidence

示例加权匹配：
> A=1, B=1.0   -> strong match
> A=1, B=0.5   -> weak match
> A=1, B=0.0   -> miss penalty
> A=0, B=1.0   -> leakage penalty
> A=0, B=0.5   -> weak leakage penalty

---

#### 6. Mountain Map

定义：
> B_mountain(u,v) = sum_{p_j in P} exp(-m * d(cell(u,v), p_j))

或使用平方距离形式：
> B_mountain(u,v) = sum_{p_j in P} exp(-d(cell(u,v), p_j)^2 / (2*sigma^2))

其中：

- `p_j` 是 WDM 中的缺陷点。
- `d(cell(u,v), p_j)` 是 cell 中心到缺陷点的距离。
- `m` 或 `sigma` 控制距离衰减速度。

含义：
> 每个 WDM 缺陷点都会在周围形成一个随距离衰减的“山峰”。
> 多个缺陷点的影响叠加后，形成一张连续的 defect density surface。

与 soft map 的区别：
> soft map 通常是先聚合到 grid，再对 grid map 做平滑。
> mountain map 更强调从原始缺陷点直接生成连续密度地形。

优点：
> 适合 WDM 原始散点数据。
> 能把离散点转换成连续空间结构。
> 对 cluster、ring、scratch 等空间结构更友好。
> 可与 WMHD / density similarity 结合。

缺点：
> 计算成本比 binary/count map 更高。
> 参数 m / sigma 对结果影响明显。
> 如果缺陷点很多，需要考虑加速或先聚合再计算。

适合使用的相似度：
> NCC
> Cosine similarity
> WMHD-like score
> shape / density similarity
> Coverage / Leakage on continuous density

推荐用途：
> 作为 Hsu 2020 mountain function 思路的 WDM-WBM 适配版本。
> 适合作为第二阶段或论文增强 baseline。

---

#### 六种 map 的对比总结

| Map 类型                  | 是否保留点数 | 是否保留强度 | 是否平滑邻域 | 可解释性 | 推荐阶段                    |
| ----------------------- | ------ | ------ | ------ | ---- | ----------------------- |
| binary map              | 否      | 否      | 否      | 很强   | 最基础 baseline            |
| count map               | 是      | 是      | 否      | 强    | density 前置表达            |
| normalized density map  | 部分     | 是      | 否      | 中等   | 第一阶段重点                  |
| soft map                | 部分     | 是      | 是      | 中等   | 第一阶段重点                  |
| 0/0.5/1 three-value map | 弱保留    | 弱/强两级  | 可选     | 很强   | 企业 baseline / SIMI-like |
| mountain map            | 是      | 是      | 是      | 中等   | 论文增强 baseline           |

推荐实现顺序：
> 1. binary map
> 2. count map
> 3. normalized density map
> 4. 0 / 0.5 / 1 three-value map
> 5. soft map
> 6. mountain map

第一轮实验不需要全部做复杂版本，建议先完成：
> binary map + normalized density map + three-value map

随后再加入：
> soft map + mountain map

---

## 关键设计二：尺寸归一化相似度

不同产品 WBM 尺寸不同，所有相似度分量应尽量使用比例或归一化距离，而不是绝对像素数。

推荐归一化方式：
> Coverage = matched_defect_mass / target_defect_mass
> Leakage  = candidate_mass_outside_target_region / candidate_defect_mass
> Location = centroid_distance / wafer_grid_diagonal
> Size     = min(defect_mass_A, defect_mass_B) / max(defect_mass_A, defect_mass_B)

这样可以保证：
> 10x10 产品和 30x30 产品下的得分具有可比性。

---

## 关键设计三：可解释单图得分

第一版建议不要只使用单一相似度，而是拆成多个可解释分量：
> Score(A, B_i) = w_cov   * Coverage(A, T_p(B_i))
>               - w_leak  * Leakage(T_p(B_i), A)
>               + w_shape * ShapeSimilarity(A, T_p(B_i))
>               + w_loc   * LocationSimilarity(A, T_p(B_i))
>               + w_size  * SizeSimilarity(A, T_p(B_i))

其中：

- `Coverage`：WDM 是否覆盖 WBM 的失效区域。
- `Leakage`：WDM 是否在 WBM 不需要的位置产生过多缺陷。
- `ShapeSimilarity`：整体形状或结构是否相似。
- `LocationSimilarity`：缺陷区域位置是否一致。
- `SizeSimilarity`：缺陷规模、面积或质量是否接近。

第一版可简化为：
> Score(A, B_i) = Coverage(A, B_i) - beta * Leakage(B_i, A)

第二版加入：
> Shape + Location + Size

第三版尝试：
> entropy-based adaptive weighting

---

## 参考论文与 baseline 设计

### Baseline 1: Basic Pixel/Grid Similarity

目的：建立最基础的直接比较方法。

候选方法：
> Dice
> IoU
> NCC
> Cosine similarity
> Hausdorff / Chamfer distance

作用：作为最低基线，证明后续方法不是只优于很弱的对照。

---

### Baseline 2: SIMI-like Weighted Overlap

参考：
> A Novel Wafer-Map Similarity Search System with High Speed and Accuracy

可借鉴点：
> Morphological closing
> Spatial filter
> 0 / 0.5 / 1 three-value map
> weighted map subtraction / SIMI Ratio

当前适配方式：
> WDM scatter -> target WBM grid -> three-value map -> weighted overlap score

优点：
> 快、可解释、适合作为企业 baseline。

---

### Baseline 3: Mountain / WMHD-like Matching

参考：
> Hsu et al. 2020, Similarity matching of wafer bin maps

可借鉴点：
> Mountain Function
> Weighted Modified Hausdorff Distance
> Outlier penalty

当前适配方式：
> WDM scatter -> mountain / density surface
> WBM -> comparable defect surface
> score = matched distance + unmatched / leakage penalty

优点：
> 适合稀疏点与粗粒度失效图之间的匹配。

---

### Baseline 4: Tensor Voting + WBBS / MBBS

参考：
> Tensor Voting Based Similarity Matching
> Wang and Wang 2023, WBBS

可借鉴点：
> Tensor voting structural saliency
> Best-Buddies Similarity
> Weighted Best-Buddies Similarity
> partial matching and outlier tolerance

当前适配方式：
> WDM / WBM -> point or saliency representation
> score by mutual nearest neighbor matching

优点：
> 适合局部匹配，能够缓解 WDM 中存在无关散点的问题。

---

### Baseline 5: Shape / Location / Size Similarity

参考：
> Kang et al. 2024, Similarity searching by shape, location, and size

可借鉴点：
> shape similarity
> location similarity
> size similarity
> entropy-based weighting

当前适配方式：
> WBM vs mapped WDM
> 分别计算 shape / location / size
> 再融合为整体得分

优点：
> 可解释性强，适合作为论文主要对比和改进方向。

---

### Baseline 6: Density Feature Retrieval Optional

参考：
> Wafer Defect Map Similarity Search Using Deep Learning

可借鉴点：
> defect density gray image
> feature extractor
> nearest neighbor retrieval

当前定位：
> 可作为后续深度特征检索 baseline，不建议第一版主打。

---

## 推荐第一阶段实验顺序

### Experiment 1: WDM 表达方式对比

目的：验证 WDM 散点映射成哪种 grid 表达最适合匹配 WBM。

比较：
> binary map
> count map
> normalized density map
> soft map
> 0 / 0.5 / 1 three-value map
> mountain map

指标：
> top-1 / top-3 人工通过率
> coverage
> leakage
> runtime
> case visualization

---

### Experiment 2: 单图相似度方法对比

目的：完成参考论文方法的第一轮 baseline。

比较：
> Dice / IoU / NCC
> SIMI-like score
> Coverage-Leakage score
> WMHD / mountain-like score
> WBBS / MBBS score
> Shape-Location-Size score

输出：
> 每个 query WBM 的 top-k WDM
> score decomposition
> A vs top-1 / top-3 WDM visualization

---

### Experiment 3: 不同 WBM 尺寸下的鲁棒性

目的：验证产品感知 grid mapping 是否必要。

分组：
> small grid WBM: around 10x10
> medium grid WBM: around 20x20
> large grid WBM: around 30x30

比较：
> fixed resize strategy
> product-aware target-grid strategy

重点观察：
> 相似度是否因尺寸差异产生偏置
> top-k 结果是否稳定
> coverage / leakage 是否可比

---

### Experiment 4: 可解释分量消融

目的：证明不是单纯相似度堆叠，而是每个分量有作用。

比较：
> Coverage only
> Coverage - Leakage
> Coverage - Leakage + Shape
> Coverage - Leakage + Shape + Location
> Coverage - Leakage + Shape + Location + Size
> Entropy-weighted version

输出：
> top-k pass rate
> score distribution
> failure case analysis

---

### Experiment 5: 单图 baseline 的上限分析

目的：为后续是否需要组合搜索提供证据，而不是直接主张组合搜索必要。

分析方式：
> 1. 找出最佳单张 WDM 得分低但局部区域明显相关的 case。
> 2. 找出 top-3 多张 WDM 分别覆盖 WBM 不同区域的 case。
> 3. 统计人工审核中“单张无法完整解释，但多张可能解释”的比例。

结论可能有两种：
> 若最佳单张已经足够解释多数 WBM，则组合搜索不是第一优先级。
> 若大量 case 需要多张 WDM 共同解释，则组合搜索作为第二阶段自然成立。

---

## 评估方案

### 1. 人工审核

生产数据可能缺少真实标签，因此第一阶段以工程审核为主。

建议输出：
> Query WBM
> Top-1 / Top-3 / Top-5 WDM
> mapped WDM grid
> overlay visualization
> score decomposition

人工判断：
> 是否空间位置一致
> 是否形状相似
> 是否大小合理
> 是否存在明显无关散点
> 是否可解释 WBM 主要失效区域

指标：
> top-1 pass rate
> top-3 pass rate
> mean expert score
> failure category distribution

---

### 2. Synthetic / Controlled Evaluation

如果缺少标注，可以构造可控评估集。

方式：
> 从真实 WDM 或 WBM pattern 生成 pseudo query
> 保留已知来源 WDM 作为 ground truth
> 测试方法能否把真实来源排在 top-k

指标：
> top-k accuracy
> mean reciprocal rank
> recall@k

---

### 3. 工程运行指标

企业场景需要报告速度。

指标：
> single query runtime
> average candidate scoring time
> top-k retrieval time
> memory usage

---

## 实现里程碑

### Milestone 1: 数据与映射闭环

目标：完成 WDM 散点到目标 WBM grid 的产品感知映射。

任务：
> 1. 读取 WBM shape / product grid 信息
> 2. 将 WDM defect points 映射到目标 WBM grid
> 3. 生成 binary / count / density / soft map
> 4. 输出 WBM 与 mapped WDM 的可视化对比

产出：
> mapping module
> visualization examples
> different WBM size case study

---

### Milestone 2: 单图基础相似度 baseline

任务：
> 1. Dice / IoU / NCC / Cosine
> 2. SIMI-like weighted overlap
> 3. Coverage-Leakage score
> 4. top-k retrieval pipeline

产出：
> baseline ranking results
> top-k visualization
> runtime report

---

### Milestone 3: 论文方法对比

任务：
> 1. Mountain / WMHD-like score
> 2. Tensor Voting / WBBS or MBBS score
> 3. Shape-Location-Size score
> 4. entropy-based weighting optional

产出：
> method comparison table
> ablation results
> failure case analysis

---

### Milestone 4: 产品尺寸鲁棒性实验

任务：
> 1. 按 WBM grid size 分组
> 2. 比较 fixed resize 与 target-grid mapping
> 3. 分析 score normalization 是否有效

产出：
> small / medium / large grid results
> cross-product robustness analysis

---

### Milestone 5: 单图上限与组合搜索决策

任务：
> 1. 分析最佳单张 WDM 失败 case
> 2. 判断失败是否来自多源组合需求
> 3. 决定是否进入组合搜索第二阶段

产出：
> single-WDM limitation report
> combination-search motivation evidence

---

## 与组合搜索的关系

当前阶段不否定组合搜索，而是将其放在更合理的位置：
> Stage 1: single-WDM retrieval baseline
> Stage 2: analyze whether single WDM is insufficient
> Stage 3: if necessary, extend to multi-WDM combinatorial explanation

这样可以回应企业方意见：
> 先完成单图搜索基线，证明现有方法能达到什么水平。
> 只有当单图搜索无法解释复杂 WBM 时，再引入组合搜索。

也可以支撑论文叙事：
> 本文首先系统研究 WBM-WDM 单图跨源检索问题，提出产品感知映射与可解释多分量相似度；随后通过失败案例分析讨论多 WDM 组合解释的必要性。

---

## 推荐论文结构调整
> 1. Introduction
>    - WBM-WDM cross-source matching motivation
>    - difficulty: WBM grid map vs WDM scatter map
>    - difficulty: product-dependent WBM resolution
>    - need for interpretable single-WDM retrieval baseline
> 2. Related Work
>    - WBM similarity search
>    - density / mountain representation
>    - tensor voting / WBBS / WMHD
>    - shape-location-size similarity
> 3. Problem Formulation
>    - single-WDM retrieval
>    - product-aware WDM-to-WBM mapping
>    - normalized similarity components
> 4. Method
>    - product-aware grid mapping
>    - WDM representation variants
>    - coverage-leakage similarity
>    - shape/location/size score
>    - optional adaptive weighting
> 5. Experiments
>    - baseline comparison
>    - different WBM size analysis
>    - ablation study
>    - production case study / expert audit
>    - single-WDM limitation analysis
> 6. Discussion
>    - when single-WDM retrieval is sufficient
>    - when multi-WDM combination may be needed
>    - limitations and future work
> 7. Conclusion

---

## 一句话总结

当前阶段建议先完成：
> Product-aware single-WDM retrieval baseline

而不是直接主打组合搜索。

真正的研究价值不在于“计算一个相似度”，而在于：
> 将 WDM 散点在不同产品尺寸下映射到目标 WBM grid，
> 并设计可解释、可归一化、可对比的单图检索评分体系。

组合搜索保留为第二阶段，用单图 baseline 的失败案例来证明其必要性。
