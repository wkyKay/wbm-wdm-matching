# Proposed Local Learned Cluster Retrieval

本文档是主方法的内部实现设计文档，用于约束后续代码实现、实验调用和与 baseline 的公平对比；主方法的核心边界是：proposal、显式几何特征和 local token matching 与 `partial_match` 保持一致，cluster embedding 使用 WaPIRL-style 自监督对比学习重新实现，不直接调用 `WaPIRL/` 目录中的代码。

## Method Logic

主方法对应 `tasks/plan.md` 中的 `Proposed, local learned cluster retrieval`，目标是在 wafer map 中先提取少量稳定的 defect cluster proposal，再把每张 map 表示成 cluster token set，最后通过局部 token matching 得到 map-level retrieval score。它和 `partial_match` 的差异只应该集中在 cluster 的 shape representation：`partial_match` 使用 handcrafted descriptor，主方法使用 learned cluster embedding，因此 proposal 方法、cluster 的 `area / centroid / bbox / geometry_type` 等显式字段、`area_affinity`、`position_affinity`、`type_affinity`、query-token 加权聚合方式都应保持一致。

对比学习部分采用 WaPIRL-style 思路，但训练对象从 whole wafer map 改成由同一个 proposal token 派生出的固定尺寸 masked patch。每个 cluster 生成两种增强视图作为正样本，memory bank 或 batch 内其他 cluster embedding 作为负样本，训练 `encoder + projection head`，保存 checkpoint 时保留 `encoder` 和 `projector`，retrieval 阶段只使用 `encoder` 产生 cluster embedding。该实现应在 `proposed/` 内独立复写，允许参考 `Wafer-DenseIR/tasks/wapirl_pretrain.py` 的训练流程和 `utils/loss.py` 的 NCE 形式，但不要直接 import `WaPIRL/` 或 `Wafer-DenseIR/` 的训练任务代码。

## Data Protocol

所有正式实验必须使用 `shared/` 已经冻结的 manifest，避免每个方法单独抽样导致评价不可比。训练阶段只读取 train split 中的 wafer maps 来生成 cluster crops，并且不使用 pattern label；validation split 用于选择 checkpoint、matching 权重和 proposal 超参数；test split 只用于最终 retrieval ranking，并且必须使用相同的 query manifest 和 candidate manifest。

默认正式 manifest：

```text
artifacts/splits/wm38k_seed2026_sig_70_10_20.csv
artifacts/splits/wm38k_seed2026_test_queries_2000.csv
artifacts/splits/wm38k_seed2026_test_candidates_1000.csv
```

正式 ranking 输出 schema 与其他方法一致：

```text
query_id,rank,candidate_id,similarity_score
```

## Code Structure

建议实现为以下文件结构，其中 `core/proposal.py` 只做轻量 adapter，直接调用 `partial_match.core.clustering.cluster` 以保证主方法和传统 baseline 使用完全相同的 proposal 逻辑；如果未来需要把 proposal 抽到更通用的共享包，应先保证 `partial_match` 和 `proposed` 同时切换到同一个共享实现。设计上要把 proposal、patch 构造、对比学习和 retrieval matching 拆成四层，层与层之间只能通过稳定 token schema 传递数据，避免 proposal 后续修改时牵动 encoder 训练代码。

```text
proposed/
├── README_INTERNAL.md
├── __init__.py
├── configs/
│   ├── __init__.py
│   └── task_configs.py
├── core/
│   ├── __init__.py
│   ├── proposal.py
│   ├── cluster_patches.py
│   ├── learned_descriptors.py
│   └── matching.py
├── datasets/
│   ├── __init__.py
│   ├── wm38k_maps.py
│   └── cluster_contrastive.py
├── models/
│   ├── __init__.py
│   ├── encoder.py
│   └── head.py
├── tasks/
│   ├── __init__.py
│   ├── cluster_pretrain.py
│   └── learned_retrieval.py
├── utils/
│   ├── __init__.py
│   ├── loss.py
│   ├── logging.py
│   └── optimization.py
├── run_cluster_pretrain.py
└── run_learned_retrieval_pipeline.py
```

## Module Boundaries

主方法的代码边界应按下面的方向依赖组织，除 `core/proposal.py` 之外，对比学习和检索代码都不应直接 import `partial_match.core.clustering`，这样后续替换 proposal 时只需要改 adapter 或配置，不需要改 dataset、encoder、loss 和 matching 逻辑。

```text
raw wafer map
 -> ProposalProvider
 -> ClusterToken[]
 -> PatchBuilder
 -> ContrastiveDataset / Encoder
 -> LearnedTokenRecord[]
 -> Matcher
 -> rankings.csv
```

稳定接口一：`ProposalProvider`。输入是 `map_id`、`raw_map` 和 proposal config，输出是 `List[ClusterToken]`。第一版 `PartialMatchProposalProvider` 内部调用 `partial_match.core.clustering.cluster`，未来如果 proposal 改成新版算法，只新增 `NewProposalProvider` 或修改 provider 内部实现，其他模块仍消费同样的 `ClusterToken` schema。

```python
class ProposalProvider:
    def extract(self, map_id: int, raw_map: np.ndarray) -> list[ClusterToken]:
        ...
```

稳定接口二：`ClusterToken`。这是 proposal 和 learned encoder 之间的唯一契约，字段应完全来自 proposal 输出或由 proposal 输出确定，patch builder 和 encoder 不允许修改这些字段。建议第一版用 dataclass 或 TypedDict 明确字段，至少包含 `map_id`、`token_id`、`pixels`、`area`、`area_ratio`、`centroid_row`、`centroid_col`、`bbox_row_min`、`bbox_row_max`、`bbox_col_min`、`bbox_col_max`、`geometry_type`、`proposal_method`、`proposal_signature`。

```python
@dataclass(frozen=True)
class ClusterToken:
    map_id: int
    token_id: int
    pixels: tuple[tuple[int, int], ...]
    area: float
    area_ratio: float
    centroid_row: float
    centroid_col: float
    bbox_row_min: int
    bbox_row_max: int
    bbox_col_min: int
    bbox_col_max: int
    geometry_type: str
    proposal_method: str
    proposal_signature: str
```

稳定接口三：`PatchBuilder`。输入是 `raw_map` 和 `ClusterToken`，输出固定尺寸 tensor 以及可选 patch metadata。它只能读取 token 字段，不能改变 token 字段；如果 patch 策略从 proposal-centered fixed window 改成 full-map masked input，也只替换 `PatchBuilder`，不修改 proposal provider、contrastive loss 或 retrieval matcher。

```python
class PatchBuilder:
    def build(self, raw_map: np.ndarray, token: ClusterToken) -> PatchSample:
        ...
```

稳定接口四：`LearnedTokenRecord`。这是 retrieval matching 消费的 token 记录，等价于 `partial_match.core.descriptors.clusters_to_records` 的输出，只把 `shape_descriptor` 替换为 learned embedding。它应包含 `embedding`、`shape_descriptor`、`area`、`area_ratio`、`pos`、`geometry_type`、`cluster`，从而让 `core/matching.py` 能复用同一套 area/position/type/matching 语义。

```python
@dataclass
class LearnedTokenRecord:
    map_id: int
    token_id: int
    embedding: np.ndarray
    shape_descriptor: np.ndarray
    area: float
    area_ratio: float
    pos: np.ndarray
    geometry_type: str
    cluster: ClusterToken
```

## Replaceable Proposal Design

为了让 proposal 后续可插拔，proposal 配置要单独序列化，并且每次训练和检索都要保存当前 proposal 配置到 `proposal_config.json`。对比学习 checkpoint 也必须记录训练时使用的 `proposal_config` 和 `patch_config`，因为 cluster-level encoder 学到的是某一种 token 定义下的表示；如果 proposal 发生语义变化，例如 token 数量、ring-aware 逻辑或粘连拆分逻辑改变，原则上应重新训练 cluster encoder，不能直接把旧 checkpoint 当作同一实验条件下的主方法结果。

缓存也要按边界拆开，避免一个 artifact 隐含多个阶段的状态。建议输出：

```text
proposal_tokens.csv
proposal_tokens_meta.json
cluster_patches_manifest.csv
cluster_pretrain/best_model.pt
retrieval/tokens.csv
retrieval/embeddings.npz
retrieval/rankings.csv
```

`proposal_tokens.csv` 是 proposal 阶段产物，可被 contrastive pretrain 和 retrieval 共同读取；`cluster_patches_manifest.csv` 只记录 patch 构造参数和 token 引用，不作为新的 token 定义；`embeddings.npz` 只缓存 learned descriptor，不缓存或覆盖 proposal 字段。后续修改 proposal 时，应只删除或重建 proposal/token 相关缓存，encoder 和 retrieval 代码路径不需要改；后续修改 patch 策略或 encoder 时，不应影响 proposal token metadata。

## Consistency Checks

正式比较前必须提供一个一致性检查脚本或 pipeline 子步骤，建议命名为 `proposed/scripts/check_proposal_consistency.py`，输入 `partial_match` 的 `tokens.csv` 和 `proposed` 的 `proposal_tokens.csv`，检查同一 `map_id, token_id` 下的 `area`、`centroid_row`、`centroid_col`、`bbox_*`、`geometry_type`、`proposal_type`、`proposal_source` 是否一致。对于 `pixels` 这类不适合直接放 CSV 的字段，应使用 `proposal_signature`，例如对排序后的 pixel 坐标和关键配置做稳定 hash。

一致性检查失败时，不能把该结果作为“同 proposal”主实验，只能标为 proposal ablation 或先同步修改 `partial_match`，让两个方法重新共享同一个 proposal provider。

### File Responsibilities

`configs/task_configs.py` 负责解析训练和检索参数，包括 `data_file`、`split_manifest`、`query_manifest`、`candidate_manifest`、proposal 参数、patch 参数、encoder 参数、optimizer 参数和输出目录。这里的参数命名尽量与 `partial_match/run_proposal_retrieval_pipeline.py` 以及 `Wafer-DenseIR/run_wapirl_pretrain.py` 对齐，减少实验脚本维护成本。

`core/proposal.py` 负责定义 `ProposalProvider`、`ClusterToken` 和 `PartialMatchProposalProvider`，把 raw wafer map 转换为 `defect_mask / valid_mask`，然后调用 `partial_match.core.clustering.cluster(..., method='retrieval_compact', min_area=5, top_k=6, enable_ring_aware=True, ...)`，返回与 `partial_match` 一致的 cluster token。该文件不要实现新的 proposal 算法，除非是显式 ablation；正式比较时还应保存 `proposal_signature` 或 token metadata 表，用来确认同一 `map_id` 下的 `token_id / pixels / area / centroid / bbox / geometry_type` 与 `partial_match` 完全一致。

`core/cluster_patches.py` 负责根据已经确定的 cluster `pixels / bbox / centroid` 生成固定尺寸 proposal-centered masked patch，例如 `64x64` 或 `96x96` 的模型输入。这个 patch 只是 learned encoder 的输入视图，不是新的 proposal，也不能回写或修改 token 的 `area / centroid / bbox / pixels` 等几何字段。默认输入建议为 `channel 0 = local defect context`、`channel 1 = proposal mask`、`channel 2 = local valid/context mask`；不要把每个 proposal 的最小 bbox 单独 resize 到同一尺寸作为主方案，因为这种 resize 会改变不同 cluster 的相对尺度和形状比例，导致它与 `partial_match` 的显式面积、bbox 和形状统计不再处在同一个处理语义下。

`datasets/cluster_contrastive.py` 负责在 train split 上抽取所有有效 cluster token，并为每个 token返回 `x`、`x_t`、`idx`、`map_id`、`token_id`、`cluster_meta`。增强只作用于 fixed-size masked patch 的图像视图，不改变 proposal token 本身，不改变 label，不读取 test query/candidate manifest。

`models/encoder.py` 和 `models/head.py` 负责 cluster encoder 与 projection head，第一版建议使用小型 ResNet 或轻量 CNN，输入尺寸与 `cluster_patches.py` 固定，输出 embedding 维度如 `128 / 256`。projection head 只用于对比学习，retrieval 阶段使用 encoder 输出或 encoder 后的 normalized embedding，不使用 projector 输出，除非作为 ablation 单独记录。

`utils/loss.py` 复写 WaPIRL-style NCE loss，输入为 `anchors / positives / negatives`，使用 cosine similarity 和 temperature-scaled cross entropy；如果使用 memory bank，负样本来自 train cluster memory，若使用 batch negatives，则要在文档和 config 中明确模式。

`tasks/cluster_pretrain.py` 负责 memory bank 初始化、train/valid epoch、checkpoint 保存、history 输出和 best checkpoint 选择。best checkpoint 只根据 valid contrastive loss 或 valid top1 proxy 指标选择，不能使用 test label metrics。

`core/learned_descriptors.py` 负责加载 checkpoint，对 train/valid/test 中需要参与检索的 map 提取 proposal tokens，并为每个 token 计算 learned embedding，同时保留和 `partial_match` 一致的 `area`、`area_ratio`、`pos`、`geometry_type`、`cluster` 字段。这里必须先固定 proposal token，再从 token 生成 patch 和 embedding；不能先通过 patch、resize 或模型结果反向影响 token 是否保留、token 面积或 token 位置。该模块应支持从已保存的 `proposal_tokens.csv` 读取 token，避免 retrieval 阶段和 pretrain 阶段因为重复运行 proposal 而出现隐性版本差异。

`core/matching.py` 负责主方法的 token matching，建议从 `partial_match.core.descriptors` 的匹配公式复写或封装为同等逻辑，只把 `shape_sim = handcrafted_descriptor_dot` 替换为 `shape_sim = learned_embedding_cosine`。为保证实验解释清楚，`sigma_pos`、`sigma_area`、`topk_match` 和 query token 权重默认值应与 `partial_match` 一致。

`tasks/learned_retrieval.py` 负责在 fixed query/candidate manifest 上生成 `rankings.csv`，并调用 `evaluation/evaluate_rankings.py` 生成 `label_metrics.json` 和 `label_metrics_flat.csv`。它不负责重新划分数据，也不使用 label 控制排序。

`run_cluster_pretrain.py` 是 cluster-level 对比学习入口，输出 checkpoint、history、config、proposal config、patch config 和可选 embedding 诊断图。`run_learned_retrieval_pipeline.py` 是正式检索入口，输出 `proposal_tokens.csv`、`rankings.csv`、`tokens.csv`、`embeddings.npz`、`match_details.csv`、`label_metrics.json` 和 `label_metrics_flat.csv`。

## Implementation Order

第一步先实现 `core/proposal.py`、`core/cluster_patches.py` 和 `datasets/cluster_contrastive.py`，用 train split 跑一个小样本 smoke test，确认每张 map 的 proposal 数量、token metadata 与 `partial_match` 完全一致，同时确认 patch 尺寸统一且只作为 encoder 输入视图。第二步实现 `utils/loss.py`、`models/encoder.py`、`models/head.py` 和 `tasks/cluster_pretrain.py`，先用 `--max_train_clusters 512 --epochs 1 --device cpu` 或小 GPU 配置确认 checkpoint 可保存。第三步实现 `core/learned_descriptors.py`、`core/matching.py` 和 `tasks/learned_retrieval.py`，在 3 个 query、每个 query 5 个 candidate 的临时 manifest 上验证 ranking schema。第四步接入正式 candidate manifest，运行官方 `evaluation/` 指标，并与 `partial_match` 在相同 query/candidate 下比较。

## Commands

Cluster-level contrastive pretraining on the fixed train split:

```bash
cd /Users/kayw/Documents/trae_projects/match-test/wbm-wdm-matching
python3 proposed/run_cluster_pretrain.py \
  --data-file ../data/wm38k/Wafer_Map_Datasets.npz \
  --split-manifest artifacts/splits/wm38k_seed2026_sig_70_10_20.csv \
  --split train \
  --valid-split valid \
  --proposal-method retrieval_compact \
  --min-area 5 \
  --top-k-proposals 6 \
  --patch-size 96 \
  --embedding-dim 256 \
  --projector-size 256 \
  --batch-size 128 \
  --num-workers 4 \
  --epochs 100 \
  --num-negatives 1024 \
  --temperature 0.07 \
  --device cuda \
  --out-dir artifacts/proposed/cluster_pretrain/wm38k_seed2026
```

Small smoke pretraining run:

```bash
cd /Users/kayw/Documents/trae_projects/match-test/wbm-wdm-matching
python3 proposed/run_cluster_pretrain.py \
  --data-file ../data/wm38k/Wafer_Map_Datasets.npz \
  --split-manifest artifacts/splits/wm38k_seed2026_sig_70_10_20.csv \
  --split train \
  --valid-split valid \
  --max-train-clusters 512 \
  --max-valid-clusters 128 \
  --epochs 1 \
  --batch-size 32 \
  --device cpu \
  --out-dir artifacts/proposed/smoke_pretrain
```

Formal learned local retrieval using the fixed test query set and controlled candidate pool:

```bash
cd /Users/kayw/Documents/trae_projects/match-test/wbm-wdm-matching
python3 proposed/run_learned_retrieval_pipeline.py \
  --data-file ../data/wm38k/Wafer_Map_Datasets.npz \
  --split-manifest artifacts/splits/wm38k_seed2026_sig_70_10_20.csv \
  --query-manifest artifacts/splits/wm38k_seed2026_test_queries_2000.csv \
  --candidate-manifest artifacts/splits/wm38k_seed2026_test_candidates_1000.csv \
  --checkpoint artifacts/proposed/cluster_pretrain/wm38k_seed2026/best_model.pt \
  --checkpoint-key encoder \
  --split test \
  --proposal-method retrieval_compact \
  --min-area 5 \
  --top-k-proposals 6 \
  --patch-size 96 \
  --topk-match 1 \
  --sigma-pos 0.35 \
  --sigma-area 1.0 \
  --metric-k 1 5 10 \
  --device cuda \
  --out-dir artifacts/proposed/retrieval/wm38k_seed2026_test_candidates_1000
```

Expected formal outputs:

```text
artifacts/proposed/retrieval/wm38k_seed2026_test_candidates_1000/
├── configs.json
├── rankings.csv
├── tokens.csv
├── embeddings.npz
├── match_details.csv
├── label_metrics.json
└── label_metrics_flat.csv
```

Official evaluation can also be run independently:

```bash
cd /Users/kayw/Documents/trae_projects/match-test/wbm-wdm-matching
python3 evaluation/evaluate_rankings.py \
  --rankings artifacts/proposed/retrieval/wm38k_seed2026_test_candidates_1000/rankings.csv \
  --split-manifest artifacts/splits/wm38k_seed2026_sig_70_10_20.csv \
  --query-manifest artifacts/splits/wm38k_seed2026_test_queries_2000.csv \
  --candidate-manifest artifacts/splits/wm38k_seed2026_test_candidates_1000.csv \
  --out artifacts/proposed/retrieval/wm38k_seed2026_test_candidates_1000/label_metrics.json \
  --flat-out artifacts/proposed/retrieval/wm38k_seed2026_test_candidates_1000/label_metrics_flat.csv \
  --k 1 5 10
```

## Fairness Constraints

主方法与 `partial_match` 比较时，必须固定相同的 `proposal-method`、`min-area`、`top-k-proposals`、`topk-match`、`sigma-pos`、`sigma-area`、query manifest 和 candidate manifest；若调整其中任何参数，应同时运行 `partial_match` 的对应配置，或者把该结果标为 ablation。固定尺寸 masked patch 只是 learned descriptor 的输入，不属于 proposal 处理，因此它不能改变 token 划分、token 数量、token 几何字段或 candidate/query 选择；正式结果中应保留 token metadata，便于和 `partial_match` 的 `tokens.csv` 做一致性检查。主方法训练只能使用 train/valid split，不允许用 test label 选择 checkpoint；retrieval 阶段可以读取 test map 图像本身生成 proposal 和 embedding，但不能读取 label 参与排序。所有正式结果以 `evaluation/` 的 `label_metrics.json` 为准，方法内部 metric 只能作为诊断。

## Relationship To Baselines

`partial_match` 是同 proposal、同 local matching、不同 embedding 的传统 baseline，因此它回答 learned cluster embedding 是否优于 handcrafted cluster descriptor。`Wafer-DenseIR` 是 whole-map / dense learned retrieval baseline，它回答 proposal-based cluster-set matching 是否比 proposal-free dense/whole-map representation 更适合该检索任务。`WaPIRL/` 是参考论文代码，不属于当前主方法直接依赖；主方法只采用 WaPIRL-style contrastive learning idea，并在 cluster-level 数据对象上独立实现。
