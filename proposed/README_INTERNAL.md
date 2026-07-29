# Proposed Local Learned Cluster Retrieval

主方法将每张 wafer map 表示为少量局部缺陷 Token：使用 `arc-ring-residual` proposal 分离外围环/弧与内部残差缺陷，为每个 Token 学习局部形状嵌入，再结合显式位置和尺度信息进行贪心一对一匹配。相较于 `partial_match`，主方法只替换形状描述子：传统基线使用手工 Zernike/moment 与几何描述子，主方法使用 WaPIRL-style 自监督学习的 Token embedding。

当前实现复用 `partial_match` 的 `arc-ring-residual` proposal、局部评分函数和 WM38K 数据读取器；对比学习、局部图块构造、编码器及检索入口实现在本目录中。因此 `proposed/` 当前不能在删除 `partial_match/` 后独立运行。

## Run Commands

所有命令从仓库根目录执行。训练与检索必须使用相同的 `--min-area`、`--top-k-proposals`、`--patch-size`、`--encoder` 和 `--embedding-dim`；改变 proposal、局部图块或 encoder 后，应重新训练 checkpoint。

### 1. Contrastive pretraining

```bash
python3 proposed/run_cluster_pretrain.py \
  --data-file ../data/wm38k/Wafer_Map_Datasets.npz \
  --split-manifest artifacts/splits/wm38k_seed2026_sig_70_10_20.csv \
  --split train \
  --valid-split valid \
  --min-area 5 \
  --top-k-proposals 5 \
  --patch-size 64 \
  --encoder resnet18 \
  --embedding-dim 256 \
  --projector-size 256 \
  --batch-size 128 \
  --num-workers 4 \
  --epochs 100 \
  --num-negatives 1024 \
  --temperature 0.07 \
  --device cuda \
  --out-dir artifacts/proposed/cluster_pretrain/wm38k_seed2026_resnet18
```

最小 smoke test：在上述命令中增加 `--max-train-clusters 512 --max-valid-clusters 128 --epochs 1 --batch-size 32`。

### 2. Experiment A retrieval

```bash
python3 proposed/run_learned_retrieval_pipeline.py \
  --data-file ../data/wm38k/Wafer_Map_Datasets.npz \
  --split-manifest artifacts/splits/wm38k_seed2026_sig_70_10_20.csv \
  --query-manifest artifacts/splits/wm38k_seed2026_test_queries_2000.csv \
  --candidate-manifest artifacts/splits/wm38k_seed2026_test_candidates_1000.csv \
  --checkpoint artifacts/proposed/cluster_pretrain/wm38k_seed2026_resnet18/best_model.pt \
  --checkpoint-key encoder \
  --split test \
  --min-area 5 \
  --top-k-proposals 5 \
  --patch-size 64 \
  --encoder resnet18 \
  --embedding-dim 256 \
  --sigma-pos 0.35 \
  --sigma-scale 1.5 \
  --min-token-score 0.30 \
  --min-relative-token-area 0.10 \
  --scale-ratio-min 0.20 \
  --device cuda \
  --out-dir artifacts/proposed/retrieval/wm38k_seed2026_resnet18_test_candidates_1000
```

### 3. Experiment B retrieval

```bash
python3 proposed/run_retrieval_B.py \
  --b-data <experiment-b-data.npz> \
  --b-split-manifest <experiment-b-split.csv> \
  --b-queries <experiment-b-queries.csv> \
  --b-candidates <experiment-b-candidates.csv> \
  --b-preferences <experiment-b-preferences.csv> \
  --checkpoint artifacts/proposed/cluster_pretrain/wm38k_seed2026_resnet18/best_model.pt \
  --checkpoint-key encoder \
  --encoder resnet18 \
  --embedding-dim 256 \
  --device cuda \
  --out-dir artifacts/preference_b/proposed
```

## Data Protocol

正式实验使用冻结的共享 manifest，避免不同方法采用不同抽样：

```text
artifacts/splits/wm38k_seed2026_sig_70_10_20.csv
artifacts/splits/wm38k_seed2026_test_queries_2000.csv
artifacts/splits/wm38k_seed2026_test_candidates_1000.csv
```

训练仅使用 `train` Token，validation 仅用于 checkpoint 选择，test 仅用于最终检索。类别标签不参与 proposal、训练损失或排序，仅用于检索完成后的官方评价。Experiment A 的 ranking schema 固定为：

```text
query_id,rank,candidate_id,similarity_score
```

## End-to-End Method

```text
raw wafer map
 -> arc-ring-residual proposal
 -> immutable ClusterToken[]
 -> 3-channel 64x64 patch
 -> self-supervised encoder pretraining
 -> normalized learned Token embeddings
 -> hard gates + greedy one-to-one matching
 -> rankings.csv and label metrics
```

### 1. Arc-ring-residual proposal

`core/proposal.py` converts raw maps to `defect_mask = (raw_map == 2)` and `valid_mask = (raw_map == 1) | (raw_map == 2)`, then calls `partial_match.core.arc_ring_retrieval.prepare_tokens`.

普通8连通域容易将外缘环带与内部团状或中心缺陷粘连成一个大区域。`arc-ring-residual` 先在晶圆外缘的径向 band 内检测环/弧，再从原始缺陷掩膜中移除这些像素，最后对残差执行8连通域分解。Token 的几何类别包括 `edge_ring`、`ring_arc`、`line`、`blob`、`central` 和 `irregular`。类别用于 proposal 排序、边缘 extent 惩罚和结果解释，不是跨图匹配的类别硬约束。

每个 `ClusterToken` 固化保存像素集合、面积、有效晶圆相对面积、质心、边界框、PCA 特征值、径向位置、角度覆盖、proposal 来源和稳定的 `proposal_signature`。patch 和 encoder 只能读取这些字段，不会反向改变 Token。

| Proposal 参数 | 默认值 | 含义 |
|---|---:|---|
| `--min-area` | `5` | 去除面积小于5像素的候选区域 |
| `--top-k-proposals` | `5` | 每图的目标 Token 数量 |

环/弧优先，剩余名额由残差区域的重要性补充。注意：当前底层实现在有效环/弧多于 `top_k` 时可能返回超过该目标数量的 Token；正式实验应检查生成的 `proposal_tokens.csv`。

### 2. Token patch

`core/cluster_patches.py` 以 Token 质心为中心，直接从原始 $52\times52$ map 裁取固定大小的局部窗口。默认窗口为 $64\times64$，越界位置补0，不 resize Token bbox 或原始窗口。输入为3通道：

1. 通道0：局部全部缺陷掩膜；
2. 通道1：当前 Token 像素掩膜；
3. 通道2：晶圆有效区域掩膜。

| Patch 参数 | 默认值 | 含义 |
|---|---:|---|
| `--patch-size` | `64` | 直接裁取的局部窗口边长 |

### 3. Contrastive pretraining

训练集的每个 Token 是一个无标签样本。`ClusterContrastiveDataset` 为同一个原始 patch 构造两个独立增强视图 `x` 和 `x_t`；二者均被增强，`x` 不是未增强原图。当前增强及默认值如下：

| 增强 | 默认值 | 作用 |
|---|---:|---|
| 旋转 | 从 $0,90,180,270^\circ$ 采样 | 三通道同步 |
| 平移 | 行、列均在 $[-3,3]$ | 三通道同步，越界补0 |
| dropout | `0.02` | 仅通道0、1 |
| 伯努利噪声 | `0.01` | 仅通道0 |

通道2仅跟随几何变换，不加入噪声或 dropout。随机数种子由训练种子和 Token 索引确定，所以同一 Token 在不同 epoch 使用同一对增强结果。

Encoder 输出单位长度 embedding；`simple` 是轻量 CNN，`resnet18` 是针对 $64\times64$ 稀疏 patch 调整的 ResNet-18，使用 $3\times3$、stride 1 stem 且不含初始 max-pool。两者之后接两层 MLP projection head，projection head 仅服务训练，检索阶段只使用 encoder 输出。

训练使用 memory bank，而不是 batch-negative 模式。训练开始前，以全部训练 Token 的投影表示初始化 memory；每一步从其他 Token 的 memory 表示中随机抽取负例。对于当前 Token，memory 以指数移动平均更新：

$$
\mathbf{m}_i \leftarrow \gamma\mathbf{m}_i+(1-\gamma)\mathbf{z}_i.
$$

训练损失是两个增强视图相对于同一 memory anchor 的温度缩放 NCE loss 的加权和。validation 不更新 memory，直接计算两个当前视图之间的对比损失。

| Training 参数 | 默认值 | 含义 |
|---|---:|---|
| `--encoder` | `simple` | `simple` 或 `resnet18` |
| `--embedding-dim` | `256` | encoder 输出维度 |
| `--encoder-width` | `32` | 仅 `simple` 的基础通道数 |
| `--projector-size` | `256` | projection head 输出维度 / memory bank 宽度 |
| `--batch-size` | `128` | 每步 Token 数 |
| `--epochs` | `100` | 训练轮数 |
| `--num-negatives` | `1024` | 每步 memory-bank 负例数 |
| `--temperature` | `0.07` | NCE 温度 |
| `--loss-weight` | `0.5` | `x_t` loss 的权重 |
| `--memory-momentum` | `0.5` | memory EMA 系数 |
| `--optimizer` | `adamw` | 可选 `adamw`、`adam`、`sgd` |
| `--learning-rate` | `1e-3` | 初始学习率 |
| `--weight-decay` | `1e-4` | 权重衰减 |
| `--scheduler` | `cosine` | 可选 `cosine`、`step`、`none` |
| `--warmup-epochs` | `0` | 当前仅影响 cosine 的退火周期，不实现独立 warmup |
| `--seed` | `2026` | Python、NumPy、PyTorch 及 Token 增强的随机种子 |

最佳 checkpoint 为 `best_model.pt`，依据 validation contrastive loss 选择；`last_model.pt` 是最终 epoch 模型。

### 4. Retrieval embedding

检索阶段不使用随机增强、projection head 或 memory bank。每个 query/candidate Token 经过同一个 encoder 得到单位长度 embedding $\mathbf{e}$，并与 proposal 阶段保存的面积、质心、PCA 和角度统计字段组成 learned Token record。

形状相似度为：

$$
A_{\mathrm{shape}}(i,j)=\max(0,\mathbf{e}_i^\top\mathbf{e}_j).
$$

### 5. Token scoring and map matching

对每个 Query--Candidate 图对，枚举两侧 Token 对。位置亲和度为：

$$
A_{\mathrm{pos}}=\exp\left(-\frac{\|\mathbf{p}_i-\mathbf{p}_j\|_2^2}{\sigma_{\mathrm{pos}}^2}\right),
$$

其中 $\mathbf{p}$ 是按 map 高、宽归一化的质心。尺度亲和度由有效区域面积比例与 PCA 长短轴尺度共同给出：

$$
A_{\mathrm{scale}}=0.3A_{\mathrm{area}}+0.7A_{\mathrm{pca}}.
$$

默认总分为：

$$
s_{ij}=0.60A_{\mathrm{shape}}+0.25A_{\mathrm{pos}}+0.15A_{\mathrm{scale}}.
$$

Token 对必须通过以下硬门槛：两侧 Token 均不小于各自图中最大 Token 面积的10%；形状相似度至少为0.30；原始像素面积比、PCA 主轴比和次轴比均至少为0.20；综合分数至少为0.30。尺度比使用 $\min(a,b)/\max(a,b)$，因此 `0.20` 约允许最多5倍尺度差异。外缘且具有明显角度跨度的 Token 还会因角度覆盖不一致而衰减形状相似度。

| Matching 参数 | 默认值 | 含义 |
|---|---:|---|
| `--sigma-pos` | `0.35` | 位置偏移容忍尺度 |
| `--sigma-scale` | `1.5` | 面积与 PCA 尺度容忍尺度 |
| `--min-token-score` | `0.30` | 综合 Token 分数门槛 |
| `--min-relative-token-area` | `0.10` | 相对本图最大 Token 面积门槛 |
| `--scale-ratio-min` | `0.20` | 面积、主轴、次轴比例硬门槛 |
| `--score-shape-weight` | `0.60` | 形状项权重 |
| `--score-position-weight` | `0.25` | 位置项权重 |
| `--score-scale-weight` | `0.15` | 尺度项权重 |
| `--scale-area-weight` | `0.30` | 尺度项内的面积权重 |
| `--scale-pca-weight` | `0.70` | 尺度项内的 PCA 权重 |

通过门槛的 Token 对按分数降序执行贪心一对一匹配。一个 Query Token 或 Candidate Token 一旦被选中，便不能再次参与匹配。最终整图分数为所有有效 Query Token 的面积平方根加权平均；未匹配 Token 记为0：

$$
S(Q,C)=\frac{\sum_i\sqrt{\max(a_i,1)}\,\hat{s}_i}
{\sum_i\sqrt{\max(a_i,1)}}.
$$

## Outputs

预训练输出目录包含：

```text
configs.json
proposal_config.json
patch_config.json
train_proposal_tokens.csv
valid_proposal_tokens.csv
train_cluster_patches_manifest.csv
valid_cluster_patches_manifest.csv
best_model.pt
last_model.pt
best_memory.pt
last_memory.pt
history.json
main.log
```

Experiment A 检索输出目录包含：

```text
configs.json
proposal_config.json
patch_config.json
proposal_tokens.csv
tokens.csv
embeddings.npz
rankings.csv
label_metrics.json
label_metrics_flat.csv
```

当前检索流程不输出 `match_details.csv`。若需要逐 Token 匹配解释，应基于 `explain_map_similarity` 单独序列化匹配结果。

## Fair Comparison Requirements

与 `partial_match` 对比时，应固定 query manifest、candidate manifest、proposal 参数、Token patch 配置和匹配参数；否则结果应视为 ablation。尤其不能让 learned patch 或 encoder 结果改变 proposal Token 的像素、面积、质心、边界框或候选集合。

训练只能使用 train/valid split；不得根据 test label 选择 checkpoint、调整超参数或修改排序。`label_metrics.json` 和 `label_metrics_flat.csv` 由 `evaluation/` 在排序完成后生成，是正式指标文件。
