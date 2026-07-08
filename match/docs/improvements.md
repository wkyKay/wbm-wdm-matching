# Local Matching Descriptor Improvement

## 原 shape descriptor 方案

原方案把每个 proposal token 表示为一组全局几何统计特征，包括 fill ratio、bbox aspect ratio、PCA elongation、compactness、orientation、angular coverage、radial standard deviation，以及粗粒度的 radial/angular pixel histogram。最终 descriptor 会做 L2 normalize，shape similarity 直接使用两个归一化向量的 dot product。

这个方案计算轻、解释性强，但实际观察中 shape 分数普遍偏高。原因是很多 wafer defect cluster 即使局部形状不同，也会共享相近的面积、bbox、compactness 和粗直方图统计；同时多数维度都是非负归一化比例或 histogram，cosine similarity 很容易保持在较高区间。因此单纯提高 hard threshold 只能过滤一部分误匹配，不能解决 descriptor 本身区分度不足的问题，还可能误杀真实 partial match。

## 改进后的 shape descriptor 方案

改进方案使用分组的 `Zernike + geometry` descriptor。对每个 proposal token，先根据 token pixels 裁剪局部 mask，padding 成正方形，resize 到固定 48x48，再映射到 unit disk，并计算 degree 8 以内的 Zernike moment magnitude 作为主要形状特征。Zernike magnitude 提供旋转不敏感的区域形状表达，比单纯全局统计更能描述 cluster mask 的空间结构。

同时保留一个较小权重的 geometry feature group，用于保存 fill ratio、aspect、elongation、compactness、angular coverage、radial standard deviation，以及在不启用 rotation tolerance 时的 orientation 信息。当前 shape similarity 不再对所有拼接特征做一次统一 cosine，而是分别计算 `moment_sim` 和 `geometry_sim`，再按 `0.75 / 0.25` 权重合成总 `shape_sim`。

## 改进原因

Zernike moments 是成熟的 binary region shape descriptor，更适合 proposal token 这种局部二值 cluster。它能编码 cluster 区域内部的空间分布，因此两个 token 即使面积、bbox ratio 或 compactness 接近，只要真实 mask 结构不同，shape 分数也更容易被拉开。保留低权重 geometry group 是为了在 proposal 有噪声时提供稳定的低频统计信号，而分组打分让权重更透明，也方便后续根据正负样本分布继续调参。

最终 match 的 score 公式和 hard gate 暂时保持不变，但 evidence table 和 review figure 会额外展示 `moment_sim`、`geometry_sim`，便于判断某个误匹配到底是 Zernike 主形状相似导致，还是 geometry 统计项拉高导致。

## token-pair score gate

在上述 descriptor 改进后，仍然需要对最终 token-pair score 增加一道门槛。原流程只要 `shape_sim` 超过 hard gate 且最终 score 大于 0，就会进入候选匹配对；这会导致 `0.0x` 级别的弱匹配也被 greedy one-to-one 选中，并以很小但不合理的分数计入最终 map score。

当前新增 `min_token_score`，默认值为 `0.10`。只有最终 token-pair score 达到该门槛的 pair 才会进入 `map_topk_matches`、`token_topk_matches` 和最终 greedy match。这样可以把 shape、position、scale 综合后仍然很弱的匹配直接排除，避免低质量 pair 稀释或污染最终匹配解释；测试脚本可通过 `--min-token-score` 调整该门槛。
