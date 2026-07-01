
• 目前的相似度可以理解成：WBM 每个失效区域 token，去 WDM token 里找一个“最像、位置
  接近、大小接近、类型兼容”的解释，然后把这些最佳解释加权平均。

  核心在 match/core/local_matching.py:824。

  单个 token pair 的分数是：

  pair_score =
    shape_sim^2
    * position_affinity
    * scale_affinity
    * type_affinity

  但有一个硬门槛：

  if shape_sim < 0.45:
      pair_score = 0

  各项含义：

  shape_sim
  token 的形状 descriptor 点积。因为 descriptor 被归一化了，所以近似可以当作
  cosine similarity。它主要比较形态特征，比如填充率、长宽比、PCA 延展度、紧凑度、
  方向、角覆盖、径向厚度、径向/角向 profile。代码里还把它平方，所以 shape 是主导
  项。

  position_affinity
  位置亲和度。比较两个 token 归一化质心距离：

  exp(-pos_dist2 / sigma_pos^2)

  越近越接近 1，越远越接近 0。默认 sigma_pos=0.35。

  scale_affinity
  尺度亲和度。比较两个 token 占有效 wafer 面积的比例：

  exp(-abs(log(wbm_area_ratio / wdm_area_ratio)) / sigma_scale)

  大小越接近越接近 1。默认 sigma_scale=1.5。

  type_affinity
  几何类型兼容度：

  同类型 = 1.0
  line/irregular、blob/irregular、central/blob = 0.6
  其他 = 0.25

  整图分数在 match/core/local_matching.py:204：

  for each WBM token:
      找 WDM token 中 pair_score 最高的那个

  final_score =
      sum(best_pair_score * sqrt(wbm_token_area))
      / sum(sqrt(wbm_token_area))

  所以它不是“两张图的整体 embedding 相似度”，而是一个WBM token 到 WDM token 的局部
  解释分数。

  一个关键限制是：当前匹配是单向 greedy best-match。也就是说：

  - 每个 WBM token 都会找一个最佳 WDM token；
  - 同一个 WDM token 可以被多个 WBM token 重复匹配；
  - 多余的 WDM token 不直接扣分；
  - WBM token 没匹配上会因为分数低而拉低平均，但没有显式 missing penalty；
  - token 之间没有整体 assignment 约束。

  所以现在的分数更适合理解为：

  WDM 是否能解释 WBM 的局部失效模式

  而不是：

  WBM 和 WDM 两张图是否在严格意义上整体相似



  - fill_ratio: 这个区域在 bbox 里有多“实”
  - aspect: bbox 长宽比
  - elongation: PCA 主轴拉长程度
  - compactness: 边界是否紧凑
  - cos(orientation) / sin(orientation): 主方向
  - angular_coverage: 围绕 wafer 中心覆盖了多少角度
  - radial_std: 距离 wafer 中心的半径是否集中

   16 个 profile 特征来自 _shape_profiles(..., bins=8)：

  8 个 radial histogram + 8 个 angular histogram

  它把 token 内部的像素相对 token 质心做统计：

  - radial histogram: 像素离 token 质心的距离分布
  - angular histogram: 像素绕 token 质心的方向分布

  比如一个圆形 blob，radial 分布会比较均匀；一条线状缺陷，angular 分布会集中在少数
  方向 bin 上。

  最后这 24 个数会拼成一个向量，并做归一化：

  desc = desc / ||desc||

  所以两个 token 的 shape 相似度就是：

  dot(desc_a, desc_b)

  也就是归一化向量点积，近似等于 cosine similarity。