# 指定固定 die 网格范围（推荐，跨 KLARF 可比）
python3 -m match.main \
  --klarf test.v1.8.klarf \
  --wbm target.png \
  --die-x-range -20 20 \
  --die-y-range -20 20

# 不指定（兼容旧行为，从缺陷数据自动推算）
python3 -m match.main --klarf test.v1.8.klarf --wbm target.png