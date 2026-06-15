import html
import os
import zipfile
from datetime import datetime, timezone


OUT = "WBM_WDM_combinatorial_solution.pptx"

W = 12192000
H = 6858000

COLORS = {
    "bg": "F8FAFC",
    "ink": "111827",
    "muted": "64748B",
    "blue": "2563EB",
    "blue2": "DBEAFE",
    "green": "059669",
    "green2": "D1FAE5",
    "orange": "EA580C",
    "orange2": "FFEDD5",
    "purple": "7C3AED",
    "purple2": "EDE9FE",
    "red": "DC2626",
    "red2": "FEE2E2",
    "gray": "E5E7EB",
    "white": "FFFFFF",
}


def esc(text):
    return html.escape(str(text), quote=True)


def fill_xml(color):
    if not color:
        return "<a:noFill/>"
    return f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'


def line_xml(color="CBD5E1", width=12000):
    if width <= 0:
        return "<a:ln><a:noFill/></a:ln>"
    return f'<a:ln w="{width}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln>'


def text_runs(text, size=1800, color="111827", bold=False):
    bold_attr = ' b="1"' if bold else ""
    return f'<a:r><a:rPr lang="zh-CN" sz="{size}"{bold_attr}><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr><a:t>{esc(text)}</a:t></a:r>'


def text_box(sid, x, y, cx, cy, text, size=1800, color="111827", bold=False, align="l"):
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{sid}" name="Text {sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/>{line_xml(color, 0)}</p:spPr>
      <p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0"/><a:lstStyle/><a:p><a:pPr algn="{align}"/>{text_runs(text, size, color, bold)}</a:p></p:txBody>
    </p:sp>
    """


def bullet_box(sid, x, y, cx, cy, bullets, size=1700, color="334155"):
    paras = []
    for b in bullets:
        paras.append(
            f'<a:p><a:pPr marL="260000" indent="-180000"><a:buChar char="•"/>'
            f'<a:defRPr sz="{size}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:defRPr></a:pPr>'
            f'{text_runs(b, size, color)}</a:p>'
        )
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{sid}" name="Bullets {sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>
      <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{''.join(paras)}</p:txBody>
    </p:sp>
    """


def shape(sid, x, y, cx, cy, text="", fill="FFFFFF", border="CBD5E1", geom="roundRect", size=1600, color="111827", bold=False):
    body = f'<a:p><a:pPr algn="ctr"/>{text_runs(text, size, color, bold)}</a:p>' if text else '<a:p/>'
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{sid}" name="Shape {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>{fill_xml(fill)}{line_xml(border, 14000)}</p:spPr>
      <p:txBody><a:bodyPr anchor="mid" wrap="square" lIns="90000" rIns="90000"/><a:lstStyle/>{body}</p:txBody>
    </p:sp>
    """


def connector(sid, x1, y1, x2, y2, color="94A3B8", width=18000):
    x = min(x1, x2)
    y = min(y1, y2)
    cx = abs(x2 - x1) or 1
    cy = abs(y2 - y1) or 1
    flip_h = ' flipH="1"' if x2 < x1 else ""
    flip_v = ' flipV="1"' if y2 < y1 else ""
    return f"""
    <p:cxnSp>
      <p:nvCxnSpPr><p:cNvPr id="{sid}" name="Connector {sid}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
      <p:spPr><a:xfrm{flip_h}{flip_v}><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="line"><a:avLst/></a:prstGeom><a:ln w="{width}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:tailEnd type="triangle"/></a:ln></p:spPr>
    </p:cxnSp>
    """


def wafer(sid, x, y, size, title, hot=None, fill="F8FAFC", accent="2563EB"):
    hot = set(hot or [])
    parts = [shape(sid, x, y, size, size, "", fill, "CBD5E1", "ellipse")]
    grid = 7
    cell = size // 10
    start_x = x + size // 2 - (grid * cell) // 2
    start_y = y + size // 2 - (grid * cell) // 2
    n = sid + 1
    for r in range(grid):
        for c in range(grid):
            dx = c - 3
            dy = r - 3
            if dx * dx + dy * dy > 11:
                continue
            color = accent if (r, c) in hot else "E2E8F0"
            border = "FFFFFF" if (r, c) in hot else "CBD5E1"
            parts.append(shape(n, start_x + c * cell, start_y + r * cell, cell - 9000, cell - 9000, "", color, border, "rect"))
            n += 1
    parts.append(text_box(n, x, y + size + 80000, size, 280000, title, 1350, COLORS["muted"], True, "ctr"))
    return "".join(parts), n + 1


def title(slide_title, subtitle=None):
    parts = [
        shape(1000, 0, 0, W, H, "", COLORS["bg"], COLORS["bg"], "rect"),
        text_box(1001, 620000, 300000, 8700000, 520000, slide_title, 3000, COLORS["ink"], True),
    ]
    if subtitle:
        parts.append(text_box(1002, 640000, 820000, 9300000, 320000, subtitle, 1450, COLORS["muted"]))
    return "".join(parts)


def footer(n):
    return text_box(9900 + n, 620000, 6450000, 5200000, 260000, "WBM-WDM 组合匹配方案 | 思路版", 1050, "94A3B8")


def slide_cover():
    parts = [shape(1, 0, 0, W, H, "", "0F172A", "0F172A", "rect")]
    parts.append(shape(2, 7600000, -700000, 4300000, 4300000, "", "1D4ED8", "1D4ED8", "ellipse"))
    parts.append(shape(3, 8550000, 3150000, 2800000, 2800000, "", "059669", "059669", "ellipse"))
    parts.append(text_box(4, 760000, 720000, 8700000, 760000, "WBM-WDM 组合匹配方案", 3600, "FFFFFF", True))
    parts.append(text_box(5, 800000, 1500000, 9000000, 500000, "面向 WBM 失效图的 WDM 组合溯源与可解释匹配框架", 1850, "CBD5E1"))
    w1, sid = wafer(10, 800000, 2600000, 1550000, "目标 WBM", {(1, 2), (2, 2), (2, 3), (3, 4), (4, 4)}, "F8FAFC", "F97316")
    w2, sid = wafer(sid, 3400000, 2600000, 1550000, "候选 WDM 序列", {(2, 2), (3, 4)}, "F8FAFC", "22C55E")
    w3, sid = wafer(sid, 6000000, 2600000, 1550000, "组合叠加结果", {(1, 2), (2, 2), (2, 3), (3, 4), (4, 4)}, "F8FAFC", "3B82F6")
    parts.extend([w1, connector(sid, 2550000, 3350000, 3300000, 3350000, "CBD5E1"), w2, connector(sid + 1, 5150000, 3350000, 5900000, 3350000, "CBD5E1"), w3])
    return "".join(parts)


def slide_problem():
    parts = [title("1. 问题定义与业务目标", "针对一张 WBM 失效分布图，在候选 WDM 序列中识别能够共同解释该失效分布的缺陷图组合。")]
    w1, sid = wafer(10, 760000, 1800000, 1450000, "WBM：结果/现象", {(1, 2), (2, 2), (2, 3), (3, 4), (4, 4)}, accent="F97316")
    parts.append(w1)
    parts.append(shape(sid, 3000000, 1850000, 1900000, 620000, "WDM 1", COLORS["green2"], COLORS["green"], size=1550, bold=True))
    parts.append(shape(sid + 1, 3000000, 2780000, 1900000, 620000, "WDM 2", COLORS["blue2"], COLORS["blue"], size=1550, bold=True))
    parts.append(shape(sid + 2, 3000000, 3710000, 1900000, 620000, "WDM 3 ... N", COLORS["purple2"], COLORS["purple"], size=1550, bold=True))
    parts.append(connector(sid + 3, 2300000, 3000000, 2920000, 3000000))
    parts.append(text_box(sid + 4, 5600000, 1700000, 4400000, 420000, "核心问题", 2300, COLORS["ink"], True))
    parts.append(bullet_box(sid + 5, 5600000, 2250000, 4750000, 2200000, ["不依赖预先定义的固定 pattern 类别", "不将匹配目标简化为单张 WDM 检索", "从候选序列中选择若干关键 WDM", "通过组合叠加解释 WBM 的空间失效分布"], 1750))
    parts.append(shape(sid + 6, 5600000, 4800000, 4600000, 600000, "输出：WDM 选择向量与 Top 组合", COLORS["orange2"], COLORS["orange"], size=1700, bold=True))
    parts.append(footer(1))
    return "".join(parts)


def slide_shift():
    parts = [title("2. 建模思路转变", "从固定类别识别转向组合优化建模，以候选 WDM 组合对 WBM 的解释能力作为排序依据。")]
    parts.append(shape(10, 800000, 1700000, 4500000, 3650000, "", "FFFFFF", "E2E8F0", "roundRect"))
    parts.append(text_box(11, 1100000, 1980000, 3900000, 360000, "传统建模倾向", 2100, COLORS["red"], True, "ctr"))
    parts.append(bullet_box(12, 1150000, 2550000, 3650000, 1700000, ["预先定义 pattern 类型体系", "基于类别或表征进行判别", "匹配结果依赖类别一致性"], 1650))
    parts.append(shape(13, 6400000, 1700000, 4500000, 3650000, "", "FFFFFF", "E2E8F0", "roundRect"))
    parts.append(text_box(14, 6700000, 1980000, 3900000, 360000, "当前方案主线", 2100, COLORS["green"], True, "ctr"))
    parts.append(bullet_box(15, 6750000, 2550000, 3650000, 1700000, ["统一 WDM 与 WBM 的空间表达", "选择多个 WDM 构造组合叠图", "以叠图对 WBM 的解释度进行排序"], 1650))
    parts.append(connector(16, 5350000, 3500000, 6300000, 3500000, COLORS["blue"], 24000))
    parts.append(shape(17, 5030000, 3220000, 660000, 520000, "升级", COLORS["blue2"], COLORS["blue"], size=1350, bold=True))
    parts.append(footer(2))
    return "".join(parts)


def slide_flow():
    parts = [title("3. 方案流程", "围绕空间统一、候选压缩、组合搜索和可视化审核，形成端到端的组合匹配流程。")]
    xs = [700000, 3000000, 5300000, 7600000, 9900000]
    labels = ["输入\nWBM + WDM序列", "空间统一\n映射到同一网格", "候选压缩\n保留高相关项", "组合搜索\n叠加与评估", "结果输出\nTop组合与叠图"]
    fills = [COLORS["orange2"], COLORS["blue2"], COLORS["green2"], COLORS["purple2"], COLORS["red2"]]
    borders = [COLORS["orange"], COLORS["blue"], COLORS["green"], COLORS["purple"], COLORS["red"]]
    for i, x in enumerate(xs):
        parts.append(shape(20 + i, x, 2050000, 1600000, 1100000, labels[i], fills[i], borders[i], "roundRect", 1450, COLORS["ink"], True))
        if i < len(xs) - 1:
            parts.append(connector(40 + i, x + 1600000, 2600000, xs[i + 1] - 120000, 2600000))
    parts.append(shape(60, 900000, 4200000, 10100000, 900000, "关键原则：输出结果应具备可解释性、可复核性和工程可落地性，而不仅是单一黑盒评分。", "FFFFFF", "CBD5E1", "roundRect", 1800, COLORS["ink"], True))
    parts.append(footer(3))
    return "".join(parts)


def slide_overlay():
    parts = [title("4. 组合叠加的必要性", "单张 WDM 可能仅反映局部缺陷来源，多张 WDM 的组合更符合复杂失效分布的形成机制。")]
    w1, sid = wafer(10, 850000, 1800000, 1350000, "WDM A", {(1, 2), (2, 2)}, accent="22C55E")
    w2, sid = wafer(sid, 2850000, 1800000, 1350000, "WDM B", {(2, 3), (3, 4)}, accent="3B82F6")
    w3, sid = wafer(sid, 4850000, 1800000, 1350000, "WDM C", {(4, 4)}, accent="A855F7")
    w4, sid = wafer(sid, 7850000, 1700000, 1600000, "叠加图", {(1, 2), (2, 2), (2, 3), (3, 4), (4, 4)}, accent="F97316")
    parts.extend([w1, w2, w3, connector(sid, 6200000, 2500000, 7750000, 2500000, COLORS["orange"], 26000), w4])
    parts.append(text_box(sid + 1, 6500000, 2200000, 900000, 360000, "+", 3300, COLORS["orange"], True, "ctr"))
    parts.append(shape(sid + 2, 1200000, 4550000, 8800000, 760000, "方案强调选择少量关键 WDM，避免无关缺陷被过度叠加，从而提升结果的解释质量。", COLORS["orange2"], COLORS["orange"], "roundRect", 1650, COLORS["ink"], True))
    parts.append(footer(4))
    return "".join(parts)


def slide_search():
    parts = [title("5. 候选空间压缩与组合搜索", "采用先筛选、再组合的两阶段策略，在搜索效率、稳定性和解释性之间取得平衡。")]
    parts.append(shape(10, 900000, 1600000, 2500000, 3600000, "完整候选集\nN 张 WDM", COLORS["gray"], "94A3B8", "trapezoid", 1700, COLORS["ink"], True))
    parts.append(shape(11, 4200000, 2050000, 2500000, 2700000, "高相关候选\nTop-M", COLORS["blue2"], COLORS["blue"], "trapezoid", 1700, COLORS["ink"], True))
    parts.append(shape(12, 7500000, 2500000, 2500000, 1800000, "最优组合集\nTop-K", COLORS["green2"], COLORS["green"], "trapezoid", 1700, COLORS["ink"], True))
    parts.append(connector(13, 3400000, 3400000, 4100000, 3400000))
    parts.append(connector(14, 6700000, 3400000, 7400000, 3400000))
    parts.append(bullet_box(15, 900000, 5400000, 9400000, 700000, ["第一版建议：先对单张 WDM 与 WBM 的相关性进行排序，保留 Top-M；再在该集合内搜索少量高质量组合。"], 1600))
    parts.append(footer(5))
    return "".join(parts)


def slide_score():
    parts = [title("6. 组合质量评估原则", "评估目标同时关注空间相似性、解释完整性和组合复杂度，避免单纯追求覆盖面积。")]
    metrics = [("覆盖一致性", "组合叠图是否覆盖 WBM 的关键失效区域", COLORS["blue2"], COLORS["blue"]), ("空间分布一致性", "整体位置、形状与强度分布是否接近", COLORS["green2"], COLORS["green"]), ("组合复杂度约束", "控制 WDM 选择数量，减少无关项干扰", COLORS["orange2"], COLORS["orange"])]
    for i, (a, b, f, c) in enumerate(metrics):
        y = 1700000 + i * 1250000
        parts.append(shape(20 + i, 1100000, y, 2300000, 760000, a, f, c, "roundRect", 1800, COLORS["ink"], True))
        parts.append(text_box(40 + i, 3700000, y + 180000, 5600000, 360000, b, 1650, COLORS["muted"]))
    parts.append(shape(70, 1650000, 5450000, 8300000, 560000, "理想输出 = 高相似度 + 合理组合规模 + 可解释叠图", COLORS["purple2"], COLORS["purple"], "roundRect", 1800, COLORS["ink"], True))
    parts.append(footer(6))
    return "".join(parts)


def slide_output():
    parts = [title("7. 结果呈现方式", "输出内容应支持工程复核，包括候选组合、选择向量、相似度排序和叠图可视化对比。")]
    w1, sid = wafer(10, 800000, 1700000, 1450000, "目标 WBM", {(1, 2), (2, 2), (2, 3), (3, 4), (4, 4)}, accent="F97316")
    w2, sid = wafer(sid, 3500000, 1700000, 1450000, "最佳单张", {(2, 2), (3, 4)}, accent="3B82F6")
    w3, sid = wafer(sid, 6200000, 1700000, 1450000, "最佳组合", {(1, 2), (2, 2), (2, 3), (3, 4), (4, 4)}, accent="22C55E")
    parts.extend([w1, w2, w3])
    parts.append(shape(sid, 8850000, 1820000, 2300000, 550000, "Top-1: WDM 2+5+8", COLORS["green2"], COLORS["green"], size=1400, bold=True))
    parts.append(shape(sid + 1, 8850000, 2600000, 2300000, 550000, "Top-2: WDM 2+6+8", "FFFFFF", "CBD5E1", size=1400))
    parts.append(shape(sid + 2, 8850000, 3380000, 2300000, 550000, "Top-3: WDM 1+5+8", "FFFFFF", "CBD5E1", size=1400))
    parts.append(shape(sid + 3, 1400000, 5100000, 8500000, 620000, "工程审核重点：验证组合叠图与 WBM 失效区域的一致性，降低对单一数值评分的依赖。", COLORS["blue2"], COLORS["blue"], "roundRect", 1650, COLORS["ink"], True))
    parts.append(footer(7))
    return "".join(parts)


def slide_value():
    parts = [title("8. 方案价值与验证路径", "第一阶段目标是建立无标签条件下可解释、可验证、可迭代的组合匹配基线。")]
    left = ["将匹配问题转化为组合解释问题", "输出候选组合、选择向量与排序结果", "通过叠图可视化支撑人工复核", "为后续配准与几何鲁棒性优化预留空间"]
    right = ["组合方案相对最佳单张是否有稳定提升", "Top-3 结果的人工审核通过率", "WDM 选择数量分布是否合理", "不同样本与参数下结果是否稳定"]
    parts.append(shape(10, 1000000, 1650000, 4700000, 3600000, "", "FFFFFF", "E2E8F0", "roundRect"))
    parts.append(text_box(11, 1300000, 1950000, 4100000, 360000, "方案价值", 2100, COLORS["green"], True, "ctr"))
    parts.append(bullet_box(12, 1300000, 2500000, 3900000, 1850000, left, 1650))
    parts.append(shape(20, 6500000, 1650000, 4700000, 3600000, "", "FFFFFF", "E2E8F0", "roundRect"))
    parts.append(text_box(21, 6800000, 1950000, 4100000, 360000, "验证路径", 2100, COLORS["blue"], True, "ctr"))
    parts.append(bullet_box(22, 6800000, 2500000, 3900000, 1850000, right, 1650))
    parts.append(footer(8))
    return "".join(parts)


SLIDES = [slide_cover, slide_problem, slide_shift, slide_flow, slide_overlay, slide_search, slide_score, slide_output, slide_value]


def slide_xml(body):
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    {body}
  </p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''


def content_types(n):
    slides = "".join(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' for i in range(1, n + 1))
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
{slides}
</Types>'''


def presentation_xml(n):
    ids = "".join(f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, n + 1))
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{n + 1}"/></p:sldMasterIdLst>
<p:sldIdLst>{ids}</p:sldIdLst>
<p:sldSz cx="{W}" cy="{H}" type="wide"/><p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>'''


def presentation_rels(n):
    rels = [f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>' for i in range(1, n + 1)]
    rels.append(f'<Relationship Id="rId{n + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>')
    rels.append(f'<Relationship Id="rId{n + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>')
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{"".join(rels)}</Relationships>'


ROOT_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''

SLIDE_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>'''

MASTER_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" cy="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>'''

MASTER_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>'''

LAYOUT_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
<p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'''

LAYOUT_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>'''

THEME_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme"><a:themeElements><a:clrScheme name="Office"><a:dk1><a:srgbClr val="111827"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="334155"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="059669"/></a:accent2><a:accent3><a:srgbClr val="EA580C"/></a:accent3><a:accent4><a:srgbClr val="7C3AED"/></a:accent4><a:accent5><a:srgbClr val="DC2626"/></a:accent5><a:accent6><a:srgbClr val="0891B2"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHLink><a:srgbClr val="7C3AED"/></a:folHLink></a:clrScheme><a:fontScheme name="Office"><a:majorFont><a:latin typeface="Arial"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Arial"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Arial"/></a:minorFont></a:fontScheme><a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>'''


def write_pptx(path):
    n = len(SLIDES)
    now = datetime.now(timezone.utc).isoformat()
    core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>WBM-WDM Combinatorial Matching Solution</dc:title><dc:creator>OpenCode</dc:creator><cp:lastModifiedBy>OpenCode</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>'''
    app = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>OpenCode</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>{n}</Slides></Properties>'''
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(n))
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)
        z.writestr("ppt/presentation.xml", presentation_xml(n))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(n))
        z.writestr("ppt/slideMasters/slideMaster1.xml", MASTER_XML)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", MASTER_RELS)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", LAYOUT_XML)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", LAYOUT_RELS)
        z.writestr("ppt/theme/theme1.xml", THEME_XML)
        for i, fn in enumerate(SLIDES, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(fn()))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", SLIDE_RELS)


if __name__ == "__main__":
    output = os.path.abspath(OUT)
    write_pptx(output)
    print(output)
