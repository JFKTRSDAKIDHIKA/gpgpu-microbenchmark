"""Two-level topology figure: 2-way split (clean) + 4-way split (weaker)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

T = np.load("nuca/table4.npy")
qm = {}
with open("nuca/quadrants.csv") as f:
    next(f)
    for line in f:
        s, g = line.strip().split(",")
        qm[int(s)] = int(g)

GHZ = 1.98
W, H = 1720, 980
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
F = "C:/Windows/Fonts/msyh.ttc"; FB = "C:/Windows/Fonts/msyhbd.ttc"


def f(sz, bold=False):
    try:
        return ImageFont.truetype(FB if bold else F, sz)
    except Exception:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", sz)


INK, GRAY = (25, 25, 30), (110, 110, 120)
COLS = [(52, 110, 200), (230, 140, 40), (60, 150, 90), (190, 70, 80)]

d.text((50, 26), "H20 L2 两级拓扑测绘：78 SM × 1024 地址（2 MiB 对齐）",
       font=f(32, True), fill=INK)
d.text((50, 74), "每格 64 次同地址依赖链求均值 · 单卡 natural-selection · 1.98 GHz · 矩阵 289.9 cyc 均值",
       font=f(18), fill=GRAY)

# ---- panel A: 4x4 heatmap
AX, AY, CW, CH = 120, 200, 150, 110
d.text((AX, AY - 46), "A. 地址组 × SM 组（cyc）", font=f(21, True), fill=INK)
lo, hi = T.min(), T.max()


def shade(v):
    t = (v - lo) / (hi - lo)
    return (int(240 - 70 * t), int(242 - 150 * t), int(248 - 170 * t))


for i in range(4):
    for j in range(4):
        x0 = AX + j * CW; y0 = AY + i * CH
        d.rectangle([x0, y0, x0 + CW - 5, y0 + CH - 5], fill=shade(T[i, j]))
        txt = f"{T[i,j]:.0f}"
        tw = d.textlength(txt, font=f(24, True))
        d.text((x0 + (CW - tw) / 2, y0 + 34), txt, font=f(24, True), fill=INK)
for j in range(4):
    d.text((AX + j * CW + 46, AY - 24), f"SM{j}", font=f(17, True), fill=COLS[j])
for i in range(4):
    d.text((AX - 44, AY + i * CH + 38), f"A{i}", font=f(17, True), fill=GRAY)

# ---- panel B: SM id strip
BX, BY, BW = 830, 250, 800
d.text((BX, BY - 60), "B. 78 个 SM 的分组（按逻辑 id 排列）", font=f(21, True), fill=INK)
for s in range(78):
    x = BX + s * (BW / 78.0)
    g = qm.get(s, 0)
    d.rectangle([x, BY, x + BW / 78.0 - 1, BY + 34], fill=COLS[g])
d.text((BX, BY + 42), "SM id 0", font=f(15), fill=GRAY)
d.text((BX + BW - 60, BY + 42), "77", font=f(15), fill=GRAY)
for g in range(4):
    n = sum(1 for v in qm.values() if v == g)
    d.rectangle([BX + g * 195, BY + 76, BX + g * 195 + 16, BY + 92], fill=COLS[g])
    d.text((BX + g * 195 + 24, BY + 74), f"组{g}：{n} 个", font=f(17), fill=COLS[g])

# ---- panel C: level-1 numbers
CX, CY = 830, 470
d.text((CX, CY - 40), "C. 一级划分（PC1，84.1% 方差）", font=f(21, True), fill=INK)
rows = [("地址组 + / SM 组 +", 306.5), ("地址组 + / SM 组 −", 276.0),
        ("地址组 − / SM 组 +", 275.3), ("地址组 − / SM 组 −", 304.9)]
for i, (lab, v) in enumerate(rows):
    y = CY + i * 40
    d.text((CX, y + 6), lab, font=f(17), fill=INK)
    x0 = CX + 250
    d.rectangle([x0, y, x0 + (v - 260) * 4, y + 26], fill=COLS[0] if v < 290 else COLS[3])
    d.text((x0 + (v - 260) * 4 + 10, y + 2), f"{v:.0f} cyc / {v/GHZ:.0f} ns", font=f(16, True),
           fill=COLS[0] if v < 290 else COLS[3])
d.text((CX, CY + 172), "对角 ≈ 276 cyc / 140 ns，反对角 ≈ 306 cyc / 155 ns，Δ = 30 cyc = 15.2 ns",
       font=f(17, True), fill=(200, 70, 70))

# ---- panel D: level-2 step
DX, DY = 830, 700
d.text((DX, DY - 36), "D. 二级台阶：每个 SM 组对自己最优地址组 vs 次优", font=f(19, True), fill=INK)
steps = []
for j in range(4):
    col = np.sort(T[:, j])
    steps.append((col[0], col[1], col[1] - col[0]))
for j, (b0, b1, st) in enumerate(steps):
    y = DY + j * 34
    d.text((DX, y + 4), f"SM{j}", font=f(16, True), fill=COLS[j])
    d.text((DX + 70, y + 4), f"最优 {b0:.0f} → 次优 {b1:.0f}", font=f(16), fill=INK)
    d.rectangle([DX + 300, y + 6, DX + 300 + st * 22, y + 24], fill=COLS[j])
    d.text((DX + 306 + st * 22, y + 4), f"+{st:.1f} cyc / {st/GHZ:.1f} ns", font=f(15, True), fill=COLS[j])

d.text((50, H - 40),
       "一级划分干净（30 cyc 交叉）；二级只有 3.5–6.7 cyc，且第 4 组为 16.4 cyc —— 能确认“半区内部还有台阶”，"
       "但不足以唯一指派到四个物理象限。",
       font=f(17), fill=GRAY)

img.save("nuca/nuca_topology.png")
print("saved", steps)
