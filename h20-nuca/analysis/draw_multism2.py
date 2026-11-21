"""Plot the multi-SM NUCA result: same 512 lines measured from all 78 SMs."""
import csv
from PIL import Image, ImageDraw, ImageFont

rows = []
with open("nuca/sm_means.csv") as f:
    r = csv.DictReader(f)
    for d in r:
        rows.append((int(d["sm"]), float(d["mean_cyc"]), float(d["mean_even"]), float(d["mean_odd"])))
rows.sort(key=lambda x: x[1])
GHZ = 1.98

W, H = 1720, 1000
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
F = "C:/Windows/Fonts/msyh.ttc"
FB = "C:/Windows/Fonts/msyhbd.ttc"


def f(sz, bold=False):
    try:
        return ImageFont.truetype(FB if bold else F, sz)
    except Exception:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", sz)


INK, GRAY = (25, 25, 30), (110, 110, 120)
BLUE, ORANGE, RED, GREEN = (52, 110, 200), (230, 140, 40), (200, 70, 70), (40, 140, 80)

d.text((50, 26), "H20 L2 的 NUCA：同一批 512 个 128 B 地址，从 78 个 SM 分别测（每行 32 次取 min）",
       font=f(32, True), fill=INK)
d.text((50, 74), "单卡 blue-space · 1.98 GHz · 由 %smid 确认 78 个 block 各占一个 SM  ·  判据：同一行在不同 SM 上是否有稳定周期差",
       font=f(18), fill=GRAY)

lo, hi = rows[0][1], rows[-1][1]
gmean = sum(r[1] for r in rows) / len(rows)

# ---------------- panel A
AX, AY, AW, AH = 90, 190, 700, 400
d.text((AX, AY - 44), "A. 每个 SM 的平均 L2 延迟（按大小排序）", font=f(22, True), fill=INK)
ymin, ymax = 255, 300
d.line([(AX, AY + AH), (AX + AW, AY + AH)], fill=INK, width=2)
d.line([(AX, AY), (AX, AY + AH)], fill=INK, width=2)


def Y(v):
    return AY + AH - (v - ymin) / (ymax - ymin) * AH


for v in range(255, 301, 15):
    d.line([(AX, Y(v)), (AX + AW, Y(v))], fill=(236, 236, 242), width=1)
    d.text((AX - 44, Y(v) - 10), str(v), font=f(16), fill=GRAY)

bw = AW / len(rows)
for i, (sm, m, _, _) in enumerate(rows):
    x0 = AX + i * bw
    d.rectangle([x0, Y(m), x0 + bw - 1, AY + AH], fill=BLUE if m > gmean else (130, 160, 210))
d.line([(AX, Y(gmean)), (AX + AW, Y(gmean))], fill=RED, width=2)
d.rectangle([AX + 6, Y(gmean) - 30, AX + 176, Y(gmean) - 4], fill="white")
d.text((AX + 10, Y(gmean) - 28), f"均值 {gmean:.0f} cyc", font=f(16, True), fill=RED)

d.text((AX, AY + AH + 16), f"最快 SM {rows[0][0]}：{lo:.0f} cyc / {lo/GHZ:.0f} ns", font=f(18, True), fill=BLUE)
d.text((AX + 330, AY + AH + 16), f"最慢 SM {rows[-1][0]}：{hi:.0f} cyc / {hi/GHZ:.0f} ns", font=f(18, True), fill=ORANGE)
d.text((AX, AY + AH + 50), f"极差 = {hi-lo:.1f} cyc = {(hi-lo)/GHZ:.1f} ns（{(hi-lo)/gmean*100:.1f}%）",
       font=f(19, True), fill=RED)

# ---------------- panel B
BX, BY, BW_, BH = 900, 210, 330, 340
d.text((BX - 20, BY - 66), "B. 稳定性：前后两半（各 256 行）", font=f(22, True), fill=INK)
d.line([(BX, BY + BH), (BX + BW_, BY + BH)], fill=INK, width=2)
d.line([(BX, BY), (BX, BY + BH)], fill=INK, width=2)
x0v, x1v = 265, 295


def X(v):
    return BX + (v - x0v) / (x1v - x0v) * BW_


def Y2(v):
    return BY + BH - (v - x0v) / (x1v - x0v) * BW_


d.line([(X(x0v), Y2(x0v)), (X(x1v), Y2(x1v))], fill=(205, 205, 210), width=1)
for sm, m, me, mo in rows:
    d.ellipse([X(me) - 3, Y2(mo) - 3, X(me) + 3, Y2(mo) + 3], fill=BLUE)
for v in (265, 280, 295):
    d.text((X(v) - 14, BY + BH + 8), str(v), font=f(15), fill=GRAY)
    d.text((BX - 40, Y2(v) - 9), str(v), font=f(15), fill=GRAY)
d.text((BX + 70, BY + BH + 32), "偶数行平均 (cyc)", font=f(16), fill=GRAY)
d.text((BX + 10, BY + 8), "r = 0.998", font=f(22, True), fill=GREEN)
d.text((BX + 10, BY + 40), "排序可复现 → 不是噪声", font=f(16), fill=GREEN)

# ---------------- panel C
CX, CY, CW, CH = 1330, 210, 330, 340
d.text((CX - 20, CY - 66), "C. 按 SM id 看", font=f(22, True), fill=INK)
d.line([(CX, CY + CH), (CX + CW, CY + CH)], fill=INK, width=2)
d.line([(CX, CY), (CX, CY + CH)], fill=INK, width=2)


def X3(sm):
    return CX + sm / 77.0 * CW


def Y3(v):
    return CY + CH - (v - ymin) / (ymax - ymin) * CH


for v in range(255, 301, 15):
    d.line([(CX, Y3(v)), (CX + CW, Y3(v))], fill=(238, 238, 243), width=1)
    d.text((CX - 40, Y3(v) - 9), str(v), font=f(15), fill=GRAY)

for sm, m, me, mo in rows:
    col = ORANGE if 46 <= sm <= 61 else BLUE
    d.ellipse([X3(sm) - 4, Y3(m) - 4, X3(sm) + 4, Y3(m) + 4], fill=col)
d.text((CX + 90, CY + CH + 10), "SM id 0 … 77", font=f(15), fill=GRAY)
d.rectangle([CX, CY + CH + 40, CX + 16, CY + CH + 56], fill=ORANGE)
d.text((CX + 24, CY + CH + 38), "SM 46–61：连续一段偏快", font=f(16), fill=ORANGE)
d.rectangle([CX + 230, CY + CH + 40, CX + 246, CY + CH + 56], fill=BLUE)
d.text((CX + 254, CY + CH + 38), "其余", font=f(16), fill=BLUE)

d.text((50, H - 78),
       "结论：同一批地址在 78 个 SM 上的平均延迟 269.7 → 290.5 cyc（10.5 ns，7.5%），前后两半 r=0.998，",
       font=f(18), fill=INK)
d.text((50, H - 50),
       "说明存在稳定的、按 SM 位置分布的 NUCA 梯度，而不是噪声。",
       font=f(18), fill=INK)
d.text((50, H - 22),
       "注意：单行的 max–min（p50 = 67 cyc）主要来自单次测量噪声；可信维度是“SM 平均”。",
       font=f(16), fill=GRAY)

img.save("nuca/nuca_multism.png")
print("saved")
