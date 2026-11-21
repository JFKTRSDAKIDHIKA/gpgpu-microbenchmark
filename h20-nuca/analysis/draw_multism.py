"""Plot the multi-SM NUCA result: same 512 lines measured from all 78 SMs."""
import csv
from PIL import Image, ImageDraw, ImageFont

rows = []
with open("nuca/sm_means.csv") as f:
    r = csv.DictReader(f)
    for d in r:
        rows.append((int(d["sm"]), float(d["mean_cyc"]), float(d["mean_even"]), float(d["mean_odd"])))
rows.sort(key=lambda x: x[1])                      # by mean
GHZ = 1.98

W, H = 1720, 980
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

# ---------------- panel A: sorted per-SM mean
AX, AY, AW, AH = 90, 180, 700, 420
d.text((AX, AY - 40), "A. 每个 SM 的平均 L2 延迟（按大小排序）", font=f(22, True), fill=INK)
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
d.text((AX + AW - 210, Y(gmean) - 26), f"均值 {gmean:.0f} cyc", font=f(17, True), fill=RED)

d.text((AX, AY + AH + 16), f"最快 SM {rows[0][0]}：{lo:.0f} cyc / {lo/GHZ:.0f} ns", font=f(18, True), fill=BLUE)
d.text((AX + 330, AY + AH + 16), f"最慢 SM {rows[-1][0]}：{hi:.0f} cyc / {hi/GHZ:.0f} ns", font=f(18, True), fill=ORANGE)
d.text((AX, AY + AH + 48), f"极差 = {hi-lo:.1f} cyc = {(hi-lo)/GHZ:.1f} ns（{(hi-lo)/gmean*100:.1f}%）",
       font=f(19, True), fill=RED)

# ---------------- panel B: even vs odd reproducibility
BX, BY, BW_, BH = 880, 200, 360, 380
d.text((BX - 10, BY - 56), "B. 稳定性：前后两半（各 256 行）", font=f(22, True), fill=INK)
d.text((BX + 30, BY - 26), "奇数行平均 (cyc)", font=f(16), fill=GRAY)
d.line([(BX, BY + BH), (BX + BW_, BY + BH)], fill=INK, width=2)
d.line([(BX, BY), (BX, BY + BH)], fill=INK, width=2)
x0v, x1v = 265, 295


def X(v):
    return BX + (v - x0v) / (x1v - x0v) * BW_


def Y2(v):
    return BY + BH - (v - x0v) / (x1v - x0v) * BW_


d.line([(X(x0v), Y2(x0v)), (X(x1v), Y2(x1v))], fill=(200, 200, 205), width=1)
for sm, m, me, mo in rows:
    d.ellipse([X(me) - 3, Y2(mo) - 3, X(me) + 3, Y2(mo) + 3], fill=BLUE)
d.text((BX + 10, BY + 10), "r = 0.998", font=f(22, True), fill=GREEN)
d.text((BX + 10, BY + 42), "排序可复现 → 不是噪声", font=f(16), fill=GREEN)

# ---------------- panel C: per-SM by id
CX, CY, CW, CH = 1310, 200, 350, 380
d.text((CX - 10, CY - 56), "C. 按 SM id 看（谁离得近）", font=f(22, True), fill=INK)
d.line([(CX, CY + CH), (CX + CW, CY + CH)], fill=INK, width=2)
d.line([(CX, CY), (CX, CY + CH)], fill=INK, width=2)


def X3(sm):
    return CX + sm / 77.0 * CW


def Y3(v):
    return CY + CH - (v - ymin) / (ymax - ymin) * CH


for sm, m, me, mo in rows:
    col = ORANGE if 46 <= sm <= 61 else BLUE
    d.ellipse([X3(sm) - 4, Y3(m) - 4, X3(sm) + 4, Y3(m) + 4], fill=col)
d.text((CX + 4, CY + CH + 14), "SM id 0 … 77", font=f(16), fill=GRAY)
d.text((CX - 6, CY + 8), "橙点：SM 46–61\n连续偏快", font=f(16), fill=ORANGE)
d.text((CX - 6, CY + 70), "蓝点：其余", font=f(16), fill=BLUE)

d.text((50, H - 66),
       "结论：同一批地址在 78 个 SM 上的平均值从 269.7 到 290.5 周期（10.5 ns，7.5%），且前后两半 r=0.998，"
       "说明存在稳定的、按 SM 位置分布的 NUCA 梯度。",
       font=f(18), fill=INK)
d.text((50, H - 36),
       "注意：单行的 max-min（p50=67 cyc）主要来自单次测量噪声；可信的是“SM 平均”这个维度。",
       font=f(17), fill=GRAY)

img.save("nuca_multism.png")
print("saved", img.size)
