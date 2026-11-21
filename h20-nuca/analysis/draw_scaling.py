"""Aggregate read bandwidth vs number of active SMs (1 block/SM, 1024 threads)."""
from PIL import Image, ImageDraw, ImageFont

DATA = [(1, 116.0), (2, 228.0), (4, 455.0), (8, 588.0)]
GHZ = 1.98
B1 = 116.0

W, H = 1500, 900
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

d.text((60, 30), "单 SM 读带宽的逐级扩展（每 SM 1 block × 1024 threads，unroll 16）",
       font=f(30, True), fill=INK)
d.text((60, 76), "H20 @ 1.98 GHz · ld.global.cg.v4.f32 · 每 SM 独立 64 KiB L2 驻留窗口 · 所有 SM 读同一 L2 分区",
       font=f(17), fill=GRAY)

# ---------------- panel A: aggregate
AX, AY, AW, AH = 110, 200, 740, 470
YMAX = 1000.0
d.text((AX, AY - 44), "A. 聚合带宽（B/cycle）", font=f(21, True), fill=INK)
d.line([(AX, AY + AH), (AX + AW, AY + AH)], fill=INK, width=2)
d.line([(AX, AY), (AX, AY + AH)], fill=INK, width=2)


def Y(v):
    return AY + AH - v / YMAX * AH


for v in range(0, 1001, 250):
    d.line([(AX, Y(v)), (AX + AW, Y(v))], fill=(236, 236, 242), width=1)
    d.text((AX - 52, Y(v) - 9), str(v), font=f(15), fill=GRAY)

bw = AW / len(DATA)
# ideal line
d.line([(AX + bw * 0.5, Y(B1)), (AX + bw * 3.5, Y(B1 * 8))], fill=GRAY, width=2)
d.text((AX + bw * 2.55, Y(B1 * 8) - 34), "线性理想 116×N", font=f(16), fill=GRAY)

for i, (n, v) in enumerate(DATA):
    x0 = AX + i * bw + 42
    x1 = AX + (i + 1) * bw - 42
    col = RED if n == 8 else BLUE
    d.rectangle([x0, Y(v), x1, AY + AH], fill=col)
    t = f"{v:.0f}"
    tw = d.textlength(t, font=f(24, True))
    d.text(((x0 + x1) / 2 - tw / 2, Y(v) - 36), t, font=f(24, True), fill=col)
    d.text(((x0 + x1) / 2 - 24, AY + AH + 10), f"{n} SM", font=f(17, True), fill=INK)

d.text((AX + bw * 3.0 - 60, Y(588) - 96), "4→8 只涨 1.29×", font=f(17, True), fill=RED)

# ---------------- panel B: per SM
BX, BY, BW, BH = 940, 200, 460, 470
YMAX2 = 130.0
d.text((BX, BY - 44), "B. 每 SM 带宽（B/cycle）", font=f(21, True), fill=INK)
d.line([(BX, BY + BH), (BX + BW, BY + BH)], fill=INK, width=2)
d.line([(BX, BY), (BX, BY + BH)], fill=INK, width=2)


def Y2(v):
    return BY + BH - v / YMAX2 * BH


for v in range(0, 131, 30):
    d.line([(BX, Y2(v)), (BX + BW, Y2(v))], fill=(236, 236, 242), width=1)
    d.text((BX - 42, Y2(v) - 9), str(v), font=f(15), fill=GRAY)

d.line([(BX, Y2(116)), (BX + BW, Y2(116))], fill=GREEN, width=2)


bw2 = BW / len(DATA)
for i, (n, v) in enumerate(DATA):
    per = v / n
    x0 = BX + i * bw2 + 22
    x1 = BX + (i + 1) * bw2 - 22
    col = RED if n == 8 else BLUE
    d.rectangle([x0, Y2(per), x1, BY + BH], fill=col)
    t = f"{per:.0f}"
    tw = d.textlength(t, font=f(20, True))
    d.text(((x0 + x1) / 2 - tw / 2, Y2(per) - 30), t, font=f(20, True), fill=col)
    d.text(((x0 + x1) / 2 - 20, BY + BH + 10), f"{n}", font=f(16, True), fill=INK)

# ---------------- text block
d.text((60, H - 156), "逐级效率（相对 116×N 的理想线性）：", font=f(18, True), fill=INK)
rows = []
for n, v in DATA:
    rows.append(f"{n} SM: {v:.0f} B/cycle = {v*GHZ:.0f} GB/s，效率 {v/(B1*n)*100:.0f}%")
d.text((60, H - 126), "   ".join(rows[:2]), font=f(17), fill=INK)
d.text((60, H - 98), "   ".join(rows[2:]), font=f(17), fill=INK)
d.text((60, H - 60),
       "4 SM 之前几乎完美线性（98%）；4→8 掉到 73.5 B/cycle/SM（63%），说明某个共享汇聚口在 455–588 B/cycle 之间饱和。",
       font=f(18, True), fill=RED)

img.save("nuca/sm_scaling.png")
print("saved")
