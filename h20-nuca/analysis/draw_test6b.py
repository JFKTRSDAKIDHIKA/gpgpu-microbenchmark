"""Plot the H20 latency-vs-working-set curve from the gpgpu-microbenchmark test6 run."""
import math
from PIL import Image, ImageDraw, ImageFont

DATA = [
    (0.0005, 34.9, 17.6), (0.001, 33.4, 16.9), (0.002, 32.7, 16.5),
    (0.0039, 32.3, 16.3), (0.0078, 32.2, 16.3), (0.0156, 32.1, 16.2),
    (0.0312, 32.0, 16.2), (0.0625, 32.0, 16.2), (0.125, 32.0, 16.2),
    (0.25, 153.9, 77.7), (0.5, 275.3, 139.0), (1, 275.1, 138.9),
    (2, 275.3, 139.0), (4, 275.3, 139.0), (8, 275.3, 139.0),
    (16, 275.3, 139.1), (32, 420.8, 212.5), (64, 648.3, 327.4),
    (128, 684.7, 345.8), (256, 685.0, 345.9), (512, 685.1, 346.0),
    (1024, 680.2, 343.5),
]

W, H = 1680, 1010
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
FONT = "C:/Windows/Fonts/msyh.ttc"
FONTB = "C:/Windows/Fonts/msyhbd.ttc"


def f(sz, bold=False):
    try:
        return ImageFont.truetype(FONTB if bold else FONT, sz)
    except Exception:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", sz)


INK, GRAY = (25, 25, 30), (110, 110, 120)
BLUE, ORANGE, RED, GREEN = (52, 110, 200), (230, 140, 40), (200, 70, 70), (40, 140, 80)

d.text((60, 30), "H20 L2 容量扫描：gpgpu-microbenchmark / hopper-h20 / test6（本地 load 延迟）",
       font=f(34, True), fill=INK)
d.text((60, 78), "指针追逐环，stride = 1 cache line，每个尺寸 50 轮  ·  H20 @ 1.98 GHz  ·  natural-selection 单卡",
       font=f(19), fill=GRAY)

PX, PY, PW, PH = 140, 170, 1180, 600
YMAX = 750.0

d.line([(PX, PY + PH), (PX + PW, PY + PH)], fill=INK, width=2)
d.line([(PX, PY), (PX, PY + PH)], fill=INK, width=2)
d.line([(PX + PW, PY), (PX + PW, PY + PH)], fill=INK, width=2)
d.text((PX - 96, PY - 26), "cyc / load", font=f(17), fill=GRAY)
d.text((PX + PW + 8, PY - 26), "ns / load", font=f(17), fill=GRAY)

for cyc in range(0, 751, 150):
    y = PY + PH - cyc / YMAX * PH
    d.line([(PX, y), (PX + PW, y)], fill=(236, 236, 242), width=1)
    d.text((PX - 62, y - 10), f"{cyc}", font=f(17), fill=GRAY)
    d.text((PX + PW + 10, y - 10), f"{cyc/1.98:.0f}", font=f(17), fill=GRAY)

lmin, lmax = math.log2(0.0005), math.log2(1024)


def X(mb):
    return PX + (math.log2(mb) - lmin) / (lmax - lmin) * PW


def Y(cyc):
    return PY + PH - cyc / YMAX * PH


for mb, lab in [(0.0005, "512B"), (0.0625, "64K"), (1, "1M"), (16, "16M"),
                (64, "64M"), (512, "512M"), (1024, "1G")]:
    x = X(mb)
    d.line([(x, PY + PH), (x, PY + PH + 8)], fill=INK, width=2)
    tw = d.textlength(lab, font=f(17))
    d.text((x - tw / 2, PY + PH + 14), lab, font=f(17), fill=GRAY)

# L2 capacity marker
xcap = X(60)
for yy in range(int(PY), int(PY + PH), 14):
    d.line([(xcap, yy), (xcap, yy + 7)], fill=ORANGE, width=2)
d.text((xcap + 8, PY + PH - 26), "L2 容量 ≈ 60 MB", font=f(17, True), fill=ORANGE)

pts = [(X(mb), Y(cyc)) for mb, cyc, ns in DATA]
d.line(pts, fill=BLUE, width=3, joint="curve")
for x, y in pts:
    d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=BLUE)

# plateau annotations
d.line([(X(0.5), Y(275.3)), (X(16), Y(275.3))], fill=GREEN, width=4)
d.text((X(0.62), Y(275.3) - 78), "L2：275 cyc / 139 ns", font=f(22, True), fill=GREEN)
d.text((X(0.62), Y(275.3) - 50), "0.5 – 16 MB 完全平坦（无第二级台阶）", font=f(17), fill=GREEN)

d.text((X(0.00055), Y(95)), "L1：32 cyc / 16 ns", font=f(20, True), fill=(120, 120, 130))

d.text((X(230), Y(726)), "HBM：680 cyc / 344 ns", font=f(21, True), fill=RED)
d.text((X(230), Y(726) + 28), "≥128 MB 平台", font=f(17), fill=RED)

d.text((X(26), Y(470)), "32 MB：421 cyc", font=f(17), fill=ORANGE)
d.text((X(66), Y(600)), "64 MB：648 cyc", font=f(17), fill=ORANGE)

d.text((60, H - 78),
       "对照自测（l2lat_pred / chase）：L2 命中 280–295 cyc、DRAM 658–710 cyc —— 与本套件 275 / 680 一致。",
       font=f(18), fill=GRAY)
d.text((60, H - 46),
       "注意：本套件每个尺寸只输出平均值，无法分辨同一尺寸内不同 L2 切片的近/远差异；NUCA 需要逐行采样。",
       font=f(18), fill=GRAY)

img.save("h20_test6_latency.png")
print("saved", img.size)
