"""Draw the L2 NUCA latency figure from the measured data (H20, l2lat_pred + chase)."""
from PIL import Image, ImageDraw, ImageFont

W, H = 1680, 1200
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

FONT = "C:/Windows/Fonts/msyh.ttc"
FONTB = "C:/Windows/Fonts/msyhbd.ttc"


def f(sz, bold=False):
    try:
        return ImageFont.truetype(FONTB if bold else FONT, sz)
    except Exception:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", sz)


INK = (25, 25, 30)
GRAY = (110, 110, 120)
BLUE = (52, 110, 200)
ORANGE = (230, 140, 40)
RED = (200, 70, 70)

d.text((60, 32), "H20 L2 NUCA 验证：近端 / 远端分区与 DRAM 的延迟差异",
       font=f(36, True), fill=INK)
d.text((60, 82),
       "数据：l2lat_pred.cu（谓词时钟，60 MiB 逐行扫描，491520 行）+ chase.cu（指针追逐交叉验证）  ·  clock = 1.98 GHz",
       font=f(19), fill=GRAY)

# ---------------------------------------------------------------- panel A
AX, AY, AW, AH = 90, 200, 660, 320
bins = [(224, 255, 10095, "edge"), (256, 287, 253019, "near"),
        (288, 319, 227660, "far"), (320, 351, 746, "edge")]
maxc = max(b[2] for b in bins)

d.text((AX, AY - 56), "A. warm（1 MiB 分块预热，全部命中 L2）延迟分布", font=f(23, True), fill=INK)
d.line([(AX, AY + AH), (AX + AW, AY + AH)], fill=INK, width=2)
d.line([(AX, AY - 6), (AX, AY + AH)], fill=INK, width=2)
bw = AW / len(bins)
for i, (lo, hi, c, kind) in enumerate(bins):
    x0 = AX + i * bw + 16
    x1 = AX + (i + 1) * bw - 16
    h = AH * c / maxc
    col = {"near": BLUE, "far": ORANGE, "edge": (175, 175, 185)}[kind]
    d.rectangle([x0, AY + AH - h, x1, AY + AH], fill=col)
    txt = f"{c/491520*100:.1f}%"
    tw = d.textlength(txt, font=f(19, True))
    if h > 60:
        d.text(((x0 + x1) / 2 - tw / 2, AY + AH - h + 10), txt, font=f(19, True), fill="white")
    else:
        d.text(((x0 + x1) / 2 - tw / 2, AY + AH - 44), txt, font=f(19, True), fill=GRAY)
    d.text(((x0 + x1) / 2 - 50, AY + AH + 10), f"{lo}-{hi}", font=f(18), fill=GRAY)

ly = AY + AH + 48
d.rectangle([AX, ly, AX + 26, ly + 22], fill=BLUE)
d.text((AX + 36, ly - 2), "近端 ≈272 cyc / 137 ns", font=f(19, True), fill=BLUE)
d.rectangle([AX + 300, ly, AX + 326, ly + 22], fill=ORANGE)
d.text((AX + 336, ly - 2), "远端 ≈304 cyc / 153 ns", font=f(19, True), fill=ORANGE)
d.text((AX, ly + 38), "Δ = 32 cyc ≈ 16 ns（两簇 51.5% vs 46.3%，与 2 × 48 片分区一致）",
       font=f(19, True), fill=RED)

# ---------------------------------------------------------------- panel B
BX, BY = 850, 240
NAMEW, BARW = 200, 400
rows = [
    ("L2 近端分区", 272, 137, BLUE),
    ("L2 远端分区", 304, 153, ORANGE),
    ("L2 平均", 282, 142, (120, 150, 190)),
    ("60 MiB 逐行", 322, 163, (150, 150, 160)),
    ("DRAM（1 GiB）", 710, 359, RED),
]
d.text((BX, BY - 56), "B. 各层级实测延迟", font=f(23, True), fill=INK)
d.text((BX + NAMEW, BY - 28), "cycles", font=f(17), fill=GRAY)
scale = BARW / 780.0
for i, (name, cyc, ns, col) in enumerate(rows):
    y = BY + i * 62
    d.text((BX, y + 8), name, font=f(18), fill=INK)
    x0 = BX + NAMEW
    d.rectangle([x0, y, x0 + cyc * scale, y + 30], fill=col)
    d.text((x0 + BARW + 14, y + 2), f"{cyc} cyc / {ns} ns", font=f(18, True), fill=col)
d.line([(BX + NAMEW, BY - 10), (BX + NAMEW, BY + len(rows) * 62 - 8)], fill=GRAY, width=1)

# ---------------------------------------------------------------- panel C
SX, SY = 90, 700
d.text((SX, SY - 16), "C. 结构示意：96 个 L2 切片 = 2 个分区 × 48 片，78 个 SM 共享同一套 L2",
       font=f(23, True), fill=INK)


def partition(x, y, label, col):
    d.rectangle([x - 8, y - 28, x + 8 * 40 + 8, y + 6 * 28 + 8], outline=col, width=3)
    for r in range(6):
        for c in range(8):
            d.rectangle([x + c * 40, y + r * 28, x + c * 40 + 32, y + r * 28 + 22],
                        fill=col, outline=None)
    d.text((x, y - 54), label, font=f(20, True), fill=col)


partition(SX + 150, SY + 70, "L2 partition 0（48 片 / 30 MiB）", BLUE)
partition(SX + 700, SY + 70, "L2 partition 1（48 片 / 30 MiB）", ORANGE)

d.ellipse([SX + 16, SY + 100, SX + 116, SY + 200], outline=INK, width=3)
d.text((SX + 34, SY + 132), "SM\n×78", font=f(19, True), fill=INK)

d.line([(SX + 120, SY + 140), (SX + 146, SY + 140)], fill=BLUE, width=4)
d.text((SX + 150, SY + 258), "近端（同分区）：≈272 cyc / 137 ns", font=f(19, True), fill=BLUE)

d.line([(SX + 66, SY + 230), (SX + 66, SY + 302), (SX + 690, SY + 302), (SX + 690, SY + 250)],
       fill=ORANGE, width=4)
d.text((SX + 330, SY + 312), "远端（跨分区）：≈304 cyc / 153 ns", font=f(19, True), fill=ORANGE)

DY = SY + 370
d.rectangle([SX + 150, DY, SX + 1320, DY + 60], outline=RED, width=3)
d.text((SX + 172, DY + 14), "HBM3（60 MiB 之外）  冷数据 ≈710 cyc / 359 ns ；指针追逐 1 GiB = 658 cyc / 332 ns",
       font=f(20, True), fill=RED)
d.line([(SX + 66, SY + 302), (SX + 66, DY + 30), (SX + 146, DY + 30)], fill=RED, width=3)

d.text((90, H - 36),
       "注：两簇由 32 周期分箱给出、中心取箱中点；1 周期级确认需细分箱并固定 SM 归属。",
       font=f(17), fill=GRAY)

img.save("nuca_latency.png")
print("saved", img.size)
