"""Single-SM read bandwidth vs unroll factor (1 SM, 1 block, 1024 threads)."""
from PIL import Image, ImageDraw, ImageFont

DATA = [(8, 90.0), (16, 116.0), (32, 111.0)]
GHZ = 1.98
PEAK = 128.0          # NCU: 9.98 KB/cycle / 78 SMs

W, H = 1400, 880
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

d.text((60, 30), "单 SM 读带宽 vs 循环展开（1 SM · 1 block · 1024 threads · L2 驻留）",
       font=f(30, True), fill=INK)
d.text((60, 76), "H20 @ 1.98 GHz · ld.global.cg.v4.f32 · 每 SM 独立 64 KiB 窗口 · 每点 n=4000×unroll 次操作",
       font=f(17), fill=GRAY)

PX, PY, PW, PH = 130, 190, 780, 470
YMAX = 140.0
d.line([(PX, PY + PH), (PX + PW, PY + PH)], fill=INK, width=2)
d.line([(PX, PY), (PX, PY + PH)], fill=INK, width=2)
d.line([(PX + PW, PY), (PX + PW, PY + PH)], fill=INK, width=2)


def Y(v):
    return PY + PH - v / YMAX * PH


for v in range(0, 141, 20):
    d.line([(PX, Y(v)), (PX + PW, Y(v))], fill=(236, 236, 242), width=1)
    d.text((PX - 46, Y(v) - 9), str(v), font=f(15), fill=GRAY)
    d.text((PX + PW + 10, Y(v) - 9), f"{v*GHZ:.0f}", font=f(15), fill=GRAY)

d.text((PX - 96, PY - 26), "B/cycle", font=f(16), fill=GRAY)
d.text((PX + PW + 8, PY - 26), "GB/s", font=f(16), fill=GRAY)

# peak line
d.line([(PX, Y(PEAK)), (PX + PW, Y(PEAK))], fill=RED, width=2)
for x in range(PX, PX + PW, 16):
    d.line([(x, Y(PEAK)), (x + 8, Y(PEAK))], fill="white", width=2)
d.text((PX + PW - 250, Y(PEAK) - 30), "NCU 标称 128 B/cycle / SM", font=f(17, True), fill=RED)

bw = PW / len(DATA)
for i, (u, v) in enumerate(DATA):
    x0 = PX + i * bw + 55
    x1 = PX + i * bw + bw - 55
    col = GREEN if v == max(dv for _, dv in DATA) else BLUE
    d.rectangle([x0, Y(v), x1, PY + PH], fill=col)
    t = f"{v:.0f}"
    tw = d.textlength(t, font=f(30, True))
    d.text(((x0 + x1) / 2 - tw / 2, Y(v) - 46), t, font=f(30, True), fill=col)
    t2 = f"{v*GHZ:.0f} GB/s · {v/PEAK*100:.0f}% of peak"
    tw2 = d.textlength(t2, font=f(16))
    d.text(((x0 + x1) / 2 - tw2 / 2, PY + PH + 40), t2, font=f(16), fill=GRAY)
    d.text(((x0 + x1) / 2 - 40, PY + PH + 12), f"unroll {u}", font=f(18, True), fill=INK)

# annotations
best_u, best = max(DATA, key=lambda t: t[1])
d.text((60, H - 120),
       f"峰值出现在 unroll = {best_u}：116 B/cycle = 230 GB/s，达到 NCU 标称口宽的 91%。",
       font=f(19), fill=INK)
d.text((60, H - 86),
       "unroll 从 8 加倍到 16 带来 +29%（更多在途请求填满 MLP）；再翻到 32 反而略降 4%（39 个在途块接近饱和，额外寄存器压力/环路开销）。",
       font=f(17), fill=INK)
d.text((60, H - 52),
       "对照：512 threads + unroll 8 只有 47.6 B/cycle（94 GB/s）——1024 线程把单 SM 读口压到了 ~90% 峰值。",
       font=f(17), fill=GRAY)

img.save("nuca/sm_unroll.png")
print("saved")
