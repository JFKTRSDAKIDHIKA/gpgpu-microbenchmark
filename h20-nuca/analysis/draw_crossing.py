"""Draw the crossing (checkerboard) evidence."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sm, ln, cyc = [], [], []
with open("nuca/multism3.csv") as f:
    for line in f:
        a, b, c = line.strip().split(",")
        sm.append(int(a)); ln.append(int(b)); cyc.append(float(c))

sms = sorted(set(sm)); lines = sorted(set(ln))
S, L = len(sms), len(lines)
sidx = {s: i for i, s in enumerate(sms)}
lidx = {l: i for i, l in enumerate(lines)}
M = np.zeros((L, S))
for a, b, c in zip(sm, ln, cyc):
    M[lidx[b], sidx[a]] = c

GHZ = 1.98
R = M - M.mean(0, keepdims=True); R = R - R.mean()
U, s, Vt = np.linalg.svd(R, full_matrices=False)
u = U[:, 0]; v = Vt[0]
Agp = u >= 0
Bgp = v < 0
c = {(1, 0): M[np.ix_(Agp, Bgp)].mean(), (1, 1): M[np.ix_(Agp, ~Bgp)].mean(),
     (0, 0): M[np.ix_(~Agp, Bgp)].mean(), (0, 1): M[np.ix_(~Agp, ~Bgp)].mean()}
d_fast = (c[(1, 0)] + c[(0, 1)]) / 2
d_slow = (c[(1, 1)] + c[(0, 0)]) / 2

per_addr = M[:, Bgp].mean(1) - M[:, ~Bgp].mean(1)

W, H = 1680, 720
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
F = "C:/Windows/Fonts/msyh.ttc"; FB = "C:/Windows/Fonts/msyhbd.ttc"


def f(sz, bold=False):
    try:
        return ImageFont.truetype(FB if bold else F, sz)
    except Exception:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", sz)


INK, GRAY = (25, 25, 30), (110, 110, 120)
BLUE, ORANGE, RED, GREEN = (52, 110, 200), (230, 140, 40), (200, 70, 70), (40, 140, 80)

d.text((60, 28), "H20 L2 NUCA 的交叉验证：同一地址在两组 SM 上快慢互换（checkerboard）",
       font=f(30, True), fill=INK)
d.text((60, 74), "512 个地址 × 78 个 SM，每格 64 次同地址依赖链求均值；分组由残差矩阵 PC1（84.7% 方差）的符号给出",
       font=f(17), fill=GRAY)

# ---- panel A: 2x2 heatmap
AX, AY, CW, CH = 120, 190, 190, 130
d.text((AX, AY - 46), "A. 2×2 交叉表（cyc）", font=f(21, True), fill=INK)


def shade(val):
    t = (val - 270) / 40.0
    t = max(0.0, min(1.0, t))
    return (int(235 - 60 * t), int(240 - 130 * t), int(250 - 150 * t))


grid = {(1, 0): c[(1, 0)], (1, 1): c[(1, 1)], (0, 0): c[(0, 0)], (0, 1): c[(0, 1)]}
for r in range(2):
    for q in range(2):
        x0 = AX + q * CW; y0 = AY + r * CH
        val = grid[(1 - r, q)]
        d.rectangle([x0, y0, x0 + CW - 4, y0 + CH - 4], fill=shade(val))
        txt = f"{val:.0f}"
        tw = d.textlength(txt, font=f(26, True))
        d.text((x0 + (CW - tw) / 2, y0 + 40), txt, font=f(26, True), fill=INK)
d.text((AX - 96, AY + 40), "地址组 +", font=f(18, True), fill=INK)
d.text((AX - 96, AY + CH + 40), "地址组 −", font=f(18, True), fill=INK)
d.text((AX + 20, AY - 24), "SM 组 −", font=f(17, True), fill=GRAY)
d.text((AX + CW + 20, AY - 24), "SM 组 +", font=f(17, True), fill=GRAY)

d.text((AX, AY + 2 * CH + 26), f"对角（快）{d_fast:.0f} cyc / {d_fast/GHZ:.0f} ns", font=f(19, True), fill=GREEN)
d.text((AX + 330, AY + 2 * CH + 26), f"反对角（慢）{d_slow:.0f} cyc / {d_slow/GHZ:.0f} ns", font=f(19, True), fill=RED)
d.text((AX, AY + 2 * CH + 58), f"交叉差 Δ = {d_slow-d_fast:.1f} cyc = {(d_slow-d_fast)/GHZ:.1f} ns（{(d_slow/d_fast-1)*100:.1f}%）",
       font=f(20, True), fill=RED)

# ---- panel B: per-address difference histogram
BX, BY, BW, BH = 900, 210, 620, 300
d.text((BX, BY - 46), "B. 每个地址的 (SM组+ 均值 − SM组− 均值)", font=f(21, True), fill=INK)
vals = per_addr
lo, hi = -40, 40
bins = np.linspace(lo, hi, 41)
cnt, _ = np.histogram(vals, bins)
mx = cnt.max()
d.line([(BX, BY + BH), (BX + BW, BY + BH)], fill=INK, width=2)
d.line([(BX, BY), (BX, BY + BH)], fill=INK, width=2)
for i in range(len(cnt)):
    x0 = BX + (bins[i] - lo) / (hi - lo) * BW
    x1 = BX + (bins[i + 1] - lo) / (hi - lo) * BW
    h = BH * cnt[i] / mx
    d.rectangle([x0 + 1, BY + BH - h, x1 - 1, BY + BH], fill=BLUE if bins[i] < 0 else ORANGE)
d.line([(BX + BW / 2, BY), (BX + BW / 2, BY + BH)], fill=GRAY, width=1)
for xv in (-30, 0, 30):
    x = BX + (xv - lo) / (hi - lo) * BW
    d.text((x - 14, BY + BH + 8), str(xv), font=f(15), fill=GRAY)
d.text((BX + BW / 2 - 60, BY + BH + 32), "差值 (cyc)", font=f(16), fill=GRAY)
d.text((BX + 6, BY + 6), f"双峰：−{np.abs(vals[vals<0]).mean():.0f} / +{vals[vals>0].mean():.0f} cyc", font=f(18, True), fill=INK)

# ---- panel C: SM group strip
CX, CY = 900, 606
d.text((CX, CY - 34), "C. SM 分组（蓝 = 组−，橙 = 组+）", font=f(19, True), fill=INK)
for i, s in enumerate(sms):
    x = CX + i * 7.2
    col = BLUE if Bgp[i] else ORANGE
    d.rectangle([x, CY, x + 6, CY + 26], fill=col)
d.text((CX, CY + 32), "SM id 0 → 77", font=f(15), fill=GRAY)

d.text((60, H - 40),
       f"地址组：{Agp.sum()} / {(~Agp).sum()} 个；SM 组：{Bgp.sum()} / {(~Bgp).sum()} 个。同一地址在两组 SM 上相差 ~30 cyc，且随地址组翻转。",
       font=f(17), fill=GRAY)

img.save("nuca/nuca_crossing.png")
print("saved", {k: round(v, 1) for k, v in c.items()}, round(d_fast, 1), round(d_slow, 1))
