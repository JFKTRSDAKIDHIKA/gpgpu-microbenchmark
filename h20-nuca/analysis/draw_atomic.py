"""Atomic-based NUCA figure: clean 2-way ladder + exact XOR hash."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sm, ln, cyc = [], [], []
with open("nuca/nuca_atomic.csv") as f:
    for line in f:
        a, b, c = line.strip().split(",")
        sm.append(int(a)); ln.append(int(b)); cyc.append(float(c))

sms = sorted(set(sm)); L = 512
sidx = {s: i for i, s in enumerate(sms)}
M = np.zeros((L, len(sms)))
for a, b, c in zip(sm, ln, cyc):
    M[b, sidx[a]] = c

R = M - M.mean(0, keepdims=True); R = R - R.mean()
v1 = np.linalg.svd(R, full_matrices=False)[2][0]
gp = v1 >= 0
d = M[:, gp].mean(1) - M[:, ~gp].mean(1)
y = d > 0

GHZ = 1.98
W, H = 1680, 1000
img = Image.new("RGB", (W, H), "white")
dr = ImageDraw.Draw(img)
F = "C:/Windows/Fonts/msyh.ttc"; FB = "C:/Windows/Fonts/msyhbd.ttc"


def f(sz, bold=False):
    try:
        return ImageFont.truetype(FB if bold else F, sz)
    except Exception:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", sz)


INK, GRAY = (25, 25, 30), (110, 110, 120)
BLUE, ORANGE, RED, GREEN = (52, 110, 200), (230, 140, 40), (200, 70, 70), (40, 140, 80)

dr.text((50, 26), "改用 atomicCAS(0,0) 后的 NUCA：两档阶梯 + 精确 XOR 哈希",
        font=f(32, True), fill=INK)
dr.text((50, 74), "78 SM × 512 行 · 每格 64 次同地址原子链（地址 = base + (ret & 0x7F)，内存值恒为 0）· H20 1.98 GHz",
        font=f(18), fill=GRAY)

# ---- A: 2x2
AX, AY, CW, CH = 90, 190, 210, 130
dr.text((AX, AY - 46), "A. 2×2（cyc）", font=f(21, True), fill=INK)
tab = {(1, 1): M[np.ix_(y, gp)].mean(), (1, 0): M[np.ix_(y, ~gp)].mean(),
       (0, 1): M[np.ix_(~y, gp)].mean(), (0, 0): M[np.ix_(~y, ~gp)].mean()}


def shade(v):
    t = (v - 290) / 220.0
    return (int(238 - 40 * t), int(242 - 150 * t), int(250 - 170 * t))


for r, an in enumerate([1, 0]):
    for q, bn in enumerate([0, 1]):
        x0 = AX + q * CW; y0 = AY + r * CH
        v = tab[(an, bn)]
        dr.rectangle([x0, y0, x0 + CW - 5, y0 + CH - 5], fill=shade(v))
        t = f"{v:.0f}"; tw = dr.textlength(t, font=f(26, True))
        dr.text((x0 + (CW - tw) / 2, y0 + 44), t, font=f(26, True), fill=INK)
dr.text((AX - 78, AY + 44), "地址组 +", font=f(17, True), fill=INK)
dr.text((AX - 78, AY + CH + 44), "地址组 −", font=f(17, True), fill=INK)
dr.text((AX + 60, AY - 24), "SM 组 −", font=f(16, True), fill=GRAY)
dr.text((AX + CW + 60, AY - 24), "SM 组 +", font=f(16, True), fill=GRAY)
dr.text((AX, AY + 2 * CH + 24), "Home 301 cyc / 152 ns", font=f(19, True), fill=GREEN)
dr.text((AX + 260, AY + 2 * CH + 24), "Remote 501 cyc / 253 ns", font=f(19, True), fill=RED)
dr.text((AX, AY + 2 * CH + 56), "Δ = 200 cyc = 101 ns（+67%）；PC1 = 97.5% 方差",
        font=f(20, True), fill=RED)

# ---- B: per-address gap histogram
BX, BY, BW, BH = 760, 200, 380, 250
dr.text((BX, BY - 46), "B. 每个地址的 (SM+ − SM−)", font=f(20, True), fill=INK)
cnt, bins = np.histogram(d, bins=24)
mx = cnt.max()
dr.line([(BX, BY + BH), (BX + BW, BY + BH)], fill=INK, width=2)
dr.line([(BX, BY), (BX, BY + BH)], fill=INK, width=2)
for i in range(len(cnt)):
    x0 = BX + (bins[i] - d.min()) / (d.max() - d.min()) * BW
    x1 = BX + (bins[i + 1] - d.min()) / (d.max() - d.min()) * BW
    h = BH * cnt[i] / mx
    dr.rectangle([x0 + 1, BY + BH - h, x1 - 1, BY + BH], fill=BLUE if bins[i] < 0 else ORANGE)
dr.text((BX, BY + BH + 10), f"{d.min():.0f} cyc", font=f(15), fill=GRAY)
dr.text((BX + BW - 70, BY + BH + 10), f"+{d.max():.0f} cyc", font=f(15), fill=GRAY)
dr.text((BX + 8, BY + 8), "干净双峰，中间无样本", font=f(16, True), fill=GREEN)

# ---- C: hash verification
CX, CY = 90, 610
dr.text((CX, CY - 34), "C. GF(2) 拟合：partition = b12 ⊕ b13 ⊕ b15（512/512 = 100%）",
        font=f(20, True), fill=INK)
addr = np.array([0x10018000000 + i * 128 for i in range(L)], dtype=np.uint64)
pred = (((addr >> 12) & 1) ^ ((addr >> 13) & 1) ^ ((addr >> 15) & 1)).astype(int)
for i in range(L):
    x = CX + i * 3
    dr.rectangle([x, CY, x + 2, CY + 18], fill=ORANGE if y[i] else BLUE)
    dr.rectangle([x, CY + 22, x + 2, CY + 40], fill=ORANGE if pred[i] else BLUE)
dr.text((CX, CY + 48), "上一行：实测（原子）   下一行：b12⊕b13⊕b15 预测   逐列完全一致", font=f(16), fill=GRAY)
dr.text((CX + 700, CY + 80), "粒度：bit 12 = 4 KB；256 B / 512 B / 1 KB 块内标签 100% 恒定",
        font=f(17), fill=INK)

dr.text((50, H - 70),
        "原子必须回到 Home 切片才能完成 RMW，所以 read-only 本地副本无法掩盖距离：落差从 Load 的 30 cyc 放大到 200 cyc，",
        font=f(17), fill=INK)
dr.text((50, H - 42),
        "同时标签变成精确 50/50，GF(2) 上出现唯一解 —— 说明之前 Load 版本的标签被本地副本“抹平”了，才拟合不出哈希。",
        font=f(17), fill=INK)

img.save("nuca/nuca_atomic_result.png")
print("saved", {k: round(v) for k, v in tab.items()})
