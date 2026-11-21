"""Interleave granularity: at which bit does the partition label start to change?"""
import numpy as np

sm, ln, cyc = [], [], []
with open("nuca/multism4.csv") as f:
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

R = M - M.mean(0, keepdims=True); R = R - R.mean()
v1 = np.linalg.svd(R, full_matrices=False)[2][0]
gp = v1 >= 0
d = M[:, gp].mean(1) - M[:, ~gp].mean(1)
y = (d > 0).astype(np.int64)

print("within-group label purity (group = fixed bits >= k):")
print(f"{'group size':>10} {'lowest bit':>10} {'purity':>8} {'groups':>8}")
for gs in [1, 2, 4, 8, 16, 32, 64, 128]:
    lowbit = 7 + int(np.log2(gs))
    n = L // gs
    pure = 0
    for g in range(n):
        seg = y[g * gs:(g + 1) * gs]
        pure += max(seg.sum(), gs - seg.sum())
    print(f"{gs:>10} {lowbit:>10} {pure/L*100:>7.1f}% {n:>8}")

# pairwise: do two lines that differ only in bits 7..k always share a label?
print("\nflip rate when only low bits differ (bit 7..k vary, bits >k fixed):")
for k in range(7, 14):
    mask = (1 << (k + 1)) - 1
    groups = {}
    for i in range(L):
        groups.setdefault(i & ~mask, []).append(i)
    flips = tot = 0
    for _, g in groups.items():
        if len(g) < 2: continue
        vals = y[g]
        flips += int((vals != vals[0]).sum()); tot += len(vals) - 1
    if tot:
        print(f"  bits 7..{k} vary: flip rate {flips/tot*100:5.1f}%  (n={tot})")
