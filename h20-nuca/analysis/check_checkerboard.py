"""Confirm the 2-partition (checkerboard) NUCA structure.

If PC1 of the double-centred matrix is the dominant mode, addresses and SMs each
split into two groups with opposite signs -> a 2x2 table where the diagonal is
slow and the anti-diagonal is fast (crossing).
"""
import numpy as np

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
R = M - M.mean(0, keepdims=True)
R = R - R.mean()
U, s, Vt = np.linalg.svd(R, full_matrices=False)
u, v = U[:, 0] * np.sqrt(s[0]), Vt[0] * np.sqrt(s[0])

print(f"grand mean {M.mean():.1f} cyc; PC1 carries {s[0]**2/ (s**2).sum()*100:.1f}% of residual variance")

# address loadings: bimodal?
q = np.percentile(u, [1, 25, 50, 75, 99])
print(f"address loadings u: p1 {q[0]:+.1f}  p25 {q[1]:+.1f}  p50 {q[2]:+.1f}  p75 {q[3]:+.1f}  p99 {q[4]:+.1f}")
neg, pos = (u < 0).sum(), (u >= 0).sum()
print(f"  addresses with u<0: {neg}   u>=0: {pos}")

q = np.percentile(v, [1, 25, 50, 75, 99])
print(f"SM loadings      v: p1 {q[0]:+.1f}  p25 {q[1]:+.1f}  p50 {q[2]:+.1f}  p75 {q[3]:+.1f}  p99 {q[4]:+.1f}")
print(f"  SMs with v<0: {(v<0).sum()}   v>=0: {(v>=0).sum()}")

# 2x2 table: mean latency for address group x SM group
A = u < 0
B = v < 0
cells = {}
for an, am in (("addr+", ~A), ("addr-", A)):
    for sn, smm in (("SM-", B), ("SM+", ~B)):
        cells[(an, sn)] = M[np.ix_(am, smm)].mean()
print("\n2x2 mean latency (cyc):")
print(f"            {'SM group -':>12} {'SM group +':>12}")
for an in ("addr+", "addr-"):
    print(f"  {an:6} {cells[(an,'SM-')]:12.1f} {cells[(an,'SM+')]:12.1f}")

diag = (cells[("addr+", "SM-")] + cells[("addr-", "SM+")]) / 2
anti = (cells[("addr+", "SM+")] + cells[("addr-", "SM-")]) / 2
print(f"\ndiagonal (fast) {diag:.1f} cyc, anti-diagonal (slow) {anti:.1f} cyc, "
      f"crossing gap {anti-diag:.1f} cyc = {(anti-diag)/GHZ:.1f} ns")
print(f"ratio: {anti/diag:.3f}")

# per-SM: is the SM grouping stable?
print("\nSM group members (v<0):", [sms[i] for i in np.where(B)[0]])
