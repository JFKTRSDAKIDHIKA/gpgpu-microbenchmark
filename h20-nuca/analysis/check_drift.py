"""Check whether the address x SM interaction is a time-drift artifact.

Each block walks the 512 addresses starting at a different rotation (start = b*7),
so any drift over the kernel's runtime maps onto a different address per block and
survives row-mean removal. Here we detrend each block's series over its own
processing order and redo the SVD.
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

# reconstruct each block's processing order
D = M.copy()
trends = []
for j, s in enumerate(sms):
    start = (s * 7) % L          # b == block index == position in sid list
    a_order = np.arange(L)
    idx_order = (start + a_order) % L
    y = M[idx_order, j]
    p = np.polyfit(a_order, y, 1)
    trends.append((p[0], s, y.std(), p[0] * L))
    D[idx_order, j] = y - np.polyval(p, a_order)

tr = np.array([t[3] for t in trends])
print(f"linear drift per block over its 512 addresses: mean {tr.mean():+.1f} cyc, "
      f"min {tr.min():+.1f}, max {tr.max():+.1f}")

def report(tag, X):
    R = X - X.mean(axis=0)
    R = R - R.mean()
    sv = np.linalg.svd(R, compute_uv=False)
    e = sv**2 / (sv**2).sum()
    print(f"{tag}: PC1 {e[0]*100:.1f}%  PC2 {e[1]*100:.1f}%  PC3 {e[2]*100:.1f}%")

report("raw            ", M)
report("detrended      ", D)

# also: does the SM effect survive detrending?
smm = D.mean(axis=0)
o = np.argsort(smm)
print(f"per-SM mean after detrending: min {smm[o[0]]:.1f} (SM {sms[o[0]]})  "
      f"max {smm[o[-1]]:.1f} (SM {sms[o[-1]]})  spread {smm[o[-1]]-smm[o[0]]:.1f} cyc")
