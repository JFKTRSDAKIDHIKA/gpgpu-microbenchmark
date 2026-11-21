"""Two-level topology mapping from the 78 x 1024 matrix.

Level 1: PC1 -> 2 SM groups / 2 address groups (the ~30 cycle partition gap).
Level 2: PC2 -> split each half again (the 4-12 cycle left/right gap).
"""
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

GHZ = 1.98
print(f"matrix {L} x {S}, grand mean {M.mean():.1f} cyc, per-address sd across SMs "
      f"{np.median(M.std(1, ddof=1)):.1f} cyc")

R = M - M.mean(0, keepdims=True)
R = R - R.mean()
U, s, Vt = np.linalg.svd(R, full_matrices=False)
e = s**2 / (s**2).sum()
print(f"variance: PC1 {e[0]*100:.1f}%  PC2 {e[1]*100:.1f}%  PC3 {e[2]*100:.1f}%")

u1, u2 = U[:, 0], U[:, 1]
v1, v2 = Vt[0], Vt[1]

# ---------- level 1
A1 = u1 >= 0
B1 = v1 >= 0
tab = {}
for an, am in (("+", A1), ("-", ~A1)):
    for bn, bm in (("+", B1), ("-", ~B1)):
        tab[(an, bn)] = M[np.ix_(am, bm)].mean()
print("\n[L1] 2x2 (cyc):")
print(f"          SM+ {tab[('+','+')]:7.1f}   SM- {tab[('+','-')]:7.1f}")
print(f"   addr-  SM+ {tab[('-','+')]:7.1f}   SM- {tab[('-','-')]:7.1f}")
diag = (tab[("+", "+")] + tab[("-", "-")]) / 2
anti = (tab[("+", "-")] + tab[("-", "+")]) / 2
print(f"   diagonal {diag:.1f}  anti {anti:.1f}  gap {anti-diag:.1f} cyc = {(anti-diag)/GHZ:.1f} ns")

# ---------- level 2: k-means in the (PC1,PC2) plane
def kmeans(X, k, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = X[lab == j].mean(0)
    return lab, c

SMs = np.stack([v1, v2], 1)
Ad = np.stack([u1, u2], 1)
slab, scen = kmeans(SMs, 4, seed=3)
alab, acen = kmeans(Ad, 4, seed=3)

# order both by angle for a stable presentation
sorder = np.argsort(np.arctan2(scen[:, 1], scen[:, 0]))
aorder = np.argsort(np.arctan2(acen[:, 1], acen[:, 0]))
smap = {old: new for new, old in enumerate(sorder)}
amap = {old: new for new, old in enumerate(aorder)}
slab = np.array([smap[x] for x in slab])
alab = np.array([amap[x] for x in alab])

T = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        T[i, j] = M[np.ix_(alab == i, slab == j)].mean()

print("\n[L2] 4x4 table (rows = address group, cols = SM group), cyc:")
print("        " + "".join(f"  SM{j}    " for j in range(4)))
for i in range(4):
    print(f"  A{i}  " + "".join(f"{T[i,j]:8.1f} " for j in range(4)))

# match each address group to its fastest SM group
match = T.argmin(1)
print("\naddress group -> fastest SM group:", match.tolist())
best = T.min(1)
print("per-address-group best latency:", [f"{b:.1f}" for b in best])
# second level: for each SM group, best vs 2nd best
print("\nper-SM-group: best / 2nd best address group (the L2 left-right step):")
for j in range(4):
    col = np.sort(T[:, j])
    print(f"  SM{j}: best {col[0]:.1f}  second {col[1]:.1f}  step {col[1]-col[0]:.1f} cyc "
          f"= {(col[1]-col[0])/GHZ:.1f} ns")

print("\nSM groups:")
for j in range(4):
    members = [sms[i] for i in np.where(slab == j)[0]]
    print(f"  SM{j} ({len(members)}): {members}")
print("address group sizes:", [int((alab == i).sum()) for i in range(4)])

np.save("nuca/table4.npy", T)
with open("nuca/quadrants.csv", "w") as f:
    f.write("sm,group\n")
    for j in range(4):
        for i in np.where(slab == j)[0]:
            f.write(f"{sms[i]},{j}\n")
print("\nwrote nuca/quadrants.csv")
