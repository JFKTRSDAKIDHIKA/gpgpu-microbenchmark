"""Decisive NUCA test on the 78x512 same-address matrix.

Steps:
 1. per-SM mean (common mode)
 2. remove it, look at the address x SM residual
 3. SVD: does a second (address-dependent) component carry real variance?
 4. k-means(2) on address profiles: are the two centroids complementary (r < 0)?
 5. per-address fast-half / slow-half agreement across addresses
"""
import numpy as np

sm, ln, cyc = [], [], []
with open("nuca/multism3.csv") as f:
    for line in f:
        a, b, c = line.strip().split(",")
        sm.append(int(a)); ln.append(int(b)); cyc.append(float(c))

sms = sorted(set(sm))
lines = sorted(set(ln))
S, L = len(sms), len(lines)
sidx = {s: i for i, s in enumerate(sms)}
lidx = {l: i for i, l in enumerate(lines)}
M = np.zeros((L, S))          # address x SM
for a, b, c in zip(sm, ln, cyc):
    M[lidx[b], sidx[a]] = c

GHZ = 1.98
print(f"matrix {L} addresses x {S} SMs, grand mean {M.mean():.1f} cyc")

# 1. per-SM common mode
sm_mean = M.mean(axis=0)
order = np.argsort(sm_mean)
print(f"per-SM mean: min {sm_mean[order[0]]:.1f} (SM {sms[order[0]]})  "
      f"max {sm_mean[order[-1]]:.1f} (SM {sms[order[-1]]})  "
      f"spread {sm_mean[order[-1]]-sm_mean[order[0]]:.1f} cyc = "
      f"{(sm_mean[order[-1]]-sm_mean[order[0]])/GHZ:.1f} ns")

# stderr of a per-SM mean
per_addr_sd = M.std(axis=1, ddof=1)
print(f"per-address sd across SMs: median {np.median(per_addr_sd):.1f} cyc; "
      f"expected SE of a 512-address mean = {np.median(per_addr_sd)/np.sqrt(L):.2f} cyc")

# 2. residual after removing the SM common mode
R = M - sm_mean[None, :]
print(f"\nvariance: total {M.var():.1f} | explained by SM mean {sm_mean.var():.1f} "
      f"({sm_mean.var()/M.var()*100:.1f}%) | residual {R.var():.1f} "
      f"({R.var()/M.var()*100:.1f}%)")

# 3. SVD of the residual: is there address-dependent structure?
U, sv, Vt = np.linalg.svd(R - R.mean(), full_matrices=False)
expl = sv**2 / (sv**2).sum()
print(f"SVD of residual: PC1 {expl[0]*100:.1f}%  PC2 {expl[1]*100:.1f}%  PC3 {expl[2]*100:.1f}%")
print(f"  PC1 loadings range [{Vt[0].min():+.2f}, {Vt[0].max():+.2f}] (mixed signs => crossing)")
print(f"  PC2 loadings range [{Vt[1].min():+.2f}, {Vt[1].max():+.2f}]")

# permutation null: how much PC1 does pure noise give?
rng = np.random.default_rng(0)
null = []
for _ in range(5):
    Rn = R.copy()
    for j in range(S):
        Rn[:, j] = rng.permutation(Rn[:, j])
    _, svn, _ = np.linalg.svd(Rn - Rn.mean(), full_matrices=False)
    null.append((svn[0]**2 / (svn**2).sum()))
print(f"  PC1 on column-shuffled (no address structure) data: {np.mean(null)*100:.1f}%")

# 4. two clusters of addresses; are their SM profiles complementary?
def kmeans2(X, iters=50):
    c = X[np.random.default_rng(1).choice(len(X), 2, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for k in range(2):
            if (lab == k).any():
                c[k] = X[lab == k].mean(0)
    return lab, c

lab, cen = kmeans2(R)
r_cent = np.corrcoef(cen)[0, 1]
print(f"\nk-means(2) on address profiles: cluster sizes {np.bincount(lab).tolist()}, "
      f"centroid correlation r = {r_cent:+.3f}")
print("  (r ~ -1 => complementary crossing; r ~ 0 => no address-dependent affinity)")

# 5. per-address fast half: is it the same SMs every time?
fast_sets = []
for i in range(L):
    o = np.argsort(M[i])
    fast_sets.append(set(o[: S // 2]))
base = fast_sets[0]
jac = [len(base & s) / len(base | s) for s in fast_sets[1:]]
print(f"\nfast-half (39 SMs) Jaccard vs address 0: mean {np.mean(jac):.3f} "
      f"(random baseline ~0.33)")
