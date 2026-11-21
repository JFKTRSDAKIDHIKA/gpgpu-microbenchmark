"""Follow-up: periodicity scan + quadratic (b_i*b_j) GF(2) fit."""
import numpy as np, itertools

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

# ---- periodicity scan
print("period scan (accuracy of 'label depends only on i mod P'):")
best = []
for P in [4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]:
    acc = 0
    for r in range(P):
        idx = np.arange(r, L, P)
        if len(idx) == 0: continue
        maj = np.bincount(y[idx], minlength=2).argmax()
        acc += (y[idx] == maj).sum()
    acc /= L
    best.append((acc, P))
    if acc > 0.7:
        print(f"  P={P:4d}: {acc*100:.1f}%")
print("  best periods:", [(P, f"{a*100:.1f}%") for a, P in sorted(best)[-4:]])

# ---- quadratic GF(2) fit: const + linear + all pairwise products
def gf2_solve(A, b):
    A = A.copy() % 2; b = b.copy() % 2
    m, n = A.shape
    aug = np.concatenate([A, b[:, None]], 1)
    row = 0; piv = []
    for col in range(n):
        p = next((r for r in range(row, m) if aug[r, col]), None)
        if p is None: continue
        aug[[row, p]] = aug[[p, row]]
        for r in range(m):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        piv.append(col); row += 1
        if row == m: break
    for r in range(row, m):
        if aug[r, :n].sum() == 0 and aug[r, n]:
            return None, piv
    sol = np.zeros(n, dtype=np.int64)
    for i, c in enumerate(piv):
        sol[c] = aug[i, n]
    return sol, piv

BASE = 0x10018000000
addr = np.array([BASE + i * 128 for i in range(L)], dtype=np.uint64)
bits = list(range(7, 17))
B = np.stack([((addr >> b) & 1).astype(np.int64) for b in bits], 1)

lin = np.concatenate([np.ones((L, 1), dtype=np.int64), B], 1)
sol, piv = gf2_solve(lin, y)
print(f"\nlinear  : rank {len(piv)}/{lin.shape[1]}, exact={sol is not None}")

pairs = [(i, j) for i, j in itertools.combinations(range(len(bits)), 2)]
Q = np.stack([B[:, i] * B[:, j] for i, j in pairs], 1)
quad = np.concatenate([lin, Q], 1)
solq, pivq = gf2_solve(quad, y)
if solq is None:
    print(f"quadratic: {quad.shape[1]} features, no exact solution")
    # best-effort: least squares over reals then round, report accuracy
    w, *_ = np.linalg.lstsq(quad.astype(float), y.astype(float), rcond=None)
    pred = (quad @ w > 0.5).astype(int)
    print(f"quadratic (real LSQ + threshold): accuracy {(pred == y).mean()*100:.1f}%")
else:
    pred = (quad @ solq) % 2
    print(f"quadratic: exact, accuracy {(pred == y).mean()*100:.1f}%, non-zero terms "
          f"{int(solq.sum())}")
    names = ["1"] + [f"b{b}" for b in bits]
    for k, (i, j) in enumerate(pairs):
        names.append(f"b{bits[i]}b{bits[j]}")
    print("  terms:", [names[i] for i in np.where(solq)[0]])
