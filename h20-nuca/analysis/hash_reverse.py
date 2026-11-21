"""Reverse-engineer the partition hash over GF(2).

Labels: for each 128 B line i, compare latency on SM_group+ vs SM_group- (PC1 split).
Solve y = c0 ^ (c7*b7) ^ ... ^ (c16*b16)  over GF(2).
Buffer is 2 MiB aligned, 128 KiB span -> bits 7..16 vary, 17+ constant.
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

# --- PC1 split of SMs -> two groups
R = M - M.mean(0, keepdims=True); R = R - R.mean()
U, s, Vt = np.linalg.svd(R, full_matrices=False)
v1 = Vt[0]
gp = v1 >= 0                      # "SM group +"
d = M[:, gp].mean(1) - M[:, ~gp].mean(1)   # + means faster on group-
y = (d > 0).astype(np.int64)      # label per line
print(f"SM groups: {gp.sum()} / {(~gp).sum()};  label balance: {y.sum()} / {L} (={y.mean():.3f})")

# monotonicity check: is |d| large compared to noise?
sd = M[:, gp].std(1, ddof=1) / np.sqrt(gp.sum())
print(f"per-line |d| median {np.median(np.abs(d)):.1f} cyc, noise sd ~{np.median(sd):.2f} cyc"
      f" -> separation {np.median(np.abs(d))/np.median(sd):.1f} sigma")

# --- features: bits 7..16 of the address (== physical low bits inside a 2 MiB page)
BASE = 0x10018000000
addr = np.array([BASE + i * 128 for i in range(L)], dtype=np.uint64)
bits = list(range(7, 17))
X = np.zeros((L, len(bits) + 1), dtype=np.int64)
X[:, 0] = 1
for j, b in enumerate(bits):
    X[:, j + 1] = ((addr >> b) & 1).astype(np.int64)

# --- GF(2) gaussian elimination
def gf2_solve(A, b):
    A = A.copy() % 2
    b = b.copy() % 2
    m, n = A.shape
    aug = np.concatenate([A, b[:, None]], 1)
    row = 0
    pivots = []
    for col in range(n):
        p = None
        for r in range(row, m):
            if aug[r, col]:
                p = r; break
        if p is None:
            continue
        aug[[row, p]] = aug[[p, row]]
        for r in range(m):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        pivots.append(col)
        row += 1
        if row == m:
            break
    # consistency: any row 0..0 | 1 ?
    for r in range(row, m):
        if aug[r, :n].sum() == 0 and aug[r, n]:
            return None, pivots, aug
    sol = np.zeros(n, dtype=np.int64)
    for i, col in enumerate(pivots):
        sol[col] = aug[i, n]
    return sol, pivots, aug

sol, piv, aug = gf2_solve(X, y)
if sol is None:
    print("no exact linear solution (nonlinear / other bits involved)")
else:
    pred = (X @ sol) % 2
    acc = (pred == y).mean()
    print(f"\nGF(2) linear fit: accuracy {acc*100:.2f}%  (rank {len(piv)} / {X.shape[1]})")
    names = ["const"] + [f"bit{b}" for b in bits]
    mask = [names[i] for i in range(len(sol)) if sol[i]]
    print("  y = " + " ^ ".join(mask) if mask else "constant")
    print("  coefficients:", {names[i]: int(sol[i]) for i in range(len(sol))})

    # which single bit alone predicts best?
    print("\n  single-bit accuracy:")
    for j, b in enumerate(bits):
        col = X[:, j + 1]
        a1 = max((col == y).mean(), (col != y).mean())
        if a1 > 0.55:
            print(f"    bit{b}: {a1*100:.1f}%")

# --- does the label alternate with a short period? (interleave granularity)
print("\nlabel runs (first 64 lines):", "".join(str(v) for v in y[:64]))
runs = 1 + int((y[1:] != y[:-1]).sum())
print(f"total runs over {L} lines: {runs} (random ~{L//2 + 1})")

# period detection: autocorrelation of y at small lags
for lag in (1, 2, 4, 8, 16, 32):
    if lag < L:
        agree = (y[:-lag] == y[lag:]).mean()
        print(f"  lag {lag:3d}: agreement {agree*100:.1f}%")
