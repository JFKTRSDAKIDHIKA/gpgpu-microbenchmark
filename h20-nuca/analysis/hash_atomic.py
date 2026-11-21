import numpy as np

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
y = ((M[:, gp].mean(1) - M[:, ~gp].mean(1)) > 0).astype(int)
print("label balance:", y.sum(), "/", L)

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
            return None, len(piv)
    sol = np.zeros(n, dtype=int)
    for i, c in enumerate(piv):
        sol[c] = aug[i, n]
    return sol, len(piv)

BASE = 0x10018000000

def build(bits):
    cols = [np.ones(L, dtype=int)]
    for b in bits:
        cols.append(((np.array([BASE + i * 128 for i in range(L)], dtype=np.uint64) >> b) & 1).astype(int))
    return np.stack(cols, 1)

for bits in [range(7, 17), range(8, 17), range(9, 17)]:
    X = build(bits)
    sol, rank = gf2_solve(X, y)
    tag = f"bits {bits.start}..{bits.stop-1}"
    if sol is None:
        print(f"{tag:12s}: no exact solution (rank {rank}/{X.shape[1]})")
    else:
        acc = ((X @ sol) % 2 == y).mean()
        names = ["1"] + [f"b{b}" for b in bits]
        print(f"{tag:12s}: EXACT {acc*100:.1f}%  mask = " +
              " ^ ".join(names[i] for i in np.where(sol)[0]))

# granularity on atomic labels
print("\n256B-block purity:")
for gs in (2, 4, 8):
    n = L // gs
    pure = sum(1 for g in range(n) if len(set(y[g*gs:(g+1)*gs])) == 1)
    print(f"  {gs*128:5d} B: {pure/n*100:.1f}%")
