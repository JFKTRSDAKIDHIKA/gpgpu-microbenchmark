import numpy as np, itertools

sm, ln, cyc = [], [], []
with open("nuca/multism4.csv") as f:
    for line in f:
        a, b, c = line.strip().split(",")
        sm.append(int(a)); ln.append(int(b)); cyc.append(float(c))

sms = sorted(set(sm)); L = 1024
sidx = {s: i for i, s in enumerate(sms)}
M = np.zeros((L, len(sms)))
for a, b, c in zip(sm, ln, cyc):
    M[b, sidx[a]] = c

R = M - M.mean(0, keepdims=True); R = R - R.mean()
v1 = np.linalg.svd(R, full_matrices=False)[2][0]
gp = v1 >= 0
y = ((M[:, gp].mean(1) - M[:, ~gp].mean(1)) > 0).astype(int)

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

def build(bits, page_feat):
    cols = [np.ones(L, dtype=int)]
    for b in bits:
        cols.append(((np.arange(L) * 128 + 0x10018000000) >> b & 1).astype(int))
    if page_feat:
        cols.append((np.arange(L) >= 512).astype(int))   # second 64 KiB half
    return np.stack(cols, 1)

for label, bits, pf in [
    ("bits 8..16            ", range(8, 17), False),
    ("bits 8..16 + 64K half ", range(8, 17), True),
    ("bits 8..20            ", range(8, 21), False),
]:
    X = build(bits, pf)
    sol, rank = gf2_solve(X, y)
    if sol is None:
        print(f"{label}: no exact solution (rank {rank}/{X.shape[1]})")
    else:
        acc = ((X @ sol) % 2 == y).mean()
        print(f"{label}: EXACT, accuracy {acc*100:.1f}%, rank {rank}/{X.shape[1]}, "
              f"mask bits {[b for i,b in enumerate(['1']+[f'b{b}' for b in bits]+(['half'] if pf else [])) if sol[i]]}")
