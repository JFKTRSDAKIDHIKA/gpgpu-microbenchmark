import numpy as np

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

print(f"{'group':>12} {'lowest bit':>10} {'pure groups':>12}")
for gs in [2, 4, 8, 16, 32, 64]:
    n = L // gs
    pure = sum(1 for g in range(n) if len(set(y[g * gs:(g + 1) * gs])) == 1)
    print(f"{gs*128:>9} B {8 + int(np.log2(gs)) - 1:>10} {pure/n*100:>11.1f}%")

# run-length distribution
runs, cur = [], 1
for i in range(1, L):
    if y[i] == y[i - 1]:
        cur += 1
    else:
        runs.append(cur); cur = 1
runs.append(cur)
runs = np.array(runs)
print(f"\nruns: {len(runs)}, mean {runs.mean():.2f} lines ({runs.mean()*128:.0f} B), "
      f"median {np.median(runs):.0f}, max {runs.max()}")
