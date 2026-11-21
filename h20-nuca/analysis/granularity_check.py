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
d = M[:, gp].mean(1) - M[:, ~gp].mean(1)
y = (d > 0).astype(int)

print("y[0:32] =", "".join(map(str, y[:32])))
pairs_pure = sum(1 for g in range(512) if y[2 * g] == y[2 * g + 1])
print(f"adjacent pairs (i, i+1) pure: {pairs_pure}/512 = {pairs_pure/512*100:.1f}%")
print(f"pairs (2g,2g+1) same as above")

# same thing but grouped by address bit 7 clear (identical to above)
print("y[0:8] =", y[:8].tolist(), " y[8:16] =", y[8:16].tolist())
