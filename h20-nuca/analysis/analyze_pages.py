import numpy as np

rows = []
with open("nuca/hash_pages.csv") as f:
    for line in f:
        a, b, c = line.strip().split(",")
        rows.append((int(a), int(b), float(c)))

sms = sorted({r[0] for r in rows})
N = 4 * 512
S = len(sms)
sidx = {s: i for i, s in enumerate(sms)}
M = np.zeros((N, S))
for a, t, c in rows:
    M[t, sidx[a]] = c

R = M - M.mean(0, keepdims=True); R = R - R.mean()
v1 = np.linalg.svd(R, full_matrices=False)[2][0]
gp = v1 >= 0
print(f"SM groups {gp.sum()}/{ (~gp).sum()}, grand mean {M.mean():.1f} cyc")

BASE = 0x10018000000
PGSZ = 2 << 20
addr = np.array([BASE + (t // 512) * PGSZ + (t % 512) * 128 for t in range(N)], dtype=np.uint64)
pred = (((addr >> 12) & 1) ^ ((addr >> 13) & 1) ^ ((addr >> 15) & 1)).astype(int)

d = M[:, gp].mean(1) - M[:, ~gp].mean(1)
y = (d > 0).astype(int)
print(f"label balance {y.sum()}/{N}")
print(f"overall accuracy of b12^b13^b15: {(pred == y).mean()*100:.1f}%")

print("\nper page:")
for p in range(4):
    sl = slice(p * 512, (p + 1) * 512)
    yp, pp = y[sl], pred[sl]
    acc = (yp == pp).mean()
    inv = (yp == (1 - pp)).mean()
    med = np.median(np.abs(d[sl]))
    print(f"  page {p}: accuracy {acc*100:5.1f}%  inverted {inv*100:5.1f}%  "
          f"median|gap| {med:5.1f} cyc  base=0x{BASE + p*PGSZ:x}")

# alternative masks that include page bits
print("\nother candidate masks:")
for name, m in [("b12^b13^b15", [12, 13, 15]),
                ("b12^b13^b15^b21", [12, 13, 15, 21]),
                ("b12^b13^b15^b21^b22", [12, 13, 15, 21, 22])]:
    pr = np.zeros(N, dtype=int)
    for b in m:
        pr ^= ((addr >> b) & 1).astype(int)
    print(f"  {name:22s}: {(pr == y).mean()*100:5.1f}%")
