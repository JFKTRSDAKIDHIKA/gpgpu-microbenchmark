import numpy as np

sm, ln, cyc = [], [], []
with open("nuca/nuca_atomic.csv") as f:
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
print(f"atomic matrix {L} x {S}, mean {M.mean():.1f} cyc ({M.mean()/GHZ:.0f} ns)")
print(f"per-address sd across SMs: median {np.median(M.std(1, ddof=1)):.1f} cyc")

smm = M.mean(0)
o = np.argsort(smm)
print(f"per-SM mean: {smm[o[0]]:.1f} (SM {sms[o[0]]}) .. {smm[o[-1]]:.1f} (SM {sms[o[-1]]}), "
      f"spread {smm[o[-1]]-smm[o[0]]:.1f} cyc = {(smm[o[-1]]-smm[o[0]])/GHZ:.1f} ns")
print(f"stderr of a per-SM mean: {np.std(smm)/np.sqrt(S):.2f} cyc")

R = M - M.mean(0, keepdims=True); R = R - R.mean()
U, s, Vt = np.linalg.svd(R, full_matrices=False)
e = s**2 / (s**2).sum()
print(f"variance: PC1 {e[0]*100:.1f}%  PC2 {e[1]*100:.1f}%  PC3 {e[2]*100:.1f}%")

gp = Vt[0] >= 0
d = M[:, gp].mean(1) - M[:, ~gp].mean(1)
y = d > 0
print(f"\nSM split {gp.sum()}/{ (~gp).sum()};  label balance {y.sum()}/{L} = {y.mean():.3f}")
print(f"per-address |gap| median {np.median(np.abs(d)):.1f} cyc, "
      f"p10 {np.percentile(np.abs(d),10):.1f}, p90 {np.percentile(np.abs(d),90):.1f}")

tab = {}
for an, am in (("+", y), ("-", ~y)):
    for bn, bm in (("+", gp), ("-", ~gp)):
        tab[(an, bn)] = M[np.ix_(am, bm)].mean()
print("\n2x2 (cyc):")
print(f"             SM+ {tab[('+','+')]:8.1f}   SM- {tab[('+','-')]:8.1f}")
print(f"     addr-   SM+ {tab[('-','+')]:8.1f}   SM- {tab[('-','-')]:8.1f}")
diag = (tab[("+","+")] + tab[("-","-")]) / 2
anti = (tab[("+","-")] + tab[("-","+")]) / 2
print(f"diagonal {diag:.1f}  anti {anti:.1f}  gap {anti-diag:.1f} cyc = {(anti-diag)/GHZ:.1f} ns "
      f"({(anti/diag-1)*100:.0f}%)")

# how well does the 2-way split separate? overlap check
g_fast = np.where(y, M[:, gp].mean(1), M[:, ~gp].mean(1))
g_slow = np.where(y, M[:, ~gp].mean(1), M[:, gp].mean(1))
print(f"\nper-address fast-home {np.median(g_fast):.1f} vs remote {np.median(g_slow):.1f} cyc")
