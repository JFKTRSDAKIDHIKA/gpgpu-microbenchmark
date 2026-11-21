"""Stability analysis of the same-address multi-SM measurement."""
import csv
from collections import defaultdict
import statistics as st

rows = []
with open("nuca/multism.csv") as f:
    for line in f:
        sm, ln, cyc = line.strip().split(",")
        rows.append((int(sm), int(ln), int(cyc)))

sm_lines = defaultdict(dict)
for sm, ln, cyc in rows:
    sm_lines[sm][ln] = cyc

sms = sorted(sm_lines)
lines = sorted(sm_lines[sms[0]])
assert len(sms) == 78 and len(lines) == 512, (len(sms), len(lines))

def sm_means(lns):
    out = {}
    for sm in sms:
        out[sm] = st.mean(sm_lines[sm][l] for l in lns)
    return out

even = [l for l in lines if l % 2 == 0]
odd = [l for l in lines if l % 2 == 1]
me, mo = sm_means(even), sm_means(odd)
m_all = sm_means(lines)

def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db)

xs = [me[s] for s in sms]
ys = [mo[s] for s in sms]
r = pearson(xs, ys)

vals = sorted(m_all.values())
print(f"SMs={len(sms)} lines={len(lines)}")
print(f"per-SM mean: min={vals[0]:.1f} max={vals[-1]:.1f} spread={vals[-1]-vals[0]:.1f} cyc "
      f"= {(vals[-1]-vals[0])/1.98:.1f} ns, global={st.mean(vals):.1f}")
print(f"stderr of a per-SM mean (over 512 lines): {st.stdev(vals)/len(vals)**0.5:.2f} cyc")
print(f"even-half vs odd-half per-SM correlation: r = {r:.3f}")
print("  -> r close to 1 means the SM ordering is a stable property, not noise")

# per-line spread
spreads = []
for l in lines:
    v = [sm_lines[s][l] for s in sms]
    spreads.append(max(v) - min(v))
spreads.sort()
q = lambda p: spreads[int(p * (len(spreads) - 1))]
print(f"per-line spread across SMs: p10={q(.1)} p50={q(.5)} p90={q(.9)} max={spreads[-1]} cyc")

# is a given line's 'fastest SM' reproducible across halves?
# use even-line data to pick the fastest SM for the *odd* lines? not applicable;
# instead: split SMs into two groups by even-half mean, check consistency on odd half
fast_even = sorted(sms, key=lambda s: me[s])[:8]
fast_odd = sorted(sms, key=lambda s: mo[s])[:8]
print("fastest-8 SMs (even half):", fast_even)
print("fastest-8 SMs (odd  half):", fast_odd)
print("overlap:", len(set(fast_even) & set(fast_odd)), "/ 8")

# dump per-SM means for plotting
with open("nuca/sm_means.csv", "w") as f:
    f.write("sm,mean_cyc,mean_even,mean_odd\n")
    for s in sms:
        f.write(f"{s},{m_all[s]:.2f},{me[s]:.2f},{mo[s]:.2f}\n")
print("wrote nuca/sm_means.csv")
