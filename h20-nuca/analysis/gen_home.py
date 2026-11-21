import numpy as np
sm, ln, cyc = [], [], []
for line in open("nuca/nuca_atomic.csv"):
    a,b,c = line.strip().split(",")
    sm.append(int(a)); ln.append(int(b)); cyc.append(float(c))
sms = sorted(set(sm)); L=512
sidx = {s:i for i,s in enumerate(sms)}
M = np.zeros((L,len(sms)))
for a,b,c in zip(sm,ln,cyc): M[b,sidx[a]]=c
R = M-M.mean(0,keepdims=True); R=R-R.mean()
v1 = np.linalg.svd(R, full_matrices=False)[2][0]
gp = v1>=0
d = M[:,gp].mean(1)-M[:,~gp].mean(1)
y = d>0                      # y=1 -> SM group gp is the slow one for this address
# home partition of each SM: 0 if faster on y=0 addresses
home = {}
for j,s in enumerate(sms):
    m0 = M[~y, j].mean(); m1 = M[y, j].mean()
    home[s] = 0 if m0 < m1 else 1
arr = [home.get(i,0) for i in range(78)]
print("counts:", arr.count(0), arr.count(1))
print("static const int home_table[78] = {" + ",".join(str(v) for v in arr) + "};")
