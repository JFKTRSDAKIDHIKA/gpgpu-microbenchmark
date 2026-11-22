import numpy as np
M = np.array([
[-1,204,203,201,199,202,203,199],
[205,-1,199,197,195,198,200,196],
[202,198,-1,194,192,195,197,193],
[202,198,196,-1,192,195,197,193],
[199,195,194,192,-1,193,194,190],
[202,198,197,195,193,-1,197,193],
[203,199,198,196,194,197,-1,194],
[200,196,194,192,190,193,195,-1]], float)
n = 8
pairs = [(0,1),(2,3),(4,5),(6,7)]
same = [min(M[i,j],M[j,i]) for i,j in pairs]
cross = [min(M[i,j],M[j,i]) for i in range(n) for j in range(i+1,n) if (i,j) not in pairs]
print("consecutive-ID pairs :", [int(x) for x in same], "mean %.1f" % np.mean(same))
print("all other pairs      : mean %.1f  min %d  max %d" % (np.mean(cross), min(cross), max(cross)))
print("row means:", [round(x,1) for x in M[M>-1].reshape(8,-1).mean(1)] if False else [round(np.mean([v for v in M[i] if v>0]),1) for i in range(n)])
print("col means:", [round(np.mean([M[i][j] for i in range(n) if M[i][j]>0]),1) for j in range(n)])
print("overall: min %d max %d mean %.1f spread %.1f%%" % (M[M>0].min(), M[M>0].max(), M[M>0].mean(), (M[M>0].max()/M[M>0].min()-1)*100))
print("symmetry: max |M[i,j]-M[j,i]| =", int(max(abs(M[i,j]-M[j,i]) for i in range(n) for j in range(n) if i!=j)))
