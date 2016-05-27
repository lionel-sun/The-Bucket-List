import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix, csr_matrix
import sys
import math
import timeit

f = pd.read_csv(sys.argv[1], names=['IN','OUT'], comment="#")
n = f.max().max()+1
cc = f["OUT"].value_counts()
B = dok_matrix((n,n), dtype=float)
for i in range(len(f)):
	x = f.iloc[i,0]
	y = f.iloc[i,1]
	B.update({(x,y) : cc[y]})

M = B.tocsr()

p = np.ones(n)*np.true_divide(1,n)
conv = 0.00000000010
it = 0
a = float(sys.argv[2])
last = 0
err = np.inf
start = timeit.default_timer()

while (err >= conv):
	q = np.copy(p)
	p = a * csr_matrix.dot(M,p) + ((1-a)/n)
	diff=p.sum() - q.sum()
	err = abs(diff-last)
	it += 1
	p= p/sum(p)
	last = diff	

ttime = timeit.default_timer() - start

x = np.unique(p)
y = []

for i in x:
	y.append(len(p[p > i]) + 1)
	
plott = plt.scatter(x,y)
plt.subplot(1,1,1)
plt.loglog(x,y)
fig = plott.get_figure()
fig.savefig("figure1.png")

print(n)
print(M.getnnz())
print(float(M.getnnz()) / (M.shape[1]**2))
print(it)
print(ttime)
print(p.min())
print(p.max())
print(np.average(p))
print(p.sum())
