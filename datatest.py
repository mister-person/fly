import pandas as pd
import numpy as np

con = pd.read_parquet("Drosophila_brain_model/Connectivity_783.parquet").to_numpy()
neurons = pd.read_csv("Drosophila_brain_model/Completeness_783.csv")
print(len(con))
print(con[0])

exc = set()
inh = set()
n = set()
syn_count = 0
for a in con[:, :]:
    syn_count += a[4]
    n.add((a[0], a[2]))
    if a[5] == 1:
        exc.add((a[0], a[2]))
    elif a[5] == -1:
        inh.add((a[0], a[2]))
    else:
        print(a)

print(len(n))
print(len(exc), len(inh), len(exc) + len(inh))
print("total synapses:", syn_count)

print(con[:, 4][:20])
by_count = np.argsort(con[:, 4])
for x in con[by_count[-40:]][::-1]:
    print(list(x))

print(exc.intersection(inh))
