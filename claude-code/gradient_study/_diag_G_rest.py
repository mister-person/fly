"""4n G seed4 resting point: why is there no gradient pushing w(1->2) back UP?

w = [289, 420, 1262, 538];  N1=[144,344]  N2=[376]  N3=[33,133,233,333,422]
target N3 = [33,133,233,291,333,433,491].  The missing mark at 291 needs N2 near 244, and
N1@144 -> dt=100 gives 420*HK[100] ~ 6.5e-3 against th 7.0e-3 -- just short.  So a demand
at N2@244 would give g(1->2) > 0 via eps[(1,2)][244].  Find out what is there.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

E = [[0, 1], [1, 2], [0, 3], [2, 3]]
C = np.array(E, np.int32)
N, TRUE = 4, [250., 500., 1200., 700.]
params = G.mkparams(520)
T = {n: G.sp(G.fsim(C, N, np.array(TRUE, np.float32), params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
w = np.array([289., 420., 1262., 538.])

print(f"w {w.tolist()}   W_CRIT {G.W_CRIT:.1f}   th {G.TH:.4e}")
V = G.fsim(C, N, w, params); s = {p: G.sp(V, p) for p in range(N)}
print(f"  N1={s[1]}  N2={s[2]}  N3={s[3]}")
print(f"  target N3={T[3]}   missing marks: "
      f"{[t for t in T[3] if all(abs(t-q)>5 for q in s[3])]}\n")

print("what N2 would need, to mark 291:")
for dt in (95, 100, 105, 110):
    print(f"   N1@144 + dt={dt} -> N2@{144+dt}:  w1*HK[{dt}] = {w[1]*G.HK[dt]:.4e}"
          f"  {'FIRES' if w[1]*G.HK[dt] >= G.TH else 'short by ' + format(G.TH - w[1]*G.HK[dt], '.2e')}")
need = G.TH / float(G.HK[100])
print(f"   -> w(1->2) needed for N2@244 from N1@144: {need:.1f}  (currently {w[1]:.0f},"
      f" true {TRUE[1]:.0f})\n")

for mask in (1, 0):
    G.OCCL_MASK = mask
    eps, L, vsub, wr = G.traces(C, N, w, s, params.steps, {3: T[3]}, V)
    e12 = eps[(1, 2)]
    g = np.zeros(4)
    for n in range(N):
        for si in inc[n]:
            g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
    nzL2 = np.nonzero(L[2])[0]
    print(f"=== OCCL_MASK={mask} ===")
    print(f"   eps[(1,2)][244] = {e12[244]:.4e}   (nonzero => a demand there WOULD act)")
    print(f"   L[2][244]       = {L[2][244]:+.4e}")
    print(f"   L[2] nonzero at {nzL2.tolist()[:12]}{'...' if len(nzL2) > 12 else ''}")
    if len(nzL2):
        ov = [int(t) for t in nzL2 if e12[t] != 0]
        print(f"   of those, overlapping eps[(1,2)]: {ov[:12] if ov else 'NONE'}")
    print(f"   g = {np.array2string(g, precision=3)}")
    print(f"   vsub[2] at 244 = {vsub[2][244]:.4e}  (th {G.TH:.4e}) -> "
          f"{'below' if vsub[2][244] < G.TH else 'above'}")
    print()
