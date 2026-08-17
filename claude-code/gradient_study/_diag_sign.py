"""Is the direct demand pointed the RIGHT WAY, independent of magnitude?

Adam is per-parameter (mh/sqrt(vh), GLOBAL_NORM=0), so a uniform scale factor mostly
washes out and what survives is the SIGN of each component.  So "12x too hot" may be the
wrong complaint.  Measure sign agreement with (true - w) over random points, old vs new.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

E = [[0, 1], [1, 2], [0, 3], [2, 3]]
C = np.array(E, np.int32)
N, TRUE = 4, np.array([250., 500., 1200., 700.])
EDGE = ["w(0->1)", "w(1->2)", "w(0->3)", "w(2->3)"]
params = G.mkparams(520)
T = {n: G.sp(G.fsim(C, N, TRUE.astype(np.float32), params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}


def grad(w, new):
    G.NEW_DEMAND = new
    G.SHARP_GAIN = 0.0 if new else 1.0
    G.OCCL_GAIN = 0.0 if new else 1.0
    G.OCCL_MASK = 0 if new else 1
    G.MOVE_GAIN = 0.0 if new else 0.25
    V = G.fsim(C, N, w, params); s = {p: G.sp(V, p) for p in range(N)}
    eps, L, vs, wr = G.traces(C, N, w, s, params.steps, {3: T[3]}, V)
    g = np.zeros(4)
    for n in range(N):
        for si in inc[n]:
            g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
    return g


rng = np.random.default_rng(0)
pts = [np.array([289., 420., 1262., 538.])]                     # the resting point
pts += [TRUE * rng.uniform(0.5, 1.5, 4) for _ in range(40)]

for tag, new in (("OLD (req+sharp+occl+move)", 0), ("NEW (direct demand)", 1)):
    agree = np.zeros(4); live = np.zeros(4); zero = np.zeros(4)
    for w in pts:
        g = grad(np.array(w, float), new)
        want = TRUE - w
        for i in range(4):
            if abs(want[i]) < 1e-9:
                continue
            if g[i] == 0.0:
                zero[i] += 1
            else:
                live[i] += 1
                agree[i] += float(np.sign(g[i]) == np.sign(want[i]))
    print(f"=== {tag} ===")
    for i in range(4):
        pct = 100.0 * agree[i] / live[i] if live[i] else float('nan')
        print(f"   {EDGE[i]}: correct sign {agree[i]:.0f}/{live[i]:.0f} = {pct:5.1f}%"
              f"   (zero gradient at {zero[i]:.0f}/{len(pts)} points)")
    tot_l, tot_a = live.sum(), agree.sum()
    print(f"   overall {tot_a:.0f}/{tot_l:.0f} = {100.0*tot_a/tot_l:.1f}% correct,"
          f" {zero.sum():.0f} zero-gradient components\n")
