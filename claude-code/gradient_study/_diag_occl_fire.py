"""Does the occlusion rule fire on 3n D seed0, and does it point the right way?

Ground truth at that point: N1@223 must move LATER into (242,275], which needs w(0->1) to
go DOWN (243 -> true 200).  So we want g(0->1) to become NEGATIVE.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

C = np.array([[0, 1], [0, 2], [1, 2]], np.int32)
N, OUTS = 3, [2]
params = G.mkparams(520)
T = {n: G.sp(G.fsim(C, N, np.array([200., 1200., 700.], np.float32), params), n)
     for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
w = np.array([243., 940., 379.])

print(f"true w [200,1200,700]   probe w {w.tolist()}   (want g(0->1) NEGATIVE)")
for og in (0.0, 1.0):
    G.OCCL_GAIN = og
    V = G.fsim(C, N, w, params); s = {p: G.sp(V, p) for p in range(N)}
    eps, L, vsub, wreq = G.traces(C, N, w, s, params.steps, {o: T[o] for o in OUTS}, V)
    g = np.zeros(len(w))
    for n in range(N):
        for si in inc[n]:
            g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
    at = [(int(q), round(float(L[1][q]), 5)) for q in s[1]]
    print(f"\n  OCCL_GAIN={og}")
    print(f"    N1={s[1]}  N2={s[2]}")
    print(f"    L[1] at N1's spikes: {at}")
    print(f"    g = [{g[0]:+.3e}, {g[1]:+.3e}, {g[2]:+.3e}]")
    print(f"    g(0->1) sign: {'NEGATIVE (correct -> N1 later)' if g[0] < 0 else 'positive (WRONG -> N1 earlier)'}")
