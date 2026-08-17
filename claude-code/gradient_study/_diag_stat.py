"""Stationarity at the TRUE weights under each SHARP variant -- the property that must
survive any change to the demand construction."""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from _diag import CASES
import grad_trace as G

for tag, sg, sf in [("sharp0", 0.0, 1), ("collapse", 1.0, 0), ("collapse+flip", 1.0, 1)]:
    G.SHARP_GAIN = sg; G.SHARP_FLIP = sf
    worst = 0.0; out = []
    for name, (E, N, outs, Wl) in CASES.items():
        C = np.array(E, np.int32); W = np.array(Wl)
        params = G.mkparams(520)
        T = {n: G.sp(G.fsim(C, N, np.array(Wl, np.float32), params), n) for n in range(N)}
        V = G.fsim(C, N, W, params); s = {p: G.sp(V, p) for p in range(N)}
        eps, L, vs, wr = G.traces(C, N, W, s, params.steps, {o: T[o] for o in outs}, V)
        inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
        g = np.zeros(len(W))
        for n in range(N):
            for si in inc[n]:
                g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
        m = float(np.abs(g).max()); worst = max(worst, m)
        out.append(f"{name}={m:.1e}")
    print(f"  {tag:14s} max|g| at truth: {'  '.join(out)}   WORST {worst:.3e}")
