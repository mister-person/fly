"""4n G seeds 4 and 5: the two that survive every hyperparameter setting.

Best config reaches 6/8 (REQ_GAIN 0.3, TRUST 5, LR 10, MOVE_GAIN 0.25, OCCL_MASK 1) and
several distinct configs reach 6/8 recovering DIFFERENT seed sets, so these two are not a
tuning problem.  Dump the live trajectory (cb hook -- the return value is a KEEP_BEST
fossil), the final spike structure, and the gradient at the resting point.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

E = [[0, 1], [1, 2], [0, 3], [2, 3]]
C = np.array(E, np.int32)
N, TRUE = 4, [250., 500., 1200., 700.]
EDGE = ["w(0->1)", "w(1->2)", "w(0->3)", "w(2->3)"]
G.REQ_GAIN, G.TRUST, G.MOVE_GAIN = 0.3, 5.0, 0.25
params = G.mkparams(520)
T = {n: G.sp(G.fsim(C, N, np.array(TRUE, np.float32), params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
CRIT = G.TH / float(G.HK.max())

print(f"true w {TRUE}   critical {CRIT:.1f}")
print(f"   N1 {T[1]}   N2 {T[2]}   N3 {T[3]}\n")

for seed in (4, 5):
    print(f"########## seed{seed}")
    w0 = (np.array(TRUE, np.float32) *
          np.random.default_rng(seed).uniform(0.5, 1.5, 4)).astype(float)
    print(f"  start {np.round(w0,0).tolist()}")
    hist = []

    def cb(it, w, upd, g, spall, vsub, L):
        hist.append((it, w.copy(), float(np.abs(upd).max()),
                     len(spall[1]), len(spall[2]), len(spall[3])))
        if it % 500 == 0 or it == 1:
            print(f"    it{it:5d} w={np.round(w,0).tolist()} |upd|={np.abs(upd).max():.2e}"
                  f"  N1={spall[1]} N2={spall[2]} N3n={len(spall[3])}")

    w = G.train(C, N, [3], w0.copy(), T, params, rounds=3200, lr=10.0, cb=cb)
    # where did it stop moving?
    frozen_at = None
    for it, ww, u, a, b, c in hist:
        if u == 0.0 and frozen_at is None:
            frozen_at = it
        elif u != 0.0:
            frozen_at = None
    live = hist[-1][1]
    print(f"    LIVE   {np.round(live,0).tolist()}   frozen from it{frozen_at}")
    print(f"    RETURNED (KEEP_BEST) {np.round(w,0).tolist()}")
    for lab, ww in (("LIVE", live), ("RETURNED", w)):
        V = G.fsim(C, N, ww, params); s = {p: G.sp(V, p) for p in range(N)}
        eps, L, vs, wr = G.traces(C, N, ww, s, params.steps, {3: T[3]}, V)
        g = np.zeros(4)
        for n in range(N):
            for si in inc[n]:
                g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
        print(f"    [{lab}] N1={s[1]} N2={s[2]}")
        print(f"           N3={s[3]}")
        print(f"           vs  {T[3]}")
        print(f"           g={np.array2string(g, precision=2)}   "
              f"maxL: L3={np.abs(L[3]).max():.1e} L2={np.abs(L[2]).max():.1e} "
              f"L1={np.abs(L[1]).max():.1e}")
        print(f"           w1={ww[1]:.0f} vs critical {CRIT:.1f} -> "
              f"{'SUB-critical' if ww[1] < CRIT else 'supra'}")
    print()
