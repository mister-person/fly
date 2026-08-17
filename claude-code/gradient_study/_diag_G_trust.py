"""4n G seeds 5/7: the demand arrives amplified and correctly signed, yet the weights
barely move.  Is the TRUST region throttling the step -- and is the reported endpoint even
the live one, or another KEEP_BEST fossil?
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
EDGE = ["w(0->1)", "w(1->2)", "w(0->3)", "w(2->3)"]

for seed in (5, 7):
    print(f"########## seed{seed}")
    w0 = (np.array(TRUE, np.float32) *
          np.random.default_rng(seed).uniform(0.5, 1.5, 4)).astype(float)
    print(f"  start {np.round(w0,0).tolist()}   true {TRUE}")

    def cb(it, w, upd, g, spall, vsub, L):
        if it % 400:
            return
        print(f"    it{it:5d} w={np.round(w,0).tolist()} |upd|={np.abs(upd).max():.3e}"
              f"  N1={spall[1]} N2={spall[2]}  N3n={len(spall[3])}")

    w = G.train(C, N, [3], w0.copy(), T, params, rounds=3200, lr=G.LR, cb=cb)
    print(f"    RETURNED (KEEP_BEST) {np.round(w,0).tolist()}")

    # trust-region analysis at the returned point
    V = G.fsim(C, N, w, params); s = {p: G.sp(V, p) for p in range(N)}
    eps, L, vsub, wreq = G.traces(C, N, w, s, params.steps, {3: T[3]}, V)
    g = np.zeros(4)
    for n in range(N):
        for si in inc[n]:
            g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
    m = 0.1 * g; v = 0.001 * g * g
    mh = m / (1 - 0.9); vh = v / (1 - 0.999)
    step = G.LR / (1.0 + G.DECAY * 1)
    prop = step * mh / (np.sqrt(vh) + 1e-18)
    worst = 0.0; who = None
    for n in range(N):
        if not s[n] or len(inc[n]) == 0:
            continue
        sl_tr = np.diff(vsub[n], prepend=vsub[n][0])
        for s_ in s[n]:
            if not (0 <= s_ < params.steps):
                continue
            dv = sum(prop[si] * eps[(int(C[si, 0]), n)][s_] for si in inc[n])
            sl = max(abs(float(sl_tr[s_])), G.SLOPE_FLOOR * G.TH)
            ds = abs(dv) / sl
            if ds > worst:
                worst, who = ds, (n, s_, sl)
    sc = min(1.0, G.TRUST / worst) if worst > 0 else 1.0
    print(f"    g        = {np.array2string(g, precision=2)}")
    print(f"    prop     = {np.array2string(prop, precision=3)}   (pre-trust)")
    print(f"    worst predicted shift {worst:.2f} steps from N{who[0]}@{who[1]}"
          f" (slope {who[2]:.2e})   TRUST={G.TRUST}")
    print(f"    trust scale = {sc:.5f}   -> actual step {np.array2string(prop*sc, precision=4)}")
    for i in range(4):
        need = TRUE[i] - w[i]
        st = prop[i] * sc
        it_needed = abs(need / st) if st != 0 else float('inf')
        print(f"      {EDGE[i]}: {w[i]:7.1f} -> {TRUE[i]:7.1f} (need {need:+7.1f}), "
              f"step {st:+.4f}/iter -> {it_needed:,.0f} iters" if st != 0 else
              f"      {EDGE[i]}: {w[i]:7.1f} -> {TRUE[i]:7.1f} (need {need:+7.1f}), STEP ZERO")
    print()
