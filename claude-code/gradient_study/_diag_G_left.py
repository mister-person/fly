"""4n G's three remaining failures (seeds 2, 3, 5) with KICK_GAIN on.

Are they still FREEZING (kick fires and does not help), or failing some other way?  Track
how often the kick fires, whether the run keeps moving, and where it ends up.
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
print(f"true w {TRUE}   N1 {T[1]}  N2 {T[2]}  N3 {T[3]}")
print(f"W_CRIT {G.W_CRIT:.1f}  KICK_GAIN {G.KICK_GAIN}\n")

for seed in (2, 3, 5):
    w0 = (np.array(TRUE, np.float32) *
          np.random.default_rng(seed).uniform(0.5, 1.5, 4)).astype(float)
    hist = []

    def cb(it, w, upd, g, spall, vsub, L):
        hist.append((it, w.copy(), float(np.abs(upd).max()),
                     len(spall[1]), len(spall[2]), len(spall[3])))

    w = G.train(C, N, [3], w0.copy(), T, params, rounds=3200, lr=G.LR, cb=cb)
    V = G.fsim(C, N, w, params)
    live = hist[-1]
    frozen = sum(1 for h in hist if h[2] == 0.0)
    # how much does the live weight vector move over the last 500 iterations?
    tail = np.array([h[1] for h in hist[-500:]])
    travel = float(np.abs(tail[-1] - tail[0]).max()) if len(tail) > 1 else 0.0
    counts = [(h[3], h[4], h[5]) for h in hist]
    seen = sorted(set(counts))
    print(f"seed{seed}: start {np.round(w0,0).tolist()}")
    print(f"   live end w={np.round(live[1],0).tolist()}  (N1,N2,N3 counts {live[3]},{live[4]},{live[5]})")
    print(f"   zero-update iterations: {frozen}/{len(hist)}   last-500 travel: {travel:.2f}")
    print(f"   distinct (N1,N2,N3) count states visited: {seen[:6]}{'...' if len(seen)>6 else ''}")
    print(f"   returned w={np.round(w,0).tolist()}  N3={G.sp(V,3)}")
    print(f"   target                                  {T[3]}")
    print(f"   w(1->2) live {live[1][1]:.0f} vs W_CRIT {G.W_CRIT:.1f} -> "
          f"{'SUB' if live[1][1] < G.W_CRIT else 'supra'}\n")
