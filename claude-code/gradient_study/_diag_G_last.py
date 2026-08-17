"""4n G's remaining failures at the current defaults (KICK_GAIN=10, KICK_STALL=2): where
do they end up and why?

    N0 --250--> N1 --500--> N2 --700--> N3        (N3 = output)
    N0 ------------1200-------------->  N3
    true  N1=[173,373]  N2=[244,444]  N3=[33,133,233,291,333,433,491]

Track the live trajectory (the return value is a KEEP_BEST fossil), the count states
visited, whether the kick fires, and the gradient at rest.
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
params = G.mkparams(520)
T = {n: G.sp(G.fsim(C, N, np.array(TRUE, np.float32), params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}

# which seeds fail at current defaults?
fails = []
for sd in range(8):
    w0 = (np.array(TRUE, np.float32) *
          np.random.default_rng(sd).uniform(0.5, 1.5, 4)).astype(float)
    w = G.train(C, N, [3], w0.copy(), T, params, rounds=3200, lr=G.LR)
    if G.sp(G.fsim(C, N, w, params), 3) != T[3]:
        fails.append(sd)
print(f"true {TRUE}  N1={T[1]} N2={T[2]}")
print(f"target N3={T[3]}")
print(f"KICK_GAIN={G.KICK_GAIN} KICK_STALL={G.KICK_STALL}  failing seeds {fails}\n")

for seed in fails:
    w0 = (np.array(TRUE, np.float32) *
          np.random.default_rng(seed).uniform(0.5, 1.5, 4)).astype(float)
    hist = []

    def cb(it, w, upd, g, spall, vsub, L):
        hist.append((it, w.copy(), float(np.abs(upd).max()), float(np.abs(g).max()),
                     len(spall[1]), len(spall[2]), len(spall[3])))

    w = G.train(C, N, [3], w0.copy(), T, params, rounds=3200, lr=G.LR, cb=cb)
    live = hist[-1][1]
    states = [(h[4], h[5], h[6]) for h in hist]
    # when was the count last correct?
    good = [i for i, st in enumerate(states) if st == (2, 2, 7)]
    near = [i for i, st in enumerate(states) if st[:2] == (2, 2)]
    tail = np.array([h[1] for h in hist[-500:]])
    travel = float(np.abs(tail[-1] - tail[0]).max())
    V = G.fsim(C, N, live, params); s = {p: G.sp(V, p) for p in range(N)}
    eps, L, vs, wr = G.traces(C, N, live, s, params.steps, {3: T[3]}, V)
    g = np.zeros(4)
    for n in range(N):
        for si in inc[n]:
            g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
    print(f"seed{seed}: start {np.round(w0,0).tolist()}")
    print(f"   LIVE end {np.round(live,0).tolist()}   last-500 travel {travel:.2f}")
    print(f"   N1={s[1]}  N2={s[2]}")
    print(f"   N3={s[3]}")
    print(f"   count states visited: {sorted(set(states))}")
    print(f"   iterations with hidden counts (2,2): {len(near)}   fully correct: {len(good)}")
    print(f"   g at rest = {np.array2string(g, precision=2)}")
    for i in range(4):
        need = TRUE[i] - live[i]
        d = "->" if abs(need) > 15 else "ok"
        print(f"      {EDGE[i]}: {live[i]:7.1f}  true {TRUE[i]:7.1f}  need {need:+7.1f} {d}"
              f"   g {g[i]:+.2e} {'CORRECT' if g[i]*need > 0 else ('zero' if g[i]==0 else 'WRONG')}")
    print()
