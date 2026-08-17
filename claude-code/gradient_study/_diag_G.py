"""4n G: 0/8 in every configuration.  What exactly blocks it?

    N0 --240--> N1 --500--> N2 --700--> N3        (N3 = output)
    N0 --1200-> N3
  true w [250, 500, 1200, 700]
  N1 [173,373]  N2 [244,444]  N3 [33,133,233,291,333,433,491]; marks 291 and 491.

Both hidden weights are SUB/near-critical: w(0->1)=250 < 444.5 so N1 accumulates, and
w(1->2)=500 is only just above 444.5, so N2 fires once per N1 spike but with little margin.
Check: (a) how wide is the region of w(1->2) that keeps N2 alive, (b) is the target
reachable from each seed by any single weight, (c) what the gradient says at the true point
and at a stuck point.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

E = [[0, 1], [1, 2], [0, 3], [2, 3]]
C = np.array(E, np.int32)
N, OUTS = 4, [3]
TRUE = [250., 500., 1200., 700.]
params = G.mkparams(520)
T = {n: G.sp(G.fsim(C, N, np.array(TRUE, np.float32), params), n) for n in range(N)}
CRIT = G.TH / float(G.HK.max())
NAMES = {0: "w(0->1)", 1: "w(1->2)", 2: "w(0->3)", 3: "w(2->3)"}


def sim(w):
    V = G.fsim(C, N, np.array(w, np.float32), params)
    return {n: G.sp(V, n) for n in range(N)}


print(f"critical single-spike weight {CRIT:.1f};  true w {TRUE}")
print(f"   N1 {T[1]}  N2 {T[2]}  N3 {T[3]}\n")

print("=== (a) how much slack does each weight have, others held TRUE? ===")
for e in range(4):
    ok = []
    for x in range(20, 3001, 10):
        w = list(TRUE); w[e] = float(x)
        if sim(w)[3] == T[3]:
            ok.append(x)
    if ok:
        runs = []; st = ok[0]; pv = ok[0]
        for x in ok[1:]:
            if x - pv > 10:
                runs.append((st, pv)); st = x
            pv = x
        runs.append((st, pv))
        width = sum(b - a for a, b in runs)
        print(f"   {NAMES[e]}: exact for {runs}  (true {TRUE[e]:.0f}, total width {width})")
    else:
        print(f"   {NAMES[e]}: NEVER exact alone")

print("\n=== (b) is N2 alive?  w(1->2) vs N2's spike count (others TRUE) ===")
prev = None
for x in range(300, 1201, 25):
    w = list(TRUE); w[1] = float(x)
    s = sim(w)
    key = (len(s[2]), tuple(s[3]))
    if key != prev:
        print(f"   w(1->2)={x:5d}  N1={s[1]} N2={s[2]}  N3={s[3]}")
        prev = key

print("\n=== (c) gradient at TRUE, and at each seed's endpoint ===")
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}


def grad(w):
    w = np.array(w, float)
    V = G.fsim(C, N, w, params); s = {p: G.sp(V, p) for p in range(N)}
    eps, L, vsub, wreq = G.traces(C, N, w, s, params.steps, {o: T[o] for o in OUTS}, V)
    g = np.zeros(len(w))
    for n in range(N):
        for si in inc[n]:
            g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
    return g, s


g, s = grad(TRUE)
print(f"   TRUE      g={np.array2string(g, precision=2)}  (must be 0)")
for seed in range(4):
    w0 = (np.array(TRUE, np.float32) *
          np.random.default_rng(seed).uniform(0.5, 1.5, 4)).astype(float)
    w = G.train(C, N, OUTS, w0.copy(), T, params, rounds=1600, lr=G.LR)
    g, s = grad(w)
    print(f"   seed{seed} w={np.round(w,0).tolist()}  N1={s[1]} N2={s[2]}")
    print(f"          N3={s[3]}  vs {T[3]}")
    print(f"          g={np.array2string(g, precision=2)}")
