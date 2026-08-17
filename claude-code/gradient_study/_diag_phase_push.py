"""8n M seed7: nudge the weights so the PHASE improves, then see what pushes back.

seed7 (all counts right) has N5 found [186,386,586,786,943,1033] vs target
[194,343,497,616,794,943].  The 2nd and 3rd found spikes are 43 and 89 steps LATE -- the
found train runs at period 200 while the target's gaps are ~150 -- so fixing them needs
MORE drive (fire earlier), i.e. LARGER fan-in weights.

Move partway from the stuck weights toward the true ones, confirm the phase improves, then
decompose the gradient at that better point: which term wants to undo it?
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G
from _diag import CASES, steps_for

E, N, outs, Wl = CASES["8n M"]
C = np.array(E, np.int32)
params = G.mkparams(steps_for("8n M"))
W = np.array(Wl, float)
T = {n: G.sp(G.fsim(C, N, np.array(Wl, np.float32), params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}

w0 = (np.array(Wl, np.float32) * np.random.default_rng(7).uniform(0.5, 1.5, len(Wl))).astype(float)
live = {}
G.train(C, N, outs, w0.copy(), T, params, rounds=3200, lr=G.LR,
        cb=lambda it, w, *a: live.update(w=w.copy()))
stuck = live["w"]


def show(w, tag):
    V = G.fsim(C, N, np.asarray(w, np.float32), params)
    F = {n: G.sp(V, n) for n in range(N)}
    errs = []
    for o in outs:
        f, t = F[o], T[o]
        errs.append(99.0 if len(f) != len(t)
                    else float(np.mean([abs(a - b) for a, b in zip(f, t)])))
    print(f"  {tag}: N5 {F[5]}")
    print(f"        N3 {F[3][:4]}  N4 {F[4][:4]}   mean|dt| "
          + " ".join(f"{e:.1f}" for e in errs))
    return F, float(np.mean(errs))


print(f"true w {[int(x) for x in W]}")
print(f"target N5 {T[5]}\n")
F0, e0 = show(stuck, "STUCK    ")
for frac in (0.25, 0.5, 0.75, 1.0):
    w = stuck + frac * (W - stuck)
    show(w, f"toward true {frac:>4.2f}")

# pick the nudge that improves phase most, then decompose the gradient there
best = None
for frac in (0.1, 0.2, 0.3, 0.4, 0.5):
    w = stuck + frac * (W - stuck)
    _, e = show(w, f"    scan {frac:>4.2f}") if False else (None, None)
    V = G.fsim(C, N, np.asarray(w, np.float32), params)
    F = {n: G.sp(V, n) for n in range(N)}
    errs = []
    for o in outs:
        f, t = F[o], T[o]
        errs.append(99.0 if len(f) != len(t)
                    else float(np.mean([abs(a - b) for a, b in zip(f, t)])))
    m = float(np.mean(errs))
    if best is None or m < best[0]:
        best = (m, frac, w)
print(f"\nbest nudge: {best[1]:.2f} of the way to truth, mean|dt| {best[0]:.1f} "
      f"(stuck was {e0:.1f})")
w = best[2]
F, _ = show(w, "NUDGED   ")

print("\nGRADIENT AT THE NUDGED POINT -- does it push back toward the stuck weights?")
print("(need = true - w; a gradient with the OPPOSITE sign is undoing the nudge)\n")
TERMS = [("full", {}),
         ("no MOVE_GAIN", dict(MOVE_GAIN=0.0)),
         ("no CREATE", dict(CREATE=0.0)),
         ("no TIM_GAIN", dict(TIM_GAIN=0.0)),
         ("no suppression (SUPP_FIX=0)", dict(SUPP_FIX=0)),
         ("no OCCL", dict(OCCL_MASK=0, OCCL_GAIN=0.0))]
saved = {k: getattr(G, k) for k in
         ("MOVE_GAIN", "CREATE", "TIM_GAIN", "SUPP_FIX", "OCCL_MASK", "OCCL_GAIN")}
V = G.fsim(C, N, np.asarray(w, np.float32), params)
s = {n: G.sp(V, n) for n in range(N)}
fanin = [si for si in range(len(Wl)) if int(C[si, 1]) in outs]
# the FAN-OUT edges (N0->N1..N4) are the ones driven by HIDDEN demand, so they are where
# CREATE / TIM_GAIN act; the fan-in edges see only the output seeding.
fanout = [si for si in range(len(Wl)) if int(C[si, 1]) not in outs]
for tag, over in TERMS:
    for k, v in saved.items():
        setattr(G, k, v)
    for k, v in over.items():
        setattr(G, k, v)
    eps, L, vs, wr = G.traces(C, N, w, s, params.steps, {o: T[o] for o in outs}, V)
    g = np.zeros(len(w))
    for n in range(N):
        for si in inc[n]:
            g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
    need = W - w
    def rep(idx, lbl):
        a = sum(1 for si in idx if g[si] * need[si] > 0)
        o_ = sum(1 for si in idx if g[si] * need[si] < 0)
        z = sum(1 for si in idx if g[si] == 0)
        return (f"{lbl} {a}a/{o_}O/{z}z  sum {float(np.dot(g[idx], need[idx])):+.2e}")
    print(f"  {tag:28s} " + rep(fanout, "FANOUT") + "   " + rep(fanin, "fanin"))
    for si in fanout:
        print(f"        w(N{int(C[si,0])}->N{int(C[si,1])}) {w[si]:6.1f} need "
              f"{need[si]:+7.1f}  g {g[si]:+.2e} "
              f"{'agree' if g[si]*need[si]>0 else ('OPPOSE' if g[si]*need[si]<0 else 'zero')}")
for k, v in saved.items():
    setattr(G, k, v)
