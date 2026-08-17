"""8n M stuck: reviving N3/N4 should immediately improve the loss.  Why doesn't it happen?

Check three things at the stuck point:
  (a) how much would the loss actually improve if N3/N4 came back?
  (b) is there any DEMAND on N3/N4 at all (L[3], L[4])?
  (c) is there any GRADIENT on their input edges w(0->3), w(0->4)?
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
W = np.array(Wl, np.float32)
T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
w = np.array([489., 529., 129., 135., 107., 315., 20., 29.,
              20., 422., 20., 28., 33., 381., 20., 24.])


def err(ww):
    V = G.fsim(C, N, np.asarray(ww, np.float32), params)
    tot = 0.0
    for o in outs:
        f, t = G.sp(V, o), T[o]
        tot += 99.0 if len(f) != len(t) else float(np.mean([abs(a - b) for a, b in zip(f, t)]))
    return tot / len(outs)


V = G.fsim(C, N, w, params)
s = {n: G.sp(V, n) for n in range(N)}
print(f"stuck w = {[int(x) for x in w]}")
print(f"   N3 {s[3]} (true {T[3]})   N4 {s[4]} (true {T[4]})")
print(f"   output error {err(w):.2f}\n")

print("(a) scan w(0->3) alone -- when does N3 revive, and does the loss improve?")
prev = None
for x in range(120, 401, 10):
    ww = w.copy(); ww[2] = float(x)
    V2 = G.fsim(C, N, np.asarray(ww, np.float32), params)
    n3 = G.sp(V2, 3)
    e = err(ww)
    if (len(n3), round(e, 2)) != prev:
        print(f"   w(0->3)={x:4d}  N3 fires {len(n3):2d}x {n3[:4]}  output err {e:6.2f}")
        prev = (len(n3), round(e, 2))

print("\n    same for w(0->4):")
prev = None
for x in range(120, 401, 10):
    ww = w.copy(); ww[3] = float(x)
    V2 = G.fsim(C, N, np.asarray(ww, np.float32), params)
    n4 = G.sp(V2, 4)
    e = err(ww)
    if (len(n4), round(e, 2)) != prev:
        print(f"   w(0->4)={x:4d}  N4 fires {len(n4):2d}x {n4[:4]}  output err {e:6.2f}")
        prev = (len(n4), round(e, 2))

print("\n    both revived together (w(0->3)=250, w(0->4)=200 = TRUE):")
ww = w.copy(); ww[2] = 250.; ww[3] = 200.
V2 = G.fsim(C, N, np.asarray(ww, np.float32), params)
print(f"       N3 {G.sp(V2,3)}  N4 {G.sp(V2,4)}   output err {err(ww):.2f}")

eps, L, vsub, wreq = G.traces(C, N, w, s, params.steps, {o: T[o] for o in outs}, V)
g = np.zeros(len(w))
for n in range(N):
    for si in inc[n]:
        g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
print("\n(b) demand on the dead neurons:")
for n in (3, 4):
    nz = np.nonzero(L[n])[0]
    print(f"   L[{n}]: {len(nz)} nonzero, max|L| {np.abs(L[n]).max():.3e}"
          + (f"  at {nz.tolist()[:6]}" if len(nz) else "   <- EMPTY"))
print("\n(c) gradient on their input edges:")
for si, lbl in ((2, "w(0->3)"), (3, "w(0->4)")):
    print(f"   {lbl}: w={w[si]:.0f} true={Wl[si]:.0f} need {Wl[si]-w[si]:+.0f}"
          f"   g={g[si]:+.3e}   eps sum {eps[(0, int(C[si,1]))].sum():.3e}")
