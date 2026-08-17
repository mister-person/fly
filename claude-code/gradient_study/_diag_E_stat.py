"""3n E: truth is not a stationary point.  g = [4.95e-06, 0, 0] with the output EXACT.

If every target is hit and nothing is spurious, the output's demand L[2] should be
identically zero and nothing can reach N1.  Find which term is nonzero.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

C = np.array([[0, 1], [0, 2], [1, 2]], np.int32)
N, OUTS, Wl = 3, [2], [260., 1200., 950.]
params = G.mkparams(520)
W = np.array(Wl)
T = {n: G.sp(G.fsim(C, N, np.array(Wl, np.float32), params), n) for n in range(N)}
V = G.fsim(C, N, W, params)
s = {p: G.sp(V, p) for p in range(N)}
eps, L, vsub, wreq = G.traces(C, N, W, s, params.steps, {2: T[2]}, V)

print(f"true w {Wl}")
print(f"   N1 {s[1]}   (true {T[1]})")
print(f"   N2 {s[2]}")
print(f"   tgt {T[2]}   exact = {s[2] == T[2]}\n")

for n in (2, 1):
    nz = np.nonzero(L[n])[0]
    print(f"L[{n}]: {len(nz)} nonzero  max|L|={np.abs(L[n]).max():.3e}")
    for t in nz[:12]:
        print(f"     t={t:4d}  L={L[n][t]:+.4e}")

print(f"\nthreshold {G.TH:.4e}")
print("vsub[2] at each TARGET t and at t-1  (the hinge reads both):")
for t in T[2]:
    v0 = vsub[2][t] if 0 <= t < params.steps else float('nan')
    v1 = vsub[2][t - 1] if 0 <= t - 1 < params.steps else float('nan')
    up = max(0.0, G.TH - v0)
    dn = min(0.0, G.TH - v1)
    flag = ""
    if up != 0:
        flag += "  UNDER-DRIVEN at t"
    if dn != 0:
        flag += "  ALREADY ABOVE th at t-1"
    print(f"   t={t:4d}  vsub(t)={v0:.4e}  vsub(t-1)={v1:.4e}"
          f"   hinge_up={up:+.2e} hinge_dn={dn:+.2e}{flag}")

print("\nsimulated V (what the simulator actually did) at the same points:")
for t in T[2]:
    print(f"   t={t:4d}  V(t)={float(V[t,2]):.4e}  V(t-1)={float(V[t-1,2]):.4e}")
