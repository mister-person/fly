"""How flat is 3n D seed0 really?  Scan each weight over its whole range and record the
OUTPUT TRAIN, not just the local derivative.  If the objective is piecewise constant with
the true solution behind a discontinuity, no LOCAL rule of any sign can point at it.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

C = np.array([[0, 1], [0, 2], [1, 2]], np.int32)
N, OUTS = 3, [2]
params = G.mkparams(520)
TRUE = [200., 1200., 700.]
T = {n: G.sp(G.fsim(C, N, np.array(TRUE, np.float32), params), n) for n in range(N)}
BASE = [243., 940., 379.]        # seed0 endpoint: N1@223, no 293 spike
NAMES = {0: "w(0->1)", 1: "w(0->2)", 2: "w(1->2)"}


def err(out):
    t = T[2]
    return 99.0 if len(out) != len(t) else float(np.mean([abs(a - b) for a, b in zip(out, t)]))


print(f"target N2 {T[2]}   true w {TRUE}   base (seed0) {BASE}\n")
for e in (2, 0):
    print(f"===== scanning {NAMES[e]} alone, others at seed0 values =====")
    prev = None
    for x in range(20, 3001, 20):
        w = np.array(BASE, float); w[e] = x
        V = G.fsim(C, N, w, params)
        out = G.sp(V, 2); n1 = G.sp(V, 1)
        key = (tuple(out), tuple(n1))
        if key != prev:                      # only print where something CHANGES
            print(f"   {NAMES[e]}={x:5d}  N1={str(n1):18s} N2={str(out):48s} err={err(out):5.2f}")
            prev = key
    print()

print("===== joint scan: does ANY single-weight change reach the target? =====")
for e in (0, 1, 2):
    best = None
    for x in range(20, 3001, 5):
        w = np.array(BASE, float); w[e] = x
        out = G.sp(G.fsim(C, N, w, params), 2)
        ee = err(out)
        if best is None or ee < best[0]:
            best = (ee, x, out)
    print(f"   best over {NAMES[e]}: err={best[0]:.2f} at {best[1]}  -> {best[2]}")
