"""FASTPROP must be EXACT, not merely close: verify the batched propagation against the loop."""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import importlib

def fields(case, seed, fast, dens):
    os.environ["F_FASTPROP"] = str(fast); os.environ["F_DENSITY"] = str(dens)
    import field_trace as F
    importlib.reload(F)
    from _diag import CASES, steps_for
    E, N, outs, Wl = CASES[case]
    C = np.array(E, np.int32); p = F.mkparams(steps_for(case))
    W = np.array(Wl, np.float32)
    T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
    w = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    sp = {n: F.sp(F.fsim(C, N, np.asarray(w, np.float32), p), n) for n in range(N)}
    g, Fl, L, ep = F.gradient(C, N, w, sp, p.steps, {o: T[o] for o in outs})
    return Fl, g, N

worst = 0.0
for case in ("4n F", "3n L", "4n V", "8n M", "14n Q"):
    for dens in (0, 1):
        for seed in (0, 3):
            A, ga, N = fields(case, seed, 0, dens)
            B, gb, _ = fields(case, seed, 1, dens)
            d = max(float(np.abs(np.asarray(A[n]) - np.asarray(B[n])).max()) for n in range(N))
            dg = float(np.abs(ga - gb).max())
            worst = max(worst, d, dg)
            print(f"  {case:<6} dens={dens} seed={seed}: max|dF|={d:.3e}  max|dg|={dg:.3e}")
print(f"\nWORST DIFFERENCE ANYWHERE: {worst:.3e}")
