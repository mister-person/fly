"""The precomputed refractory-shadow mask must reproduce the per-reset loop EXACTLY."""
import os, sys, importlib
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

# 1. the mask itself, against the loop it replaces, on random spike trains
os.environ["F_NO_OCC_MASK"] = "0"
import field_trace as F
importlib.reload(F)
rng = np.random.default_rng(0)
bad = 0
for trial in range(300):
    T = int(rng.integers(200, 1100))
    rs = sorted(rng.choice(np.arange(0, T), size=int(rng.integers(0, 40)), replace=False).tolist())
    qs = np.arange(0, T - F.DELAY_ITERS)
    arr = qs + F.DELAY_ITERS
    ref = np.ones(len(qs), bool)
    for r in rs:
        ref &= ~((arr > r) & (arr < r + F.REFRAC_ITERS))
    got = ~F._occ_mask(rs, T)[arr]
    if not np.array_equal(ref, got):
        bad += 1
print(f"mask vs loop over 300 random spike trains: {'IDENTICAL' if bad == 0 else str(bad)+' MISMATCHES'}")

# 2. end to end, fields and gradients, mask path against loop path
def run(case, seed, nomask, dens, fast):
    os.environ["F_NO_OCC_MASK"] = str(nomask)
    os.environ["F_DENSITY"] = str(dens); os.environ["F_FASTPROP"] = str(fast)
    import field_trace as FF
    importlib.reload(FF)
    from _diag import CASES, steps_for
    E, N, outs, Wl = CASES[case]
    C = np.array(E, np.int32); p = FF.mkparams(steps_for(case))
    W = np.array(Wl, np.float32)
    T = {n: FF.sp(FF.fsim(C, N, W, p), n) for n in range(N)}
    w = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    sp = {n: FF.sp(FF.fsim(C, N, np.asarray(w, np.float32), p), n) for n in range(N)}
    g, Fl, L, ep = FF.gradient(C, N, w, sp, p.steps, {o: T[o] for o in outs})
    return Fl, g, N

worst = 0.0
for case in ("4n F", "3n L", "4n V", "8n M", "14n Q", "50n A"):
    for dens, fast in ((0, 0), (0, 1), (1, 1), (4, 0)):
        for seed in (0, 3):
            A, ga, N = run(case, seed, 1, dens, fast)
            B, gb, _ = run(case, seed, 0, dens, fast)
            d = max(float(np.abs(np.asarray(A[n]) - np.asarray(B[n])).max()) for n in range(N))
            dg = float(np.abs(ga - gb).max())
            worst = max(worst, d, dg)
    print(f"  {case:<6}: worst so far {worst:.3e}")
print(f"\nWORST DIFFERENCE, mask vs loop, anywhere: {worst:.3e}")
