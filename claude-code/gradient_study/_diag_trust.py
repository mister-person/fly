"""Is the TRUST region throttling 3n D seed2?  Recompute its `worst` per neuron at the
stuck point, and report the slope each spike is divided by."""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, grad_trace as G

C = np.array([[0, 1], [0, 2], [1, 2]], np.int32)
N, OUTS = 3, [2]
params = G.mkparams(520)
W = np.array([200., 1200., 700.], np.float32)
T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
w = np.array([194., 967., 887.])

V = G.fsim(C, N, w, params); spall = {p: G.sp(V, p) for p in range(N)}
eps, L, vsub, wreq = G.traces(C, N, w, spall, params.steps, {o: T[o] for o in OUTS}, V)
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
g = np.zeros(len(w))
for n in range(N):
    for si in inc[n]:
        g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))

# reproduce Adam's first step exactly as train() does at it=1
m = 0.1 * g; v = 0.001 * g * g
mh = m / (1 - 0.9); vh = v / (1 - 0.999)
if G.GLOBAL_NORM:
    vh = np.full_like(vh, float(vh.max()))
step = G.LR / (1.0 + G.DECAY * 1)
prop = step * mh / (np.sqrt(vh) + 1e-18)
print(f"g            = {g}")
print(f"GLOBAL_NORM  = {G.GLOBAL_NORM}   TRUST = {G.TRUST}  SLOPE_FLOOR = {G.SLOPE_FLOOR}")
print(f"prop (pre-trust) = {prop}\n")

worst = 0.0; detail = []
for n in range(N):
    if not spall[n] or len(inc[n]) == 0:
        continue
    sl_tr = np.diff(vsub[n], prepend=vsub[n][0])
    for s_ in spall[n]:
        if not (0 <= s_ < params.steps):
            continue
        dv = sum(prop[si] * eps[(int(C[si, 0]), n)][s_] for si in inc[n])
        raw = float(sl_tr[s_])
        sl = max(abs(raw), G.SLOPE_FLOOR * G.TH)
        ds = abs(dv) / sl
        detail.append((n, s_, dv, raw, sl, ds, abs(raw) < G.SLOPE_FLOOR * G.TH))
        worst = max(worst, ds)

print(f"{'neuron':>7} {'spike':>6} {'dv':>12} {'raw slope':>12} {'used slope':>12} {'pred shift':>11}  floored")
for n, s_, dv, raw, sl, ds, fl in sorted(detail, key=lambda x: -x[5]):
    print(f"{'N'+str(n):>7} {s_:>6} {dv:>12.3e} {raw:>12.3e} {sl:>12.3e} {ds:>11.2f}  {'YES' if fl else ''}")

print(f"\nworst = {worst:.2f}   TRUST = {G.TRUST}   scale factor = {min(1.0, G.TRUST/worst):.4f}")
print(f"w(0->2) step: {prop[1]:+.4f} -> {prop[1]*min(1.0, G.TRUST/worst):+.6f}")
print(f"   at that rate, 967 -> 1200 needs "
      f"{233/abs(prop[1]*min(1.0, G.TRUST/worst)):,.0f} iterations")
