"""Two follow-ups on N1's gradients:
 (A) seed0 WEAK: why is g(1->2)=0 when w(1->2) is the edge that actually needs to grow?
     Hypothesis: N1@223 sits one cycle early, in the epoch BEFORE the 293 target, so it is
     masked out of the 293 epoch and its eligibility for driving 293 reads 0.
 (B) seed2 LATE: the no-flip ENDPOINT decomposes identically with/without flip, so the
     divergence must be earlier.  Evaluate g(0->1) at the seed2 START and a few points and
     see where flip first changes the sign at N1's actual spike.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

C = np.array([[0, 1], [0, 2], [1, 2]], np.int32)
N, OUTS = 3, [2]
params = G.mkparams(520)
T = {n: G.sp(G.fsim(C, N, np.array([200., 1200., 700.], np.float32), params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}

print("=== (A) seed0 WEAK: w=[243,940,379], N1@223, target N2 spike at 293 ===")
w = np.array([243., 940., 379.])
V = G.fsim(C, N, w, params); spall = {p: G.sp(V, p) for p in range(N)}
eps, L, vsub, wreq = G.traces(C, N, w, spall, params.steps, {o: T[o] for o in OUTS}, V)
e12 = eps[(1, 2)]
print(f"   N1 fires {spall[1]}   (true 246; it is one cycle EARLY)")
print(f"   L[2] at target 293 = {L[2][293]:+.3e}   (hinge deficit, wants drive)")
print(f"   eps[(1->2)] at 293 = {e12[293]:.3e}   <- N1's PSP at 293, AFTER epoch masking")
raw = float(w[2]) * G.HK[293 - 223] if 0 <= 293 - 223 < G.KWIN else 0.0
print(f"   UNMASKED w12*h(293-223) = {G.HK[293-223]:.3e} (kernel) -> would be {G.HK[293-223]:.3e}")
print(f"   g(1->2)=dot(L[2],eps12) = {float(np.dot(L[2], e12)):+.3e}")
print(f"   nonzero eps12 near 293: "
      f"{[(int(t), round(float(e12[t]),8)) for t in np.nonzero(e12)[0] if 260 <= t <= 300]}")

print("\n=== (B) seed2 LATE: where does flip first change N1's gradient? ===")
w0 = (np.array([200., 1200., 700.], np.float32) *
      np.random.default_rng(2).uniform(0.5, 1.5, 3)).astype(float)
print(f"   seed2 start w={np.round(w0,0).tolist()}")
for lab, ww in [("start", w0)]:
    for flip in (0, 1):
        G.SHARP_FLIP = flip
        V = G.fsim(C, N, ww, params); s = {p: G.sp(V, p) for p in range(N)}
        eps, L, vs, wr = G.traces(C, N, ww, s, params.steps, {o: T[o] for o in OUTS}, V)
        g01 = float(np.dot(L[1], eps[(0, 1)]))
        nz = np.nonzero(L[1])[0]
        atsp = [(int(q), round(float(L[1][q]), 5)) for q in s[1]]
        print(f"   [{lab} flip={flip}] N1={s[1]}  g(0->1)={g01:+.3e}  "
              f"L[1]@spikes={atsp}  L1nz=[{nz.min() if len(nz) else '-'}..{nz.max() if len(nz) else '-'}]")
