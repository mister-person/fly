"""3n D seed2: the output is 5 steps late on every N0-driven spike, w(0->2)=967 vs true
1200.  Single presynaptic, direct edge -- this should be the most identifiable weight in
the net.  Scan it, and read the live gradient at the stuck point."""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, grad_trace as G

C = np.array([[0, 1], [0, 2], [1, 2]], np.int32)
N, OUTS, Wl = 3, [2], [200., 1200., 700.]
params = G.mkparams(520)
W = np.array(Wl, np.float32)
T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}

STUCK = np.array([194., 967., 887.])


def outs_at(w):
    V = G.fsim(C, N, np.asarray(w, float), params)
    return G.sp(V, 1), G.sp(V, 2)


def grad_at(w):
    w = np.asarray(w, float)
    V = G.fsim(C, N, w, params); spall = {p: G.sp(V, p) for p in range(N)}
    eps, L, vsub, wreq = G.traces(C, N, w, spall, params.steps, {o: T[o] for o in OUTS}, V)
    inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
    g = np.zeros(len(w))
    for n in range(N):
        for si in inc[n]:
            g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
    return g, L, vsub, spall


print(f"TRUE     w={Wl}  N1={T[1]}  N2={T[2]}")
n1, n2 = outs_at(STUCK)
print(f"STUCK    w={STUCK.tolist()}  N1={n1}  N2={n2}")
print(f"         offsets {[a - b for a, b in zip(n2, T[2])]}\n")

print("scan w(0->2) with w(0->1), w(1->2) held at the stuck values:")
for x in range(700, 1601, 50):
    w = STUCK.copy(); w[1] = x
    a, b = outs_at(w)
    err = (99.0 if len(b) != len(T[2])
           else float(np.mean([abs(p - q) for p, q in zip(b, T[2])])))
    g, _, _, _ = grad_at(w)
    print(f"   w(0->2)={x:5d}  N2={str(b):46s} err={err:5.2f}  g(0->2)={g[1]:+.3e}")

print("\ngradient at the stuck point:")
g, L, vsub, spall = grad_at(STUCK)
print(f"   g = {g}")
print(f"   (sign convention: update is +g direction via Adam)")
print("\n   L[2] nonzero entries (time: value), and N2 target vs actual:")
nz = np.nonzero(L[2])[0]
for t in nz:
    print(f"      t={t:4d}  L={L[2][t]:+.3e}  vsub={vsub[2][t]:.4e}  th={G.TH:.4e}")
print(f"   N2 actual {spall[2]}")
print(f"   N2 target {T[2]}")
