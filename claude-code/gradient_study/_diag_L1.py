"""What actually drives g(0->1) on 3n D?  Decompose the hidden neuron's learning signal
L[1] at seed2's STARTING weights, under several gain settings."""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, grad_trace as G

C = np.array([[0, 1], [0, 2], [1, 2]], np.int32)
N, OUTS = 3, [2]
params = G.mkparams(520)
W = np.array([200., 1200., 700.], np.float32)
T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
w = (W * np.random.default_rng(2).uniform(0.5, 1.5, len(W))).astype(float)

print(f"REQ_GAIN={G.REQ_GAIN} SUPP_GAIN={G.SUPP_GAIN} SHARP_GAIN={G.SHARP_GAIN} "
      f"REQ_SELFNORM={G.REQ_SELFNORM}")
print(f"seed2 start w={np.round(w,0).tolist()}   true N1={T[1]}  true N2={T[2]}\n")

V = G.fsim(C, N, w, params); spall = {p: G.sp(V, p) for p in range(N)}
eps, L, vsub, wreq = G.traces(C, N, w, spall, params.steps, {o: T[o] for o in OUTS}, V)
print(f"actual N1={spall[1]}  actual N2={spall[2]}")

inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
g = np.zeros(len(w))
for n in range(N):
    for si in inc[n]:
        g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
print(f"g = {g}\n")

for n in (1, 2):
    nz = np.nonzero(L[n])[0]
    tot = float(np.abs(L[n]).sum())
    print(f"L[{n}]: {len(nz)} nonzero, sum|L|={tot:.4e}, max={np.abs(L[n]).max():.4e}")
    if len(nz):
        span = f"[{nz.min()}..{nz.max()}]"
        pos = int((L[n][nz] > 0).sum()); neg = int((L[n][nz] < 0).sum())
        print(f"      span {span}   {pos} positive, {neg} negative")
        top = nz[np.argsort(-np.abs(L[n][nz]))[:8]]
        print("      top: " + ", ".join(f"t={t}:{L[n][t]:+.2e}" for t in sorted(top)))
print()
print(f"eps[(0,1)] sum = {eps[(0,1)].sum():.4e}   nonzero at "
      f"{np.nonzero(eps[(0,1)])[0][:5].tolist()}...")
print(f"contribution g(0->1) = dot(L[1], eps[(0,1)]) = {np.dot(L[1], eps[(0,1)]):.4e}")
