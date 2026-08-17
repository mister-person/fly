"""Trace 3n D seed2 through a real single train() call: is w(0->2) climbing toward 1200
at all, and what is the trust scale doing?"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, grad_trace as G

C = np.array([[0, 1], [0, 2], [1, 2]], np.int32)
N, OUTS = 3, [2]
params = G.mkparams(520)
W = np.array([200., 1200., 700.], np.float32)
T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
w = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(W))).astype(float)
print(f"seed{seed} start w={np.round(w,0).tolist()}  true={W.tolist()}")
print(f"   target N2={T[2]}  N1={T[1]}\n")


def cb(it, w, upd, g, spall, vsub, L):
    if it % 100:
        return
    f = spall[2]
    off = ([a - b for a, b in zip(f, T[2])] if len(f) == len(T[2])
           else f"CNT{len(f)}")
    print(f"  it{it:5d} w={np.round(w,0).tolist()}  g={np.array2string(g, precision=2)}"
          f"  upd={np.array2string(upd, precision=3)}  N1={spall[1]} off={off}")


w = G.train(C, N, OUTS, w, T, params, rounds=1600, lr=G.LR, cb=cb)
V = G.fsim(C, N, w, params)
print(f"\n  FINAL w={np.round(w,0).tolist()} N2={G.sp(V,2)} target={T[2]}")
