"""Trace any case/seed under a chosen SHARP_GAIN, showing the HIDDEN neurons too.

usage: _diag_case.py <case> <seed> <sharp_gain> [rounds]
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from _diag import CASES   # dict: name -> (edges, N, outs, w_true)
import grad_trace as G

name = sys.argv[1]; seed = int(sys.argv[2])
G.SHARP_GAIN = float(sys.argv[3])
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 1600

E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32); W = np.array(Wl, np.float32)
params = G.mkparams(520)
T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
w = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(W))).astype(float)

print(f"=== {name} seed{seed}  SHARP_GAIN={G.SHARP_GAIN}  edges={E} ===")
print(f"   true w={Wl}   start w={np.round(w,0).tolist()}")
for n in range(N):
    print(f"   TRUE N{n}: {T[n]}{'  <- OUT' if n in outs else ''}")
print()

o = outs[0]


def cb(it, w, upd, g, spall, vsub, L):
    if it % 200:
        return
    f = spall[o]
    off = ([a - b for a, b in zip(f, T[o])] if len(f) == len(T[o]) else f"CNT{len(f)}")
    hid = "  ".join(f"N{n}={spall[n]}" for n in range(N) if n != o and n != 0)
    print(f"  it{it:5d} w={np.round(w,0).tolist()} {hid}  off={off}")


w = G.train(C, N, outs, w, T, params, rounds=rounds, lr=G.LR, cb=cb)
V = G.fsim(C, N, w, params)
f = G.sp(V, o)
print(f"\n  FINAL w={np.round(w,0).tolist()}  out={f}")
print(f"        target={T[o]}   {'EXACT' if f == T[o] else 'MISS'}")
for n in range(N):
    if n != o:
        print(f"        N{n}: got {G.sp(V,n)}  true {T[n]}")
