"""How many requested times does the sharpening actually select?"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from _diag import CASES
import grad_trace as G

G.SHARP_MULTI = 1
G.SHARP_DEBUG = 1
for name in ("3n D", "over-demand"):
    E, N, outs, Wl = CASES[name]
    C = np.array(E, np.int32); params = G.mkparams(520)
    T = {n: G.sp(G.fsim(C, N, np.array(Wl, np.float32), params), n) for n in range(N)}
    for seed in (2, 3):
        w = (np.array(Wl, np.float32) * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
        G.LAST_SHARP.clear()
        V = G.fsim(C, N, w, params); s = {p: G.sp(V, p) for p in range(N)}
        G.traces(C, N, w, s, params.steps, {o: T[o] for o in outs}, V)
        hid = [n for n in range(N) if n not in outs and n != 0]
        print(f"{name} seed{seed}: hidden {hid}  true " +
              ", ".join(f"N{n}={T[n]}" for n in hid))
        for n in hid:
            got = G.LAST_SHARP.get(n)
            if got is None:
                print(f"      N{n}: NO sharpening entry (R[n].max()<=0?)")
            else:
                taus, npool, cands = got
                print(f"      N{n}: taus={taus}  pools={npool}  candidates={cands[:12]}"
                      f"{'...' if len(cands) > 12 else ''}  ({len(cands)} total)")
