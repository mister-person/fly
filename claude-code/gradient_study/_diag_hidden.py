"""Test: does SHARP_GAIN help exactly the cases whose HIDDEN neurons fire ~once?

The sharpening sign-flip (grad_trace.py:701) assumes the neuron should fire ONCE, at tau:
every spike further than SHARP_WIN from tau has its demand sign forced, and any spike
LATER than tau is forced to "needs more drive".  For a hidden neuron with several
legitimate spikes that inflates its input weight without bound.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from _diag import CASES
import grad_trace as G

DELTA = {"chain": 0, "3-cycle": +2, "2-cycle": -1, "over-demand": -2, "3n A": 0, "3n D": +2}
CRIT = G.TH / float(G.HK.max())
print(f"critical weight (single spike can fire) = {CRIT:.1f}\n")
print(f"{'case':12s} {'sharp delta':>11}  hidden neurons: spike counts (true)")
for name, (E, N, outs, Wl) in CASES.items():
    C = np.array(E, np.int32); W = np.array(Wl, np.float32)
    params = G.mkparams(520)
    T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
    hid = [n for n in range(N) if n not in outs and n != 0]
    desc = ", ".join(f"N{n}:{len(T[n])}" for n in hid) or "(none)"
    mx = max((len(T[n]) for n in hid), default=0)
    print(f"{name:12s} {DELTA.get(name,0):>+11d}  {desc:24s} max={mx}")
    for n in hid:
        print(f"{'':26s}N{n} true spikes {T[n]}")
