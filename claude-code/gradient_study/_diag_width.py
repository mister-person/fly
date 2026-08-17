"""How WIDE is the exact-recovery set in each weight, per case?

If a case is exact only on a near-zero-measure set, requiring an exact output match is a
needle hunt regardless of how good the gradient is -- a different complaint from "the
gradient points the wrong way".  Scan each weight at unit resolution with the others held
at truth, and report the width of the exactly-correct interval.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import multiprocessing as mp
from _diag import CASES
import grad_trace as G


def _job(a):
    name, e = a
    E, N, outs, Wl = CASES[name]
    C = np.array(E, np.int32); params = G.mkparams(520)
    T = {n: G.sp(G.fsim(C, N, np.array(Wl, np.float32), params), n) for n in range(N)}
    o = outs[0]
    lo, hi = max(20, int(Wl[e] * 0.3)), min(3000, int(Wl[e] * 2.5) + 2)
    ok = []
    for x in range(lo, hi + 1):
        w = list(Wl); w[e] = float(x)
        if G.sp(G.fsim(C, N, np.array(w, np.float32), params), o) == T[o]:
            ok.append(x)
    return name, e, Wl[e], (min(ok), max(ok), len(ok)) if ok else None


if __name__ == "__main__":
    jobs = [(nm, e) for nm, (E, N, o, W) in CASES.items() for e in range(len(W))]
    with mp.get_context("spawn").Pool(16) as p:
        res = p.map(_job, jobs)
    cur = None
    for name, e, tw, r in res:
        if name != cur:
            print(f"\n=== {name} ===")
            cur = name
        if r is None:
            print(f"    w{e} (true {tw:6.0f}): NEVER exact alone")
        else:
            a, b, n = r
            print(f"    w{e} (true {tw:6.0f}): exact on [{a},{b}]  width {b-a+1}"
                  f"   rel {100.0*(b-a+1)/max(tw,1):.1f}% of true")
