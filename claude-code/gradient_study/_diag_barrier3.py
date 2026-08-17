"""Is chain seed2's parking point [456,415,626] actually a trap, and is it JOINT?

A single sub-critical weight recovers from anywhere (see _diag_barrier2.py: 10/10 at both
depths, from as low as 120).  So if [456,415,626] is stuck, it is stuck because SEVERAL
weights are wrong together, not because w(1->2) is below 444.5.
Test: start exactly there, and also start there with each single weight restored to truth.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp

E = [[0, 1], [1, 2], [2, 3]]
N, OUTS, TRUE = 4, [3], [500., 500., 500.]
STUCK = [456., 415., 626.]
NAMES = ["w(0->1)", "w(1->2)", "w(2->3)"]

STARTS = [("stuck point as-is", list(STUCK))]
for i in range(3):
    s = list(STUCK); s[i] = TRUE[i]
    STARTS.append((f"stuck but {NAMES[i]} restored to 500", s))
STARTS.append(("only w(1->2) sub-critical (others true)", [500., 415., 500.]))


def _job(a):
    label, start = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    C = np.array(E, np.int32); params = G.mkparams(520)
    T = {n: G.sp(G.fsim(C, N, np.array(TRUE, np.float32), params), n) for n in range(N)}
    w = G.train(C, N, OUTS, np.array(start, float), T, params, rounds=3200, lr=G.LR)
    V = G.fsim(C, N, w, params)
    return label, start, np.round(w, 0).tolist(), G.sp(V, 3), T[3], \
        {n: G.sp(V, n) for n in range(N)}


def main():
    with mp.get_context("spawn").Pool(len(STARTS)) as p:
        res = p.map(_job, STARTS)
    print(f"true {TRUE}   chain seed2 parks at {STUCK}   critical 444.5\n")
    for label, start, w, got, tgt, s in res:
        ok = got == tgt
        print(f"  {label}")
        print(f"      start {start} -> {w}")
        print(f"      out={got}  {'OK' if ok else 'STUCK   target ' + str(tgt)}")
        if not ok:
            print(f"      N1={s[1]}  N2={s[2]}")


if __name__ == "__main__":
    main()
