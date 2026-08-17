"""Why does MOVE_GAIN cost over-demand 7/8 -> 5/8?

over-demand: edges [[0,1],[1,2],[0,2]], out N2, true w [250,700,300].
N1 is a sub-critical accumulator firing TWICE (173,373); output N2 = [140,220,399].
Compare each seed with the timing demand off and on.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp

E, N, OUTS, TRUE = [[0, 1], [1, 2], [0, 2]], 3, [2], [250., 700., 300.]


def _job(a):
    seed, mg = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    G.MOVE_GAIN = mg
    C = np.array(E, np.int32); params = G.mkparams(520)
    W = np.array(TRUE, np.float32)
    T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, 3)).astype(float)
    w = G.train(C, N, OUTS, w0.copy(), T, params, rounds=3200, lr=G.LR)
    V = G.fsim(C, N, w, params)
    return seed, mg, np.round(w0, 0).tolist(), np.round(w, 0).tolist(), \
        {n: G.sp(V, n) for n in range(N)}, T


def main():
    jobs = [(s, mg) for s in range(8) for mg in (0.0, 0.25)]
    with mp.get_context("spawn").Pool(16) as p:
        res = p.map(_job, jobs)
    T = res[0][5]
    print(f"true w {TRUE}   N1 {T[1]} (fires twice)   N2 target {T[2]}\n")
    by = {}
    for seed, mg, w0, w, s, _ in res:
        by.setdefault(seed, {})[mg] = (w0, w, s)
    for seed in range(8):
        d = by[seed]
        r0, r1 = d[0.0], d[0.25]
        ok0 = r0[2][2] == T[2]; ok1 = r1[2][2] == T[2]
        tag = "" if ok0 == ok1 else ("  <-- LOST by MOVE_GAIN" if ok0 else "  <-- GAINED")
        print(f"seed{seed}  start {r0[0]}{tag}")
        for mg, r, ok in ((0.0, r0, ok0), (0.25, r1, ok1)):
            print(f"    MOVE={mg:<5} w={r[1]}  N1={r[2][1]}  N2={r[2][2]}"
                  f"  {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
