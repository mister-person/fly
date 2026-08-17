"""*** CHEATING *** upper bound: train with the TRUE hidden spike times as targets."""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp
from _suite_mp import CASES


def _job(a):
    name, E, N, outs, Wl, seed, oracle = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    C = np.array(E, np.int32); params = G.mkparams(520); W = np.array(Wl, np.float32)
    T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
    G.ORACLE_T = ({n: T[n] for n in range(N) if n not in outs and n != 0}
                  if oracle else {})
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    w = G.train(C, N, outs, w0.copy(), T, params, rounds=3200, lr=G.LR)
    V = G.fsim(C, N, w, params)
    return name, oracle, seed, all(G.sp(V, o) == T[o] for o in outs)


def main():
    jobs = [(nm, E, N, o, W, s, orc) for (nm, E, N, o, W) in CASES
            for s in range(8) for orc in (0, 1)]
    with mp.get_context("spawn").Pool(16) as p:
        res = p.map(_job, jobs)
    tot = {0: 0, 1: 0}
    print(f"{'case':12s} {'normal':>8} {'ORACLE(cheat)':>15}")
    for nm, _E, _N, _o, _W in CASES:
        a = sum(ok for n, orc, s, ok in res if n == nm and not orc)
        b = sum(ok for n, orc, s, ok in res if n == nm and orc)
        tot[0] += a; tot[1] += b
        flag = "  <-- oracle WORSE" if b < a else ""
        print(f"{nm:12s} {a:>6}/8 {b:>13}/8{flag}")
    print(f"{'TOTAL':12s} {tot[0]:>6}/72 {tot[1]:>13}/72")


if __name__ == "__main__":
    main()
