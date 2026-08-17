"""Does the sub-critical barrier fail at DEPTH?

A sub-critical START recovers fine when the affected neuron is observed or one hop from the
output (_find_barrier.py: A 8/8, B 7/8, every sub-critical seed recovered).  So the barrier
alone is not the failure.  Chain seed2 parks at w(1->2)=415 with the output TWO hops away.
Force the sub-critical value instead of hoping a random seed lands there, at both depths.

MUST be a file, not `python3 -c`: spawn workers re-import __main__, which for -c is empty,
so the pool hangs waiting on workers that cannot find the job function.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp

XS = (120, 200, 280, 360, 420)
NETS = [
    ("B 3n  N0->N1->N2      out=N2, barrier 1 hop from output",
     [[0, 1], [1, 2]], 3, [2], [500., 500.], 1),
    ("C 4n  N0->N1->N2->N3  out=N3, barrier 2 hops from output",
     [[0, 1], [1, 2], [2, 3]], 4, [3], [500., 500., 500.], 1),
]


def _job(a):
    tag, E, N, outs, Wtrue, idx, X = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    C = np.array(E, np.int32); params = G.mkparams(520)
    T = {n: G.sp(G.fsim(C, N, np.array(Wtrue, np.float32), params), n) for n in range(N)}
    start = list(Wtrue); start[idx] = float(X)
    w = G.train(C, N, outs, np.array(start, float), T, params, rounds=3200, lr=G.LR)
    V = G.fsim(C, N, w, params); o = outs[0]
    return tag, X, np.round(w, 0).tolist(), G.sp(V, o), T[o], \
        {n: G.sp(V, n) for n in range(N)}


def main():
    jobs = [(tag, E, N, o, W, idx, X) for (tag, E, N, o, W, idx) in NETS for X in XS]
    with mp.get_context("spawn").Pool(min(16, len(jobs))) as p:
        res = p.map(_job, jobs)
    print("critical single-spike weight 444.5;  true w(1->2)=500\n")
    for tag, *_ in NETS:
        print(f"=== {tag} ===")
        for t, X, w, got, tgt, s in [r for r in res if r[0] == tag]:
            ok = got == tgt
            print(f"    start w(1->2)={X:4d} -> {w}")
            print(f"        out={got}  {'OK' if ok else 'FAIL   target ' + str(tgt)}")
            if not ok:
                print(f"        hidden: " + "  ".join(f"N{n}={s[n]}" for n in sorted(s)
                                                      if n != 0 and n != tgt))
        print()


if __name__ == "__main__":
    main()
