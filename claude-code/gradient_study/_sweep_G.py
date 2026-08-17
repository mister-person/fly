"""Sweep 4n G alone (graded propagation now unconditional).  Goal: 8/8 here first."""
import os, sys, itertools
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp

E, N, OUTS, TRUE = [[0, 1], [1, 2], [0, 3], [2, 3]], 4, [3], [250., 500., 1200., 700.]
GRID = {"REQ_GAIN": [0.1, 0.2, 0.3], "TRUST": [5.0, 10.0, 20.0], "LR": [5.0, 10.0, 20.0]}
KEYS = list(GRID)


def _job(a):
    cfg, seed = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    for k, v in cfg.items():
        setattr(G, k, v)
    G.MOVE_GAIN = 0.25
    C = np.array(E, np.int32); params = G.mkparams(520)
    W = np.array(TRUE, np.float32)
    T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, 4)).astype(float)
    w = G.train(C, N, OUTS, w0.copy(), T, params, rounds=3200, lr=cfg.get("LR", G.LR))
    return tuple(sorted(cfg.items())), seed, G.sp(G.fsim(C, N, w, params), 3) == T[3]


def main():
    cfgs = [dict(zip(KEYS, c)) for c in itertools.product(*(GRID[k] for k in KEYS))]
    jobs = [(c, s) for c in cfgs for s in range(8)]
    with mp.get_context("spawn").Pool(16) as p:
        res = p.map(_job, jobs)
    agg = {}
    for key, seed, ok in res:
        agg.setdefault(key, []).append((seed, ok))
    for key, v in sorted(agg.items(), key=lambda kv: -sum(o for _, o in kv[1])):
        n = sum(o for _, o in v)
        ok = sorted(s for s, o in v if o)
        print(f"  {n}/8  {dict(key)}  ok={ok}")


if __name__ == "__main__":
    main()
