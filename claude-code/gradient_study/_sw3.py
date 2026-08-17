"""4n G with the barrier clamp."""
import os, sys, itertools
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp
E, N, OUTS, TRUE = [[0, 1], [1, 2], [0, 3], [2, 3]], 4, [3], [250., 500., 1200., 700.]
GRID = {"OCCL_RELOC": [0, 1], "REQ_GAIN": [0.3, 1.0], "TRUST": [5.0, 10.0]}
KEYS = list(GRID)
def _job(a):
    cfg, seed = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    for k, v in cfg.items():
        setattr(G, k, v)
    C = np.array(E, np.int32); params = G.mkparams(520); W = np.array(TRUE, np.float32)
    T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, 4)).astype(float)
    w = G.train(C, N, OUTS, w0.copy(), T, params, rounds=3200, lr=G.LR)
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
        print(f"  {sum(o for _,o in v)}/8  {dict(key)}  ok={sorted(s for s,o in v if o)}")
if __name__ == "__main__":
    main()
