"""Does 4n G still reach 8/8 without unconditional GRADE_PROP?  Never tested -- the
ablation held GRADE_PROP=1 fixed throughout."""
import os, sys, itertools
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp
E, N, OUTS, TRUE = [[0, 1], [1, 2], [0, 3], [2, 3]], 4, [3], [250., 500., 1200., 700.]
GRID = {"GRADE_PROP": [0, 1], "OCCL_FROMDEMAND": [0, 1], "REQ_GAIN": [0.3, 3.0]}
KEYS = list(GRID)
def _job(a):
    cfg, seed = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    G.LN_RELOC = 1; G.TRUST = 5.0
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
        agg.setdefault(key, []).append(ok)
    for k, v in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
        print(f"   {sum(v)}/8   {dict(k)}")
if __name__ == "__main__":
    main()
