"""Which components does 4n G actually need?  Add back one at a time from MINIMAL.

The single-point ablation at the resting point said REQ_GAIN contributes ~nothing
(+4.196e-07 -> +4.552e-07), but MINIMAL (request off) scores 0/8 while the full config
scores 8/8.  A point ablation measures the gradient AT a state; it says nothing about
whether that state is ever reached.  So measure by running.
"""
import os, sys, itertools
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp

E, N, OUTS, TRUE = [[0, 1], [1, 2], [0, 3], [2, 3]], 4, [3], [250., 500., 1200., 700.]
BASE = {"REQ_GAIN": 0.0, "SHARP_GAIN": 0.0, "MOVE_GAIN": 0.0, "OCCL_GAIN": 0.0,
        "TRUST": 5.0, "LN_RELOC": 1, "OCCL_MASK": 1}
ON = {"REQ_GAIN": 0.3, "SHARP_GAIN": 1.0, "MOVE_GAIN": 0.25, "OCCL_GAIN": 1.0}
COMPS = list(ON)


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
    cfgs = []
    labels = {}
    for r in range(len(COMPS) + 1):
        for combo in itertools.combinations(COMPS, r):
            c = dict(BASE)
            for k in combo:
                c[k] = ON[k]
            cfgs.append(c)
            labels[tuple(sorted(c.items()))] = "+".join(combo) if combo else "MINIMAL"
    jobs = [(c, s) for c in cfgs for s in range(8)]
    with mp.get_context("spawn").Pool(16) as p:
        res = p.map(_job, jobs)
    agg = {}
    for key, seed, ok in res:
        agg.setdefault(key, []).append(ok)
    rows = sorted(((sum(v), labels[k]) for k, v in agg.items()), reverse=True)
    for n, lab in rows:
        print(f"   {n}/8   {lab}")


if __name__ == "__main__":
    main()
