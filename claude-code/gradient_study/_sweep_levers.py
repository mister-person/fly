"""8n M: which lever fixes a correctly-signed but tiny net signal?

Diagnosis: g(0->4) is positive 78% of iterations, never zero, mean +1.72e-08, against
per-iteration updates of ~0.7.  The weight travels ~2000 units for a NET +33 -- a random
walk with a weak drift, not a blocked gradient.  Candidate levers:
    CREATE  raise the signal
    LR      shrink the step (less walk per unit drift)
    BETA1   momentum (average the reversals out)
Report mean output error and how often the two rare accumulators are alive at the end.
"""
import os, sys, itertools
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp

GRID = {"CREATE": [0.3, 1.0], "LR": [2.0, 10.0], "BETA1": [0.9, 0.97]}
SEEDS = 4
KEYS = list(GRID)


def _job(a):
    cfg, seed = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    from _diag import CASES, steps_for
    for k, v in cfg.items():
        setattr(G, k, v)
    E, N, outs, Wl = CASES["8n M"]
    C = np.array(E, np.int32); params = G.mkparams(steps_for("8n M"))
    W = np.array(Wl, np.float32)
    T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    w = G.train(C, N, outs, w0.copy(), T, params, rounds=3200, lr=cfg["LR"])
    V = G.fsim(C, N, w, params)
    ok = all(G.sp(V, o) == T[o] for o in outs)
    err = 0.0
    for o in outs:
        f, t = G.sp(V, o), T[o]
        err += 99.0 if len(f) != len(t) else float(np.mean([abs(a - b) for a, b in zip(f, t)]))
    alive = sum(1 for n in (3, 4) if G.sp(V, n))
    w34 = (float(w[2]), float(w[3]))
    return tuple(sorted(cfg.items())), seed, ok, err / len(outs), alive, w34


def main():
    import numpy as np
    cfgs = [dict(zip(KEYS, c)) for c in itertools.product(*(GRID[k] for k in KEYS))]
    jobs = [(c, s) for c in cfgs for s in range(SEEDS)]
    with mp.get_context("spawn").Pool(16) as p:
        res = p.map(_job, jobs)
    agg = {}
    for key, seed, ok, err, alive, w34 in res:
        agg.setdefault(key, []).append((ok, err, alive, w34))
    rows = []
    for k, v in agg.items():
        d = dict(k)
        rows.append((np.mean([x[1] for x in v]), sum(x[0] for x in v),
                     sum(x[2] for x in v), d,
                     np.mean([x[3][0] for x in v]), np.mean([x[3][1] for x in v])))
    print("true w(0->3)=250  w(0->4)=200\n")
    print(f"{'meanErr':>8} {'exact':>6} {'alive/8':>9}  {'w03':>6} {'w04':>6}   config")
    for e, ok, al, d, a3, a4 in sorted(rows):
        print(f"{e:8.2f} {ok:>6} {al:>9}  {a3:6.0f} {a4:6.0f}   {d}")


if __name__ == "__main__":
    main()
