"""Multi-start over SHARP_GAIN, selected by OBSERVABLE output error.

SHARP_GAIN=0 and SHARP_GAIN=1 fail on largely DISJOINT seeds (chain: {0,2,3,5,6} vs
{1,4,5,6,7}), so running both and keeping the better run is worth ~6/48 over either alone.
This is not cheating: the selection uses only mean |t_found - t_target| on the OUTPUT
neurons, which is computable from the given targets -- the same quantity KEEP_BEST already
uses inside train().  The true weights are never consulted.

SHARP_GAIN CANNOT be passed by env var here: pool workers are REUSED across jobs, and
grad_trace reads its config into module globals at import time, so the second job to land
on a worker finds grad_trace already in sys.modules and silently keeps the FIRST job's
setting (observed: both columns identical, sharp1 scoring 29 instead of its true 30).
Set the module attribute directly after import instead.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp

from _suite_mp import CASES

STARTS = [
    # EVERY start must specify EVERY key that any other start varies.  Workers are reused
    # and grad_trace's config lives in module globals, so a key left unset silently
    # inherits the previous job's value -- that is exactly how "collapse" scored 35 here
    # while scoring 38 standalone (it was picking up no-occl's OCCL_MASK=0).
    ("sharp0",        {"SHARP_GAIN": 0.0, "SHARP_FLIP": 1, "OCCL_MASK": 1, "OCCL_GAIN": 1.0}),
    ("collapse",      {"SHARP_GAIN": 1.0, "SHARP_FLIP": 0, "OCCL_MASK": 1, "OCCL_GAIN": 1.0}),
    ("collapse+flip", {"SHARP_GAIN": 1.0, "SHARP_FLIP": 1, "OCCL_MASK": 1, "OCCL_GAIN": 1.0}),
    ("no-occl",       {"SHARP_GAIN": 1.0, "SHARP_FLIP": 0, "OCCL_MASK": 0, "OCCL_GAIN": 0.0}),
]


def _job(a):
    name, E, N, outs, Wl, seed, rounds, tag, env = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    for _k, _v in env.items():       # module globals, NOT env -- workers are reused
        setattr(G, _k, _v)
    C = np.array(E, np.int32); W = np.array(Wl, np.float32)
    params = G.mkparams(520)
    T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
    w = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(W))).astype(float)
    w = G.train(C, N, outs, w, T, params, rounds=rounds, lr=G.LR)
    V = G.fsim(C, N, w, params)
    # OBSERVABLE score only: how far the produced output train is from the given target
    err = 0.0
    for o in outs:
        f = G.sp(V, o); t = T[o]
        err += (99.0 if len(f) != len(t)
                else float(np.mean([abs(p - q) for p, q in zip(f, t)])))
    exact = all(G.sp(V, o) == T[o] for o in outs)
    return name, seed, tag, err / max(len(outs), 1), exact


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 1600
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = [(nm, E, N, o, W, s, rounds, tag, env)
            for (nm, E, N, o, W) in CASES for s in range(seeds)
            for tag, env in STARTS]
    with mp.get_context("spawn").Pool(16) as pool:
        res = pool.map(_job, jobs)

    per = {}
    for nm, s, tag, err, exact in res:
        per.setdefault((nm, s), {})[tag] = (err, exact)

    tot_sel = 0; tot_best = {t: 0 for t, _ in STARTS}
    for nm, _E, _N, _o, _W in CASES:
        sel = []
        for s in range(seeds):
            d = per[(nm, s)]
            for t, _ in STARTS:
                tot_best[t] += int(d[t][1])
            pick = min(STARTS, key=lambda ts: d[ts[0]][0])[0]   # observable err only
            if d[pick][1]:
                sel.append(s)
        tot_sel += len(sel)
        cols = "  ".join(f"{t}={sum(per[(nm,s)][t][1] for s in range(seeds))}"
                         for t, _ in STARTS)
        print(f"  {nm:12s} {cols}   SELECTED {len(sel)}/{seeds}  ok={sel}")
    print(f"\n   single-setting totals: " +
          "  ".join(f"{t}={v}/{len(CASES)*seeds}" for t, v in tot_best.items()))
    print(f"   MULTI-START (observable selection): {tot_sel}/{len(CASES)*seeds}"
          f"   (rounds={rounds})")


if __name__ == "__main__":
    main()
