"""Field pathway first, then grad_trace to finish the timing.

    python3 _hybrid_suite.py [field_rounds] [grad_rounds] [seeds]

The two pathways fail differently.  field_trace gets the spike COUNT right on 93% of
outputs (mean |Δcount| 0.19) and then sits a few timesteps off -- 3-cycle's six non-exact
seeds average 0.8 steps of error, 8n M is 96% count-correct and 4.1 steps out.  grad_trace
has the timing machinery (the hinge, the signed TIM term, occlusion) but has to find the
count itself.  So: let the field set the structure, then let grad_trace polish it.

Reported against BOTH alone at a matched total budget, since a hybrid that just gets twice
the iterations would look good for the wrong reason.
"""
import os, sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
from _suite_mp import CASES


def _job(args):
    name, E, N, outs, Wl, seed, r_field, r_grad, mode = args
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _diag import steps_for
    import field_trace as F
    import grad_trace as G
    C = np.array(E, np.int32); W = np.array(Wl, np.float32)
    params = F.mkparams(steps_for(name))
    T = {n: F.sp(F.fsim(C, N, W, params), n) for n in range(N)}
    w = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(W))).astype(float)
    if mode == "field":
        w = F.train(C, N, outs, w, T, params, rounds=r_field + r_grad, lr=F.LR)
    elif mode == "grad":
        w = G.train(C, N, outs, np.asarray(w, float), T, params,
                    rounds=r_field + r_grad, lr=G.LR)
    elif mode == "hybrid":                      # field first, then grad_trace
        w = F.train(C, N, outs, w, T, params, rounds=r_field, lr=F.LR)
        w = G.train(C, N, outs, np.asarray(w, float), T, params, rounds=r_grad, lr=G.LR)
    elif mode == "reverse":                     # grad_trace first, then field
        w = G.train(C, N, outs, np.asarray(w, float), T, params, rounds=r_grad, lr=G.LR)
        w = F.train(C, N, outs, np.asarray(w, float), T, params, rounds=r_field, lr=F.LR)
    V = F.fsim(C, N, np.asarray(w, np.float32), params)
    per = []
    for o in outs:
        f, t = F.sp(V, o), T[o]
        dc = len(f) - len(t)
        per.append((dc, [abs(a - b) for a, b in zip(f, t)] if dc == 0 else None))
    exact = all(dc == 0 and max(dt) == 0 for dc, dt in per)
    return name, mode, seed, exact, per


def main():
    import numpy as np
    rf = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    rg = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    modes = [("field", rf, rg), ("grad", rf, rg), ("hybrid", rf, rg),
             ("reverse", rf, rg)]
    jobs = [(nm, E, N, o, W, s, a, b, m)
            for (nm, E, N, o, W) in CASES for s in range(seeds)
            for m, a, b in modes]
    with mp.get_context("spawn").Pool(min(16, len(jobs))) as pool:
        res = pool.map(_job, jobs)

    for m, a, b in modes:
        rows = [r for r in res if r[1] == m]
        tot = sum(1 for r in rows if r[3])
        dcs = [abs(dc) for _n, _m, _s, _e, per in rows for dc, _dt in per]
        hit = [dc == 0 for _n, _m, _s, _e, per in rows for dc, _dt in per]
        dts = [np.mean(dt) for _n, _m, _s, _e, per in rows for dc, dt in per if dc == 0]
        lab = {"field": f"field only ({a+b})", "grad": f"grad_trace only ({a+b})",
               "hybrid": f"HYBRID  field {a} -> grad {b}",
               "reverse": f"REVERSE grad {b} -> field {a}"}[m]
        print(f"  {lab:<34} exact {tot}/{len(rows)}   |Δcount| {np.mean(dcs):.2f}   "
              f"count-ok {100*np.mean(hit):.0f}%   |Δt| {np.mean(dts):.2f}")
    print()
    for nm, _E, _N, _o, _W in CASES:
        line = f"  {nm:<12}"
        for m, _a, _b in modes:
            k = sum(1 for r in res if r[0] == nm and r[1] == m and r[3])
            line += f"  {m[:6]} {k}/{seeds}"
        print(line)


if __name__ == "__main__":
    main()
