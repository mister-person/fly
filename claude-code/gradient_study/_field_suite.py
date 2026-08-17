"""Suite runner for the field pathway, with a GRADED error, not just pass/fail.

    python3 _field_suite.py [rounds] [seeds] [case-substring]

Exact recovery is the headline, but it hides most of what a run does.  A case whose output
is one timestep late on every spike and one whose output has the wrong number of spikes both
score zero, and they are nothing like each other -- the first is essentially solved.  So each
output neuron reports two things, in the order they matter:

    Δcount   len(fired) - len(target).  Categorical: you either have the right number of
             spikes or you do not, and no amount of timing work fixes a miscount.
    |Δt|     ONLY when the count matches, the per-spike timing error.  Undefined otherwise,
             and averaging it over count-mismatched outputs would be meaningless, so those
             are excluded and their share is reported separately.

Per case: exact recovery, the mean |Δcount|, what fraction of outputs got the count right,
and the timing error over just those.
"""
import os, sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
from _suite_mp import CASES


def _job(args):
    name, E, N, outs, Wl, seed, rounds = args
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import field_trace as F
    from _diag import steps_for
    C = np.array(E, np.int32); W = np.array(Wl, np.float32)
    params = F.mkparams(steps_for(name))
    T = {n: F.sp(F.fsim(C, N, W, params), n) for n in range(N)}
    w = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(W))).astype(float)
    w = F.train(C, N, outs, w, T, params, rounds=rounds, lr=F.LR)
    V = F.fsim(C, N, np.asarray(w, np.float32), params)
    per = []
    for o in outs:
        f, t = F.sp(V, o), T[o]
        dc = len(f) - len(t)
        dt = ([abs(a - b) for a, b in zip(f, t)] if dc == 0 else None)
        per.append((dc, dt))
    exact = all(dc == 0 and max(dt) == 0 for dc, dt in per)
    return name, seed, exact, per


def main():
    import numpy as np
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    only = sys.argv[3] if len(sys.argv) > 3 else None
    # "!sub" EXCLUDES instead of selecting -- needed to compare a variant across the whole
    # suite minus the slow 50n cases without listing every other case by name
    # A COMMA LIST selects several cases, which is what iterating on a variant needs: the
    # full 18-case suite is ~10 min a run and most of it is cases the change cannot affect.
    # DEV_SET below is the discriminating subset -- the cases where the bump and density
    # pathways disagree most, in both directions.
    DEV_SET = "chain,4n F,3n A,5n H,14n Q"
    if only in ("dev", "DEV"):
        only = DEV_SET
    if only and only.startswith("!"):
        cases = [c for c in CASES if only[1:].lower() not in c[0].lower()]
    elif only and "," in only:
        want = [x.strip().lower() for x in only.split(",") if x.strip()]
        cases = [c for c in CASES if any(x == c[0].lower() for x in want)]
    else:
        cases = [c for c in CASES if only is None or only.lower() in c[0].lower()]
    jobs = [(nm, E, N, o, W, s, rounds)
            for (nm, E, N, o, W) in cases for s in range(seeds)]
    with mp.get_context("spawn").Pool(min(16, len(jobs))) as pool:
        res = pool.map(_job, jobs)

    tot = 0
    g_dc, g_hit, g_dt, g_all = [], [], [], []
    for nm, _E, _N, _o, _W in cases:
        rows = [r for r in res if r[0] == nm]
        ok = sorted(s for _n, s, e, _p in rows if e)
        tot += len(ok)
        dcs = [abs(dc) for _n, _s, _e, per in rows for dc, _dt in per]
        hit = [dc == 0 for _n, _s, _e, per in rows for dc, _dt in per]
        dts = [np.mean(dt) for _n, _s, _e, per in rows for dc, dt in per if dc == 0]
        mxs = [max(dt) for _n, _s, _e, per in rows for dc, dt in per if dc == 0]
        g_dc += dcs; g_hit += hit; g_dt += dts; g_all += mxs
        tim = (f"|Δt| mean {np.mean(dts):5.1f} max {max(mxs):3.0f}" if dts
               else "|Δt| --  (no count-matched output)")
        print(f"  {nm:<12} exact {len(ok)}/{seeds}   Δcount {np.mean(dcs):4.2f}"
              f"  count-ok {100*np.mean(hit):3.0f}%   {tim}   ok={ok}")
    print(f"   TOTAL exact {tot}/{len(jobs)}   mean |Δcount| {np.mean(g_dc):.2f}   "
          f"count-ok {100*np.mean(g_hit):.0f}%   "
          f"|Δt| over those: mean {np.mean(g_dt):.2f} median {np.median(g_dt):.1f} "
          f"max {max(g_all):.0f}   (rounds={rounds})")


if __name__ == "__main__":
    main()
