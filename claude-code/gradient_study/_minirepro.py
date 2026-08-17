"""Smallest net that reproduces the hidden-neuron suppression bias.

    python3 _minirepro.py [rounds]

The 50n nets take ~5 min a run, which is too slow to iterate on the actual defect.  The defect
itself needs only three ingredients, none of which requires 50 neurons:

  1. many HIDDEN neurons per OUTPUT, so the guessed targets outvote the ground truth;
  2. SUB-CRITICAL weights, so no neuron is fired by one arrival and the bump-based count guess
     has to infer coincidences rather than read off a chain;
  3. inhibition, since the inhibitory population is what collapses hardest.

So the search reuses _bignets.generate at small N and scores candidates by the diagnostic that
actually matters -- hidden share of suppression mass, and hidden L+/L- -- rather than by
whether they fail.  A case that fails for some unrelated reason is not a reproduction.

Candidates that pass the demand screen are then trained, to confirm they fail the SAME WAY:
weights eroding toward zero on both signs.
"""
import os, sys, json
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import multiprocessing as mp
import field_trace as F
import _bignets as B

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = int(os.environ.get("M_STEPS", "1040"))

# FIRST ATTEMPT, kept as a note: at N=10 with 1-2 outputs the demand signature reproduces
# easily (hidden L+/L- 0.41-0.89, hidden share 89-100%) and the weights DO erode (|w| exc
# 0.83-0.95, inh 0.51-0.87) -- yet count-ok is 1.00 on every candidate.  A biased hidden guess
# is necessary but not sufficient.  With 24 edges and one output the system is so
# underdetermined that many weight settings reproduce the target counts, so a 15% erosion costs
# nothing.  The 50n nets carry 97 target spikes across 10 outputs; there the same erosion has
# nowhere to hide.  So the second search holds the bias fixed and raises CONSTRAINT DENSITY --
# target spikes per weight -- which is the ingredient that was actually missing.


def demand_split(E, N, outs, Wl, seed):
    """(hidden L+/L-, hidden share of L-, output L+/L-) at the perturbed start."""
    C = np.array(E, np.int32)
    p = F.mkparams(STEPS)
    W = np.array(Wl, np.float32)
    T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
    w = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    V = F.fsim(C, N, np.asarray(w, np.float32), p)
    spall = {n: F.sp(V, n) for n in range(N)}
    _F, L, _Lc, _ep, _PR, _Fc = F.build(C, N, w, spall, p.steps,
                                   {o: list(T[o]) for o in outs})
    inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
    o = [0.0, 0.0]; h = [0.0, 0.0]
    for n in range(N):
        if not len(inc[n]):
            continue
        v = L[n]
        s = o if n in outs else h
        s[0] += float(v[v > 0].sum()); s[1] += float(-v[v < 0].sum())
    return (h[0] / max(h[1], 1e-9), h[1] / max(h[1] + o[1], 1e-9),
            o[0] / max(o[1], 1e-9))


def screen(args):
    N, n_out, seed = args
    try:
        E, _N, outs, W, inh = B.generate(seed, N=N, n_start=2, n_out=n_out, fan_in=3,
                                         frac_inh=0.25, w_lo=60.0, w_hi=380.0,
                                         w_start=(650.0, 950.0))
    except Exception:
        return None
    C = np.array(E, np.int32)
    p = F.mkparams(STEPS)
    V = F.fsim(C, N, np.asarray(W, np.float32), p)
    sp = {n: F.sp(V, n) for n in range(N)}
    live = sum(1 for n in range(1, N) if sp[n])
    oc = [len(sp[o]) for o in outs]
    if live < 0.7 * (N - 1) or not all(3 <= c <= 20 for c in oc) or not inh:
        return None
    if max(len(sp[n]) for n in range(1, N)) > 45:
        return None
    hr, hs, orr = demand_split(E, N, outs, W, seed=0)
    return dict(N=N, n_out=n_out, seed=seed, edges=len(E), live=live, oc=oc,
                inh=len(inh), h_ratio=hr, h_share=hs, o_ratio=orr,
                dens=sum(oc) / len(E),          # target spikes per weight
                subcrit=float(np.mean([abs(w) < B.W_CRIT for w in W])), E=E, W=W,
                outs=outs)


def trial(args):
    c, rounds = args
    E, N, outs, Wl = c["E"], c["N"], c["outs"], c["W"]
    C = np.array(E, np.int32)
    p = F.mkparams(STEPS)
    W = np.array(Wl, np.float32)
    T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
    wt = np.asarray(Wl, float)
    exc, inh = wt > 0, wt < 0
    oks, mags, dts, exa = [], [], [], []
    for seed in (0, 1, 2, 3):
        w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
        wf = np.asarray(F.train(C, N, outs, w0.copy(), T, p, rounds=rounds, lr=F.LR), float)
        Vh = F.fsim(C, N, np.asarray(wf, np.float32), p)
        ok, d = [], []
        for o in outs:
            f, t = list(F.sp(Vh, o)), list(T[o])
            ok.append(len(f) == len(t))
            if len(f) == len(t):
                d += [abs(a - b) for a, b in zip(f, t)]
        oks.append(np.mean(ok)); dts.append(np.mean(d) if d else np.nan)
        exa.append(all(ok) and (not d or max(d) == 0))
        mags.append((np.mean(np.abs(wf[exc])) / np.mean(np.abs(wt[exc])),
                     np.mean(np.abs(wf[inh])) / np.mean(np.abs(wt[inh]))))
    m = np.array(mags)
    return (c, float(np.mean(oks)), float(m[:, 0].mean()), float(m[:, 1].mean()),
            float(np.nanmean(dts)) if not np.all(np.isnan(dts)) else float("nan"),
            int(sum(exa)))


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    jobs = [(N, no, s) for N in (14, 18, 22, 26) for no in (4, 6) for s in range(60)]
    with mp.Pool(14) as pool:
        cands = [c for c in pool.map(screen, jobs) if c]
    # the reproduction signature: hidden guess suppression-dominated AND dominant in mass,
    # while the outputs still ask correctly -- that last part is what makes it the same bug
    hits = [c for c in cands if c["h_ratio"] < 1.0 and c["h_share"] > 0.85
            and c["o_ratio"] > 1.0]
    hits.sort(key=lambda c: (c["N"], -c["dens"]))
    print(f"{len(cands)} usable nets, {len(hits)} match the 50n demand signature\n")
    print(f"{'N':>3} {'out':>3} {'seed':>4} {'edges':>5} {'sub%':>5} {'sp/w':>5} | "
          f"{'hid L+/L-':>9} {'hid share':>9} {'out L+/L-':>9}")
    for c in hits[:14]:
        print(f"{c['N']:>3} {c['n_out']:>3} {c['seed']:>4} {c['edges']:>5} "
              f"{100*c['subcrit']:>4.0f}% {c['dens']:>5.2f} | {c['h_ratio']:>9.2f} "
              f"{100*c['h_share']:>8.0f}% {c['o_ratio']:>9.2f}")
    if not hits:
        sys.exit("no candidate matched")
    print(f"\ntraining the {min(6, len(hits))} smallest, {rounds} rounds x 4 seeds ...")
    with mp.Pool(min(6, len(hits))) as pool:
        out = pool.map(trial, [(c, rounds) for c in hits[:6]])
    print(f"\n{'N':>3} {'out':>3} {'seed':>4} {'sp/w':>5} | {'exact':>5} {'count-ok':>8} "
          f"{'|dt|':>6} | {'|w| exc':>7} {'|w| inh':>7}   (repro = low ok, both |w| < 1)")
    for c, ok, me, mi, dt, ex in out:
        print(f"{c['N']:>3} {c['n_out']:>3} {c['seed']:>4} {c['dens']:>5.2f} | "
              f"{ex:>3}/4 {ok:>8.2f} {dt:>6.1f} | {me:>7.3f} {mi:>7.3f}")
    json.dump([[c["N"], c["n_out"], c["seed"], c["E"], c["W"], c["outs"], ok, me, mi, dt, ex]
               for c, ok, me, mi, dt, ex in out], open(f"{HERE}/minirepro.json", "w"))
    print("\nwrote minirepro.json")
