"""Add SUPPRESSION (the missing ingredient) to the local solve and re-test.

Diagnosis: with nudge+robust and TRUE inputs the 50-neuron neurons still fail,
almost entirely by firing EXTRA spikes -> the solve fits the target crossings but
never constrains V_sub < th elsewhere.  Add upper-bound cuts: solve, find the
neuron's extra crossings, require V_sub < (1-m)*th there, re-solve.  (This is what
tp_50neuron's QP did; here we cut iteratively on the closed-form solve.)
"""

import sys, os, dataclasses, types
sys.path.insert(0, "/workspace/project/gradient_study")
sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")
for _n, _a in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        for _k, _v in _a.items():
            setattr(_m, _k, _v)
        sys.modules[_n] = _m
import numpy as np
import jax.numpy as jnp
from homotopy_core import hard_sim as _hard_sim
import grad_robust_recurrent as RR

TH = RR.TH
hk_nud = RR.hk_nud
hk = RR.hk
MARGIN = 0.15


def epoch_of(t, targets):
    prev = 0
    for tt in sorted(targets):
        if t <= tt:
            return prev, tt
        prev = tt
    return prev, None


def vsub_at(t, pre_times_by_syn, w, targets):
    """V_sub at time t, accumulation reset at the most recent target before t."""
    prev, _ = epoch_of(t, targets)
    return sum(w[k] * sum(hk(t - s) for s in pts if prev < s < t)
               for k, pts in enumerate(pre_times_by_syn))


def solve_supp(pre_times_by_syn, targets, lo, hi, rounds=4):
    tg = sorted(targets)
    # base fit rows (nudged) + robust handled inside RR.solve; we add upper cuts here
    up_times = []                       # times where we require V_sub < cap
    w = RR.solve(pre_times_by_syn, targets, lo, hi, robust=True)
    if w is None:
        return None
    n = len(w)
    for _ in range(rounds):
        # find EXTRA crossings of V_sub in [0..last target+50], not within 10 of a target
        cross = []
        for t in range(20, tg[-1] + 60):
            if vsub_at(t, pre_times_by_syn, w, targets) >= TH and \
               vsub_at(t - 1, pre_times_by_syn, w, targets) < TH:
                if all(abs(t - tt) > 10 for tt in tg):
                    cross.append(t)
        new = [c for c in cross if c not in up_times]
        if not new:
            break
        up_times += new
        # build augmented system: fit rows (=th) + upper rows (=(1-m)th), solve LS, clip
        Arows, b = [], []
        prev = 0
        for tstar in tg:
            Arows.append([sum(hk_nud(tstar - s) for s in pts if prev < s <= tstar)
                          for pts in pre_times_by_syn]); b.append(TH); prev = tstar
        for ut in up_times:
            pv, _ = epoch_of(ut, targets)
            Arows.append([sum(hk(ut - s) for s in pts if pv < s < ut)
                          for pts in pre_times_by_syn]); b.append((1 - MARGIN) * TH)
        A = np.array(Arows); bb = np.array(b)
        wl, *_ = np.linalg.lstsq(A, bb, rcond=None)
        w = np.clip(wl, lo, hi)
    return w


def main():
    print("Single pass, TRUE inputs: robust vs robust+SUPPRESSION (whole-net match)")
    for ci in range(3):
        tc, params, C, w_true, N, outs = RR.build(ci)
        Cj = jnp.array(C)
        fs = lambda w: np.array(_hard_sim(jnp.array(np.asarray(w, np.float32)), params, Cj, N, jnp.array([0])))
        tv = fs(w_true); T = {n: RR.spikes_of(tv, n) for n in range(N)}
        lo, hi = w_true * 0.1, w_true * 5.0
        inc = {n: (np.where(C[:, 1] == n)[0], C[np.where(C[:, 1] == n)[0], 0]) for n in range(N)}
        res = {}
        for tag, fn in [("robust", lambda pt, t, l, h: RR.solve(pt, t, l, h, robust=True)),
                        ("robust+supp", solve_supp)]:
            w = w_true.astype(float).copy()
            for n in range(1, N):
                if not T[n]:
                    continue
                syn, pres = inc[n]
                if len(syn) == 0:
                    continue
                sol = fn([T[int(p)] for p in pres], T[n], lo[syn], hi[syn])
                if sol is not None:
                    w[syn] = sol
            V = fs(w)
            net = sum(1 for n in range(N) if RR.spikes_of(V, n) == T[n])
            extra = sum(1 for n in range(N) if len(RR.spikes_of(V, n)) > len(T[n]))
            sp = {n: len(RR.spikes_of(V, n)) for n in outs}
            res[tag] = (net, extra, sp)
        tgt = {n: len(T[n]) for n in outs}
        print(f"  case{ci} (tgt {[tgt[n] for n in outs]}):")
        for tag, (net, extra, sp) in res.items():
            print(f"     {tag:12s}: net={net}/{N}  extra-firing={extra}  out={[sp[n] for n in outs]}")


if __name__ == "__main__":
    main()
