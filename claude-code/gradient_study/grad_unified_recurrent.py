"""The unified sub-threshold-voltage objective as the local solver in target
propagation, on the project's three 50-neuron recurrent cases (the hard benchmark).

Per neuron n, given its presynaptic neurons' current spike times and n's TARGET
times, the incoming weights solve the per-epoch linear system
    V_sub(t_j*) = Σ_k w_k A[j,k] = th ,  A[j,k]=Σ_{pre-k spikes in epoch j} hk(t_j*-t)
by least squares (the closed form of the unified objective).  Iterate: simulate,
solve every neuron locally, damped-update, repeat.

ORACLE run: every neuron gets its TRUE spike times (isolates the local solver from
target assignment).  Compare output spike-count recovery to the project's best
(soft homotopy: case0 7/2/1, case1 4/2/7, case2 3/3/7 target).
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
import jax_spiking_model as sim
from homotopy_core import hard_sim as _hard_sim
from test_cases import RECURRENT_CASES, _make_recurrent_weights
import grad_unified as U

TH = U.TH
hk = U.hk


def build(ci):
    tc = RECURRENT_CASES[ci]
    conns, tw = _make_recurrent_weights(tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
                                        tc["num_neurons"], tc["output_neurons"])
    params = dataclasses.replace(sim.default_params, steps=1000)
    return (tc, params, np.array(conns, np.int32), np.array(tw, np.float32),
            tc["num_neurons"], tc["output_neurons"])


def spikes_of(V, n, th):
    return np.where(V[:, n] >= th)[0].tolist()


def hk_eval(dt, nudge):
    # nudge: aim V_sub at (t*-0.5) -> discrete crossing centered on t* (not the edge)
    return 0.5 * (hk(dt - 1) + hk(dt)) if nudge else hk(dt)


def local_solve(pre_times_by_syn, targets, lo, hi, nudge=False):
    """Closed-form least-squares of the unified objective, clipped to [lo,hi]."""
    tg = sorted(targets); prev = 0; rows = []
    for tstar in tg:
        rows.append([sum(hk_eval(tstar - t, nudge) for t in pts if prev < t <= tstar)
                     for pts in pre_times_by_syn])
        prev = tstar
    A = np.array(rows)
    if A.size == 0 or A.shape[0] == 0:
        return None
    b = np.full(A.shape[0], TH)
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.clip(w, lo, hi)


def run_case(ci, nudge, seeds=3, iters=12, alpha=0.5):
    tc, params, C, w_true, N, outs = build(ci)
    Cj = jnp.array(C)

    def fsim(w):
        return np.array(_hard_sim(jnp.array(np.asarray(w, np.float32)), params, Cj, N, jnp.array([0])))

    target_v = fsim(w_true)
    T_true = {n: spikes_of(target_v, n, TH) for n in range(N)}
    lo, hi = w_true * 0.1, w_true * 5.0
    inc = {n: (np.where(C[:, 1] == n)[0], C[np.where(C[:, 1] == n)[0], 0]) for n in range(N)}

    best = None
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        w = (w_true * rng.uniform(0.5, 1.5, len(w_true))).astype(np.float64)
        for _ in range(iters):
            V = fsim(w)
            sp_all = {p: spikes_of(V, p, TH) for p in range(N)}
            for n in range(1, N):
                if not T_true[n]:
                    continue
                syn, pres = inc[n]
                if len(syn) == 0:
                    continue
                pre_times = [sp_all[int(p)] for p in pres]
                sol = local_solve(pre_times, T_true[n], lo[syn], hi[syn], nudge=nudge)
                if sol is not None:
                    w[syn] = (1 - alpha) * w[syn] + alpha * sol
        V = fsim(w)
        sp = {n: len(spikes_of(V, n, TH)) for n in outs}
        cnt_ok = sum(sp[n] == len(T_true[n]) for n in outs)
        loss = float(sum(np.sum((target_v[:, n] - V[:, n]) ** 2) for n in outs))
        if best is None or cnt_ok > best[0] or (cnt_ok == best[0] and loss < best[2]):
            best = (cnt_ok, sp, loss, seed)
    tgt = {n: len(T_true[n]) for n in outs}
    return tc["name"], tgt, best


def main():
    print("Unified target-prop (oracle targets) on the 50-neuron cases: EDGE vs NUDGED")
    print(f"{'case':20s} {'target':>14s}  {'EDGE out':>16s} {'cnt':>4s} {'loss':>9s}   "
          f"{'NUDGED out':>16s} {'cnt':>4s} {'loss':>9s}")
    for ci in range(3):
        name, tgt, be = run_case(ci, nudge=False)
        _, _, bn = run_case(ci, nudge=True)
        tstr = " ".join(f"{tgt[n]}" for n in tgt)
        def fmt(b):
            cnt, sp, loss, _ = b
            return " ".join(f"{sp[n]}" for n in sp), cnt, loss
        se, ce, le = fmt(be); sn, cn, ln = fmt(bn)
        print(f"{name:20s} {tstr:>14s}  {se:>16s} {ce}/3 {le:>9.2e}   "
              f"{sn:>16s} {cn}/3 {ln:>9.2e}")
    print("\nsoft-homotopy ref: case0 EXACT 7/2/1, case1 EXACT 4/2/7, case2 unsolved")


if __name__ == "__main__":
    main()
