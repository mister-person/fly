"""Does the compounding failure survive a MORE ACCURATE local solver?

Same unrolled/feed-forward depth-compounding chain as tp_compounding.py
(N0->N1->...->NL, all true w=500), but each hop solved with the UNIFIED objective:
V_sub(t*)=th using the EXACT probe kernel (not the analytic impulse response),
least-squares per epoch.  Compare output error vs depth to the old TP.
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
import grad_unified as U

params = dataclasses.replace(sim.default_params, steps=1000)
TH = U.TH
hk = U.hk


def run(C, N, w):
    v = np.array(_hard_sim(jnp.array(np.asarray(w, np.float32)), params,
                           jnp.array(C, np.int32), N, jnp.array([0])))
    return {n: np.where(v[:, n] >= TH)[0].tolist() for n in range(N)}, v


def solve_hop(pre_spikes, tgt):
    """Unified V_sub(t*)=th: aim the EDGE of the step-interval (crossing at t*)."""
    prev = 0; ws = []
    for Tj in tgt:
        A = sum(hk(Tj - t) for t in pre_spikes if prev < t < Tj)
        if A > 1e-12:
            ws.append(TH / A)
        prev = Tj
    return float(np.median(ws)) if ws else None


def solve_hop_nudged(pre_spikes, tgt):
    """Nudge to the right STEP: the crossing lands at t* for w in
    [th/A(t*), th/A(t*-1)); aim the MIDPOINT so it sits squarely on t*,
    not on the rounding edge.  (= target level th + half the per-step rise.)"""
    prev = 0; ws = []
    for Tj in tgt:
        Aj = sum(hk(Tj - t) for t in pre_spikes if prev < t < Tj)
        Ajm1 = sum(hk(Tj - 1 - t) for t in pre_spikes if prev < t < Tj - 1)
        if Aj > 1e-12 and Ajm1 > 1e-12:
            w_lo = TH / Aj          # V_sub(t*)   = th  (barely crosses at t*)
            w_hi = TH / Ajm1        # V_sub(t*-1) = th  (crosses at t*-1)
            ws.append(0.5 * (w_lo + w_hi))
        elif Aj > 1e-12:
            ws.append(TH / Aj)
        prev = Tj
    return float(np.median(ws)) if ws else None


def terr(found, tgt):
    if len(found) != len(tgt):
        return None
    return max(abs(f - t) for f, t in zip(found, tgt))


def eval_chain(L, solver):
    C = np.array([[i, i + 1] for i in range(L)], np.int32)
    N = L + 1
    T_true, _ = run(C, N, np.full(L, 500.0))
    if len(T_true[L]) < 2:
        return None, None
    w_tp = np.full(L, 500.0)
    for i in range(L):
        wi = solver(T_true[i], T_true[i + 1])
        if wi is not None:
            w_tp[i] = wi
    T_tp, _ = run(C, N, w_tp)
    return terr(T_tp[L], T_true[L]), len(T_true[L])


def main():
    print("Depth-compounding chain: EDGE (V_sub=th) vs NUDGED (center of step-interval)")
    print(f"{'L (hops)':>8} {'out_sp':>7} {'edge err':>9} {'edge/hop':>9} "
          f"{'nudged err':>11} {'nudged/hop':>11}")
    for L in [2, 4, 8, 16, 24, 32, 48, 64]:
        e_edge, nsp = eval_chain(L, solve_hop)
        e_nud, _ = eval_chain(L, solve_hop_nudged)
        if nsp is None:
            print(f"{L:>8} {'true net silent':>7}")
            continue
        def s(e): return str(e) if e is not None else "cnt-miss"
        def ph(e): return f"{e/L:.2f}" if e is not None else "-"
        print(f"{L:>8} {nsp:>7} {s(e_edge):>9} {ph(e_edge):>9} "
              f"{s(e_nud):>11} {ph(e_nud):>11}")


if __name__ == "__main__":
    main()
