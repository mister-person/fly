"""Isolate ingredient (3): the recurrent COUPLING.

Minimal feedback net  N0(input)->N1->N2->N3 with feedback N2->N1.
N1's inputs are {N0, N2(feedback)} ; N2's input is {N1} ; N3's is {N2}.

Two ways to run the local (nudge+robust) solve, both with ORACLE targets:
  A) ITERATE using the CURRENT sim's presynaptic spike times (the failing way).
  B) ONE PASS using the TARGET presynaptic times (the true times we already have
     in the oracle case) instead of the current wrong ones.

If B recovers the truth and A doesn't, the coupling failure is precisely "local
solves using wrong presynaptic times during the iteration", and the fix is to
feed each solve the target (or converged) presynaptic times.
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
import grad_robust_recurrent as RR

TH = RR.TH
params = dataclasses.replace(sim.default_params, steps=520)

# feedback net (target_prop's 4-neuron): edges w01,w12,w_fb(2->1),w23
C = np.array([[0, 1], [1, 2], [2, 1], [2, 3]], np.int32)
N = 4
w_true = np.array([500., 500., 50., 500.], np.float32)
OUT = 3


def fsim(w):
    return np.array(_hard_sim(jnp.array(np.asarray(w, np.float32)), params,
                              jnp.array(C), N, jnp.array([0])))


def sp_of(V, n):
    return np.where(V[:, n] >= TH)[0].tolist()


def solve_neuron(n, pre_times_by_syn, target, lo, hi):
    return RR.solve(pre_times_by_syn, target, lo, hi, robust=True)


def main():
    Vt = fsim(w_true)
    T_true = {n: sp_of(Vt, n) for n in range(N)}
    print("true spikes:", {n: T_true[n] for n in range(N)})
    lo, hi = w_true * 0.1, w_true * 5.0
    inc = {n: (np.where(C[:, 1] == n)[0], C[np.where(C[:, 1] == n)[0], 0]) for n in range(N)}

    def net_match(V):
        return sum(1 for n in range(N) if sp_of(V, n) == T_true[n])

    # ── A) ITERATE with current sim inputs ──────────────────────────────────
    print("\nA) iterate, CURRENT presynaptic spikes:")
    best_a = None
    for seed in range(6):
        rng = np.random.default_rng(seed)
        w = (w_true * rng.uniform(0.5, 1.5, len(w_true))).astype(float)
        for _ in range(20):
            V = fsim(w); sp_all = {p: sp_of(V, p) for p in range(N)}
            for n in range(1, N):
                syn, pres = inc[n]
                sol = solve_neuron(n, [sp_all[int(p)] for p in pres], T_true[n], lo[syn], hi[syn])
                if sol is not None:
                    w[syn] = 0.5 * w[syn] + 0.5 * sol
        V = fsim(w); nm = net_match(V)
        if best_a is None or nm > best_a[0]:
            best_a = (nm, sp_of(V, OUT), np.round(w, 0))
    print(f"   best net_match {best_a[0]}/{N}  output {best_a[1]} (true {T_true[OUT]})  w={best_a[2].tolist()}")

    # ── B) ONE PASS with TARGET presynaptic times ───────────────────────────
    print("\nB) one pass, TARGET (true) presynaptic times:")
    w = w_true.astype(float).copy()
    for n in range(1, N):
        syn, pres = inc[n]
        sol = solve_neuron(n, [T_true[int(p)] for p in pres], T_true[n], lo[syn], hi[syn])
        if sol is not None:
            w[syn] = sol
    V = fsim(w)
    print(f"   net_match {net_match(V)}/{N}  output {sp_of(V, OUT)} (true {T_true[OUT]})  "
          f"w={np.round(w,0).tolist()} (true {w_true.tolist()})")


if __name__ == "__main__":
    main()
