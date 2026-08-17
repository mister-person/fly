"""Relaxation loop (grad_infer_relax) on the 50-neuron RECURRENT cases, OUTPUT-ONLY.

Only the 3 output neurons are pinned to their true spike times; every other neuron's
target is re-inferred each iteration from its downstream neighbours' current relaxed
targets (Gauss-Seidel sweeps), then its incoming weights are fit by solve_vsub.
Compare output spike-count recovery to the oracle ceiling and the soft-homotopy ref.
"""
import sys, os, dataclasses, types, time
sys.path.insert(0, "/workspace/project/gradient_study")
sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")
for _n, _a in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        for _k, _v in _a.items():
            setattr(_m, _k, _v)
        sys.modules[_n] = _m
import numpy as np, jax.numpy as jnp
import jax_spiking_model as sim
from homotopy_core import hard_sim as _hard_sim
from test_cases import RECURRENT_CASES, _make_recurrent_weights
from grad_fork_test import solve_vsub
from grad_infer_relax import infer_relax
import grad_unified as U

TH = U.TH
params = dataclasses.replace(sim.default_params, steps=1000)


def build(ci):
    tc = RECURRENT_CASES[ci]
    conns, tw = _make_recurrent_weights(tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
                                        tc["num_neurons"], tc["output_neurons"])
    return tc, np.array(conns, np.int32), np.array(tw, np.float32), tc["num_neurons"], tc["output_neurons"]


def fsim(C, N, w):
    return np.array(_hard_sim(jnp.array(np.asarray(w, np.float32)), params, jnp.array(C), N, jnp.array([0])))


def sp(V, n): return np.where(V[:, n] >= TH)[0].tolist()


def run_case(ci, seeds=3, rounds=15, alpha=0.5, sweeps=10, true_pre=False):
    """true_pre=True: feed each local solve the TRUE presynaptic spike times (exact
    fan-in linear system) instead of the running sim's -- isolates target-assignment
    error from presynaptic-timing error."""
    tc, C, w_true, N, outs = build(ci)
    tv = fsim(C, N, w_true); T_true = {n: sp(tv, n) for n in range(N)}
    inc = {n: (np.where(C[:, 1] == n)[0], C[np.where(C[:, 1] == n)[0], 0]) for n in range(N)}
    lo, hi = w_true * 0.1, w_true * 5.0
    out_t = {o: T_true[o] for o in outs}
    best = None
    for seed in range(seeds):
        w = (w_true * np.random.default_rng(seed).uniform(0.5, 1.5, len(w_true))).astype(float)
        tgt = {}
        for _ in range(rounds):
            V = fsim(C, N, w); spall = {p: sp(V, p) for p in range(N)}
            pre_src = T_true if true_pre else spall
            tgt = infer_relax(C, N, out_t, spall, tgt, sweeps=sweeps)
            for n in range(1, N):
                if n not in tgt or not tgt[n]:
                    continue
                syn, pres = inc[n]
                if len(syn) == 0:
                    continue
                sol = solve_vsub([pre_src[int(p)] for p in pres], tgt[n], robust=(len(pres) > 1))
                w[syn] = np.clip((1 - alpha) * w[syn] + alpha * sol, lo[syn], hi[syn])
        V = fsim(C, N, w)
        cnt_ok = sum(len(sp(V, o)) == len(T_true[o]) for o in outs)
        exact = all(sp(V, o) == T_true[o] for o in outs)
        loss = float(sum(np.sum((tv[:, o] - V[:, o]) ** 2) for o in outs))
        cur = (cnt_ok, exact, {o: len(sp(V, o)) for o in outs}, loss, seed)
        if best is None or cnt_ok > best[0] or (cnt_ok == best[0] and loss < best[3]):
            best = cur
    return tc["name"], {o: len(T_true[o]) for o in outs}, best


def main():
    print("Relaxation loop, OUTPUT-ONLY, on the 50-neuron recurrent cases")
    print(f"{'case':22s} {'target':>10s}  {'found':>10s} {'cnt':>4s} {'exact':>6s} {'loss':>9s}  {'t(s)':>5s}")
    for ci in range(3):
        t0 = time.time()
        name, tgt, (cnt, exact, found, loss, seed) = run_case(ci)
        ts = " ".join(str(tgt[o]) for o in tgt)
        fs = " ".join(str(found[o]) for o in found)
        print(f"{name:22s} {ts:>10s}  {fs:>10s} {cnt}/3 {str(exact):>6s} {loss:>9.2e}  {time.time()-t0:5.0f}")
    print("\nrefs: oracle-target V_sub -> case0 2/3, case1 1/3, case2 1/3 counts;")
    print("      soft homotopy -> case0 & case1 EXACT, case2 unsolved.")


if __name__ == "__main__":
    main()
