"""Recurrent test of the voltage-target method via layer-local target propagation.

Each iteration: simulate the whole net (JAX sim), then train EACH non-input
neuron's incoming weights *locally* with the single-neuron voltage-target
objective, using its presynaptic neurons' CURRENT spikes as inputs, toward that
neuron's TARGET spike times.  Target-prop with a gradient-descent local solver.

Two nets, to isolate what recurrence costs:
    chain     N0->N1->N2->N3                 (feedforward)
    feedback  N0->N1->N2->N3, plus N2->N1    (a real loop)

Conditions:
    ORACLE    every neuron gets its TRUE target times   (tests the local solver)
    OUTPUT    only N3 targeted, hidden untrained         (the inversion wall)
"""

import sys, os, types, dataclasses
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
from grad_method import lif_tangent, TH
from grad_multi_neuron import voltage_grad

params = dataclasses.replace(sim.default_params, steps=520)
T = params.steps

NETS = {
    "chain":    dict(C=np.array([[0, 1], [1, 2], [2, 3]], np.int32),
                     w=np.array([500., 500., 500.], np.float32)),
    "feedback": dict(C=np.array([[0, 1], [1, 2], [2, 1], [2, 3]], np.int32),
                     w=np.array([500., 500., 50., 500.], np.float32)),
}
N = 4
INPUT = [0]
OUTPUT = 3


def full_sim(net, w):
    V, _, _ = sim.run_sim(params, jnp.array(net["C"]), N,
                          jnp.array(np.asarray(w, np.float32)), jnp.array(INPUT))
    return np.array(V)


def spikes_of(V, n):
    return np.where(V[:, n] >= TH)[0].tolist()


def incoming(net, n):
    idx = np.where(net["C"][:, 1] == n)[0]
    return idx, net["C"][idx, 0]


def local_inputs(V, pres):
    ia = np.zeros((len(pres), T), bool)
    for k, p in enumerate(pres):
        ia[k] = V[:, p] >= TH
    return ia


def validate_local(net):
    V = full_sim(net, net["w"])
    ok = True
    for n in [1, 2, 3]:
        syn, pres = incoming(net, n)
        _, sp_local, _ = lif_tangent(net["w"][syn], local_inputs(V, pres), T)
        if sp_local != spikes_of(V, n):
            ok = False
    return V, ok


def count_err(V, targets):
    return sum(abs(len(spikes_of(V, n)) - len(t)) for n, t in targets.items())


def timing_err(V, targets):
    errs = {}
    for n, t in targets.items():
        sp = spikes_of(V, n)
        if len(sp) == len(t) and t:
            errs[n] = float(np.mean([abs(a - b) for a, b in zip(sorted(sp), sorted(t))]))
    return errs


def train_recurrent(net, targets, seed=0, iters=60, inner=30, step=4.0, alpha=0.5,
                    train_neurons=None):
    rng = np.random.default_rng(seed)
    w = (net["w"] * rng.uniform(0.5, 1.5, len(net["w"]))).astype(np.float64)
    trainable = train_neurons if train_neurons is not None else list(targets)
    for _ in range(iters):
        V = full_sim(net, w)
        for n in trainable:
            if n not in targets:
                continue
            syn, pres = incoming(net, n)
            ia = local_inputs(V, pres)
            wl = w[syn].copy()
            for _ in range(inner):
                g = voltage_grad(wl, ia, targets[n], T, suppress=True)[0]
                gn = np.linalg.norm(g)
                if gn > 1e-30:
                    wl = np.clip(wl - step * g / gn, 20, 3000)
            w[syn] = (1 - alpha) * w[syn] + alpha * wl
    return w


def main():
    for kind in ["chain", "feedback"]:
        net = NETS[kind]
        Vt, ok = validate_local(net)
        T_true = {n: spikes_of(Vt, n) for n in range(N)}
        print("=" * 68)
        print(f"NET = {kind}   (local-forward validation: {'OK' if ok else 'MISMATCH'})")
        print(f"  true spikes: " + "  ".join(f"N{n}={len(T_true[n])}sp" for n in range(N)))
        print("=" * 68)

        # ORACLE
        tgt_all = {n: T_true[n] for n in [1, 2, 3]}
        succ, best = 0, None
        for seed in range(8):
            w = train_recurrent(net, tgt_all, seed=seed)
            V = full_sim(net, w)
            ce = count_err(V, tgt_all)
            out_ok = len(spikes_of(V, OUTPUT)) == len(T_true[OUTPUT])
            succ += int(out_ok)
            if best is None or ce < best[0]:
                best = (ce, seed, V)
        te = timing_err(best[2], tgt_all)
        print(f"  ORACLE (all targets): output-count OK on {succ}/8 seeds; "
              f"best count-err={best[0]}")
        print(f"    best per-neuron timing err: "
              + "  ".join(f"N{n}={te.get(n, float('nan')):.0f}" for n in [1, 2, 3]))

        # OUTPUT-ONLY
        tgt_out = {OUTPUT: T_true[OUTPUT]}
        outs = []
        for seed in range(8):
            w = train_recurrent(net, tgt_out, seed=seed, train_neurons=[OUTPUT])
            V = full_sim(net, w)
            outs.append(len(spikes_of(V, OUTPUT)) == len(T_true[OUTPUT]))
        print(f"  OUTPUT-ONLY (hidden untrained): output-count OK on {sum(outs)}/8 seeds "
              f"(true N3={len(T_true[OUTPUT])}sp)")
        print()


if __name__ == "__main__":
    main()
