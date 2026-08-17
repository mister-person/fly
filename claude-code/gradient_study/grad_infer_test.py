"""Can the V_sub method credit-assign through hidden neurons with NO known data?

Only OUTPUT neurons get a target.  Hidden-neuron targets are INFERRED by the
backward latency message: a presynaptic neuron should fire ~lat before each
downstream target it must help produce; a fan-out neuron aggregates the demands of
all its downstream neurons.  Then the direction-free V_sub solve fits each neuron
to its (given or inferred) target, and we iterate.

Nets:
  chain   N0->N1->N2->N3            (depth: does inferred-target error compound?)
  fanout  N0->N1, N1->N2, N1->N3    (N1 must serve two outputs at once)
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
import numpy as np, jax.numpy as jnp
import jax_spiking_model as sim
from homotopy_core import hard_sim as _hard_sim
from grad_fork_test import solve_vsub
import grad_unified as U

TH = U.TH
params = dataclasses.replace(sim.default_params, steps=520)


def fsim(C, N, w):
    return np.array(_hard_sim(jnp.array(np.asarray(w, np.float32)), params, jnp.array(C), N, jnp.array([0])))


def sp(V, n): return np.where(V[:, n] >= TH)[0].tolist()


def est_latency():
    """single-edge crossing lag at a nominal weight (model knowledge, not target data)."""
    from grad_method import lif_tangent
    ia = np.zeros((1, 520), bool); ia[0, 0] = True
    for w in [500, 600, 700]:
        s = lif_tangent(np.array([float(w)]), ia, 520)[1]
        if s:
            return s[0]
    return 65


LAT = est_latency()


def infer_targets(C, N, output_targets):
    """Backward pass: hidden target = union over downstream of (their target - LAT)."""
    tgt = dict(output_targets)
    # reverse topological-ish: repeatedly fill a neuron once all its downstream are known
    outs = set(output_targets)
    for _ in range(N):
        for n in range(N - 1, 0, -1):
            if n in tgt:
                continue
            downs = C[C[:, 0] == n][:, 1]          # neurons n feeds
            if len(downs) == 0 or any(int(d) not in tgt for d in downs):
                continue
            want = sorted(set(int(t - LAT) for d in downs for t in tgt[int(d)] if t - LAT > 0))
            # merge times closer than the refractory window
            merged = []
            for t in want:
                if not merged or t - merged[-1] > params.refractory_iters:
                    merged.append(t)
            tgt[n] = merged
    return tgt


def run(name, C, N, out_ns, w_true, seeds=4, rounds=18, alpha=0.5):
    C = np.array(C, np.int32); w_true = np.array(w_true, np.float32)
    tv = fsim(C, N, w_true); T_true = {n: sp(tv, n) for n in range(N)}
    inc = {n: (np.where(C[:, 1] == n)[0], C[np.where(C[:, 1] == n)[0], 0]) for n in range(N)}
    tgt = infer_targets(C, N, {o: T_true[o] for o in out_ns})
    print(f"\n{name}  (LAT={LAT})")
    for n in range(N):
        note = "OUTPUT (given)" if n in out_ns else ("input" if n == 0 else "hidden (inferred)")
        print(f"   N{n} {note}: target={tgt.get(n, '-')}   true={T_true[n]}")
    succ = 0
    for seed in range(seeds):
        w = (w_true * np.random.default_rng(seed).uniform(0.5, 1.5, len(w_true))).astype(float)
        for _ in range(rounds):
            V = fsim(C, N, w); spall = {p: sp(V, p) for p in range(N)}
            for n in range(1, N):
                if n not in tgt or not tgt[n]:
                    continue
                syn, pres = inc[n]
                if len(syn) == 0:
                    continue
                sol = solve_vsub([spall[int(p)] for p in pres], tgt[n], robust=(len(pres) > 1))
                w[syn] = (1 - alpha) * w[syn] + alpha * sol
        V = fsim(C, N, w)
        ok = all(sp(V, o) == T_true[o] for o in out_ns)
        succ += int(ok)
    print(f"   -> outputs recovered on {succ}/{seeds} seeds")


def main():
    run("CHAIN N0->N1->N2->N3 (output N3 only)",
        [[0, 1], [1, 2], [2, 3]], 4, [3], [500., 500., 500.])
    run("FANOUT N0->N1->{N2,N3} (outputs N2,N3 only)",
        [[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 500., 500.])


if __name__ == "__main__":
    main()
