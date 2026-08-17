"""Alternative to the clustering fix: RELAXATION LOOP (recurrence-capable).

The clustering fix inferred hidden targets by a backward ACYCLIC pass -> it needs a
topological order, so it dies on recurrent graphs (0/46 hidden neurons reachable on
the 50-neuron cases).

Here the SAME gauge-anchored placement (group downstream targets one-per-neuron per
pattern-instance; anchor so the slowest edge sits at MAX_LAT) is run WITHOUT a
topological order: every outer iteration we re-infer EVERY hidden neuron's target from
the CURRENT relaxed targets of its downstream neighbours (Gauss-Seidel sweeps).  On a
cyclic graph the information flows around the loop over sweeps instead of a single
backward pass.  Hidden targets warm-start from the current sim spikes.

Result: matches clustering on all feed-forward cases (break/chain/fanout-eq/fanout-hard
4/4) AND recovers recurrent models clustering can't (3-cycle 4/4).  Its limit is
inherent to output-only supervision: a hidden neuron that fires MORE often than any of
its downstream targets reveal (2-cycle N1 fires 5x, downstream N2 4x) can't be pinned
down -- the extra spike is invisible downstream.
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
REFR = params.refractory_iters
NOM_LAT = 71
MAX_LAT = 82    # longest single-spike latency a firing edge can realise (feasibility bound)
CLUSTER_WIN = 90  # temporal width of one pattern-instance group (< inter-pattern spacing)


def fsim(C, N, w):
    return np.array(_hard_sim(jnp.array(np.asarray(w, np.float32)), params, jnp.array(C), N, jnp.array([0])))


def sp(V, n): return np.where(V[:, n] >= TH)[0].tolist()


def anchor_targets(downs, tgt):
    """Gauge-anchored group-and-place for one hidden neuron (proven on all feed-forward
    cases).  Group downstream targets one-per-neuron per pattern-instance, measure each
    edge's OFFSET vs the most-firing reference downstream, and choose the gauge so the
    latest edge sits at MAX_LAT.  Runs on CURRENT relaxed downstream targets, so it works
    inside the relaxation loop on cyclic graphs (no acyclic backward pass)."""
    from collections import Counter
    items = sorted((tstar, d) for d in downs for tstar in tgt[d])
    if not items:
        return []
    groups = []; seen = set()
    for tstar, d in items:
        if groups and d not in seen and tstar - groups[-1][0][0] <= CLUSTER_WIN:
            groups[-1].append((tstar, d)); seen.add(d)
        else:
            groups.append([(tstar, d)]); seen = {d}
    cnt = Counter(d for g in groups for _, d in g)
    ref = max(downs, key=lambda d: cnt[d])
    offs = {}
    for d in downs:
        diffs = [dict((dd, tt) for tt, dd in g)[d] - dict((dd, tt) for tt, dd in g)[ref]
                 for g in groups if d in [dd for _, dd in g] and ref in [dd for _, dd in g]]
        offs[d] = float(np.median(diffs)) if diffs else 0.0
    L_ref = MAX_LAT - max(0.0, max(offs.values()))
    want = sorted(int(round(t - L_ref)) for t, d in items if d == ref and t - L_ref > 0)
    merged = []
    for t in want:
        if not merged or t - merged[-1] > REFR:
            merged.append(t)
    return merged


def infer_relax(C, N, out_targets, spall, tgt, sweeps=4):
    """One call = `sweeps` relaxation sweeps.  `tgt` carries relaxed targets across
    outer iterations (warm start); outputs are pinned."""
    down = {n: [int(d) for d in C[C[:, 0] == n][:, 1]] for n in range(N)}
    tgt = {n: list(tgt.get(n, spall[n])) for n in range(N)}
    for o, t in out_targets.items():
        tgt[o] = list(t)
    for _ in range(sweeps):
        new = dict(tgt)
        for n in range(1, N):
            if n in out_targets or not down[n]:
                continue
            new[n] = anchor_targets(down[n], tgt)
        tgt = new
    return tgt


def run(name, C, N, out_ns, w_true, seeds=4, rounds=25, alpha=0.5, verbose=False):
    C = np.array(C, np.int32); w_true = np.array(w_true, np.float32)
    tv = fsim(C, N, w_true); T_true = {n: sp(tv, n) for n in range(N)}
    inc = {n: (np.where(C[:, 1] == n)[0], C[np.where(C[:, 1] == n)[0], 0]) for n in range(N)}
    out_t = {o: T_true[o] for o in out_ns}
    succ = 0; last = None
    for seed in range(seeds):
        w = (w_true * np.random.default_rng(seed).uniform(0.5, 1.5, len(w_true))).astype(float)
        tgt = {}
        for _ in range(rounds):
            V = fsim(C, N, w); spall = {p: sp(V, p) for p in range(N)}
            tgt = infer_relax(C, N, out_t, spall, tgt); last = tgt
            for n in range(1, N):
                if n not in tgt or not tgt[n]:
                    continue
                syn, pres = inc[n]
                if len(syn) == 0:
                    continue
                sol = solve_vsub([spall[int(p)] for p in pres], tgt[n], robust=(len(pres) > 1))
                w[syn] = (1 - alpha) * w[syn] + alpha * sol
        V = fsim(C, N, w)
        succ += int(all(sp(V, o) == T_true[o] for o in out_ns))
    tag = "" if not verbose else "".join(
        f"\n   N{n}: inferred {last.get(n, '-')}   true {T_true[n]}" for n in range(N))
    print(f"{name}: recovered {succ}/{seeds} seeds{tag}")
    return succ


def main():
    print("SIM-CORRESPONDENCE RELAXATION -- feed-forward sanity (cf clustering 4/4/4/4):")
    run("BREAK divergent  N1->N2(w900)+N3(w470)",
        [[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 900., 470.], rounds=40, verbose=True)
    run("chain  N0->N1->N2->N3", [[0, 1], [1, 2], [2, 3]], 4, [3], [500., 500., 500.], rounds=40)
    run("fanout equal", [[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 500., 500.], rounds=40)
    run("fanout hard (w500/w200)", [[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 500., 200.], rounds=40)

    print("\nSMALL RECURRENT models (clustering can't touch these -- 0 hidden reachable):")
    # 2-cycle: N1<->N2 feedback, output N3.  feedback weight small to stay stable.
    run("2-cycle  N0->N1->N2->N3, N2->N1(fb)",
        [[0, 1], [1, 2], [2, 1], [2, 3]], 4, [3], [500., 500., 60., 500.], rounds=40, verbose=True)
    # 3-cycle: N1->N2->N3->N1 loop, output N3.
    run("3-cycle  N0->N1->N2->N3, N3->N1(fb)",
        [[0, 1], [1, 2], [2, 3], [3, 1]], 4, [3], [500., 500., 500., 60.], rounds=40)
    # cycle + fan-out: N1<->N2, N2->{N3,N4} outputs (divergent latency inside a loop)
    run("cycle+fanout  N1<->N2, N2->{N3,N4}",
        [[0, 1], [1, 2], [2, 1], [2, 3], [2, 4]], 5, [3, 4],
        [500., 500., 60., 700., 400.], rounds=40)


if __name__ == "__main__":
    main()
