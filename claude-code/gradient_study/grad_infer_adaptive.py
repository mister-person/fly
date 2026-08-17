"""Divergent-latency inference for a fan-out hidden neuron.

A fixed backward latency LAT splits ONE hidden spike into two inferred targets
when the fan-out edges have very different latencies (e.g. w900->lat38 and
w470->lat82).  Weight-derived latencies (measured or model) fix that case but
destabilise the uniform cases (transient-weight jitter, depth compounding).

The stable fix is GAUGE-ANCHORED CLUSTERING (infer_clustered): group downstream
targets one-per-neuron, treat each edge's latency as a single global unknown, and
choose the gauge so the latest edge sits at MAX_LAT.  Weight-free and schedule-free.
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
DEFAULT_LAT = 71


def fsim(C, N, w):
    return np.array(_hard_sim(jnp.array(np.asarray(w, np.float32)), params, jnp.array(C), N, jnp.array([0])))


def sp(V, n): return np.where(V[:, n] >= TH)[0].tolist()


CLUSTER_WIN = 90   # < inter-pattern spacing (100); groups downstream spikes that
                   # must share one presynaptic spike, regardless of per-edge latency.
MAX_LAT = 82       # longest single-spike latency an edge can realise (weak firing weight)


def infer_clustered(C, N, out_targets):
    """Weight-free hidden-target inference for a fan-out neuron.

    A fan-out neuron fires ONCE and serves several downstream neurons, each at a
    single FIXED latency L_d (one weight = one latency).  We (1) group downstream
    target spikes so each group has at most one spike per downstream neuron
    (a repeated neuron starts a new group -- no window tuning), (2) measure each
    edge's latency OFFSET relative to a reference downstream, and (3) choose the
    gauge so the LATEST edge sits at MAX_LAT -- i.e. as late as physically firable,
    so no edge is asked for an impossible long latency.  Stable (from fixed
    targets, not transient weights) and free of divergent-latency doubling."""
    from collections import Counter
    tgt = dict(out_targets)
    for _ in range(N):
        for n in range(N - 1, 0, -1):
            if n in tgt:
                continue
            downs = [int(d) for d in C[C[:, 0] == n][:, 1]]
            if not downs or any(d not in tgt for d in downs):
                continue
            pool = sorted((t, d) for d in downs for t in tgt[d])
            groups = []; seen = set()
            for t, d in pool:
                if groups and d not in seen and t - groups[-1][0][0] <= CLUSTER_WIN:
                    groups[-1].append((t, d)); seen.add(d)
                else:
                    groups.append([(t, d)]); seen = {d}
            cnt = Counter(d for g in groups for _, d in g)
            ref = max(downs, key=lambda d: cnt[d])            # most-firing downstream
            offs = {}
            for d in downs:
                diffs = [dict((dd, tt) for tt, dd in g)[d] - dict((dd, tt) for tt, dd in g)[ref]
                         for g in groups if d in [dd for _, dd in g] and ref in [dd for _, dd in g]]
                offs[d] = float(np.median(diffs)) if diffs else 0.0
            L_ref = MAX_LAT - max(0.0, max(offs.values()))    # latest edge -> MAX_LAT
            want = sorted(int(round(t - L_ref)) for t, d in pool if d == ref and t - L_ref > 0)
            merged = []
            for t in want:
                if not merged or t - merged[-1] > REFR:
                    merged.append(t)
            tgt[n] = merged
    return tgt


def run(name, C, N, out_ns, w_true, seeds=4, rounds=25, alpha=0.5, verbose=False):
    C = np.array(C, np.int32); w_true = np.array(w_true, np.float32)
    tv = fsim(C, N, w_true); T_true = {n: sp(tv, n) for n in range(N)}
    inc = {n: (np.where(C[:, 1] == n)[0], C[np.where(C[:, 1] == n)[0], 0]) for n in range(N)}
    out_t = {o: T_true[o] for o in out_ns}
    succ = 0
    last_tgt = None
    for seed in range(seeds):
        w = (w_true * np.random.default_rng(seed).uniform(0.5, 1.5, len(w_true))).astype(float)
        tgt = infer_clustered(C, N, out_t); last_tgt = tgt
        for r in range(rounds):
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
        succ += int(all(sp(V, o) == T_true[o] for o in out_ns))
    print(f"{name}: recovered {succ}/{seeds} seeds")
    if verbose:
        for n in range(N):
            print(f"   N{n}: inferred {last_tgt.get(n,'-')}   true {T_true[n]}")


def main():
    print("Gauge-anchored clustered inference (weight-free, no schedule):")
    run("BREAK case  N1->N2(w900,lat38)+N3(w470,lat82)",
        [[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 900., 470.], verbose=True)
    print("\nregression checks (should still pass):")
    run("chain  N0->N1->N2->N3", [[0, 1], [1, 2], [2, 3]], 4, [3], [500., 500., 500.])
    run("fanout equal", [[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 500., 500.])
    run("fanout hard (w500/w200)", [[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 500., 200.])


if __name__ == "__main__":
    main()
