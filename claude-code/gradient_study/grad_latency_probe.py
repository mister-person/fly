"""Two probes prompted by review questions:

Q1  Latency depends on the RECEIVING neuron's voltage (a bypass that pre-charges it
    shortens the edge's effective latency).  Does gauge-anchored clustered inference
    still recover, given it only uses target-derived RELATIVE offsets + one absolute
    gauge, and the solve handles accumulation?

Q2  The weight-derived-latency jitter that regressed the uniform cases -- is it
    variance (smaller learning rate fixes it) or bias (depth compounding, LR can't)?
    Re-run the model_lat-every-iteration scheme across alpha and rounds.
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
from grad_infer_adaptive import infer_clustered, DEFAULT_LAT

TH = U.TH
params = dataclasses.replace(sim.default_params, steps=520)
REFR = params.refractory_iters


def fsim(C, N, w):
    return np.array(_hard_sim(jnp.array(np.asarray(w, np.float32)), params, jnp.array(C), N, jnp.array([0])))


def sp(V, n): return np.where(V[:, n] >= TH)[0].tolist()


def model_lat(w):
    for dt in range(1, 400):
        if w * U.hk(dt) >= TH:
            return dt
    return DEFAULT_LAT


def iterate(C, N, out_ns, w_true, infer, seeds=4, rounds=25, alpha=0.5):
    C = np.array(C, np.int32); w_true = np.array(w_true, np.float32)
    tv = fsim(C, N, w_true); T_true = {n: sp(tv, n) for n in range(N)}
    inc = {n: (np.where(C[:, 1] == n)[0], C[np.where(C[:, 1] == n)[0], 0]) for n in range(N)}
    out_t = {o: T_true[o] for o in out_ns}
    succ = 0; last = None
    for seed in range(seeds):
        w = (w_true * np.random.default_rng(seed).uniform(0.5, 1.5, len(w_true))).astype(float)
        for _ in range(rounds):
            V = fsim(C, N, w); spall = {p: sp(V, p) for p in range(N)}
            tgt = infer(C, N, out_t, w, spall); last = tgt
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
    return succ, T_true, last


# --- inference variants (signature infer(C,N,out_t,w,spall)) ---
def inf_cluster(C, N, out_t, w, spall):
    return infer_clustered(C, N, out_t)


def inf_modellat(C, N, out_t, w, spall):
    """Fixed backward message with per-edge model_lat(current weight)."""
    lat = {(int(C[i, 0]), int(C[i, 1])): model_lat(w[i]) for i in range(len(C))}
    tgt = dict(out_t)
    for _ in range(N):
        for n in range(N - 1, 0, -1):
            if n in tgt:
                continue
            downs = [int(d) for d in C[C[:, 0] == n][:, 1]]
            if not downs or any(d not in tgt for d in downs):
                continue
            want = sorted(set(int(round(t - (lat.get((n, d)) or DEFAULT_LAT)))
                              for d in downs for t in tgt[d] if t - (lat.get((n, d)) or DEFAULT_LAT) > 0))
            merged = []
            for t in want:
                if not merged or t - merged[-1] > REFR:
                    merged.append(t)
            tgt[n] = merged
    return tgt


NETS = {
    "chain N0->N1->N2->N3":        ([[0, 1], [1, 2], [2, 3]], 4, [3], [500., 500., 500.]),
    "fanout equal":                ([[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 500., 500.]),
    "fanout hard (w500/w200)":     ([[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 500., 200.]),
    "BREAK divergent (w900/w470)": ([[0, 1], [1, 2], [1, 3]], 4, [2, 3], [500., 900., 470.]),
}
# voltage-dependent-latency net: N2 gets N1 (chain) AND N0 (bypass) -> bypass pre-charges
# N2, so N1->N2 effective latency is much shorter than the from-rest value.  Output N2.
VNET = ([[0, 1], [1, 2], [0, 2]], 3, [2], [500., 500., 300.])


def main():
    print("Q1a: voltage-dependent latency (N2 = chain N1 + bypass N0; output N2 only)")
    C, N, out, wt = VNET
    tv = fsim(np.array(C, np.int32), N, np.array(wt, np.float32))
    latvd = sp(tv, 2)[0] - sp(tv, 1)[0]
    print(f"   true N1->N2 effective latency = {latvd} (vs from-rest ~71; bypass shortens it)")
    s, T, last = iterate(C, N, out, wt, inf_cluster)
    print(f"   clustered inference: recovered {s}/4   inferred N1={last.get(1)}  true N1={T[1]}")

    # Q1b: fan-out where the two edges have EQUAL weight but DIFFERENT effective
    # latency because a bypass (N0->N2) pre-charges only N2.  Tests whether the
    # relative-offset clustering copes with voltage-driven (not weight-driven) latency.
    print("\nQ1b: fan-out, equal weights, latency split by VOLTAGE (bypass pre-charges N2)")
    C2 = [[0, 1], [1, 2], [1, 3], [0, 2]]; N2n = 4; out2 = [2, 3]; wt2 = [500., 500., 500., 300.]
    tv2 = fsim(np.array(C2, np.int32), N2n, np.array(wt2, np.float32))
    l2 = sp(tv2, 2)[0] - sp(tv2, 1)[0]; l3 = sp(tv2, 3)[0] - sp(tv2, 1)[0]
    print(f"   N1->N2 eff lat={l2}, N1->N3 eff lat={l3}  (same weight, split by pre-charge)")
    s2, T2, last2 = iterate(C2, N2n, out2, wt2, inf_cluster)
    print(f"   clustered inference: recovered {s2}/4   inferred N1={last2.get(1)}  true N1={T2[1]}")

    print("\nQ2: is the weight-latency jitter variance (LR fixes) or bias (LR can't)?")
    print(f"   {'net':30s} {'cluster':>8s} " + " ".join(f"mlat a={a}".rjust(11) for a in [0.5, 0.2, 0.1]))
    for name, (C, N, out, wt) in NETS.items():
        sc, *_ = iterate(C, N, out, wt, inf_cluster)
        row = f"   {name:30s} {sc}/4".rjust(0)
        cells = f"   {name:30s} {str(sc)+'/4':>8s} "
        for a in [0.5, 0.2, 0.1]:
            rounds = int(25 * 0.5 / a)   # more rounds for smaller LR (equal total travel)
            sm, *_ = iterate(C, N, out, wt, inf_modellat, alpha=a, rounds=rounds)
            cells += f"{str(sm)+'/4':>11s} "
        print(cells)


if __name__ == "__main__":
    main()
