"""All three fixes on the 50-neuron cases: nudge (1) + drift-robust split (2),
iterated (3).  Local solve per neuron:
  fit:      A[j,k] = sum_{pre-k in epoch j} hk_nudged(t*_j - t)   (V_sub=th, centered)
  robust:   equalize the derivative-weighted contributions {w_k D[j,k]} across
            inputs -> penalty w^T Q w,  Q = sum_j diag(d_j)(I - 11^T/n) diag(d_j),
            D[j,k] = sum h'(t*_j - t).  Minimise K||Aw-th||^2 + w^T Q w.
Oracle hidden targets.  Compare to fit-only (edge) baseline.
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
def hk(dt): return U.hk(dt)
def hk_nud(dt): return 0.5 * (U.hk(dt - 1) + U.hk(dt))     # center-of-step
def hkp(dt): return U.hk(dt) - U.hk(dt - 1)                # derivative


def build(ci):
    tc = RECURRENT_CASES[ci]
    conns, tw = _make_recurrent_weights(tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
                                        tc["num_neurons"], tc["output_neurons"])
    params = dataclasses.replace(sim.default_params, steps=1000)
    return tc, params, np.array(conns, np.int32), np.array(tw, np.float32), tc["num_neurons"], tc["output_neurons"]


def spikes_of(V, n): return np.where(V[:, n] >= TH)[0].tolist()


def solve(pre_times_by_syn, targets, lo, hi, robust, K=1e8):
    tg = sorted(targets); prev = 0; Arows = []; Drows = []
    for tstar in tg:
        Arows.append([sum(hk_nud(tstar - t) for t in pts if prev < t <= tstar) for pts in pre_times_by_syn])
        Drows.append([sum(hkp(tstar - t) for t in pts if prev < t <= tstar) for pts in pre_times_by_syn])
        prev = tstar
    A = np.array(Arows); D = np.array(Drows)
    if A.size == 0 or A.shape[0] == 0:
        return None
    b = np.full(A.shape[0], TH); n = A.shape[1]
    if robust and n > 1:
        Jm = np.ones((n, n)) / n
        Q = np.zeros((n, n))
        for dj in D:
            Dg = np.diag(dj); Q += Dg @ (np.eye(n) - Jm) @ Dg
        M = K * (A.T @ A) + Q
        try:
            w = np.linalg.solve(M, K * (A.T @ b))
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(A, b, rcond=None)
    else:
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.clip(w, lo, hi)


def run_case(ci, robust, seeds=3, iters=12, alpha=0.5):
    tc, params, C, w_true, N, outs = build(ci)
    Cj = jnp.array(C)
    fsim = lambda w: np.array(_hard_sim(jnp.array(np.asarray(w, np.float32)), params, Cj, N, jnp.array([0])))
    tv = fsim(w_true); T_true = {n: spikes_of(tv, n) for n in range(N)}
    lo, hi = w_true * 0.1, w_true * 5.0
    inc = {n: (np.where(C[:, 1] == n)[0], C[np.where(C[:, 1] == n)[0], 0]) for n in range(N)}
    best = None
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        w = (w_true * rng.uniform(0.5, 1.5, len(w_true))).astype(np.float64)
        for _ in range(iters):
            V = fsim(w); sp_all = {p: spikes_of(V, p) for p in range(N)}
            for nn in range(1, N):
                if not T_true[nn]:
                    continue
                syn, pres = inc[nn]
                if len(syn) == 0:
                    continue
                sol = solve([sp_all[int(p)] for p in pres], T_true[nn], lo[syn], hi[syn], robust)
                if sol is not None:
                    w[syn] = (1 - alpha) * w[syn] + alpha * sol
        V = fsim(w); sp = {n: len(spikes_of(V, n)) for n in outs}
        cnt = sum(sp[n] == len(T_true[n]) for n in outs)
        loss = float(sum(np.sum((tv[:, n] - V[:, n]) ** 2) for n in outs))
        net = sum(1 for n in range(N) if len(spikes_of(V, n)) == len(T_true[n]))
        if best is None or cnt > best[0] or (cnt == best[0] and loss < best[2]):
            best = (cnt, sp, loss, net)
    return tc["name"], {n: len(T_true[n]) for n in outs}, best


def main():
    print("50-neuron, oracle targets: FIT-only vs FIT+NUDGE+DRIFT-ROBUST")
    print(f"{'case':20s} {'tgt':>10s}  {'fit out':>14s} {'c':>2s} {'net':>5s} {'loss':>8s}   "
          f"{'robust out':>14s} {'c':>2s} {'net':>5s} {'loss':>8s}")
    for ci in range(3):
        name, tgt, bf = run_case(ci, robust=False)
        _, _, br = run_case(ci, robust=True)
        ts = "/".join(str(tgt[n]) for n in tgt)
        def f(b): return "/".join(str(b[1][n]) for n in b[1]), b[0], b[3], b[2]
        sf, cf, nf, lf = f(bf); sr, cr, nr, lr = f(br)
        print(f"{name:20s} {ts:>10s}  {sf:>14s} {cf} {nf:>5} {lf:>8.2e}   "
              f"{sr:>14s} {cr} {nr:>5} {lr:>8.2e}")
    print("\nsoft-homotopy ref: case0 EXACT 7/2/1, case1 EXACT 4/2/7, case2 unsolved")


if __name__ == "__main__":
    main()
