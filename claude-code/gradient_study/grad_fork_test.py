"""Try every idea on the 3-neuron fork  N0->N1->N2 + bypass N0->N2.

A) N2 ALONE, true inputs: V_sub fit + suppression -> does it prefer chain>bypass?
B) whole fork, oracle targets, iterated local V_sub solve -> output recovery.
C) OUTPUT-ONLY: infer N1's target by backward latency message, then iterate.
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
from grad_method import lif_tangent
import grad_unified as U

TH = U.TH
hk = U.hk
def hk_nud(dt): return 0.5 * (hk(dt - 1) + hk(dt))
def hkp(dt): return hk(dt) - hk(dt - 1)

params = dataclasses.replace(sim.default_params, steps=400)
C = np.array([[0, 1], [1, 2], [0, 2]], np.int32)   # w01(chain1), w12(chain2), w02(bypass)
N = 3
w_true = np.array([500., 500., 50.], np.float32)


def fsim(w):
    return np.array(_hard_sim(jnp.array(np.asarray(w, np.float32)), params, jnp.array(C), N, jnp.array([0])))


def sp(V, n): return np.where(V[:, n] >= TH)[0].tolist()


def isolated_N2(w_chain, w_bypass, n1_sp, n0_sp):
    ia = np.zeros((2, 400), bool)
    for t in n1_sp: ia[0, t] = True
    for t in n0_sp: ia[1, t] = True
    return lif_tangent(np.array([w_chain, w_bypass]), ia, 400)[1]


def vsub_N2(w, n1_sp, n0_sp, targets, t):
    """no-reset V_sub at time t; accumulation resets at the most recent target."""
    prev = 0
    for tt in sorted(targets):
        if t <= tt: break
        prev = tt
    return (w[0] * sum(hk(t - s) for s in n1_sp if prev < s < t) +
            w[1] * sum(hk(t - s) for s in n0_sp if prev < s < t))


def solve_vsub(pre_list, targets, robust=True):
    """Closed-form nudged V_sub least-squares over the incoming weights (n inputs).
    No suppression -- it clamps the legitimate ramp; the latency-credited fit
    already prefers the right path."""
    tg = sorted(targets); prev = 0; Arows = []; Drows = []
    for tt in tg:
        Arows.append([sum(hk_nud(tt - s) for s in pts if prev < s <= tt) for pts in pre_list])
        Drows.append([sum(hkp(tt - s) for s in pts if prev < s <= tt) for pts in pre_list])
        prev = tt
    A = np.array(Arows); D = np.array(Drows); b = np.full(len(A), TH); n = A.shape[1]
    if robust and n > 1 and A.shape[0] >= 1:
        Jm = np.ones((n, n)) / n; Q = np.zeros((n, n))
        for dj in D:
            Dg = np.diag(dj); Q += Dg @ (np.eye(n) - Jm) @ Dg
        try:
            w = np.linalg.solve(1e8 * (A.T @ A) + Q + 1e-6 * np.eye(n), 1e8 * (A.T @ b))
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(A, b, rcond=None)
    else:
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.clip(w, 20, 3000)


def solve_N2(n1_sp, n0_sp, targets, robust=True, suppress=False):
    return solve_vsub([n1_sp, n0_sp], targets, robust=robust)


def main():
    tv = fsim(w_true); T = {n: sp(tv, n) for n in range(N)}
    print("true spikes:", {n: T[n] for n in range(N)}, " (w_true chain=500,500 bypass=50)")

    print("\nA) N2 ALONE, true inputs, V_sub solve variants:")
    for name, rob, sup in [("plain", False, False), ("+robust", True, False),
                           ("+suppress", False, True), ("+robust+suppress", True, True)]:
        w = solve_N2(T[1], T[0], T[2], robust=rob, suppress=sup)
        spk = isolated_N2(w[0], w[1], T[1], T[0])
        print(f"   {name:18s} chain={w[0]:.0f} bypass={w[1]:.0f} -> N2 {spk} (tgt {T[2]})")

    inc = {n: (np.where(C[:, 1] == n)[0], C[np.where(C[:, 1] == n)[0], 0]) for n in range(N)}

    def iterate(targets_by_neuron, seed, rounds=15, alpha=0.5):
        w = (w_true * np.random.default_rng(seed).uniform(0.5, 1.5, 3)).astype(float)
        for _ in range(rounds):
            V = fsim(w); spall = {p: sp(V, p) for p in range(N)}
            for n in [1, 2]:
                if n not in targets_by_neuron:
                    continue
                syn, pres = inc[n]
                sol = solve_vsub([spall[int(p)] for p in pres], targets_by_neuron[n], robust=(len(pres) > 1))
                w[syn] = (1 - alpha) * w[syn] + alpha * sol
        return w, fsim(w)

    print("\nB) whole fork, ORACLE targets (N1 and N2), iterated plain V_sub solve:")
    for seed in range(4):
        w, V = iterate({1: T[1], 2: T[2]}, seed)
        ok = sp(V, 2) == T[2]
        print(f"   seed{seed}: w={np.round(w,0).tolist()} -> N2 {sp(V,2)}  {'RECOVERED' if ok else 'fail'}")

    print("\nC) OUTPUT-ONLY: infer N1 target by backward latency, iterate:")
    lat = T[2][0] - T[1][0]
    # infer an N1 target time before EACH N2 target, keeping N1's natural extra spikes
    n1_inf = sorted(set([t - lat for t in T[2]]))
    print(f"   inferred N1 target (N2tgt - {lat}): {n1_inf}  (true N1 {T[1]})")
    for seed in range(4):
        w, V = iterate({1: n1_inf, 2: T[2]}, seed)
        ok = sp(V, 2) == T[2]
        print(f"   seed{seed}: w={np.round(w,0).tolist()} -> N2 {sp(V,2)}  {'RECOVERED' if ok else 'fail'}")


if __name__ == "__main__":
    main()
