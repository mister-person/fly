"""Unified sub-threshold-voltage objective on (A) MULTI-INPUT neurons and
(B) a HIDDEN neuron via target propagation.

Per-epoch objective, now over a weight VECTOR:
    V_sub(t_j*) = Σ_k w_k · A[j,k] = th ,   A[j,k] = Σ_{pre-k spikes in epoch j} hk(t_j*-t)
    L(w) = Σ_j (Σ_k w_k A[j,k] - th)^2   -> linear least squares in w.

(A) one output neuron, several inputs (real forward = lif_tangent with a weight
    vector).  Tests reachable, coincident (degenerate) inputs, and over-determined
    (more target constraints than weights -> least-squares compromise).
(B) chain N0->N1->N2 (JAX sim): assign the hidden N1 a target time, solve each
    neuron's incoming weights locally with the unified objective, iterate.
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
from grad_method import lif_tangent, TH
import grad_unified as U

T = U.T
hk = U.hk


def A_matrix(pre_times_by_syn, targets):
    tg = sorted(targets); prev = 0; rows = []
    for tstar in tg:
        rows.append([sum(hk(tstar - t) for t in pts if prev < t <= tstar)
                     for pts in pre_times_by_syn])
        prev = tstar
    return np.array(rows)                       # (n_targets, n_syn)


def unified_solve(pre_times_by_syn, targets, w0, lo=20.0, hi=5000.0, iters=3000, lr=3.0):
    """Least-squares descent of Σ_j (Σ_k w_k A[j,k] - th)^2 over the weight vector."""
    A = A_matrix(pre_times_by_syn, targets)
    w = np.array(w0, float); m = np.zeros_like(w); v = np.zeros_like(w)
    if A.size == 0:
        return w
    for t in range(1, iters + 1):
        r = A @ w - TH                          # (n_targets,)
        g = 2.0 * (A.T @ r)                      # (n_syn,)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
        w = np.clip(w - lr * mh / (np.sqrt(vh) + 1e-12), lo, hi)
    return w


# ══════════════════════════════════════════════════════════════════════════════
# (A) MULTI-INPUT readout neuron
# ══════════════════════════════════════════════════════════════════════════════
def make_inputs(times_by_line):
    K = len(times_by_line)
    ia = np.zeros((K, T), bool)
    for k, ts in enumerate(times_by_line):
        for t in ts:
            ia[k, t] = True
    return ia


def spikes_multi(w, ia):
    return lif_tangent(np.asarray(w, float), ia, T)[1]


def part_A():
    print("=" * 66)
    print("(A) MULTI-INPUT readout neuron (unified least-squares over w vector)")
    print("=" * 66)

    # A1: 3 staggered inputs, single reachable output target
    ia = make_inputs([[15], [30], [45]])
    pre = [[15], [30], [45]]
    w = unified_solve(pre, [95], [200., 200., 200.])
    print(f"  A1 staggered 3-input, target [95]: w={np.round(w,0).tolist()} "
          f"-> spikes {spikes_multi(w, ia)}")

    # A2: coincident inputs (both at t=15) -> degenerate; LS must not blow up
    ia = make_inputs([[15], [15]])
    pre = [[15], [15]]
    w = unified_solve(pre, [80], [200., 200.])
    print(f"  A2 coincident 2-input, target [80]: w={np.round(w,0).tolist()} "
          f"(equal split, min-norm) -> spikes {spikes_multi(w, ia)}")

    # A3: OVER-DETERMINED: 2 inputs firing 3x, output wants 3 IRREGULAR spikes
    lines = [[15, 115, 215], [50, 150, 250]]
    ia = make_inputs(lines)
    targets = [90, 205, 285]                       # irregular spacings (115, 80)
    w = unified_solve(lines, targets, [200., 200.])
    sp = spikes_multi(w, ia)
    rms = (np.sqrt(np.mean([(a - b) ** 2 for a, b in zip(sorted(sp), targets)]))
           if len(sp) == len(targets) else float('nan'))
    print(f"  A3 OVER-DETERMINED (3 targets, 2 weights): w={np.round(w,0).tolist()} "
          f"-> spikes {sp} (target {targets}) RMS={rms:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# (B) HIDDEN neuron: chain N0->N1->N2, unified objective as the local TP solver
# ══════════════════════════════════════════════════════════════════════════════
params = dataclasses.replace(sim.default_params, steps=T)
C = np.array([[0, 1], [1, 2]], np.int32)
N = 3


def full_sim(w):
    V, _, _ = sim.run_sim(params, jnp.array(C), N,
                          jnp.array(np.asarray(w, np.float32)), jnp.array([0]))
    return np.array(V)


def sp_of(V, n):
    return np.where(V[:, n] >= TH)[0].tolist()


def part_B():
    print("\n" + "=" * 66)
    print("(B) HIDDEN neuron: chain N0->N1->N2, unified objective as local TP solver")
    print("=" * 66)
    w_true = np.array([500., 500.], np.float32)
    Vt = full_sim(w_true)
    T_true = {n: sp_of(Vt, n) for n in range(N)}
    print(f"  true spikes: N0={T_true[0]} N1={T_true[1]} N2={T_true[2]}")
    T2_target = T_true[2]                          # output target = true N2 times

    for label, n1_target in [("ORACLE N1 target", T_true[1]),
                             ("INVERTED N1 target (N2-latency)", None)]:
        if n1_target is None:
            lat = T_true[2][0] - T_true[1][0]
            n1_target = [t - lat for t in T2_target]
        # local solves, iterated (N1's spikes feed N2)
        w = (w_true * np.random.default_rng(0).uniform(0.6, 1.4, 2)).astype(float)
        for _ in range(12):
            V = full_sim(w)
            n0_sp, n1_sp = sp_of(V, 0), sp_of(V, 1)
            w0new = unified_solve([n0_sp], n1_target, [w[0]])[0]        # N1 from N0
            w1new = unified_solve([sp_of(full_sim([w0new, w[1]]), 1)], T2_target, [w[1]])[0]  # N2 from N1
            w = np.array([0.5 * w[0] + 0.5 * w0new, 0.5 * w[1] + 0.5 * w1new])
        V = full_sim(w)
        print(f"  {label:34s}: w={np.round(w,0).tolist()}  "
              f"N1={sp_of(V,1)} (tgt {list(n1_target)})  N2={sp_of(V,2)} (tgt {T2_target})")


if __name__ == "__main__":
    part_A()
    part_B()
