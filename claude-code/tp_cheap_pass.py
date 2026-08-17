"""Does recovering weights from a LONGER observation window close the TP gap?

tp_cheap.py PART A showed the real net drifts: at the 1000-step task window most
high-fan-in neurons are under-determined (#epochs < #inputs, cond=inf), but
extending to 2000/4000 steps makes every system over-determined and well-
conditioned (cond -> single digits). The weights are static, so we can OBSERVE
the true net for longer to constrain them, then DEPLOY on the original 1000-step
task and measure the same output loss.

This is a cheaper, more realistic "more information" than sub-threshold voltage
trajectories: it's just a longer recording of spike times we already assume access to.

Baseline to beat: oracle TP on the 1000-step window = 9.89e-3 (RIDGE=1e-3).
"""

import sys, os, types, dataclasses
os.environ.setdefault("LOSS", "st")
for _n, _attrs in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n); [setattr(_m, k, v) for k, v in _attrs.items()]; sys.modules[_n] = _m
sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import jax.numpy as jnp
from scipy.optimize import nnls, minimize

from homotopy_core import hard_sim as _hard_sim
import jax_spiking_model as sim
from test_cases import RECURRENT_CASES, _make_recurrent_weights

CASE_IDX = int(os.environ.get("CASE", "2"))
RIDGE    = float(os.environ.get("RIDGE", "1e-3"))
EVAL_STEPS = 1000

tc = RECURRENT_CASES[CASE_IDX]
conns, tw = _make_recurrent_weights(
    tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
    tc["num_neurons"], tc["output_neurons"])

th  = sim.default_params.threshold
gsw = sim.default_params.global_synapse_weight
delay = sim.default_params.delay_iters
refr  = sim.default_params.refractory_iters
nd = float(sim.default_params.neuron_decay)
rd = float(sim.default_params.rise_decay)
A_ext = jnp.array([0])

C_np = np.array(conns, dtype=np.int32)
C    = jnp.array(C_np)
N    = tc["num_neurons"]
outs = tc["output_neurons"]
edges = C_np
w_true = np.array(tw, np.float32)
lo, hi = w_true * 0.1, w_true * 5.0

MAX_H = 600
h = np.zeros(MAX_H)
_R = _V = 0.0
for t in range(MAX_H):
    _R = (_R + (1.0 if t == delay else 0.0)) * rd
    _V = (_V - _R) * nd + _R
    h[t] = _V

def pre_of(n):
    idx = np.where(edges[:, 1] == n)[0]
    return idx, edges[idx, 0]

def spikes_of(V_np, n):
    return np.where(V_np[:, n] >= th)[0].tolist()

def sim_true(steps):
    p = dataclasses.replace(sim.default_params, steps=steps)
    v = np.array(_hard_sim(jnp.array(w_true), p, C, N, A_ext))
    return v, {n: spikes_of(v, n) for n in range(N)}

# B_MODE: "th" -> RHS = th/gsw (biased low); "vact" -> RHS = actual crossing V/gsw
B_MODE = os.environ.get("B_MODE", "th")

def contrib(n, pre_spikes, eval_times, epoch_starts):
    syn_idxs, pres = pre_of(n)
    rows = []
    for j, Tj in enumerate(eval_times):
        Tp = epoch_starts[j]
        row = np.zeros(len(pres))
        for ci, p in enumerate(pres):
            for tk in pre_spikes.get(int(p), []):
                if Tp < tk < Tj:
                    dt = Tj - tk
                    if 0 < dt < MAX_H:
                        row[ci] += h[dt]
        rows.append(row)
    return syn_idxs, pres, np.array(rows)

def nnls_ridge(A, b, rf):
    if rf <= 0:
        return nnls(A, b)[0]
    lam = rf * np.trace(A.T @ A) / max(A.shape[1], 1)
    return nnls(np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])]),
                np.concatenate([b, np.zeros(A.shape[1])]))[0]

def recover_neuron(n, pre_spikes, tgt, margin, ridge, v_obs=None):
    syn_idxs, pres = pre_of(n)
    if len(pres) == 0 or len(tgt) == 0:
        return None
    tgt = list(tgt); starts = [0] + tgt[:-1]
    _, _, A_lo = contrib(n, pre_spikes, tgt, starts)
    if B_MODE == "vact" and v_obs is not None:
        b = np.array([v_obs[Tj, n] / gsw for Tj in tgt])
    else:
        b = np.full(len(tgt), th / gsw)
    valid = A_lo.max(axis=1) > 1e-12
    A_lo, b = A_lo[valid], b[valid]
    if len(A_lo) == 0:
        return None
    w_nnls = nnls_ridge(A_lo, b, ridge)
    if margin <= 0:
        return syn_idxs, np.clip(w_nnls, lo[syn_idxs], hi[syn_idxs])
    nt, nte = [], []
    for j in range(len(tgt) - 1):
        e = tgt[j] + refr + 2; nx = tgt[j + 1]
        if e >= nx: continue
        for f in [0.25, 0.45, 0.65, 0.85]:
            nt.append(int(e + f * (nx - e))); nte.append(tgt[j])
    A_up = np.zeros((0, len(pres))); b_up = np.zeros(0)
    if nt:
        _, _, A_up = contrib(n, pre_spikes, nt, nte)
        b_up = np.full(len(nt), (1 - margin) * th / gsw)
        v = A_up.max(axis=1) > 1e-12
        A_up, b_up = A_up[v], b_up[v]
    cons = [{'type': 'ineq', 'fun': lambda w, A=A_lo, bb=b: A @ w - bb}]
    if len(A_up) > 0:
        cons.append({'type': 'ineq', 'fun': lambda w, A=A_up, bb=b_up: bb - A @ w})
    bnds = [(float(lo[si]), float(hi[si])) for si in syn_idxs]
    r = minimize(lambda w: 0.5 * float(np.dot(w - w_nnls, w - w_nnls)), w_nnls.copy(),
                 jac=lambda w: w - w_nnls, method='SLSQP', constraints=cons,
                 bounds=bnds, options={'ftol': 1e-10, 'maxiter': 2000})
    w_sol = r.x if (r.success or r.status in [0, 4]) else w_nnls
    return syn_idxs, np.clip(w_sol, lo[syn_idxs], hi[syn_idxs])

def tp_pass(rec_spikes, margin, v_obs=None):
    w = w_true.copy()
    for n in range(N):
        if n == 0 or not rec_spikes[n]:
            continue
        res = recover_neuron(n, rec_spikes, rec_spikes[n], margin, RIDGE, v_obs)
        if res is not None:
            idxs, ws = res
            w[idxs] = ws
    return w

# ── eval loss always on the 1000-step task window ──────────────────────────
tv_eval, T_eval = sim_true(EVAL_STEPS)
def out_loss(w):
    v = np.array(_hard_sim(jnp.array(w.astype(np.float32)),
                           dataclasses.replace(sim.default_params, steps=EVAL_STEPS),
                           C, N, A_ext))
    return float(sum(np.sum((tv_eval[:, n] - v[:, n])**2) for n in outs))

def count_match(w):
    v = np.array(_hard_sim(jnp.array(w.astype(np.float32)),
                           dataclasses.replace(sim.default_params, steps=EVAL_STEPS),
                           C, N, A_ext))
    T = {n: spikes_of(v, n) for n in range(N)}
    return sum(1 for n in range(N) if T[n] == T_eval[n])

print(f"Case {CASE_IDX}  RIDGE={RIDGE}  eval window={EVAL_STEPS}")
print("Recover weights from an OBSERVATION window of W steps; deploy & score on")
print(f"the {EVAL_STEPS}-step task. Baseline (W={EVAL_STEPS}) reproduces oracle TP.\n")
def weight_err(w):
    return 100 * float(np.mean(np.abs(w - w_true) / (w_true + 1e-9)))

print(f"{'W_obs':>6} {'margin':>6} {'out_loss':>11} {'wErr_all':>9} {'count_match':>11}")

print(f"(B_MODE={B_MODE})\n")
best = {}
for W in [1000, 2000, 4000, 6000, 50000]:
    v_obs, T_obs = sim_true(W)
    for margin in [0.0, 0.05, 0.10, 0.15]:
        w = tp_pass(T_obs, margin, v_obs)
        L = out_loss(w)
        cm = count_match(w)
        we = weight_err(w)
        key = W
        if key not in best or L < best[key][0]:
            best[key] = (L, margin, cm, we)
        print(f"{W:>6} {margin:>6.2f} {L:>11.4e} {we:>8.1f}% {cm:>11}")
    print()

print("Best per observation window:")
for W in sorted(best):
    L, m, cm, we = best[W]
    print(f"  W={W:>5}: loss={L:.4e}  wErr_all={we:.1f}%  (margin={m}, count_match={cm})")
print("\nDone.")
