"""Cheap cures for high-fan-in ill-conditioning, BEFORE building the
voltage-trajectory machinery.

Root cause (tp_diagnose / tp_fanin): high-fan-in neurons have near-collinear
contribution columns -> cond(A) explodes -> weight error blows up. tp_fanin
found TWO cures:
  (1) MORE EPOCHS: if input timing drifts across epochs, extra rows sample h at
      different lags and decorrelate the columns (SWEEP B: cond 4.6e7 -> 19).
      FREE in the real net IF its activity actually drifts (vs periodic).
  (2) DIRECTIONAL REGULARIZATION: truncated-SVD / spectral cut solves in the
      well-conditioned subspace only. Pure post-processing of existing A; needs
      no new data. Compare against the uniform ridge we use now.

This script:
  PART A  measures, per high-cond neuron, how cond(A) and rank evolve as we
          EXTEND the sim window (1000/2000/4000 steps) -> is drift available?
  PART B  compares weight-recovery error of uniform ridge vs truncated-SVD on
          those same neurons (oracle: true inputs+targets).
It does NOT yet run a full TP pass; it decides which cure is worth wiring in.
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
from scipy.optimize import nnls

from homotopy_core import hard_sim as _hard_sim
import jax_spiking_model as sim
from test_cases import RECURRENT_CASES, _make_recurrent_weights

CASE_IDX = int(os.environ.get("CASE", "2"))
RIDGE    = float(os.environ.get("RIDGE", "1e-3"))

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

def build_A(n, pre_spikes, tgt):
    tgt = list(tgt); starts = [0] + tgt[:-1]
    _, _, A = contrib(n, pre_spikes, tgt, starts)
    b = np.full(len(tgt), th / gsw)
    if A.size == 0:
        return A, b
    keep = A.max(axis=1) > 1e-12
    return A[keep], b[keep]

def condnum(A):
    return float(np.linalg.cond(A)) if (A.ndim == 2 and A.shape[0] >= A.shape[1] > 1) else np.inf

def werr(w, wt, idxs):
    return 100 * np.mean(np.abs(w - wt[idxs]) / (wt[idxs] + 1e-9))

# ── recovery variants (post-processing of the SAME A) ──────────────────────
def rec_ridge(A, b, rf):
    if rf <= 0:
        return nnls(A, b)[0]
    lam = rf * np.trace(A.T @ A) / max(A.shape[1], 1)
    return nnls(np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])]),
                np.concatenate([b, np.zeros(A.shape[1])]))[0]

def rec_tsvd(A, b, rcond):
    """Truncated-SVD least squares, then clip negatives to 0 and refit on kept
    columns. Directional: drops directions with singular value < rcond*smax."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    smax = s[0] if len(s) else 0.0
    keep = s > rcond * smax
    s_inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    w = Vt.T @ (s_inv * (U.T @ b))
    w = np.clip(w, 0, None)
    return w

# ═══════════════════════════════════════════════════════════════════════════
print("=" * 78)
print("Simulating true net at increasing window lengths...")
STEP_SET = [1000, 2000, 4000]
sims = {}
for s in STEP_SET:
    v, T = sim_true(s)
    sims[s] = T
    alive = sum(1 for n in range(N) if T[n])
    tot   = sum(len(T[n]) for n in range(N))
    print(f"  steps={s:>5}: {alive}/{N} alive, {tot} total spikes "
          f"(mean {tot/max(alive,1):.1f} sp/alive-neuron)")

# reference conditioning at base window; pick worst-conditioned firing neurons
T0 = sims[STEP_SET[0]]
cand = []
for n in range(1, N):
    if not T0[n]:
        continue
    idxs, pres = pre_of(n)
    if len(pres) < 2:
        continue
    A, b = build_A(n, T0, T0[n])
    if A.size == 0 or A.shape[0] < A.shape[1]:
        c = np.inf
    else:
        c = condnum(A)
    cand.append((n, len(pres), len(T0[n]), c))
cand.sort(key=lambda x: -(x[3] if np.isfinite(x[3]) else 1e300))
focus = [c[0] for c in cand[:12]]

print("\n" + "=" * 78)
print("PART A: does EXTENDING the window decorrelate columns? (drift check)")
print("=" * 78)
print("If cond(A) falls as epochs accumulate, 'more epochs' is a FREE cure.")
print("If it stays flat, the net is ~periodic and extra rows just repeat.\n")
print(f"{'n':>4} {'#in':>4} | "
      + " ".join(f"{f'{s}s':>18}" for s in STEP_SET))
print(f"{'':>4} {'':>4} | "
      + " ".join(f"{'#ep':>6}{'cond':>12}" for _ in STEP_SET))
for n in focus:
    idxs, pres = pre_of(n)
    cells = []
    for s in STEP_SET:
        A, b = build_A(n, sims[s], sims[s][n])
        nep = A.shape[0] if A.ndim == 2 else 0
        cells.append(f"{nep:>6}{condnum(A):>12.1e}")
    print(f"{n:>4} {len(pres):>4} | " + " ".join(cells))

print("\n" + "=" * 78)
print("PART B: uniform ridge vs truncated-SVD on the base window (no new data)")
print("=" * 78)
print("Oracle inputs+targets. wErr = mean |w-w_true|/w_true over that neuron's")
print("synapses. Lower is better; tests whether a directional cut beats ridge.\n")
print(f"{'n':>4} {'#in':>4} {'#ep':>4} {'cond':>9} "
      f"{'wErr_ridge':>10} {'wErr_tsvd(-2)':>13} {'wErr_tsvd(-3)':>13}")
agg = {'ridge': [], 't2': [], 't3': []}
for n in focus:
    idxs, pres = pre_of(n)
    A, b = build_A(n, T0, T0[n])
    if A.size == 0:
        continue
    c = condnum(A)
    w_r = np.clip(rec_ridge(A, b, RIDGE), lo[idxs], hi[idxs])
    w_2 = np.clip(rec_tsvd(A, b, 1e-2), lo[idxs], hi[idxs])
    w_3 = np.clip(rec_tsvd(A, b, 1e-3), lo[idxs], hi[idxs])
    e_r, e_2, e_3 = (werr(w_r, w_true, idxs), werr(w_2, w_true, idxs),
                     werr(w_3, w_true, idxs))
    agg['ridge'].append(e_r); agg['t2'].append(e_2); agg['t3'].append(e_3)
    print(f"{n:>4} {len(pres):>4} {A.shape[0]:>4} {c:>9.1e} "
          f"{e_r:>9.0f}% {e_2:>12.0f}% {e_3:>12.0f}%")
print(f"\n{'median':>13} {'':>16} "
      f"{np.median(agg['ridge']):>9.0f}% {np.median(agg['t2']):>12.0f}% "
      f"{np.median(agg['t3']):>12.0f}%")

print("\nDone.")
