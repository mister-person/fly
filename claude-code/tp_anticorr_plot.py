"""Visualize the weight-accuracy vs task-loss relationship on the 50-neuron net.

Recover weights from an OBSERVATION window of W steps (pure ridge-NNLS, margin=0),
deploy on the fixed 1000-step task, and plot BOTH metrics vs W:
  - global weight error  ||w_rec - w_true||  (mean relative %)   -> falls with W
  - output task loss (SSE on output neurons)                     -> rises with W
for b=th (biased one-hop RHS) and b=V_actual (unbiased RHS).

The "scissors" (weight error down, loss up) is the point: better weights, worse
loss. Saves tp_anticorr.png.
"""

import sys, os, types, dataclasses
os.environ.setdefault("LOSS", "st")
os.environ.setdefault("MPLBACKEND", "Agg")
for _n, _attrs in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n); [setattr(_m, k, v) for k, v in _attrs.items()]; sys.modules[_n] = _m
sys.path.insert(0, "/workspace/project")

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.optimize import nnls

from homotopy_core import hard_sim as _hard_sim
import jax_spiking_model as sim
from test_cases import RECURRENT_CASES, _make_recurrent_weights

CASE_IDX = int(os.environ.get("CASE", "2"))
RIDGE    = float(os.environ.get("RIDGE", "1e-3"))
EVAL     = 1000
WINDOWS  = [1000, 2000, 4000, 6000, 10000, 20000]

tc = RECURRENT_CASES[CASE_IDX]
conns, tw = _make_recurrent_weights(
    tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
    tc["num_neurons"], tc["output_neurons"])

th, gsw = sim.default_params.threshold, sim.default_params.global_synapse_weight
delay, refr = sim.default_params.delay_iters, sim.default_params.refractory_iters
nd, rd = float(sim.default_params.neuron_decay), float(sim.default_params.rise_decay)
A_ext = jnp.array([0])
C_np = np.array(conns, np.int32); C = jnp.array(C_np)
N = tc["num_neurons"]; outs = tc["output_neurons"]; edges = C_np
w_true = np.array(tw, np.float32)
lo, hi = w_true * 0.1, w_true * 5.0

MAX_H = 600
h = np.zeros(MAX_H); _R = _V = 0.0
for t in range(MAX_H):
    _R = (_R + (1.0 if t == delay else 0.0)) * rd
    _V = (_V - _R) * nd + _R
    h[t] = _V

def pre_of(n):
    idx = np.where(edges[:, 1] == n)[0]
    return idx, edges[idx, 0]
def spikes_of(V, n):
    return np.where(V[:, n] >= th)[0].tolist()
def sim_true(steps):
    v = np.array(_hard_sim(jnp.array(w_true),
                 dataclasses.replace(sim.default_params, steps=steps), C, N, A_ext))
    return v, {n: spikes_of(v, n) for n in range(N)}
def nnls_ridge(A, b, rf):
    lam = rf * np.trace(A.T @ A) / max(A.shape[1], 1)
    return nnls(np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])]),
                np.concatenate([b, np.zeros(A.shape[1])]))[0]

def recover(n, T, V, bmode):
    idxs, pres = pre_of(n)
    tgt = T[n]
    if len(idxs) == 0 or len(tgt) < 1:
        return None
    starts = [0] + list(tgt[:-1]); rows, b = [], []
    for j, Tj in enumerate(tgt):
        Tp = starts[j]
        rows.append([sum(h[Tj - tk] for tk in T[int(pp)]
                     if Tp < tk < Tj and 0 < Tj - tk < MAX_H) for pp in pres])
        b.append(V[Tj, n] / gsw if bmode == "vact" else th / gsw)
    A = np.array(rows); b = np.array(b)
    keep = A.max(axis=1) > 1e-12
    A, b = A[keep], b[keep]
    if A.shape[0] == 0:
        return None
    return idxs, np.clip(nnls_ridge(A, b, RIDGE), lo[idxs], hi[idxs])

def tp_pass(T, V, bmode):
    w = w_true.copy()
    for n in range(1, N):
        r = recover(n, T, V, bmode)
        if r is not None:
            idxs, ws = r; w[idxs] = ws
    return w

tv_eval, T_eval = sim_true(EVAL)
p_eval = dataclasses.replace(sim.default_params, steps=EVAL)
def out_loss(w):
    v = np.array(_hard_sim(jnp.array(w.astype(np.float32)), p_eval, C, N, A_ext))
    return float(sum(np.sum((tv_eval[:, n] - v[:, n]) ** 2) for n in outs))
def weight_err(w):
    return 100 * float(np.mean(np.abs(w - w_true) / (w_true + 1e-9)))

data = {"th": {"we": [], "loss": []}, "vact": {"we": [], "loss": []}}
print(f"{'W':>7} {'bmode':>6} {'wErr%':>7} {'loss':>11}")
for W in WINDOWS:
    v_obs, T_obs = sim_true(W)
    for bmode in ["th", "vact"]:
        w = tp_pass(T_obs, v_obs, bmode)
        we, L = weight_err(w), out_loss(w)
        data[bmode]["we"].append(we); data[bmode]["loss"].append(L)
        print(f"{W:>7} {bmode:>6} {we:>6.1f}% {L:>11.4e}")

# ── figure: two panels sharing the W axis ──────────────────────────────────
Wx = np.array(WINDOWS)
fig, (axW, axL) = plt.subplots(1, 2, figsize=(11, 4.2))
col = {"th": "#c0392b", "vact": "#2980b9"}
lab = {"th": r"$b=\theta$  (biased one-hop RHS)",
       "vact": r"$b=V_{\rm actual}$  (unbiased RHS)"}
for bmode in ["th", "vact"]:
    axW.plot(Wx, data[bmode]["we"], "o-", color=col[bmode], label=lab[bmode])
    axL.plot(Wx, np.array(data[bmode]["loss"]) * 1e3, "o-", color=col[bmode], label=lab[bmode])

axW.set_xscale("log"); axL.set_xscale("log")
axW.set_xlabel("observation window W (steps)")
axL.set_xlabel("observation window W (steps)")
axW.set_ylabel("global weight error  (mean rel. %)")
axL.set_ylabel(r"output task loss  (SSE $\times 10^{-3}$, eval@1000)")
axW.set_title("Weights get BETTER with more data")
axL.set_title("...but task loss gets WORSE")
axL.axhline(data["th"]["loss"][0] * 1e3, ls="--", lw=0.9, color="0.5",
            label="baseline (W=1000, b=$\\theta$)")
axW.axvline(EVAL, ls=":", lw=0.9, color="0.6")
axL.axvline(EVAL, ls=":", lw=0.9, color="0.6")
axW.legend(fontsize=8, frameon=False); axL.legend(fontsize=8, frameon=False)
axW.grid(alpha=0.25); axL.grid(alpha=0.25)
fig.suptitle("50-neuron TP: weight accuracy and task loss move in OPPOSITE "
             "directions (loss is compounding-limited, not weight-limited)",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("tp_anticorr.png", dpi=140)
print("\nSaved tp_anticorr.png")
