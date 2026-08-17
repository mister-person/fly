"""Minimal example: weight-recovery error vs FAN-IN (determinacy).

Diagnosis of the 50-neuron case (tp_diagnose.py) found the strongest predictor
of weight-recovery error is the number of *independent* target-spike equations
relative to the number of input synapses (eqin corr -0.53; fan-in corr +0.44),
NOT conditioning (ridge already handles correlated inputs).

This isolates that: one post-neuron fed by K inputs with distinct, well-separated
phases and slightly different periods (so their relative timing drifts across
epochs → each epoch is a genuinely independent equation, well-conditioned).

  SWEEP A: fix sim length, vary fan-in K.  A stays full rank (rank==K) but its
           condition number explodes as K grows — many inputs packed into the
           same ~100-step pre-spike window have nearly-collinear contribution
           columns — so weight error rises even though it's "identifiable".
  SWEEP B: fix K=8, add epochs.  rank hits K quickly but cond stays huge until
           enough drift accumulates to separate the columns; weight error falls
           monotonically as cond collapses.

Takeaway: recovery accuracy is governed by cond(A), not rank. High fan-in is
the STRUCTURAL cause of ill-conditioning; genuinely independent, well-separated
constraints (from input timing drift across epochs) are the cure.
"""

import sys, os, types, dataclasses
os.environ.setdefault("LOSS", "st")
for _n, _attrs in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n); [setattr(_m, k, v) for k, v in _attrs.items()]; sys.modules[_n] = _m
sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from scipy.optimize import nnls
import jax_spiking_model as sim

params = dataclasses.replace(sim.default_params, steps=1200)
th, gsw = params.threshold, params.global_synapse_weight
delay   = params.delay_iters
refr    = params.refractory_iters
nd, rd  = float(params.neuron_decay), float(params.rise_decay)

MAX_H = 700
h = np.zeros(MAX_H)
_R = _V = 0.0
for t in range(MAX_H):
    _R = (_R + (1.0 if t == delay else 0.0)) * rd
    _V = (_V - _R) * nd + _R
    h[t] = _V

def forward_lif(input_spikes, weights, steps):
    """Single LIF driven by K input trains. input_spikes: list of K spike-time
    lists; weights: list of K weights. Returns post spike times."""
    events = []
    for sp, w in zip(input_spikes, weights):
        for tk in sp:
            ta = tk + delay
            if 0 <= ta < steps:
                events.append((ta, w * gsw))
    events.sort(); ev = iter(events); nxt = next(ev, None)
    V = R = 0.0; ref = 0; out = []
    for t in range(steps):
        upd = 0.0
        while nxt and nxt[0] == t:
            upd += nxt[1]; nxt = next(ev, None)
        R = (R + upd) * rd * (ref != 1)
        V = (V - R) * nd + R
        V = V * (ref == 0)
        if V >= th and ref == 0:
            out.append(t); ref = refr + 1
        elif ref > 0:
            ref -= 1
    return out

def make_inputs(K, steps, seed=0):
    """K input trains: distinct phases (spread ~40 steps) and slightly different
    periods (100..100+K) so relative timing drifts → independent equations."""
    trains = []
    for i in range(K):
        phase  = 40 + 3 * i
        period = 100 + i          # distinct periods → drift across epochs
        trains.append([phase + k * period for k in range(steps // period + 1)
                       if phase + k * period < steps])
    return trains

def build_system(input_spikes, post_spikes):
    """A[j,k] = sum h[Tj - t'] for input k's spikes in epoch (T_{j-1}, Tj]."""
    rows = []
    Tprev = 0
    for Tj in post_spikes:
        row = [sum(h[Tj - tk] for tk in sp if Tprev < tk < Tj and 0 < Tj - tk < MAX_H)
               for sp in input_spikes]
        rows.append(row)
        Tprev = Tj
    A = np.array(rows)
    b = np.full(len(post_spikes), th / gsw)
    keep = A.max(axis=1) > 1e-12
    return A[keep], b[keep]

def nnls_ridge(A, b, rf):
    if rf <= 0:
        return nnls(A, b)[0]
    lam = rf * np.trace(A.T @ A) / max(A.shape[1], 1)
    return nnls(np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])]),
               np.concatenate([b, np.zeros(A.shape[1])]))[0]

def werr(w_rec, w_true):
    return 100 * np.mean(np.abs(w_rec - w_true) / (w_true + 1e-9))

def terr(a, b):
    if len(a) != len(b):
        return None
    return max((abs(x - y) for x, y in zip(a, b)), default=0)

rng = np.random.default_rng(0)

# ═══════════════════════════════════════════════════════════════════════════
# SWEEP A: fan-in K, fixed sim length
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 74)
print("SWEEP A: weight error vs fan-in K  (inputs well-separated, distinct phases)")
print("=" * 74)
print("A is full rank (rank==K) throughout, but cond(A) blows up with fan-in.")
print(f"\n{'K':>3} {'post_sp':>7} {'rank':>5} {'cond(A)':>9} "
      f"{'wErr_NNLS':>9} {'wErr_ridge':>10} {'timing':>7}")

STEPS = 1200
for K in [1, 2, 3, 4, 6, 8, 10]:
    inp = make_inputs(K, STEPS)
    w_true = (500.0 / K) * rng.uniform(0.7, 1.3, K)
    post = forward_lif(inp, w_true, STEPS)
    if len(post) < 2:
        print(f"{K:>3}  post silent"); continue
    A, b = build_system(inp, post)
    if A.shape[0] == 0:
        print(f"{K:>3}  no visible inputs"); continue
    rank = np.linalg.matrix_rank(A, tol=1e-9)
    cond = np.linalg.cond(A) if K > 1 else 1.0
    w_n  = nnls_ridge(A, b, 0.0)
    w_r  = nnls_ridge(A, b, 1e-3)
    post_r = forward_lif(inp, w_r, STEPS)
    te = terr(post_r, post)
    te_s = f"{te}" if te is not None else f"c{len(post_r)}/{len(post)}"
    print(f"{K:>3} {len(post):>7} {rank:>5} {cond:>9.1e} "
          f"{werr(w_n, w_true):>8.0f}% {werr(w_r, w_true):>9.0f}% {te_s:>7}")

# ═══════════════════════════════════════════════════════════════════════════
# SWEEP B: fix fan-in K=8, vary number of epochs (constraints)
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 74)
print("SWEEP B: fix fan-in K=8, add constraints by using more epochs")
print("=" * 74)
print("more epochs → more independent rows → system becomes determined.")
print(f"\n{'epochs':>6} {'rank':>5} {'cond(A)':>9} {'wErr_ridge':>10}")

K = 8
inp_full = make_inputs(K, 6000)               # long trains
w_true   = (500.0 / K) * rng.uniform(0.7, 1.3, K)
post_full = forward_lif(inp_full, w_true, 6000)
for n_ep in [2, 4, 6, 8, 12, 20, 40]:
    if len(post_full) < n_ep:
        continue
    post = post_full[:n_ep]
    Tcut = post[-1] + 1
    inp_cut = [[t for t in sp if t < Tcut] for sp in inp_full]
    A, b = build_system(inp_cut, post)
    if A.shape[0] == 0:
        continue
    rank = np.linalg.matrix_rank(A, tol=1e-9)
    cond = np.linalg.cond(A)
    w_r  = nnls_ridge(A, b, 1e-3)
    print(f"{n_ep:>6} {rank:>5} {cond:>9.1e} {werr(w_r, w_true):>9.0f}%")

print("\nDone.")
