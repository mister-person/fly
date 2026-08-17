"""Can we get better-conditioned constraints from the SAME information?

Diagnosis (tp_fanin.py): weight-recovery error at high fan-in comes from
ill-conditioning — many inputs sampled through the same smooth h at the spike
time give nearly-collinear columns. We currently collapse each inter-spike
epoch to ONE equation ("voltage = threshold at the spike time").

But within an epoch (between resets) the voltage is an exact linear
superposition at EVERY timestep:  V(t)/gsw = sum_k w_k * (sum h[t - t_k']).
Sampling V(t) densely across the epoch adds rows where the inputs are sampled
at DIFFERENT points of their h-tails (recent input on the steep rise, older one
on the flat tail), which evolve differently in t -> non-collinear rows ->
better conditioning. The trajectory is information we already have (target_v in
the oracle case; structurally determined given inputs+weights otherwise).

This script compares:
  SPARSE : one row per spike time (what TP does now)
  DENSE  : one row per sub-threshold timestep within each epoch
and reports cond(A), weight error, and residual-at-true-w (consistency check).
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

def forward_lif_trace(input_spikes, weights, steps):
    """Return (spikes, samples) where samples = list of (t, epoch_start, V)
    for every sub-threshold ACTIVE step (ref==0, no spike). epoch_start is the
    last spike time before t (0 if none) — the reset point for contributions."""
    events = []
    for sp, w in zip(input_spikes, weights):
        for tk in sp:
            ta = tk + delay
            if 0 <= ta < steps:
                events.append((ta, w * gsw))
    events.sort(); ev = iter(events); nxt = next(ev, None)
    V = R = 0.0; ref = 0; spikes = []; samples = []
    reset_end = 0            # arrival-time after which inputs contribute (refr end)
    for t in range(steps):
        upd = 0.0
        while nxt and nxt[0] == t:
            upd += nxt[1]; nxt = next(ev, None)
        R = (R + upd) * rd * (ref != 1)
        V = (V - R) * nd + R
        V = V * (ref == 0)
        if V >= th and ref == 0:
            spikes.append(t); ref = refr + 1; reset_end = t + refr
        elif ref > 0:
            ref -= 1
        else:
            # active, sub-threshold step: record trajectory sample
            if V > 1e-9:
                samples.append((t, reset_end, V))
    return spikes, samples

def contrib_row(input_spikes, reset_end, t):
    # input spike tk arrives at tk+delay; contributes only if arrival is after
    # the refractory clear (tk+delay > reset_end) and by time t (h==0 if not yet)
    return [sum(h[t - tk] for tk in sp
                if tk + delay > reset_end and 0 < t - tk < MAX_H)
            for sp in input_spikes]

def build_sparse(input_spikes, spikes):
    rows, b = [], []
    reset_end = 0
    for Tj in spikes:
        rows.append(contrib_row(input_spikes, reset_end, Tj)); b.append(th / gsw)
        reset_end = Tj + refr
    A = np.array(rows); b = np.array(b)
    keep = A.max(axis=1) > 1e-12
    return A[keep], b[keep]

def build_dense(input_spikes, samples, stride=1):
    rows, b = [], []
    for i, (t, re, V) in enumerate(samples):
        if i % stride: continue
        rows.append(contrib_row(input_spikes, re, t)); b.append(V / gsw)
    A = np.array(rows); b = np.array(b)
    if A.size == 0:
        return A, b
    keep = A.max(axis=1) > 1e-12
    return A[keep], b[keep]

def make_inputs(K, steps):
    trains = []
    for i in range(K):
        phase, period = 40 + 3 * i, 100 + i
        trains.append([phase + k * period for k in range(steps // period + 1)
                       if phase + k * period < steps])
    return trains

def werr(w, wt): return 100 * np.mean(np.abs(w - wt) / (wt + 1e-9))
def cond(A):     return np.linalg.cond(A) if A.shape[1] > 1 and A.shape[0] >= A.shape[1] else np.inf

def nnls_ridge(A, b, rf=1e-3):
    lam = rf * np.trace(A.T @ A) / max(A.shape[1], 1)
    return nnls(np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])]),
               np.concatenate([b, np.zeros(A.shape[1])]))[0]

def recover_ineq(inp, spikes, samples, dense, lo, hi):
    """Recover weights using ONLY spike times (no target voltages):
      lower bounds: V >= th at each spike time
      upper bounds: V < th at 'don't fire' steps
        dense=True  -> every sub-threshold active step + the step before each spike
        dense=False -> 4 sampled points per inter-spike interval (current TP)
    Objective: min ||w - w_nnls||^2 with w_nnls = ridge NNLS on spike equalities.
    """
    from scipy.optimize import minimize
    K = len(inp)
    A_lo, b_lo = build_sparse(inp, spikes)          # fire-at-spike lower bounds
    w0 = nnls_ridge(A_lo, b_lo)
    # upper-bound rows
    up_rows = []
    if dense:
        # every recorded sub-threshold step (strided) ...
        for i, (t, re, _V) in enumerate(samples):
            if i % 3 == 0:
                up_rows.append(contrib_row(inp, re, t))
        # ... plus the tight bracket: V < th the step BEFORE each spike
        reset_end = 0
        for Tj in spikes:
            up_rows.append(contrib_row(inp, reset_end, Tj - 1))
            reset_end = Tj + refr
    else:
        reset_end = 0
        prev = 0
        for Tj in spikes:
            for f in [0.25, 0.45, 0.65, 0.85]:
                tt = int(prev + refr + 2 + f * (Tj - (prev + refr + 2)))
                if tt > prev + refr + 2:
                    up_rows.append(contrib_row(inp, reset_end, tt))
            prev, reset_end = Tj, Tj + refr
    A_up = np.array([r for r in up_rows if max(r) > 1e-12]) if up_rows else np.zeros((0, K))
    b_up = np.full(len(A_up), th / gsw)
    cons = [{'type': 'ineq', 'fun': lambda w, A=A_lo, b=b_lo: A @ w - b}]
    if len(A_up):
        cons.append({'type': 'ineq', 'fun': lambda w, A=A_up, b=b_up: b - A @ w})
    r = minimize(lambda w: 0.5 * float(np.dot(w - w0, w - w0)), w0.copy(),
                 jac=lambda w: w - w0, method='SLSQP', constraints=cons,
                 bounds=[(lo, hi)] * K, options={'ftol': 1e-12, 'maxiter': 3000})
    return (r.x if (r.success or r.status in [0, 4]) else w0), len(A_up)

rng = np.random.default_rng(0)
STEPS = 1200

print("=" * 78)
print("Recovery from SPIKE TIMES ONLY (no target voltages):")
print("  sparse-ineq = 4 upper-bound samples/interval (current TP)")
print("  dense-ineq  = V<th at every sub-threshold step + V<th one step pre-spike")
print("  dense-eq    = uses target voltage trajectory (oracle ceiling, for ref)")
print("=" * 78)
print(f"\n{'K':>3} {'#up_sp':>6} {'#up_de':>6} {'wErr_sparseIneq':>15} "
      f"{'wErr_denseIneq':>14} {'wErr_denseEq':>12}")

for K in [2, 4, 6, 8, 10]:
    inp = make_inputs(K, STEPS)
    w_true = (500.0 / K) * rng.uniform(0.7, 1.3, K)
    spikes, samples = forward_lif_trace(inp, w_true, STEPS)
    if len(spikes) < 2:
        print(f"{K:>3}  post silent"); continue
    loW, hiW = 0.0, float(w_true.max() * 5)
    w_sp, nup_s = recover_ineq(inp, spikes, samples, dense=False, lo=loW, hi=hiW)
    w_de, nup_d = recover_ineq(inp, spikes, samples, dense=True,  lo=loW, hi=hiW)
    Ad, bd = build_dense(inp, samples)                 # equality ceiling (uses V)
    w_eq, _ = nnls(Ad, bd) if Ad.size else (w_true, None)
    print(f"{K:>3} {nup_s:>6} {nup_d:>6} {werr(w_sp, w_true):>14.0f}% "
          f"{werr(w_de, w_true):>13.0f}% {werr(w_eq, w_true):>11.0f}%")

print("\nDone.")
