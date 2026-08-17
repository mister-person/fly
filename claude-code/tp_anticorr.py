"""Minimal testbed: one-hop TP recovery improves WEIGHTS but worsens LOSS.

Not "surprising" (loss is non-convex in w; w_true gives loss 0, so the path back
to it isn't monotone). The useful content: the one-hop least-squares surrogate
`A w = th/gsw` is BIASED, so sharpening it (more/better-conditioned constraints)
walks you along a direction ANTI-ALIGNED with the true loss gradient. This is why
the "richer constraints" program (more epochs, dense voltage trajectories) can't
help the 50-neuron task loss — it optimizes the wrong objective.

Circuit (all constructed, fully controllable — no periodic-drive lock-in):
    in0, in1  (two DRIFTING trains, slightly different periods)  ->  Nh
    Nh                                                            ->  Nout
Nh is the 2-input neuron whose split is ill-determined on a short window but
well-determined once drift decorrelates the columns. Nout makes it a task:
its spike timing is the "output loss" and Nh's split error compounds into it.

Key design: the EVAL window is a strict PREFIX of the observation window. We
recover Nh's split + Nout's weight from an observation window of W steps, deploy
on a FIXED short eval window, and report:
    wErr(Nh split) vs true   —  expected to DROP as W grows (better conditioned)
    out timing err on eval   —  expected to RISE as W grows (surrogate bias)
Then repeat with b=V_actual (unbiased RHS) to test if the bias is the whole story.
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

p = sim.default_params
th, gsw = p.threshold, p.global_synapse_weight
delay, refr = p.delay_iters, p.refractory_iters
nd, rd = float(p.neuron_decay), float(p.rise_decay)

MAX_H = 600
h = np.zeros(MAX_H)
_R = _V = 0.0
for t in range(MAX_H):
    _R = (_R + (1.0 if t == delay else 0.0)) * rd
    _V = (_V - _R) * nd + _R
    h[t] = _V

def forward_lif(input_trains, weights, steps):
    """Isolated LIF. Returns (spike_times, {spike_time: V_at_spike})."""
    events = []
    for sp, w in zip(input_trains, weights):
        for tk in sp:
            ta = tk + delay
            if 0 <= ta < steps:
                events.append((ta, w * gsw))
    events.sort(); ev = iter(events); nxt = next(ev, None)
    V = R = 0.0; ref = 0; out = []; vat = {}
    for t in range(steps):
        upd = 0.0
        while nxt and nxt[0] == t:
            upd += nxt[1]; nxt = next(ev, None)
        R = (R + upd) * rd * (ref != 1)
        V = (V - R) * nd + R
        V = V * (ref == 0)
        if V >= th and ref == 0:
            out.append(t); vat[t] = V; ref = refr + 1
        elif ref > 0:
            ref -= 1
    return out, vat

def contrib_rows(input_trains, spikes):
    """A[j,k] = sum h[Tj - t'] for input k's spikes since previous reset."""
    rows = []
    reset_end = 0
    for Tj in spikes:
        row = [sum(h[Tj - tk] for tk in sp
                   if tk + delay > reset_end and 0 < Tj - tk < MAX_H)
               for sp in input_trains]
        rows.append(row)
        reset_end = Tj + refr
    return np.array(rows)

def ridge_nnls(A, b, rf=1e-3):
    if A.size == 0:
        return np.zeros(A.shape[1] if A.ndim == 2 else 0)
    lam = rf * np.trace(A.T @ A) / max(A.shape[1], 1)
    return nnls(np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])]),
                np.concatenate([b, np.zeros(A.shape[1])]))[0]

def recover_weights(input_trains, spikes, vat, bmode):
    A = contrib_rows(input_trains, spikes)
    keep = A.max(axis=1) > 1e-12 if A.size else np.array([], bool)
    A = A[keep]; sp_keep = [s for s, k in zip(spikes, keep) if k]
    if A.shape[0] == 0:
        return None, 0
    if bmode == "vact":
        b = np.array([vat[s] / gsw for s in sp_keep])
    else:
        b = np.full(len(sp_keep), th / gsw)
    return ridge_nnls(A, b), A.shape[0]

def terr(found, tgt):
    n = min(len(found), len(tgt))
    if n == 0:
        return None
    return max(abs(found[i] - tgt[i]) for i in range(n)) + 20 * abs(len(found) - len(tgt))

# ── ground truth ───────────────────────────────────────────────────────────
# Nh: 5 DRIFTING, tightly-phased (correlated) inputs -> ill-conditioned on a
# short window (columns near-collinear), decorrelated once drift accumulates.
# fan-in 6, slow period so Nh fires only ~4x in the eval window (< 6 inputs ->
# UNDER-DETERMINED on eval, exactly the full-net regime). Drift (distinct
# periods) makes longer windows over-determined.
NIN = 6
IN = [[15 + 2 * i + (130 + i) * k for k in range(1200)] for i in range(NIN)]
W_SPLIT_TRUE = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0])  # true split
CHAIN_DEPTH  = 2
W_CHAIN_TRUE = 520.0

EVAL = 560          # fixed scoring window (a PREFIX of every observation window)

def build_truth(steps):
    """Return per-layer (spikes, v_at_spike): layer 0 = Nh, then the chain."""
    layers = []
    sp, vat = forward_lif(IN, W_SPLIT_TRUE, steps)
    layers.append((sp, vat))
    for _ in range(CHAIN_DEPTH):
        sp, vat = forward_lif([layers[-1][0]], [W_CHAIN_TRUE], steps)
        layers.append((sp, vat))
    return layers

truth_eval = build_truth(EVAL)
nh_eval, out_eval = truth_eval[0][0], truth_eval[-1][0]
overshoot = [round(truth_eval[0][1][s] / th, 4) for s in nh_eval[:4]]
print(f"th-crossing overshoot at Nh (first 4 spikes): {overshoot}  (>1 = biased)")
print(f"Eval {EVAL} steps: Nh fires {len(nh_eval)}x, output (depth {CHAIN_DEPTH}) "
      f"fires {len(out_eval)}x\n")

def experiment(bmode):
    print(f"--- b = {bmode} " + "-" * 60)
    print(f"{'W_obs':>6} {'#ep_Nh':>6} {'wErr_split':>10} "
          f"{'out_err(eval)':>13}")
    for W in [EVAL, 1400, 2800, 6000, 20000, 60000]:
        layers = build_truth(W)
        nh, nh_v = layers[0]
        if len(nh) < 2:
            print(f"{W:>6}  Nh too few spikes"); continue
        # recover Nh's 5-way split
        w_split, nep = recover_weights(IN, nh, nh_v, bmode)
        werr = 100 * np.mean(np.abs(w_split - W_SPLIT_TRUE) / W_SPLIT_TRUE)
        # recover each chain weight (single input from the previous layer)
        w_chain = []
        for d in range(CHAIN_DEPTH):
            pre_sp = layers[d][0]
            post_sp, post_v = layers[d + 1]
            wc, _ = recover_weights([pre_sp], post_sp, post_v, bmode)
            w_chain.append(float(wc[0]) if wc is not None and len(wc) else W_CHAIN_TRUE)
        # deploy recovered weights on the EVAL window, measure output timing err
        sp_d, _ = forward_lif(IN, w_split, EVAL)
        for d in range(CHAIN_DEPTH):
            sp_d, _ = forward_lif([sp_d], [w_chain[d]], EVAL)
        oe = terr(sp_d, out_eval)
        oe_s = f"{oe}" if oe is not None else "silent"
        print(f"{W:>6} {nep:>6} {werr:>9.1f}% {oe_s:>13}")
    print()

experiment("th")
experiment("vact")
print("Done.")
