"""Minimal example of the 50-neuron failure cause: error COMPOUNDING.

Diagnosis (tp_diagnose.py) showed: with oracle (true) inputs, most neurons
recover weights that reproduce their OWN spikes fine, but in the full sim tiny
per-hop timing errors propagate and amplify. Two ingredients:

  1. DEPTH compounding: a feed-forward chain. Each hop's TP weight is ~1 step
     off; the full-sim output error grows roughly linearly with chain length.

  2. FRAGILITY from correlated inputs: when two inputs to a neuron are highly
     correlated (fire close together), single-hop TP recovers a WRONG weight
     split that still reproduces firing on the true inputs — but does NOT
     generalize once the inputs shift, so it breaks in the full sim.

This script demonstrates (1) with a length sweep and (2) with a 2-input probe.
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

params = dataclasses.replace(sim.default_params, steps=1000)
th, gsw = params.threshold, params.global_synapse_weight
delay   = params.delay_iters
refr    = params.refractory_iters
nd, rd  = float(params.neuron_decay), float(params.rise_decay)
A_ext   = jnp.array([0])

MAX_H = 600
h = np.zeros(MAX_H)
_R = _V = 0.0
for t in range(MAX_H):
    _R = (_R + (1.0 if t == delay else 0.0)) * rd
    _V = (_V - _R) * nd + _R
    h[t] = _V

def spikes_of(V_np, n):
    return np.where(V_np[:, n] >= th)[0].tolist()

def run(C_np, N, w):
    v = np.array(_hard_sim(jnp.array(w, jnp.float32), params,
                           jnp.array(C_np, jnp.int32), N, A_ext))
    return {n: spikes_of(v, n) for n in range(N)}, v

def full_contrib(spikes, Tprev, Tj):
    return sum(h[Tj - tk] for tk in spikes if Tprev < tk < Tj and 0 < Tj - tk < MAX_H)

def recover_1input(pre_spikes, tgt, v_ref=None, post=None):
    """Single-input TP: w = median over epochs of b_j / A_j.
    b_j = V_actual(Tj)/gsw if v_ref given (removes threshold-overshoot bias),
    else th/gsw."""
    ws = []
    Tprev = 0
    for Tj in tgt:
        A = full_contrib(pre_spikes, Tprev, Tj)
        if A > 1e-12:
            b = (float(v_ref[Tj, post]) / gsw) if v_ref is not None else (th / gsw)
            ws.append(b / A)
        Tprev = Tj
    return float(np.median(ws)) if ws else None

def terr(found, tgt):
    if len(found) != len(tgt):
        return None, f"count {len(found)}/{len(tgt)}"
    d = [f - t for f, t in zip(found, tgt)]
    return max(abs(x) for x in d), d

# ═══════════════════════════════════════════════════════════════════════════
# PART 1: DEPTH compounding — chain N0(ext)→N1→N2→...→NL, all w=500
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: depth compounding — feed-forward chain, all true w=500")
print("=" * 70)
print("Each hop TP-recovered from TRUE inputs+targets. Full-sim output error")
print("grows ~linearly with depth; b=th/gsw biases ~2/hop, b=V_actual ~1/hop.\n")
print(f"{'L':>3}  {'out_sp':>6}  {'out_err(b=th)':>13}  {'out_err(b=Vact)':>16}")

for L in [2, 4, 8, 16, 24]:
    # chain: edges 0->1, 1->2, ..., (L-1)->L  ; N = L+1 neurons
    C = np.array([[i, i + 1] for i in range(L)], np.int32)
    N = L + 1
    w_true = np.full(L, 500.0, np.float32)
    T_true, v_true = run(C, N, w_true)
    if len(T_true[L]) < 2:
        print(f"{L:>3}  output silent/too few spikes"); continue

    # oracle TP with two RHS choices: th/gsw vs V_actual/gsw
    outs_by_rhs = {}
    for tag, vref in [("th", None), ("Vact", v_true)]:
        w_tp = w_true.copy()
        for i in range(L):
            pre, post = i, i + 1
            w_i = recover_1input(T_true[pre], T_true[post], v_ref=vref, post=post)
            if w_i is not None:
                w_tp[i] = w_i
        T_tp, _ = run(C, N, w_tp)
        eo, _ = terr(T_tp[L], T_true[L])
        outs_by_rhs[tag] = f"{eo}" if eo is not None else f"cnt {len(T_tp[L])}/{len(T_true[L])}"
    print(f"{L:>3}  {len(T_true[L]):>6}  {outs_by_rhs['th']:>13}  {outs_by_rhs['Vact']:>16}")

# ═══════════════════════════════════════════════════════════════════════════
# PART 2: FRAGILITY — correlated inputs give non-generalizing weights
# N0(ext)→N1, N0(ext)→N2 with N1,N2 firing close together; N1,N2→N3.
# Single-hop TP for N3 (true inputs) reproduces N3, but the recovered split is
# wrong and breaks when N1/N2 timing shifts (as it does in a full sim).
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("PART 2: fragility — correlated inputs → non-generalizing weight split")
print("=" * 70)

C2 = np.array([[0, 1], [0, 2], [1, 3], [2, 3]], np.int32)
N2 = 4
# w1, w2 make N1, N2 fire ~simultaneously; w13=w23=300 (true split)
w_true2 = np.array([500., 480., 300., 300.], np.float32)
T2, _ = run(C2, N2, w_true2)
print(f"  N1 spikes: {T2[1][:5]}   N2 spikes: {T2[2][:5]}  (gap ~{T2[2][0]-T2[1][0]} steps)")

# recover w13,w23 via plain NNLS (no ridge) from true inputs
rows, b = [], []
Tprev = 0
for Tj in T2[3]:
    A1 = full_contrib(T2[1], Tprev, Tj)
    A2 = full_contrib(T2[2], Tprev, Tj)
    if max(A1, A2) > 1e-12:
        rows.append([A1, A2]); b.append(th / gsw)
    Tprev = Tj
A = np.array(rows); bb = np.array(b)
w_plain, _ = nnls(A, bb)
# ridge (min-norm) version
lam = 1e-3 * np.trace(A.T @ A) / 2
w_ridge, _ = nnls(np.vstack([A, np.sqrt(lam) * np.eye(2)]),
                  np.concatenate([bb, np.zeros(2)]))

print(f"  recovered splits:  plain NNLS w=[{w_plain[0]:.0f},{w_plain[1]:.0f}]   "
      f"ridge w=[{w_ridge[0]:.0f},{w_ridge[1]:.0f}]   (true 300/300)\n")

# Fragility probe: shift inputs preserving their AVERAGE (N1 +d, N2 -d).
# True neuron responds to the weighted sum; with true 300/300 an average-
# preserving shift barely moves N3. A vertex split [w_tot,0] tracks N1 fully.
def n3_shift(w13, w23, d):
    """N3 spike time (first epoch) when N1 shifted +d, N2 shifted -d."""
    n1 = [t + d for t in T2[1]]
    n2 = [t - d for t in T2[2]]
    # isolated N3 LIF driven by shifted inputs with given weights
    events = []
    for sp, w in [(n1, w13), (n2, w23)]:
        for tk in sp:
            ta = tk + delay
            if 0 <= ta < params.steps:
                events.append((ta, w * gsw))
    events.sort(); ev = iter(events); nxt = next(ev, None)
    V = R = 0.0; ref = 0
    for t in range(params.steps):
        upd = 0.0
        while nxt and nxt[0] == t:
            upd += nxt[1]; nxt = next(ev, None)
        R = (R + upd) * rd * (ref != 1)
        V = (V - R) * nd + R
        V = V * (ref == 0)
        if V >= th and ref == 0:
            return t
        elif ref > 0:
            ref -= 1
    return None

print("  Average-preserving input jitter (N1 +d, N2 -d): N3 first-spike shift")
print(f"  {'d':>3}  {'true[300,300]':>14}  {'plain[%.0f,%.0f]' % (w_plain[0],w_plain[1]):>16}  "
      f"{'ridge[%.0f,%.0f]' % (w_ridge[0],w_ridge[1]):>16}")
base = {tag: n3_shift(a, b_, 0) for tag, (a, b_) in
        [("true", (300., 300.)), ("plain", tuple(w_plain)), ("ridge", tuple(w_ridge))]}
for d in [0, 3, 6, 10]:
    cells = []
    for tag, (a, b_) in [("true", (300., 300.)), ("plain", tuple(w_plain)), ("ridge", tuple(w_ridge))]:
        t = n3_shift(a, b_, d)
        cells.append("silent" if t is None or base[tag] is None else f"{t - base[tag]:+d}")
    print(f"  {d:>3}  {cells[0]:>14}  {cells[1]:>16}  {cells[2]:>16}")

print("\n  → the [w,0] vertex split tracks N1's jitter fully; the balanced/ridge")
print("    split follows the average like the true neuron (robust to input drift).")
print("\nDone.")
