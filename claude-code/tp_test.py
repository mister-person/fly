"""Controlled TP test cases to find where the linear model breaks.

Tests:
  1. Single input, single output — TP should be exact
  2. Two inputs, fixed relative timing — TP should work (system consistent)
  3. Two inputs, drifting relative timing — TP system becomes inconsistent
  4. Two inputs firing nearly simultaneously — near-singular system
  5. High-frequency neuron (short inter-spike interval) — residual rise filter?
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
th    = params.threshold
gsw   = params.global_synapse_weight
delay = params.delay_iters
A_ext = jnp.array([0])

# Use float64 and float32 decay constants separately so we can compare
nd64  = float(params.neuron_decay)
rd64  = float(params.rise_decay)
nd32  = float(np.float32(params.neuron_decay))
rd32  = float(np.float32(params.rise_decay))
print(f"nd: f64={nd64:.10f}  f32={nd32:.10f}  diff={nd64-nd32:.2e}")
print(f"rd: f64={rd64:.10f}  f32={rd32:.10f}  diff={rd64-rd32:.2e}")

# ── impulse responses: float64 (old) vs float32 (matches sim) ─────────────
MAX_H = 600

def make_h(nd_, rd_):
    h_ = np.zeros(MAX_H)
    R = V = 0.0
    for t in range(MAX_H):
        upd   = 1.0 if t == delay else 0.0
        R     = (R + upd) * rd_
        V     = (V - R) * nd_ + R
        h_[t] = V
    return h_

h_f64 = make_h(nd64, rd64)
h_f32 = make_h(nd32, rd32)
print(f"h[71]: f64={h_f64[71]:.8f}  f32={h_f32[71]:.8f}  diff={h_f64[71]-h_f32[71]:.2e}\n")

# Use float32 h going forward
h = h_f32
nd, rd = nd32, rd32

def spikes_of(V_np, n):
    return np.where(V_np[:, n] >= th)[0].tolist()

def run_sim(C_np, N, w_np):
    C   = jnp.array(C_np, jnp.int32)
    v   = np.array(_hard_sim(jnp.array(w_np, jnp.float32), params, C, N, A_ext))
    spk = {n: spikes_of(v, n) for n in range(N)}
    return spk, v   # now returns (spike dict, voltage array)

def full_contribution(spike_times, T_prev, Tj):
    """Sum h[Tj - t_k] for all pre-spikes in (T_prev, Tj)."""
    total = 0.0
    for t_k in spike_times:
        if T_prev < t_k < Tj:
            dt = Tj - t_k
            if 0 < dt < MAX_H:
                total += h[dt]
    return total

def rhs_at(Tj, n, v_ref):
    """Right-hand side for epoch j: actual voltage / gsw if available, else th/gsw."""
    if v_ref is not None and Tj < v_ref.shape[0]:
        return float(v_ref[Tj, n]) / gsw
    return th / gsw

def tp_recover_1input(pre_spikes, tgt_spikes, post_neuron=1, v_ref=None, label=""):
    """Single-input TP: solve w s.t. w * gsw * A_j = V_actual(T_j)."""
    rows, rhs = [], []
    T_prev = 0
    for Tj in tgt_spikes:
        A = full_contribution(pre_spikes, T_prev, Tj)
        if A > 1e-12:
            rows.append(A)
            rhs.append(rhs_at(Tj, post_neuron, v_ref))
        T_prev = Tj
    if not rows:
        return None
    w_per_epoch = [r / a for r, a in zip(rhs, rows)]
    w_opt = float(np.median(w_per_epoch))
    spread = max(w_per_epoch) - min(w_per_epoch) if len(w_per_epoch) > 1 else 0
    if label:
        print(f"  {label}: per-epoch w = {[f'{x:.1f}' for x in w_per_epoch[:6]]}  "
              f"median={w_opt:.1f}  spread={spread:.1f}")
    return w_opt

def nnls_ridge(A_mat, b_vec, ridge_frac=0.0, w_prior=None):
    """Non-negative least squares with optional Tikhonov regularization.

    Solves  min ||A w - b||^2 + lam ||w - w_prior||^2   s.t.  w >= 0
    via the augmented system  [A; sqrt(lam) I] w = [b; sqrt(lam) w_prior].

    lam = ridge_frac * mean(diag(A^T A))  — scaled to the data so the same
    ridge_frac transfers across networks (large or small A magnitudes).
    ridge_frac only bites on rank-deficient (near-singular) directions; the
    well-determined directions have curvature >> lam and are barely perturbed.
    w_prior=None → prior of 0 → picks the minimum-norm split of any degenerate
    direction (equal split for proportional columns), instead of an NNLS vertex.
    """
    ncol = A_mat.shape[1]
    if ridge_frac <= 0:
        return nnls(A_mat, b_vec)
    scale = np.trace(A_mat.T @ A_mat) / max(ncol, 1)   # mean eigenvalue scale of A^T A
    lam   = ridge_frac * scale
    wp    = np.zeros(ncol) if w_prior is None else np.asarray(w_prior, float)
    A_aug = np.vstack([A_mat, np.sqrt(lam) * np.eye(ncol)])
    b_aug = np.concatenate([b_vec, np.sqrt(lam) * wp])
    return nnls(A_aug, b_aug)

def tp_recover_2inputs(pre1, pre2, tgt_spikes, post_neuron=3, v_ref=None, label="",
                        ridge_frac=0.0, w_prior=None):
    """Two-input TP: solve [A1 A2] [w1; w2] = V_actual(T_j)/gsw."""
    rows, rhs = [], []
    T_prev = 0
    for Tj in tgt_spikes:
        A1 = full_contribution(pre1, T_prev, Tj)
        A2 = full_contribution(pre2, T_prev, Tj)
        if max(A1, A2) > 1e-12:
            rows.append([A1, A2])
            rhs.append(rhs_at(Tj, post_neuron, v_ref))
        T_prev = Tj
    if not rows:
        return None, None
    A_mat = np.array(rows)
    b_vec = np.array(rhs)
    w_sol, residual = nnls_ridge(A_mat, b_vec, ridge_frac, w_prior)
    cond = np.linalg.cond(A_mat)
    if label:
        print(f"  {label}: w=[{w_sol[0]:.1f}, {w_sol[1]:.1f}]  "
              f"cond={cond:.1f}  residual={residual:.6f}  "
              f"||A*w-b||={np.linalg.norm(A_mat@w_sol - b_vec):.6f}")
    return w_sol, A_mat

def timing_err(found, target):
    if not found or not target or len(found) != len(target):
        return None, f"count {len(found)}/{len(target)}"
    diffs = [f-t for f,t in zip(found, target)]
    return max(abs(d) for d in diffs), str(diffs[:6])

# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: single input, single output — TP should be exact
# N0 (ext) → N1, weight=w
# N1 target: fires w_true steps after each N0 spike
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: single input (N0→N1)  [old: b=th/gsw  new: b=V_actual/gsw]")
print("=" * 70)
for w_true in [300, 500, 700, 1000]:
    C = np.array([[0, 1]], np.int32)
    T, v_true = run_sim(C, 2, np.array([float(w_true)]))
    n0, n1 = T[0], T[1]
    # old (b=th/gsw)
    w_old = tp_recover_1input(n0, n1, post_neuron=1, v_ref=None)
    # new (b=V_actual/gsw, float32 h)
    w_new = tp_recover_1input(n0, n1, post_neuron=1, v_ref=v_true)
    T_old, _ = run_sim(C, 2, np.array([float(w_old)]))
    T_new, _ = run_sim(C, 2, np.array([float(w_new)]))
    e_old, _ = timing_err(T_old[1], n1)
    e_new, _ = timing_err(T_new[1], n1)
    print(f"  w_true={w_true:4d}  old: w={w_old:.1f} err={w_old-w_true:+.1f} timing={e_old}st  "
          f"new: w={w_new:.1f} err={w_new-w_true:+.1f} timing={e_new}st")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: two inputs, fixed relative timing — TP should work
# N0 (ext) → N1 (w1), N0 → N2 (w2), N1 + N2 → N3 (w13, w23)
# N1 and N2 fire at FIXED latency from N0 each epoch → consistent system
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 2: two inputs, fixed relative timing — should work")
print("=" * 70)

# Topology: N0→N1 (w1), N0→N2 (w2), N1→N3 (w13), N2→N3 (w23)
C2 = np.array([[0,1],[0,2],[1,3],[2,3]], np.int32)
N2 = 4
# w1=500 (N1 fires at ~72 steps), w2=300 (N2 fires at ~100 steps)
# w13, w23 chosen so N3 fires given N1 and N2 spike times
w_true2 = np.array([500., 300., 400., 400.], np.float32)
T2, v2 = run_sim(C2, N2, w_true2)
print(f"  N0: {T2[0][:5]}  N1: {T2[1][:5]}  N2: {T2[2][:5]}  N3: {T2[3][:5]}")
for use_vref, tag, rf in [(v2, "new (b=V/gsw)      ", 0.0),
                          (v2, "new + ridge=1e-3", 1e-3),
                          (v2, "new + ridge=1e-2", 1e-2)]:
    w_sol2, _ = tp_recover_2inputs(T2[1], T2[2], T2[3], post_neuron=3,
                                    v_ref=use_vref, label=tag, ridge_frac=rf)
    if w_sol2 is not None:
        w_t2 = w_true2.copy(); w_t2[2], w_t2[3] = w_sol2[0], w_sol2[1]
        T2r, _ = run_sim(C2, N2, w_t2)
        e2, d2 = timing_err(T2r[3], T2[3])
        print(f"    {tag}: w13={w_sol2[0]:.1f} w23={w_sol2[1]:.1f}  "
              f"(true 400/400)  timing={d2} max={e2}st")
print()

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: drifting relative timing — inconsistent system
# N0 (ext) → N1, N3 (ext) → N2 where N3 has a DIFFERENT period than N0
# N1 fires every 100 steps, N2 fires every 97 steps → relative timing drifts
# N4 receives from N1 + N2
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 3: drifting relative timing — TP system inconsistent")
print("=" * 70)

# Simulate two input spike trains with different periods
n1_spikes = [72 + 100*k for k in range(9) if 72 + 100*k < 1000]
n2_spikes = [85 + 97*k  for k in range(10) if 85 + 97*k  < 1000]

# Target N4 times: find when N4 would fire if w14=w24=w
# Instead, use formula: N4 fires when sum of contributions first reaches th/gsw
# Find the target times from a forward simulation
# (For simplicity: compute the actual voltage at each step and find crossings)

def forward_lif(inputs_by_neuron, weights, steps=1000):
    """Mini forward sim of a single post-neuron with given inputs."""
    # inputs_by_neuron: dict {neuron_id: list of spike times}
    # weights: list matching order of inputs_by_neuron
    V, R, ref = 0.0, 0.0, 0
    spikes = []
    # Build spike event list
    events = []  # (time, weight)
    for (nid, w_val) in zip(inputs_by_neuron.keys(), weights):
        for t_k in inputs_by_neuron[nid]:
            t_arr = t_k + delay
            if t_arr < steps:
                events.append((t_arr, w_val * gsw))
    events.sort()
    event_iter = iter(events)
    next_ev = next(event_iter, None)
    for t in range(steps):
        upd = 0.0
        while next_ev and next_ev[0] == t:
            upd += next_ev[1]
            next_ev = next(event_iter, None)
        R = (R + upd) * rd * (ref != 1)
        V = (V - R) * nd + R
        V = V * (ref == 0)
        if V >= th and ref == 0:
            spikes.append(t)
            ref = params.refractory_iters + 1
        elif ref > 0:
            ref -= 1
    return spikes

# Find N4's true spikes given w14=w24=250
w14, w24 = 250., 250.
n4_true_spikes = forward_lif({1: n1_spikes, 2: n2_spikes}, [w14, w24])

print(f"  N1 (100-step period): {n1_spikes[:6]}")
print(f"  N2 ( 97-step period): {n2_spikes[:6]}")
print(f"  N4 targets (true):    {n4_true_spikes[:8]}")

# Show per-epoch A matrix columns to reveal inconsistency
print(f"\n  Per-epoch TP rows (A1=h[T4-t1], A2=h[T4-t2], ratio=A1/A2):")
T_prev = 0
epoch_A1, epoch_A2 = [], []
for Tj in n4_true_spikes:
    A1 = full_contribution(n1_spikes, T_prev, Tj)
    A2 = full_contribution(n2_spikes, T_prev, Tj)
    ratio = A1/A2 if A2 > 1e-12 else float('inf')
    print(f"    T={Tj:4d}  A1={A1:.5f}  A2={A2:.5f}  A1/A2={ratio:.3f}")
    epoch_A1.append(A1); epoch_A2.append(A2)
    T_prev = Tj

# Solve TP system — no v_ref here since forward_lif doesn't return voltage
# (drifting timing test; v_ref would require integrating voltage tracking)
w_sol3, A3 = tp_recover_2inputs(n1_spikes, n2_spikes, n4_true_spikes,
                                  post_neuron=0, v_ref=None, label="\n  NNLS")
if w_sol3 is not None:
    n4_rec = forward_lif({1: n1_spikes, 2: n2_spikes}, list(w_sol3))
    max_err3, diffs3 = timing_err(n4_rec, n4_true_spikes)
    print(f"    True:  w14={w14:.0f} w24={w24:.0f}")
    print(f"    Recov: w14={w_sol3[0]:.1f} w24={w_sol3[1]:.1f}")
    print(f"    N4 timing errors: {diffs3}  max={max_err3}\n")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: near-simultaneous inputs — near-singular system
# N1 and N2 fire within 1-3 steps of each other each epoch
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 4: near-simultaneous inputs — near-singular TP system")
print("=" * 70)

print("  plain NNLS vs ridge-NNLS (minimum-norm, prior=0). true=[300,300]\n")
print(f"  {'gap':>4}  {'cond':>10}  {'NNLS':>14}  "
      + "  ".join(f"ridge={rf:g}".rjust(14) for rf in [1e-3, 1e-2, 1e-1]))
for gap in [0, 1, 3, 10, 30]:
    # N1 fires at 72, 172, ...; N2 fires 'gap' steps later
    n1_s = [72 + 100*k for k in range(9) if 72 + 100*k < 1000]
    n2_s = [72 + gap + 100*k for k in range(9) if 72 + gap + 100*k < 1000]
    # Target N4 from both
    w14_t, w24_t = 300., 300.
    n4_t = forward_lif({1: n1_s, 2: n2_s}, [w14_t, w24_t])
    if len(n4_t) < 3:
        print(f"  gap={gap:3d}: N4 fires {len(n4_t)} times (not enough)"); continue

    cells, cond4 = [], None
    for rf in [0.0, 1e-3, 1e-2, 1e-1]:
        w_sol4, A4 = tp_recover_2inputs(n1_s, n2_s, n4_t, ridge_frac=rf)
        if w_sol4 is None:
            cells.append("fail"); continue
        cond4 = np.linalg.cond(A4)
        n4_r  = forward_lif({1: n1_s, 2: n2_s}, list(w_sol4))
        me, _ = timing_err(n4_r, n4_t)
        cells.append(f"[{w_sol4[0]:.0f},{w_sol4[1]:.0f}]t{me}")
    print(f"  {gap:>4}  {cond4:>10.1e}  " + "  ".join(c.rjust(14) for c in cells))

# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: High-frequency neuron (short inter-spike interval)
# Verify that refractory clears the rise filter (epoch-boundary assumption)
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("TEST 5: Short inter-spike interval — epoch-boundary assumption")
print("=" * 70)
print(f"  delay={delay}, refractory={params.refractory_iters}")
print(f"  Rise filter cleared at ref=1 (one step before refractory ends)")
print(f"  → V and R should be 0 at epoch start regardless of inter-spike interval\n")

for w_true_5 in [500, 700, 900]:
    C5 = np.array([[0, 1]], np.int32)
    T5, v5 = run_sim(C5, 2, np.array([float(w_true_5)]))
    n0_5, n1_5 = T5[0], T5[1]
    if len(n1_5) < 2:
        continue
    isis = [n1_5[i+1] - n1_5[i] for i in range(min(5, len(n1_5)-1))]
    w_old5 = tp_recover_1input(n0_5, n1_5, post_neuron=1, v_ref=None)
    w_new5 = tp_recover_1input(n0_5, n1_5, post_neuron=1, v_ref=v5)
    T_old5, _ = run_sim(C5, 2, np.array([float(w_old5)]))
    T_new5, _ = run_sim(C5, 2, np.array([float(w_new5)]))
    e_old5, _ = timing_err(T_old5[1], n1_5)
    e_new5, _ = timing_err(T_new5[1], n1_5)
    print(f"  w={w_true_5} ISIs={isis[:3]}  "
          f"old: w_rec={w_old5:.1f} err={w_old5-w_true_5:+.1f} t={e_old5}  "
          f"new: w_rec={w_new5:.1f} err={w_new5-w_true_5:+.1f} t={e_new5}")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: Validate the linear model against true simulation
# For each epoch: compare h-based prediction vs actual voltage
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("TEST 6: Linear model accuracy — h[Δt] vs true voltage (float32 h)")
print("=" * 70)

C6   = np.array([[0,1]], np.int32)
w6   = np.array([500.], np.float32)
v6   = np.array(_hard_sim(jnp.array(w6), params, jnp.array(C6, jnp.int32), 2, A_ext))
n0_6 = spikes_of(v6, 0)
n1_6 = spikes_of(v6, 1)

print(f"  Single input w=500, N1 spikes: {n1_6[:4]}")
print(f"  {'T':>5}  {'V_true':>10}  {'V_pred_f64':>12}  {'V_pred_f32':>12}  "
      f"{'err_f64':>10}  {'err_f32':>10}")
T_prev6 = 0
for Tj in n1_6[:5]:
    V_true  = float(v6[Tj, 1])
    A_j     = full_contribution(n0_6, T_prev6, Tj)
    V_f64   = float(w6[0]) * gsw * make_h(nd64, rd64)[Tj - n0_6[0]]
    V_f32   = float(w6[0]) * gsw * h[Tj - n0_6[0]]
    print(f"  {Tj:>5}  {V_true:>10.7f}  {V_f64:>12.7f}  {V_f32:>12.7f}  "
          f"  {(V_f64-V_true)/th*100:>+8.2f}%  {(V_f32-V_true)/th*100:>+8.2f}%")
    T_prev6 = Tj

print("\nDone.")
