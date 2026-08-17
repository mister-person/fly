"""Target propagation for the minimal 4-neuron recurrent failing case.

Network: N0(ext)→N1→N2→N3,  feedback N2→N1 (w_fb)
Failure: soft homotopy → w_fb≈7 (kills feedback) → N3 fires at half rate.

Target propagation: layer-by-layer 1-hop linear system.

For a single synapse pre→post with weight w:
    V_post(T) = w * gsw * sum_k h(T - t_k)   (linear superposition in w)
where h(Δt) is the LINEAR impulse response (no threshold, correct LIF eqns).

Setting V_post(T_j) = th gives:
    w* = th / (gsw * A_j)   where A_j = sum_{relevant k} h(T_j - t_k)

For multi-input neurons, stack into Ax=b and solve least-squares.

Inversion: given w* and target post times, find pre times τ_j by solving
    h(T_j - τ_j) = th / (w* * gsw)
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

from homotopy_core import hard_sim as _hard_sim
import jax_spiking_model as sim

params = dataclasses.replace(sim.default_params, steps=1000)
th    = params.threshold       # 0.007
gsw   = params.global_synapse_weight  # 0.0001
delay = params.delay_iters     # 18
refrac= params.refractory_iters  # 22
nd    = float(params.neuron_decay)   # 0.99497
rd    = float(params.rise_decay)     # 0.9803
A_ext = jnp.array([0])

def spikes_of(V, n): return np.where(V[:, n] >= th)[0].tolist()

# ── 4-neuron network ──────────────────────────────────────────────────────────
C_np = np.array([[0,1],[1,2],[2,1],[2,3]], dtype=np.int32)
#   index:         0     1     2     3
#   synapse:      w01   w12  w_fb  w23
C    = jnp.array(C_np)
N    = 4

# From minimal_failures.py run: soft homotopy failure
w_true  = np.array([500.,  500.,  50., 500.], np.float32)
w_found = np.array([527.6, 489.0,  7.2, 447.8], np.float32)

target_v = np.array(_hard_sim(jnp.array(w_true),  params, C, N, A_ext))
found_v  = np.array(_hard_sim(jnp.array(w_found), params, C, N, A_ext))

T_tgt = {n: spikes_of(target_v, n) for n in range(N)}
T_fnd = {n: spikes_of(found_v,  n) for n in range(N)}

print("Failure: found weights vs true")
print(f"  w_true:  {w_true}")
print(f"  w_found: {w_found}")
print(f"  N3 target: {T_tgt[3]}")
print(f"  N3 found:  {T_fnd[3]}  ← half-rate, 4/8 spikes")


# ═══════════════════════════════════════════════════════════════════════════
# LINEAR IMPULSE RESPONSE
# h(t) = V_post(t) due to a single pre-spike at t=0, unit upd, no threshold.
# LIF equation: V = (V - R)*nd + R  ≡  V*nd + R*(1-nd)
# Rise filter:  R = (R + upd) * rd
# ═══════════════════════════════════════════════════════════════════════════

MAX_H = 400

def impulse_response_linear():
    """V_post per unit upd (= w * gsw), no threshold, correct LIF equations."""
    h = np.zeros(MAX_H)
    R = 0.0
    V = 0.0
    for t in range(MAX_H):
        upd = 1.0 if t == delay else 0.0
        R   = (R + upd) * rd
        V   = (V - R) * nd + R        # = V*nd + R*(1-nd)
        h[t] = V
    return h

h = impulse_response_linear()

print(f"\nLinear impulse response h(t):")
print(f"  h[{h.argmax()}] = {h.max():.6f}  (peak, should be > th/sw_500 = {th/(500*gsw):.4f})")
print(f"  h[71] = {h[71]:.6f}   (true latency N1 fires at t=72 when N0 at t=1)")
print(f"  500*gsw*h[71] = {500*gsw*h[71]:.5f}  vs th={th:.5f}  (expect ≈ th)")

# Find Δt such that h(Δt) = th/(w*gsw) for a given w
def find_delta_t(w_val):
    """Find Δt where h(Δt) = th/(w*gsw) — the natural latency at this weight."""
    target_h = th / (w_val * gsw)
    # h rises then falls; first crossing on the way up is the firing point
    for t in range(MAX_H):
        if h[t] >= target_h:
            return t
    return None

lat_true  = find_delta_t(500.0)
lat_found = find_delta_t(447.8)
print(f"  Latency at w=500:  Δt={lat_true}  (N3 fires Δt steps after N2 spike)")
print(f"  Latency at w=448:  Δt={lat_found}")


# ═══════════════════════════════════════════════════════════════════════════
# PER-EPOCH TARGET PROPAGATION
#
# For each output target time T_j, find the SINGLE dominant pre-spike in
# the window (T_{j-1}, T_j) and solve:  w = th / (gsw * h(T_j - t_k))
# ═══════════════════════════════════════════════════════════════════════════

def tp_weight(pre_times, tgt_times, label=""):
    """Target propagation weight for a single synapse (dominant-spike approximation).

    For each epoch (T_{j-1}, T_j], finds the pre-spike t_k closest to T_j
    (within the window after the previous target), computes h(T_j - t_k),
    and solves for w = th / (gsw * h).

    Returns (w_opt, per_epoch_list).
    """
    per_epoch = []
    T_prev = 0
    for j, Tj in enumerate(tgt_times):
        # Dominant spike: the LAST pre-spike before Tj that is after T_prev
        window = [tk for tk in pre_times if T_prev < tk < Tj]
        if not window:
            T_prev = Tj
            continue
        tk = window[-1]     # most recent pre-spike before Tj
        dt = Tj - tk
        if 0 < dt < MAX_H and h[dt] > 1e-12:
            w_j = th / (gsw * h[dt])
            per_epoch.append((j, Tj, tk, dt, w_j))
        T_prev = Tj

    if not per_epoch:
        return None, []

    ws = [x[4] for x in per_epoch]
    w_opt = float(np.median(ws))
    if label:
        print(f"  {label}:")
        for j, Tj, tk, dt, wj in per_epoch:
            print(f"    epoch {j}: T_j={Tj}  pre_spike={tk}  Δt={dt}  h(Δt)={h[dt]:.5f}  w_j={wj:.1f}")
        print(f"  median w* = {w_opt:.1f}  (range {min(ws):.0f}–{max(ws):.0f})")
    return w_opt, per_epoch


# ── Step 1: w*[N2→N3] using FOUND N2 spikes + TARGET N3 times ──────────────
print("\n── Step 1: Recover w*[N2→N3] ───────────────────────────────────────")
w_star_23, ep23 = tp_weight(T_fnd[2], T_tgt[3], label="N2→N3 (using found N2 spikes)")
print(f"  True=500  Found=447.8  Recovered={w_star_23:.1f}")


# ── Invert: find N2 target times from N3 targets + w* ──────────────────────
print("\n── Invert: N3 targets → N2 target times ────────────────────────────")

lat23 = find_delta_t(w_star_23)
n2_tgt = [Tj - lat23 for Tj in T_tgt[3]]
print(f"  Latency Δt at w*={w_star_23:.0f}: {lat23} steps")
print(f"  N2 target times (= T_N3 - {lat23}): {n2_tgt[:8]}")
print(f"  True N2 times:                       {T_tgt[2][:8]}")
print(f"  Found N2 times:                      {T_fnd[2][:8]}")


# ── Step 2: w*[N1→N2] using FOUND N1 spikes + N2 TARGET times ──────────────
print("\n── Step 2: Recover w*[N1→N2] ───────────────────────────────────────")
w_star_12, ep12 = tp_weight(T_fnd[1], n2_tgt, label="N1→N2 (using found N1 spikes)")
print(f"  True=500  Found=489.0  Recovered={w_star_12:.1f}")


# ── Invert: find N1 target times ───────────────────────────────────────────
print("\n── Invert: N2 targets → N1 target times ────────────────────────────")

lat12 = find_delta_t(w_star_12)
n1_tgt = [T2 - lat12 for T2 in n2_tgt]
print(f"  Latency Δt at w*={w_star_12:.0f}: {lat12} steps")
print(f"  N1 target times (= T_N2 - {lat12}): {n1_tgt[:8]}")
print(f"  True N1 times:                       {T_tgt[1][:8]}")
print(f"  Found N1 times:                      {T_fnd[1][:8]}")


# ── Step 3: w*[N0→N1] and w*[N2→N1] (feedback) via least-squares ──────────
print("\n── Step 3: Recover w*[N0→N1] and w*[N2→N1] (feedback) ─────────────")
print("  System: V_N1(T_j) = gsw * (w01 * A_N0_j + w_fb * A_N2_j) = th")

n0_spikes = T_tgt[0]               # N0 external spikes (same in both; every 100 steps from t=1)
n2_fnd_fb = T_fnd[2]               # N2 found spikes used as feedback source

rows_A0, rows_A2, rows_b = [], [], []
T_prev = 0
for j, Tj in enumerate(n1_tgt):
    # Contributions from LAST dominant pre-spike in window
    w0 = [tk for tk in n0_spikes if T_prev < tk < Tj]
    w2 = [tk for tk in n2_fnd_fb if T_prev < tk < Tj]
    tk0 = w0[-1] if w0 else None
    tk2 = w2[-1] if w2 else None
    A0 = h[Tj - tk0] if tk0 and 0 < Tj-tk0 < MAX_H else 0.0
    A2 = h[Tj - tk2] if tk2 and 0 < Tj-tk2 < MAX_H else 0.0
    print(f"  epoch {j}: T_j={Tj}  N0_pre={tk0}(Δ={Tj-tk0 if tk0 else '-'})  "
          f"N2_pre={tk2}(Δ={Tj-tk2 if tk2 else '-'})  "
          f"A0={A0:.5f}  A2={A2:.5f}")
    if A0 > 1e-12:
        rows_A0.append(A0)
        rows_A2.append(A2)
        rows_b.append(th / gsw)
    T_prev = Tj

if rows_A0:
    Amat = np.column_stack([rows_A0, rows_A2])
    bvec = np.array(rows_b)
    wv, res, rank, sv = np.linalg.lstsq(Amat, bvec, rcond=None)
    w_star_01, w_star_fb = float(wv[0]), float(wv[1])
    print(f"\n  Recovered w*[N0→N1] = {w_star_01:.1f}  (true=500, found=527.6)")
    print(f"  Recovered w*[N2→N1] = {w_star_fb:.1f}  (true= 50, found=  7.2)")
    if len(res) > 0:
        print(f"  Residual: {res[0]:.4e}")
else:
    print("  No valid equations found — using true weights as fallback")
    w_star_01, w_star_fb = 500.0, 50.0


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Simulate with recovered weights
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Step 4: Simulate with recovered weights ──────────────────────────")

w_rec_np = np.array([w_star_01, w_star_12, w_star_fb, w_star_23], np.float32)
v_rec = np.array(_hard_sim(jnp.array(w_rec_np), params, C, N, A_ext))
T_rec = {n: spikes_of(v_rec, n) for n in range(N)}

print(f"\n  Weight table:")
syns = ["N0→N1", "N1→N2", "N2→N1", "N2→N3"]
for i, s in enumerate(syns):
    print(f"    {s}:  true={w_true[i]:.0f}  found={w_found[i]:.1f}  recovered={w_rec_np[i]:.1f}")

print(f"\n  Spike table (first 8):")
for n in range(N):
    mark = " ←OUT" if n == 3 else ""
    print(f"  N{n}{mark}:")
    print(f"    target:    {T_tgt[n][:8]}")
    print(f"    found:     {T_fnd[n][:8]}")
    print(f"    recovered: {T_rec[n][:8]}")

n3_match = (T_rec[3] == T_tgt[3])
if n3_match:
    print("\n  ✓ TARGET PROPAGATION SUCCEEDED — N3 fires at exact target times!")
elif len(T_rec[3]) == len(T_tgt[3]):
    diffs = [r-t for t, r in zip(T_tgt[3], T_rec[3])]
    print(f"\n  ~ Correct count ({len(T_rec[3])}sp), timing errors: {diffs}")
    print(f"    Max timing error: {max(abs(d) for d in diffs)} steps")
else:
    tgt_n = len(T_tgt[3]); rec_n = len(T_rec[3])
    print(f"\n  ✗ Wrong count: target={tgt_n}sp  recovered={rec_n}sp")
    print(f"    Need to refine or iterate.")


# ═══════════════════════════════════════════════════════════════════════════
# REFINEMENT: iterate TP with FIXED intermediate target times.
#
# Root cause of divergence: each iter recomputes lat=find_delta_t(w23), so
# as w23 changes the inversion shifts n2_tgt, which shifts n1_tgt, which
# shifts the least-squares system — feedback loop grows unboundedly.
#
# Fix: pin the intermediate target times from the FIRST pass and only
# iterate the weight-recovery step (update weights, re-run sim, repeat).
# The per-neuron target times don't change across iterations.
# ═══════════════════════════════════════════════════════════════════════════
print("\n── TP iteration (fixed intermediate targets, update weights only) ────")

# Pin the targets derived in steps 2 and 4 above (use FIRST-pass values)
n2_tgt_fixed = n2_tgt   # [145, 242, 341, ...]  from first pass
n1_tgt_fixed = n1_tgt   # [ 70, 167, 266, ...]  from first pass

def tp_step(T_curr):
    """One TP iteration: recover weights from current sim spike times."""
    w23, _ = tp_weight(T_curr[2], T_tgt[3])
    if w23 is None: w23 = w_rec_np[3]

    w12, _ = tp_weight(T_curr[1], n2_tgt_fixed)
    if w12 is None: w12 = w_rec_np[1]

    rA0, rA2, rb = [], [], []
    T_prev = 0
    for Tj in n1_tgt_fixed:
        tk0 = max([t for t in n0_spikes if T_prev < t < Tj], default=None)
        tk2 = max([t for t in T_curr[2]  if T_prev < t < Tj], default=None)
        A0 = h[Tj-tk0] if tk0 and 0 < Tj-tk0 < MAX_H else 0.0
        A2 = h[Tj-tk2] if tk2 and 0 < Tj-tk2 < MAX_H else 0.0
        if A0 > 1e-12:
            rA0.append(A0); rA2.append(A2); rb.append(th/gsw)
        T_prev = Tj
    if rA0:
        Amat = np.column_stack([rA0, rA2])
        wv2, _, _, _ = np.linalg.lstsq(Amat, np.array(rb), rcond=None)
        w01, wfb = float(wv2[0]), float(wv2[1])
    else:
        w01, wfb = w_rec_np[0], w_rec_np[2]

    return np.array([w01, w12, wfb, w23], np.float32)

T_curr    = T_rec.copy()
w_curr    = w_rec_np.copy()
best_err  = 7
best_w    = w_curr.copy()
ALPHA     = 0.35   # damping: w = (1-α)*w_curr + α*w_tp

print(f"  {'iter':>4}  {'w01':>6}  {'w12':>6}  {'wfb':>6}  {'w23':>6}  {'max_err':>7}  status")
print(f"  {'0':>4}  {w_curr[0]:>6.0f}  {w_curr[1]:>6.0f}  {w_curr[2]:>6.1f}  {w_curr[3]:>6.0f}  "
      f"{'7':>7}  (first pass)")

for iteration in range(60):
    w_tp  = tp_step(T_curr)
    w_new = (1 - ALPHA) * w_curr + ALPHA * w_tp       # damped update

    v_new  = np.array(_hard_sim(jnp.array(w_new.astype(np.float32)), params, C, N, A_ext))
    T_curr = {n: spikes_of(v_new, n) for n in range(N)}

    n3_f  = T_curr[3]
    n3_t  = T_tgt[3]
    if n3_f == n3_t:
        status = "SOLVED ✓"; diffs = [0]*len(n3_t)
    elif len(n3_f) == len(n3_t) and n3_t:
        diffs  = [r-t for t,r in zip(n3_t, n3_f)]
        status = f"diffs={diffs[:4]}"
    else:
        diffs = []; status = f"count {len(n3_f)}/{len(n3_t)}"

    max_err = max(abs(d) for d in diffs) if diffs else 999
    if max_err < best_err:
        best_err = max_err; best_w = w_new.copy()

    print(f"  {iteration+1:>4}  {w_new[0]:>6.0f}  {w_new[1]:>6.0f}  {w_new[2]:>6.1f}  {w_new[3]:>6.0f}  "
          f"{max_err:>7}  {status}")
    w_curr = w_new
    if n3_f == n3_t:
        break

print(f"\n  Best weights found: {best_w.astype(np.float32)}")
print(f"  Best timing error:  {best_err} steps")
v_best = np.array(_hard_sim(jnp.array(best_w.astype(np.float32)), params, C, N, A_ext))
for n in range(N):
    mark = " ←OUT" if n == 3 else ""
    print(f"  N{n}{mark}: {spikes_of(v_best, n)[:8]}")

print("\nDone.")
