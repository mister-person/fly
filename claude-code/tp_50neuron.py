"""Target propagation on the 50-neuron case 2.

Strategy:
  For each neuron n (except N0), given TRUE pre-synaptic spike times and
  TRUE target spike times, solve the multi-input 1-hop linear system:

    sum_k w_k * A_{k,j} = th/gsw   for each target spike j of n
    A_{k,j} = h(T_j - t_k)  summed over relevant pre-spikes of neuron k

  → least-squares solution for all incoming weights to n.

After recovering weights, run hard sim and compare to target.
Also try iterating with the recovered sim's spike times.
"""

import sys, os, types, dataclasses, time
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
from test_cases import RECURRENT_CASES, _make_recurrent_weights

CASE_IDX = int(os.environ.get("CASE", "2"))
N_ITER   = int(os.environ.get("N_ITER", "10"))
ALPHA    = float(os.environ.get("ALPHA", "0.4"))  # damping
RIDGE    = float(os.environ.get("RIDGE", "1e-3"))  # Tikhonov frac for near-singular NNLS
                                                   # (1e-3 best on case 2: oracle 1.07e-2→9.9e-3)
DENSE_UP = int(os.environ.get("DENSE_UP", "0"))    # 1 = upper-bound every sub-threshold
                                                   # step (catch near-miss overfiring)
DENSE_STRIDE = int(os.environ.get("DENSE_STRIDE", "3"))

tc = RECURRENT_CASES[CASE_IDX]
conns, tw = _make_recurrent_weights(
    tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
    tc["num_neurons"], tc["output_neurons"])

params = dataclasses.replace(sim.default_params, steps=1000)
th     = params.threshold
gsw    = params.global_synapse_weight
delay  = params.delay_iters
nd     = float(params.neuron_decay)
rd     = float(params.rise_decay)
A_ext  = jnp.array([0])

C_np   = np.array(conns, dtype=np.int32)
C      = jnp.array(C_np)
N      = tc["num_neurons"]
outs   = tc["output_neurons"]

w_true = np.array(tw, np.float32)
lo     = w_true * 0.1
hi     = w_true * 5.0

print(f"Case {CASE_IDX}: {tc['name']}  ({N} neurons, {len(tw)} synapses)")
print(f"Output neurons: {outs}")

# ── impulse response (linear, no threshold) ───────────────────────────────
MAX_H = 600
h = np.zeros(MAX_H)
R = V = 0.0
for t in range(MAX_H):
    upd = 1.0 if t == delay else 0.0
    R = (R + upd) * rd
    V = (V - R) * nd + R
    h[t] = V
print(f"Impulse response peak: h[{h.argmax()}]={h.max():.5f}  "
      f"(w=500 fires at Δt where {500*gsw}*h[Δt]≥{th:.4f})")

# ── topology helpers ──────────────────────────────────────────────────────
edges = C_np  # shape (n_syn, 2): edges[k] = [pre, post]

def pre_of(n):
    idx = np.where(edges[:, 1] == n)[0]
    return idx, edges[idx, 0]   # (syn_indices, pre_neuron_ids)

def spikes_of(V_np, n):
    return np.where(V_np[:, n] >= th)[0].tolist()

# ── true simulation ───────────────────────────────────────────────────────
target_v = np.array(_hard_sim(jnp.array(w_true), params, C, N, A_ext))
T_true   = {n: spikes_of(target_v, n) for n in range(N)}

# ── found weights (best from soft homotopy) ───────────────────────────────
found_path = f"best_weights_case{CASE_IDX}.npy"
if os.path.exists(found_path):
    w_found  = np.load(found_path).astype(np.float32)
    found_v  = np.array(_hard_sim(jnp.array(w_found), params, C, N, A_ext))
    T_found  = {n: spikes_of(found_v, n) for n in range(N)}
    found_loss = sum(np.sum((target_v[:, n] - found_v[:, n])**2) for n in outs)
    print(f"Loaded found weights (loss={found_loss:.3e})")
else:
    w_found = w_true.copy()
    T_found = T_true.copy()
    print("No saved weights; using true as baseline")

print(f"\nTrue output spikes:  "
      + "  ".join(f"N{n}={len(T_true[n])}sp" for n in outs))
print(f"Found output spikes: "
      + "  ".join(f"N{n}={len(T_found[n])}sp" for n in outs))
print(f"Alive neurons in found: "
      f"{sum(1 for n in range(N) if len(T_found[n]) > 0)}/{N}")


# ═══════════════════════════════════════════════════════════════════════════
# CORE: solve per-neuron linear systems
#
# For each non-external neuron n:
#   For each target spike T_j (j=0..m-1):
#     One equation:  sum_k w_k * A_{k,j} = th/gsw
#     A_{k,j} = h(T_j - t_last_k)  where t_last_k is the last pre-spike
#               of input k within the epoch (T_{j-1}, T_j).
#
# Stack equations → least-squares solution for w = [w_{k1→n}, ..., w_{km→n}].
# ═══════════════════════════════════════════════════════════════════════════

nd32 = float(np.float32(params.neuron_decay))
rd32 = float(np.float32(params.rise_decay))
h_f32 = np.zeros(MAX_H)
_R = _V = 0.0
for _t in range(MAX_H):
    _upd = 1.0 if _t == delay else 0.0
    _R   = (_R + _upd) * rd32
    _V   = (_V - _R) * nd32 + _R
    h_f32[_t] = _V
print(f"h[71] f64={h[71]:.8f}  f32={h_f32[71]:.8f}  diff={h[71]-h_f32[71]:.2e}")
# Use float64 h — difference from float32 is ~5e-7, negligible

def nnls_ridge(A_mat, b_vec, ridge_frac=0.0, w_prior=None):
    """NNLS with optional Tikhonov reg: min ||A w - b||^2 + lam||w - w_prior||^2, w>=0.

    lam = ridge_frac * mean(diag(A^T A)) scales with the data so one ridge_frac
    transfers across neurons/networks. Only bites on near-singular directions
    (e.g. two inputs firing near-simultaneously → proportional columns), pulling
    the split toward min-norm (equal) instead of an NNLS vertex [w_tot, 0].
    Well-conditioned systems are barely perturbed. w_prior=None → prior 0.
    """
    from scipy.optimize import nnls
    ncol = A_mat.shape[1]
    if ridge_frac <= 0:
        return nnls(A_mat, b_vec)
    scale = np.trace(A_mat.T @ A_mat) / max(ncol, 1)
    lam   = ridge_frac * scale
    wp    = np.zeros(ncol) if w_prior is None else np.asarray(w_prior, float)
    A_aug = np.vstack([A_mat, np.sqrt(lam) * np.eye(ncol)])
    b_aug = np.concatenate([b_vec, np.sqrt(lam) * wp])
    return nnls(A_aug, b_aug)

def build_contribution_matrix(n, pre_spike_times_by_neuron, eval_times, T_epoch_starts):
    """A[j, k] = sum of h(eval_times[j] - t') for pre-spikes t' of k
    within the epoch window (T_epoch_starts[j], eval_times[j]).
    """
    syn_idxs, pres = pre_of(n)
    rows = []
    for j, Tj in enumerate(eval_times):
        T_prev = T_epoch_starts[j]
        row = np.zeros(len(pres))
        for col_i, p in enumerate(pres):
            for t_k in pre_spike_times_by_neuron.get(int(p), []):
                if T_prev < t_k < Tj:
                    dt = Tj - t_k
                    if 0 < dt < MAX_H:
                        row[col_i] += h[dt]
        rows.append(row)
    return syn_idxs, pres, np.array(rows)

def tp_weights_for_neuron(n, pre_spike_times_by_neuron, tgt_times,
                           margin=0.10, v_ref=None):
    """Constrained QP for all synapses into neuron n.

    Lower constraints: at each target time Tj, voltage must reach b_lo (fire).
      b_lo[j] = V_ref[Tj, n] / gsw  if v_ref is provided (exact threshold value)
              = th / gsw             otherwise
    Upper constraints: at sampled non-target times, voltage must stay below
                       (1-margin)*th/gsw (no spurious firing).
    Objective: minimize ||w - w_nnls||^2.

    margin=0: pure NNLS (lower bounds only, no upper constraints).
    v_ref: optional voltage array from which to read actual threshold voltages.
    """
    syn_idxs, pres = pre_of(n)
    if len(pres) == 0 or len(tgt_times) == 0:
        return None

    refractory = params.refractory_iters

    # ── lower-bound rows: must fire at each target time ───────────────────
    tgt_arr         = list(tgt_times)
    epoch_starts_lo = [0] + tgt_arr[:-1]
    _, _, A_lo = build_contribution_matrix(
        n, pre_spike_times_by_neuron, tgt_arr, epoch_starts_lo)

    # Both NNLS objective and QP lower bound: use th/gsw.
    # (V_actual/gsw as NNLS RHS gives marginally higher weights, which can
    #  interact poorly with the upper-bound QP constraints at (1-margin)*th/gsw.)
    b_nnls = np.full(len(tgt_arr), th / gsw)
    b_lo   = b_nnls

    # Drop rows where no input is visible (can't fire anyway)
    valid_lo = A_lo.max(axis=1) > 1e-12
    A_lo, b_lo, b_nnls = A_lo[valid_lo], b_lo[valid_lo], b_nnls[valid_lo]
    if len(A_lo) == 0:
        return None

    # ── non-negative LS baseline (ridge resolves near-singular input splits) ─
    from scipy.optimize import minimize
    w_nnls, _ = nnls_ridge(A_lo, b_nnls, ridge_frac=RIDGE)

    if margin <= 0:
        return syn_idxs, np.clip(w_nnls, lo[syn_idxs], hi[syn_idxs])

    # ── upper-bound rows: must NOT fire at non-target times ───────────────
    # DENSE_UP=1 samples EVERY sub-threshold step (strided) to catch near-miss
    # bumps that 4-point sampling misses (prevents spurious extra spikes).
    nt_times, nt_epochs = [], []
    for j in range(len(tgt_arr) - 1):
        T_ref_end = tgt_arr[j] + refractory + 2
        T_next    = tgt_arr[j + 1]
        if T_ref_end >= T_next:
            continue
        if DENSE_UP:
            for Tnt in range(T_ref_end, T_next, DENSE_STRIDE):
                nt_times.append(Tnt); nt_epochs.append(tgt_arr[j])
        else:
            for frac in [0.25, 0.45, 0.65, 0.85]:
                Tnt = int(T_ref_end + frac * (T_next - T_ref_end))
                nt_times.append(Tnt); nt_epochs.append(tgt_arr[j])

    A_up = np.zeros((0, len(pres)))
    b_up = np.zeros(0)
    if nt_times:
        _, _, A_up = build_contribution_matrix(
            n, pre_spike_times_by_neuron, nt_times, nt_epochs)
        b_up = np.full(len(nt_times), (1.0 - margin) * th / gsw)
        # Keep only rows where at least one input is visible
        valid_up = A_up.max(axis=1) > 1e-12
        A_up, b_up = A_up[valid_up], b_up[valid_up]

    # ── QP: min ||w - w_nnls||^2  s.t. A_lo@w >= b_lo, A_up@w <= b_up ──
    constraints = [
        {'type': 'ineq', 'fun': lambda w, A=A_lo, b=b_lo: A @ w - b},
    ]
    if len(A_up) > 0:
        constraints.append(
            {'type': 'ineq', 'fun': lambda w, A=A_up, b=b_up: b - A @ w})

    bounds = [(float(lo[si]), float(hi[si])) for si in syn_idxs]

    result = minimize(
        lambda w: 0.5 * float(np.dot(w - w_nnls, w - w_nnls)),
        w_nnls.copy(),
        jac=lambda w: (w - w_nnls),
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 1e-10, 'maxiter': 2000, 'disp': False},
    )
    w_sol = result.x if (result.success or result.status in [0, 4]) else w_nnls
    return syn_idxs, np.clip(w_sol, lo[syn_idxs], hi[syn_idxs])

def tp_pass(pre_spike_times_by_neuron, margin=0.10, v_ref=None):
    """Full TP pass: recover weights for all neurons using given pre-spike times.

    v_ref: if provided, use V_ref[T_j, n] as the exact RHS instead of th/gsw.
           Only meaningful when the reference sim produces spikes at T_true[n].
    """
    w_new = w_true.copy()
    n_solved = 0
    for n in range(N):
        if n == 0:
            continue
        tgt = T_true[n]
        if not tgt:
            continue
        result = tp_weights_for_neuron(n, pre_spike_times_by_neuron, tgt,
                                        margin=margin, v_ref=v_ref)
        if result is None:
            continue
        syn_idxs, w_sol = result
        w_new[syn_idxs] = w_sol
        n_solved += 1
    return w_new, n_solved


# ═══════════════════════════════════════════════════════════════════════════
# PASS 0: TP using TRUE pre-spike times (oracle)
# Establishes how close TP can get when given the true dynamics
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("PASS 0: TP with TRUE pre-spike times (oracle)")
print("═"*70)

def eval_tp(w_tp, label, ref_loss):
    v_tp = np.array(_hard_sim(jnp.array(w_tp.astype(np.float32)), params, C, N, A_ext))
    T_tp = {n: spikes_of(v_tp, n) for n in range(N)}
    loss = float(sum(np.sum((target_v[:, n] - v_tp[:, n])**2) for n in outs))
    alive = sum(1 for n in range(N) if T_tp[n])
    n_count = sum(1 for n in range(N) if len(T_tp[n]) == len(T_true[n]) and T_true[n])
    print(f"  {label}:  loss={loss:.3e}  alive={alive}/{N}  count_match={n_count}")
    for n in outs:
        ts, rs = T_true[n], T_tp[n]
        if ts and rs and len(ts) == len(rs):
            diffs = [r-t for t,r in zip(ts, rs)]
            print(f"    N{n}: {len(rs)}sp  timing errors {diffs}  max={max(abs(d) for d in diffs)}")
        else:
            print(f"    N{n}: tgt={len(ts)}sp  tp={len(rs)}sp  {rs[:5]}")
    w_err = np.abs(w_tp - w_true)
    print(f"  Weight recovery: mean_err={w_err.mean():.1f}  "
          f"within_10%={100*np.mean(w_err/(w_true+1e-6)<0.10):.0f}%  "
          f"within_20%={100*np.mean(w_err/(w_true+1e-6)<0.20):.0f}%")
    return loss, T_tp, v_tp

# ── sweep over margin values ──────────────────────────────────────────────
print(f"\n── Margin sweep (oracle pre-spike times) ───────────────────────────")
print(f"  {'margin':>8}  {'loss':>10}  {'alive':>6}  {'cnt_match':>9}  outputs")
best_margin_loss = found_loss
best_margin_w    = None
for margin in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
    t0 = time.perf_counter()
    w_m, ns = tp_pass(T_true, margin=margin, v_ref=target_v)
    v_m = np.array(_hard_sim(jnp.array(w_m.astype(np.float32)), params, C, N, A_ext))
    T_m = {n: spikes_of(v_m, n) for n in range(N)}
    loss_m = float(sum(np.sum((target_v[:, n] - v_m[:, n])**2) for n in outs))
    alive_m = sum(1 for n in range(N) if T_m[n])
    cnt_m   = sum(1 for n in range(N) if len(T_m[n]) == len(T_true[n]) and T_true[n])
    out_str = "  ".join(f"N{n}:{len(T_m[n])}/{len(T_true[n])}" for n in outs)
    wall_m  = time.perf_counter() - t0
    flag = " ★" if loss_m < best_margin_loss else ""
    print(f"  {margin:>8.2f}  {loss_m:>10.3e}  {alive_m:>6}  {cnt_m:>9}  {out_str}  ({wall_m:.1f}s){flag}")
    if loss_m < best_margin_loss:
        best_margin_loss = loss_m
        best_margin_w    = w_m.copy()
        best_margin_val  = margin

print(f"\n  Soft homotopy baseline: loss={found_loss:.3e}")
if best_margin_w is not None:
    print(f"  Best margin ({best_margin_val}):  loss={best_margin_loss:.3e}  ← improvement!")
    w_tp0 = best_margin_w
    T_tp0 = None   # recompute below
else:
    print(f"  No margin improved over soft homotopy — using margin=0")
    w_tp0, _ = tp_pass(T_true, margin=0.0)

t0 = time.perf_counter()
loss_tp0, T_tp0, v_tp0 = eval_tp(w_tp0, "Best oracle TP", found_loss)
print(f"  (Wall: {time.perf_counter()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════════
# PASS 1: TP using FOUND pre-spike times (realistic — what TP would do)
# Dead neurons contribute nothing; alive neurons anchor the solution.
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("PASS 1: TP with FOUND pre-spike times (realistic)")
print("═"*70)

t0 = time.perf_counter()
w_tp1, n_solved1 = tp_pass(T_found, margin=best_margin_val if best_margin_w is not None else 0.10)
loss_tp1, T_tp1, v_tp1 = eval_tp(w_tp1, "Found-prior TP", found_loss)
print(f"  (Wall: {time.perf_counter()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════════
# PASS 2: TP using MIXED pre-spike times
# For neurons alive in found: use found spike times.
# For neurons dead in found:  use TRUE spike times (oracle for dead).
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("PASS 2: TP with MIXED pre-spike times (found where alive, true where dead)")
print("═"*70)

T_mixed = {}
for n in range(N):
    T_mixed[n] = T_found[n] if T_found[n] else T_true[n]

t0 = time.perf_counter()
use_margin = best_margin_val if best_margin_w is not None else 0.10
w_tp2, n_solved2 = tp_pass(T_mixed, margin=use_margin)
loss_tp2, T_tp2, v_tp2 = eval_tp(w_tp2, "Mixed-prior TP", found_loss)
print(f"  (Wall: {time.perf_counter()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════════
# ITERATION: start from oracle TP weights, iterate using sim spike times
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print(f"ITERATION from oracle TP (N_ITER={N_ITER}, ALPHA={ALPHA})")
print("═"*70)

def count_match(T_curr):
    return sum(1 for n in range(N) if len(T_curr[n]) == len(T_true[n]) and T_true[n])

def output_spikes_str(T_curr):
    parts = []
    for n in outs:
        tgt, fnd = len(T_true[n]), len(T_curr[n])
        parts.append(f"N{n}:{fnd}/{tgt}")
    return "  ".join(parts)

w_curr   = w_tp0.copy()
T_curr   = T_tp0.copy()
best_loss = loss_tp0
best_w    = w_tp0.copy()
iter_margin = use_margin

print(f"  {'iter':>4}  {'loss':>10}  {'alive':>6}  {'cnt_match':>9}  output_counts")
print(f"  {'0':>4}  {loss_tp0:>10.3e}  {sum(1 for n in range(N) if T_tp0[n]):>6}  "
      f"{count_match(T_tp0):>9}  {output_spikes_str(T_tp0)}")

for it in range(1, N_ITER + 1):
    T_mix_it = {n: (T_curr[n] if T_curr[n] else T_true[n]) for n in range(N)}
    w_tp_it, _ = tp_pass(T_mix_it, margin=iter_margin)
    w_new = (1 - ALPHA) * w_curr + ALPHA * w_tp_it
    w_new = np.clip(w_new, lo, hi).astype(np.float32)

    v_new  = np.array(_hard_sim(jnp.array(w_new), params, C, N, A_ext))
    T_curr = {n: spikes_of(v_new, n) for n in range(N)}
    loss_it = float(sum(np.sum((target_v[:, n] - v_new[:, n])**2) for n in outs))
    alive_it = sum(1 for n in range(N) if T_curr[n])

    if loss_it < best_loss:
        best_loss = loss_it; best_w = w_new.copy()

    print(f"  {it:>4}  {loss_it:>10.3e}  {alive_it:>6}  "
          f"{count_match(T_curr):>9}  {output_spikes_str(T_curr)}")
    w_curr = w_new

# ── final report ─────────────────────────────────────────────────────────
print(f"\n── Best result (loss={best_loss:.3e}) ─────────────────────────────────")
v_best = np.array(_hard_sim(jnp.array(best_w.astype(np.float32)), params, C, N, A_ext))
T_best = {n: spikes_of(v_best, n) for n in range(N)}

print(f"  {'n':>3}  {'tgt':>4}  {'fnd':>4}  {'best_tp':>7}  match")
for n in range(N):
    nt, nf, nr = len(T_true[n]), len(T_found[n]), len(T_best[n])
    if nt == 0 and nf == 0 and nr == 0: continue
    flag = "★" if nr == nt else "✗"
    mark = " ←OUT" if n in outs else ""
    print(f"  {n:>3}  {nt:>4}  {nf:>4}  {nr:>7}  {flag}{mark}")

print(f"\n  Soft homotopy (found): loss={found_loss:.3e}")
print(f"  TP best:               loss={best_loss:.3e}")
print(f"  True:                  loss=0")


# ═══════════════════════════════════════════════════════════════════════════
# WARM-START HOMOTOPY: run soft homotopy from best TP weights.
#
# Skip the low-beta warm-up stages (they exist to escape dead neurons from
# random init). Since TP already has correct spike counts, jump straight
# to high-beta where the surrogate is near-hard.
#
# Compare three starting conditions:
#   A) Cold start (random init, full homotopy) — the original soft homotopy
#   B) TP warm start (best_w), high-beta stages only
#   C) TP warm start (best_w), ALL beta stages (in case low-beta helps too)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("WARM-START HOMOTOPY from best TP weights")
print("═"*70)

from recurrent_compare import make_soft_stage, fwd_exp_conv, BETAS, TAU_SCHEDULE, TAU
import jax

target_jnp = jnp.array(target_v)
train_ns   = list(range(N))

# Soft ST loss (same formulation as recurrent_compare)
def soft_st_loss(w_, beta_):
    decay = jnp.float32(np.exp(-1.0 / TAU))
    v_    = sim.soft_sim(w_, beta_, params, C, N, A_ext)
    S_    = fwd_exp_conv(jax.nn.sigmoid(beta_ * (v_ / th - 1.0)), decay)
    St_   = fwd_exp_conv(jax.nn.sigmoid(beta_ * (target_jnp / th - 1.0)), decay)
    return sum(jnp.sum((St_[:, n] - S_[:, n])**2) for n in outs)

stage = make_soft_stage(params, C, N, A_ext, train_ns)
d0    = jnp.float32(np.exp(-1.0 / TAU_SCHEDULE[0]))
# warm-up call (JIT compile)
_ = stage(jnp.array(w_true, jnp.float32), jnp.array(w_true, jnp.float32),
          jnp.array(lo, jnp.float32), jnp.array(hi, jnp.float32),
          jnp.float32(1.0), jnp.float32(1.0), d0)

lo_j  = jnp.array(lo,     jnp.float32)
hi_j  = jnp.array(hi,     jnp.float32)
wt_j  = jnp.array(w_true, jnp.float32)

def run_homotopy_from(w_init, start_beta_idx=0, label=""):
    """Run homotopy from w_init starting at BETAS[start_beta_idx]."""
    w = jnp.array(w_init, jnp.float32)
    for i, (beta, tau_i) in enumerate(zip(BETAS, TAU_SCHEDULE)):
        if i < start_beta_idx:
            continue
        lr    = 1.0 if beta <= 2 else (0.5 if beta <= 8 else 0.2)
        decay = jnp.float32(np.exp(-1.0 / tau_i))
        w     = stage(w, wt_j, lo_j, hi_j, jnp.float32(beta), jnp.float32(lr), decay)
    v_h  = np.array(_hard_sim(w, params, C, N, A_ext))
    loss = float(sum(np.sum((target_v[:, n] - v_h[:, n])**2) for n in outs))
    T_h  = {n: spikes_of(v_h, n) for n in range(N)}
    alive_h = sum(1 for n in range(N) if T_h[n])
    cnt_h   = sum(1 for n in range(N) if len(T_h[n]) == len(T_true[n]) and T_true[n])
    print(f"  {label:45s}  loss={loss:.3e}  alive={alive_h}/{N}  cnt={cnt_h}")
    for n in outs:
        ts, rs = T_true[n], T_h[n]
        if ts and rs and len(ts) == len(rs):
            diffs = [r-t for t, r in zip(ts, rs)]
            print(f"    N{n}: {len(rs)}sp  timing_errors={diffs}  max={max(abs(d) for d in diffs)}")
        else:
            print(f"    N{n}: tgt={len(ts)}sp  found={len(rs)}sp  {rs[:5]}")
    return loss, np.array(w), T_h

# Find which beta index corresponds to start point
HIGH_BETA_START = next(i for i, b in enumerate(BETAS) if b >= 5)
print(f"BETAS={BETAS}  starting from index {HIGH_BETA_START} (beta={BETAS[HIGH_BETA_START]})\n")

# A) cold start: full homotopy from random init (NR=1 representative run)
rng = np.random.default_rng(42)
w_rand = jnp.array(w_true * rng.uniform(0.5, 1.5, len(w_true)), jnp.float32)
t0 = time.perf_counter()
loss_cold, w_cold, T_cold = run_homotopy_from(w_rand, 0, "A) cold start (full homotopy, seed=42)")
print(f"    Wall: {time.perf_counter()-t0:.1f}s")

# B) TP warm start, high-beta only
t0 = time.perf_counter()
loss_b, w_b, T_b = run_homotopy_from(best_w, HIGH_BETA_START,
    f"B) TP warm start (beta>={BETAS[HIGH_BETA_START]})")
print(f"    Wall: {time.perf_counter()-t0:.1f}s")

# C) TP warm start, all beta stages
t0 = time.perf_counter()
loss_c, w_c, T_c = run_homotopy_from(best_w, 0, "C) TP warm start (all beta stages)")
print(f"    Wall: {time.perf_counter()-t0:.1f}s")

# D) best cold-start (NR=8 like the original)
print(f"\n  Running NR=8 cold starts for fair comparison ...")
t0 = time.perf_counter()
best_cold_loss, best_cold_w = float("inf"), None
for seed in range(42, 50):
    rng_ = np.random.default_rng(seed)
    w_r  = jnp.array(w_true * rng_.uniform(0.5, 1.5, len(w_true)), jnp.float32)
    for i, (beta, tau_i) in enumerate(zip(BETAS, TAU_SCHEDULE)):
        lr    = 1.0 if beta <= 2 else (0.5 if beta <= 8 else 0.2)
        decay = jnp.float32(np.exp(-1.0 / tau_i))
        w_r   = stage(w_r, wt_j, lo_j, hi_j, jnp.float32(beta), jnp.float32(lr), decay)
    v_r  = np.array(_hard_sim(w_r, params, C, N, A_ext))
    l_r  = float(sum(np.sum((target_v[:, n] - v_r[:, n])**2) for n in outs))
    if l_r < best_cold_loss:
        best_cold_loss = l_r; best_cold_w = np.array(w_r)
wall_cold = time.perf_counter() - t0
run_homotopy_from.__doc__  # no-op to keep reference
v_best_cold = np.array(_hard_sim(jnp.array(best_cold_w), params, C, N, A_ext))
T_best_cold = {n: spikes_of(v_best_cold, n) for n in range(N)}
cnt_bc = sum(1 for n in range(N) if len(T_best_cold[n]) == len(T_true[n]) and T_true[n])
print(f"  D) best of NR=8 cold starts{' ':17s}  loss={best_cold_loss:.3e}  "
      f"alive={sum(1 for n in range(N) if T_best_cold[n])}/{N}  cnt={cnt_bc}")
for n in outs:
    ts, rs = T_true[n], T_best_cold[n]
    if ts and rs and len(ts) == len(rs):
        diffs = [r-t for t, r in zip(ts, rs)]
        print(f"    N{n}: {len(rs)}sp  timing_errors={diffs}  max={max(abs(d) for d in diffs)}")
    else:
        print(f"    N{n}: tgt={len(ts)}sp  found={len(rs)}sp  {rs[:5]}")
print(f"    Wall: {wall_cold:.1f}s")

print(f"\n── Summary ────────────────────────────────────────────────────────")
print(f"  {'Method':45s}  {'loss':>10}")
print(f"  {'Soft homotopy (NR=8, saved best)':45s}  {found_loss:>10.3e}")
print(f"  {'TP only (best, no refinement)':45s}  {best_loss:>10.3e}")
print(f"  {'A) cold start (seed=42, full homotopy)':45s}  {loss_cold:>10.3e}")
print(f"  {'D) best of NR=8 cold starts':45s}  {best_cold_loss:>10.3e}")
print(f"  {'B) TP + high-beta homotopy':45s}  {loss_b:>10.3e}")
print(f"  {'C) TP + full homotopy':45s}  {loss_c:>10.3e}")
print(f"  {'True (exact)':45s}  {'0':>10}")

print("\nDone.")
