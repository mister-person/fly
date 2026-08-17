"""Minimal failing test cases for two SNN BPTT failure modes.

From sweep results:
  - Timing failure:  4-neuron chain+bypass (k=2, N0→N1→N2→N3, bypass N0→N3)
  - Count failure:   4-neuron recurrent  (N0→N1⇄N2→N3, feedback w21=50)

Both are reproduced by the soft homotopy (not just hard BPTT).
The failure is from local minima in the soft landscape, not gradient vanishing.
"""

import sys, os, types, dataclasses, time
os.environ.setdefault("LOSS", "st")

for _n, _attrs in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n); [setattr(_m, k, v) for k, v in _attrs.items()]; sys.modules[_n] = _m
sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import jax, jax.numpy as jnp

from homotopy_core import hard_sim as _hard_sim, soft_sim
import jax_spiking_model as sim
from recurrent_compare import make_soft_stage, fwd_exp_conv, BETAS, TAU_SCHEDULE, TAU

params = dataclasses.replace(sim.default_params, steps=1000)
th     = params.threshold
A      = jnp.array([0])

NR   = int(os.environ.get("NR",   "8"))
NOPT = int(os.environ.get("NOPT", "600"))

def spikes_of(V, n):  return np.where(V[:, n] >= th)[0].tolist()

def soft_loss_at(w, C_jnp, N, target_jnp, out_ns):
    decay = jnp.float32(np.exp(-1.0 / TAU))
    def _l(w_):
        v  = soft_sim(w_, 34.0, params, C_jnp, N, A)  # high beta = near-hard
        S  = fwd_exp_conv(jax.nn.sigmoid(34.0 * (v / th - 1.0)), decay)
        St = fwd_exp_conv(jax.nn.sigmoid(34.0 * (target_jnp / th - 1.0)), decay)
        return sum(jnp.sum((St[:, n] - S[:, n])**2) for n in out_ns)
    return jax.value_and_grad(_l)(w)

def run_case(C_np, N, w_true_np, out_ns, extra_seeds=None):
    """Run soft homotopy NR times, return all results for diagnostics."""
    C_jnp     = jnp.array(C_np, jnp.int32)
    w_true    = jnp.array(w_true_np, jnp.float32)
    lo        = w_true * 0.1
    hi        = w_true * 5.0
    target_v  = np.array(_hard_sim(w_true, params, C_jnp, N, A))
    target_jnp = jnp.array(target_v)
    train_ns  = list(range(N))

    stage = make_soft_stage(params, C_jnp, N, A, train_ns)
    d0    = jnp.float32(np.exp(-1.0 / TAU_SCHEDULE[0]))
    stage(w_true, w_true, lo, hi, jnp.float32(1.0), jnp.float32(1.0), d0)

    seeds = list(range(42, 42 + NR)) + (extra_seeds or [])
    best_loss, best_w = float("inf"), w_true
    t0 = time.perf_counter()
    for seed in seeds:
        rng = np.random.default_rng(seed)
        w = w_true * jnp.array(rng.uniform(0.5, 1.5, len(w_true_np)), jnp.float32)
        for beta, tau_i in zip(BETAS, TAU_SCHEDULE):
            lr    = 1.0 if beta <= 2 else (0.5 if beta <= 8 else 0.2)
            decay = jnp.float32(np.exp(-1.0 / tau_i))
            w     = stage(w, w_true, lo, hi, jnp.float32(beta), jnp.float32(lr), decay)
        v_f  = np.array(_hard_sim(w, params, C_jnp, N, A))
        loss = float(sum(np.sum((target_v[:, n] - v_f[:, n])**2) for n in out_ns))
        if loss < best_loss:
            best_loss, best_w = loss, w
    wall = time.perf_counter() - t0
    v_f = np.array(_hard_sim(best_w, params, C_jnp, N, A))
    return best_loss, target_v, v_f, best_w, target_jnp, C_jnp, wall

def diagnose(label, C_np, N, w_true_np, out_ns, v_found, v_target, best_w, C_jnp, target_jnp):
    """Print detailed diagnosis matching the 50-neuron diagnostic format."""
    w_true = np.array(w_true_np)
    w_found = np.array(best_w)
    edges = np.array(C_np)

    def pre_of(n):
        idx = np.where(edges[:, 1] == n)[0]
        return idx, edges[idx, 0]

    print(f"\n── Spike raster (target vs found) ─────────────────────────────")
    for n in range(N):
        sp_t = spikes_of(v_target, n)
        sp_f = spikes_of(v_found, n)
        maxv = v_found[:, n].max()
        mark = " ←OUT" if n in out_ns else ""
        flag = "★" if sp_t == sp_f and sp_t else ("✗" if len(sp_t) != len(sp_f) else "~")
        print(f"  N{n}: {flag} tgt={len(sp_t)} {sp_t[:5]}  fnd={len(sp_f)} {sp_f[:5]}"
              f"  max_V={maxv:.5f} ({maxv/th*100:.0f}%th){mark}")

    print(f"\n── Output neuron voltages at target spike times ────────────────")
    for n in out_ns:
        sp_t = spikes_of(v_target, n)
        print(f"  N{n} (target={len(sp_t)}sp, found={len(spikes_of(v_found, n))}sp):")
        for tt in sp_t:
            vt, vf = v_target[tt, n], v_found[tt, n]
            gap = th - vf
            s = "FIRES" if vf >= th else f"SILENT  gap={gap:.4f} ({gap/th*100:.0f}%th)"
            print(f"    t={tt:4d}  V_target={vt:.5f}  V_found={vf:.5f}  {s}")

    print(f"\n── Pre-synaptic inputs to output neurons ───────────────────────")
    for n in out_ns:
        syn_idxs, pre = pre_of(n)
        sp_t = spikes_of(v_target, n)
        print(f"  N{n} receives from: {sorted(pre.tolist())}")
        print(f"    true_w={[f'{w_true[i]:.0f}' for i in syn_idxs]}  "
              f"found_w={[f'{w_found[i]:.0f}' for i in syn_idxs]}")
        for tt in sp_t:
            t_start = max(0, tt - 50)
            has_any = False
            for si, p in zip(syn_idxs, pre):
                tp = [s for s in spikes_of(v_target, p) if t_start <= s <= tt]
                fp = [s for s in spikes_of(v_found, p) if t_start <= s <= tt]
                if tp or fp:
                    if not has_any:
                        print(f"    @ t={tt} (window [{t_start},{tt}]):")
                        has_any = True
                    print(f"      N{p}: tw={w_true[si]:.0f}  fw={w_found[si]:.0f}  "
                          f"tgt_spikes={tp}  fnd_spikes={fp}")

    print(f"\n── Gradients at found weights (hard ST loss ≈ 0; soft ST at β=34) ──")
    loss_s, grad_s = soft_loss_at(best_w, C_jnp, N, target_jnp, out_ns)
    print(f"  soft_loss(β=34) = {float(loss_s):.4e}  ||grad|| = {float(jnp.linalg.norm(grad_s)):.4e}")
    print(f"  per-weight gradients:")
    g = np.array(grad_s)
    edges_np = np.array(C_np)
    for i in range(len(g)):
        print(f"    w[N{edges_np[i,0]}→N{edges_np[i,1]}]:  true={w_true[i]:.0f}  "
              f"found={w_found[i]:.1f}  grad={g[i]:+.4e}")

    print(f"\n── Soft landscape: loss along true→found axis ──────────────────")
    decay = jnp.float32(np.exp(-1.0 / TAU))
    def st_hard_loss(w_):
        v  = _hard_sim(w_, params, C_jnp, N, A)
        S  = fwd_exp_conv((v >= th).astype(jnp.float32), decay)
        St = fwd_exp_conv((target_jnp >= th).astype(jnp.float32), decay)
        return float(sum(jnp.sum((St[:, n] - S[:, n])**2) for n in out_ns))
    w_true_j = jnp.array(w_true_np, jnp.float32)
    print(f"  α    loss(α·true + (1-α)·found)  [hard ST]")
    for alpha in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        w_interp = alpha * w_true_j + (1-alpha) * best_w
        l = st_hard_loss(w_interp)
        print(f"  {alpha:.2f}   {l:.4e}")


# ═══════════════════════════════════════════════════════════════════════════
# FAILURE MODE 1: Timing local minimum
# 4 neurons: N0→N1→N2→N3 + bypass N0→N3
# k=2 chain — first case where soft homotopy gets stuck in wrong timing
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("FAILURE MODE 1: Timing local minimum")
print("Topology: N0(ext)→N1→N2→N3  +  bypass N0→N3")
print("True: chain=500, bypass=50.  Target: N3 fires via chain ~214 steps after N0.")
print("Wrong attractor: N3 fires via bypass ~71 steps after N0.")
print("═"*70)

C1 = np.array([[0,1],[1,2],[2,3],[0,3]], dtype=np.int32)
N1, out_ns1 = 4, [3]
w_true1 = np.array([500., 500., 500., 50.], dtype=np.float32)

loss1, target_v1, v_found1, best_w1, tgt_jnp1, C1_j, wall1 = run_case(
    C1, N1, w_true1, out_ns1)

sp_t = spikes_of(target_v1, 3); sp_f = spikes_of(v_found1, 3)
print(f"\nResult: loss={loss1:.4e}  wall={wall1:.1f}s")
print(f"Target N3: {sp_t[:6]}")
print(f"Found  N3: {sp_f[:6]}")
if sp_t and sp_f:
    diffs = [f-t for t, f in zip(sp_t[:min(len(sp_t),len(sp_f))], sp_f[:min(len(sp_t),len(sp_f))])]
    print(f"Timing errors: {diffs}  max={max(abs(d) for d in diffs)} steps")

diagnose("timing_fork", C1, N1, w_true1, out_ns1,
         v_found1, target_v1, best_w1, C1_j, tgt_jnp1)


# ═══════════════════════════════════════════════════════════════════════════
# FAILURE MODE 2: Count failure / dead output (recurrent feedback)
# 4 neurons: N0→N1→N2→N3, feedback N2→N1 (w21=50)
# Reproduces case 2's wrong-count / dead-output failure
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("FAILURE MODE 2: Wrong count / dead output (recurrent feedback)")
print("Topology: N0(ext)→N1→N2→N3,  feedback N2→N1 (w21=50)")
print("True: w01=w12=w2out=500, w_fb=50.  Recurrence shifts N1/N2 timing.")
print("═"*70)

C2 = np.array([[0,1],[1,2],[2,1],[2,3]], dtype=np.int32)
N2_n, out_ns2 = 4, [3]
w_true2 = np.array([500., 500., 50., 500.], dtype=np.float32)

loss2, target_v2, v_found2, best_w2, tgt_jnp2, C2_j, wall2 = run_case(
    C2, N2_n, w_true2, out_ns2)

sp_t2 = spikes_of(target_v2, 3); sp_f2 = spikes_of(v_found2, 3)
print(f"\nResult: loss={loss2:.4e}  wall={wall2:.1f}s")
print(f"Target N3: {sp_t2[:6]}")
print(f"Found  N3: {sp_f2[:6]}")

diagnose("recurrent_feedback", C2, N2_n, w_true2, out_ns2,
         v_found2, target_v2, best_w2, C2_j, tgt_jnp2)


# ═══════════════════════════════════════════════════════════════════════════
# HARDER VERSIONS: push until loss matches 50-neuron scale (1e-2)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("SCALING: increase recurrent feedback until loss reaches 50-neuron scale (~1e-2)")
print("═"*70)
print(f"{'w_fb':>6}  {'tgt':>5}  {'fnd':>5}  {'loss':>10}  status")

for w_fb in [50, 100, 150, 200, 300, 500]:
    C_ = np.array([[0,1],[1,2],[2,1],[2,3]], dtype=np.int32)
    w_ = np.array([500., 500., float(w_fb), 500.], dtype=np.float32)
    l, tv, fv, bw, tj, cj, w = run_case(C_, 4, w_, [3])
    sp_t_ = spikes_of(tv, 3); sp_f_ = spikes_of(fv, 3)
    status = "SOLVED" if sp_t_ == sp_f_ else ("wrong_timing" if len(sp_t_)==len(sp_f_) else "wrong_count")
    print(f"{w_fb:>6}  {len(sp_t_):>5}  {len(sp_f_):>5}  {l:>10.3e}  {status}")

print("\nDone.")
