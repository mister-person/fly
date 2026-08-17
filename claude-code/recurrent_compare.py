"""Compare soft homotopy vs hard-surrogate direct training on 50-neuron recurrent cases.

Soft homotopy:
  - sigmoid forward (soft_sim), target also soft
  - 9-stage beta annealing: 0.5 → 34
  - loss: MSE on train_neurons

Hard surrogate:
  - hard step-function forward
  - 9 Adam phases with same LR schedule
  - loss: MSE on train_neurons; backward seed only at those neurons
  - gradient: manual BPTT with narrow sigmoid-derivative surrogate (beta_surr=30)

Evaluation uses outs=[47,48,49] regardless of training set.
Same seeds, same total step budget (9 × NOPT), same restarts.

Parallelism: one process per (case × method) = 6 processes total.
JIT is compiled once per process and reused across all restarts.
"""

import sys, os, types, dataclasses, time

for _n, _attrs in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        for _k, _v in _attrs.items():
            setattr(_m, _k, _v)
        sys.modules[_n] = _m

sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import jax, jax.numpy as jnp

from homotopy_core import soft_sim, hard_sim as _hard_sim, _bptt_backward
import jax_spiking_model as sim
import jax_model_grads_hack as manual
from test_cases import RECURRENT_CASES, _make_recurrent_weights, BETAS

NOPT       = int(os.environ.get("NOPT",    "600"))
N_RESTARTS = int(os.environ.get("NR",        "8"))
PATIENCE   = int(os.environ.get("PATIENCE", "50"))
RTOL       = float(os.environ.get("RTOL",  "1e-3"))
N_WORKERS  = int(os.environ.get("NW",        "6"))
# N_TRAIN: how many neurons (from the end) to train on.
# 50 → all, 25 → second half, 3 → output neurons only.  0 → use all (same as N).
_N_TRAIN_ENV = int(os.environ.get("N_TRAIN", "0"))
# LOSS: "mse" → MSE on voltages;  "st" → spike-timing (Van Rossum) loss.
LOSS_MODE  = os.environ.get("LOSS", "mse")
# TAU: exponential kernel timescale (timesteps) for spike-timing loss.
TAU        = float(os.environ.get("TAU", "20.0"))
# TAU_START / TAU_END: anneal tau from broad→sharp alongside the beta schedule.
# Set both to enable annealing; leaving them unset keeps fixed TAU throughout.
TAU_START  = float(os.environ.get("TAU_START", str(TAU)))
TAU_END    = float(os.environ.get("TAU_END",   str(TAU)))
# One tau per beta stage (geometric interpolation over 9 stages).
_N_STAGES  = 9
TAU_SCHEDULE = [
    TAU_START * (TAU_END / TAU_START) ** (i / (_N_STAGES - 1))
    for i in range(_N_STAGES)
] if TAU_START != TAU_END else [TAU] * _N_STAGES
# SEED_BETA: surrogate beta used only in the hard ST seed (dL/d_spike → dL/dV).
# Keep this wide (low) so gradient is nonzero away from threshold; the internal
# beta_surr=30 stays sharp for the synaptic propagation inside _bptt_backward.
SEED_BETA  = float(os.environ.get("SEED_BETA", "5.0"))
TOL        = 1e-6


# ── spike-timing (Van Rossum) convolution helpers ────────────────────────────

def fwd_exp_conv(x, decay):
    """Causal exponential convolution: S[t] = decay*S[t-1] + x[t]. x: (T, N)."""
    def step(carry, x_t):
        carry = carry * decay + x_t
        return carry, carry
    _, S = jax.lax.scan(step, jnp.zeros(x.shape[-1]), x)
    return S


def rev_exp_conv(x, decay):
    """Anti-causal convolution: R[t] = x[t] + decay*x[t+1] + ... (adjoint of fwd). x: (T, N)."""
    def step(carry, x_t):
        carry = carry * decay + x_t
        return carry, carry
    _, S_rev = jax.lax.scan(step, jnp.zeros(x.shape[-1]), x[::-1])
    return S_rev[::-1]


# ── soft homotopy stage ───────────────────────────────────────────────────────

def make_soft_stage(params, C, N, A, train_ns):
    th = params.threshold

    @jax.jit
    def stage(w0, base, lo, hi, beta, lr, decay):
        tgt = soft_sim(base, beta, params, C, N, A)
        if LOSS_MODE == "st":
            S_tgt = fwd_exp_conv(jax.nn.sigmoid(beta * (tgt / th - 1.0)), decay)
        def loss_fn(w):
            v = soft_sim(w, beta, params, C, N, A)
            if LOSS_MODE == "st":
                S = fwd_exp_conv(jax.nn.sigmoid(beta * (v / th - 1.0)), decay)
                return sum(jnp.sum((S_tgt[:, n] - S[:, n]) ** 2) for n in train_ns)
            return sum(jnp.sum((tgt[:, n] - v[:, n]) ** 2) for n in train_ns)
        vg = jax.value_and_grad(loss_fn)
        def cond(c):
            _, _, _, _, _, t, _, done = c
            return (t < NOPT) & ~done
        def body(c):
            w, m, v, bw, bl, t, l_check, done = c
            l, g = vg(w)
            g = jnp.nan_to_num(g)
            bw, bl = jax.lax.cond(l < bl, lambda: (w, l), lambda: (bw, bl))
            m  = 0.9   * m + 0.1   * g
            v  = 0.999 * v + 0.001 * g * g
            t1 = (t + 1).astype(jnp.float32)
            step = (m / (1 - 0.9 ** t1)) / (jnp.sqrt(v / (1 - 0.999 ** t1)) + 1e-12)
            w_new = jnp.clip(w - lr * step, lo, hi)
            new_t = t + 1
            at_end = (new_t % PATIENCE == 0)
            rel_imp = (l_check - bl) / (jnp.abs(l_check) + 1e-10)
            done_now    = done | (at_end & (rel_imp < RTOL))
            l_check_new = jax.lax.cond(at_end, lambda: bl, lambda: l_check)
            return (w_new, m, v, bw, bl, new_t, l_check_new, done_now)
        l0 = vg(w0)[0]
        z  = jnp.zeros_like(w0)
        init = (w0, z, z, w0, l0, jnp.int32(0), l0, jnp.bool_(False))
        _, _, _, bw, _, _, _, _ = jax.lax.while_loop(cond, body, init)
        return bw
    return stage


def run_soft(params, C, N, A, outs, train_ns, true_strs, lo, hi, target_v, th, seeds):
    stage = make_soft_stage(params, C, N, A, train_ns)
    _d0 = jnp.float32(np.exp(-1.0 / TAU_SCHEDULE[0]))
    stage(true_strs, true_strs, lo, hi, jnp.float32(1.0), jnp.float32(1.0), _d0)  # warm-up JIT

    best_loss = float("inf")
    best_w    = true_strs
    t0        = time.perf_counter()

    for seed in seeds:
        rng = np.random.default_rng(seed)
        w = true_strs * jnp.array(rng.uniform(0.5, 1.5, len(true_strs)), jnp.float32)
        for beta, tau_i in zip(BETAS, TAU_SCHEDULE):
            lr    = 1.0 if beta <= 2 else (0.5 if beta <= 8 else 0.2)
            decay = jnp.float32(np.exp(-1.0 / tau_i))
            w     = stage(w, true_strs, lo, hi, jnp.float32(beta), jnp.float32(lr), decay)
        v_found = np.array(_hard_sim(w, params, C, N, A))
        hl = float(sum(np.sum((np.array(target_v)[:, n] - v_found[:, n]) ** 2) for n in outs))
        if hl < best_loss:
            best_loss = hl
            best_w    = w
        if best_loss < TOL:
            break

    return best_loss, best_w, time.perf_counter() - t0


# ── hard-surrogate direct training ───────────────────────────────────────────

def make_hard_stage(params, C, N, A, train_ns, target_v):
    th        = params.threshold
    beta_surr = 30.0
    decay     = jnp.float32(jnp.exp(-1.0 / TAU))

    def act_slope(pre_vals):
        return (pre_vals >= th) * 1.0, manual._narrow_surrogate_slope(pre_vals, th, beta_surr)

    if LOSS_MODE == "st":
        spikes_true = (target_v >= th).astype(jnp.float32)
        S_true_all  = fwd_exp_conv(spikes_true, decay)          # (T, N) – precomputed
        train_mask  = jnp.zeros(target_v.shape[1]).at[jnp.array(train_ns)].set(1.0)
    else:
        tgt_out = jnp.zeros_like(target_v)
        for n in train_ns:
            tgt_out = tgt_out.at[:, n].set(target_v[:, n])

    @jax.jit
    def vg(w):
        V, _, refs = manual._forward_with_refractory(params, C, N, w, A)
        if LOSS_MODE == "st":
            S_found = fwd_exp_conv((V >= th).astype(jnp.float32), decay)
            err     = (S_found - S_true_all) * train_mask[None, :]
            l       = jnp.sum(err ** 2)
            d_spike = rev_exp_conv(2.0 * err, decay)            # dL/d_spike: (T, N)
            # Wide surrogate in seed so gradient is nonzero well away from threshold.
            a_seed  = jax.nn.sigmoid(SEED_BETA * (V / th - 1.0))
            aV_seed = d_spike * a_seed * (1.0 - a_seed) * (SEED_BETA / th)
        else:
            l = sum(jnp.sum((tgt_out[:, n] - V[:, n]) ** 2) for n in train_ns)
            aV_seed = jnp.zeros_like(V)
            for n in train_ns:
                aV_seed = aV_seed.at[:, n].set(2.0 * (V[:, n] - tgt_out[:, n]))
        g, _ = _bptt_backward(V, refs, aV_seed, w, params, C, N, A, act_slope)
        return l, g

    @jax.jit
    def stage(w0, lo, hi, lr):
        def cond(c):
            _, _, _, _, _, t, _, done = c
            return (t < NOPT) & ~done
        def body(c):
            w, m, v, bw, bl, t, l_check, done = c
            l, g = vg(w)
            g    = jnp.nan_to_num(g)
            bw, bl = jax.lax.cond(l < bl, lambda: (w, l), lambda: (bw, bl))
            m  = 0.9   * m + 0.1   * g
            v  = 0.999 * v + 0.001 * g * g
            t1 = (t + 1).astype(jnp.float32)
            step = (m / (1 - 0.9 ** t1)) / (jnp.sqrt(v / (1 - 0.999 ** t1)) + 1e-12)
            w_new = jnp.clip(w - lr * step, lo, hi)
            new_t = t + 1
            at_end = (new_t % PATIENCE == 0)
            rel_imp = (l_check - bl) / (jnp.abs(l_check) + 1e-10)
            done_now    = done | (at_end & (rel_imp < RTOL))
            l_check_new = jax.lax.cond(at_end, lambda: bl, lambda: l_check)
            return (w_new, m, v, bw, bl, new_t, l_check_new, done_now)
        l0 = vg(w0)[0]
        z  = jnp.zeros_like(w0)
        init = (w0, z, z, w0, l0, jnp.int32(0), l0, jnp.bool_(False))
        _, _, _, bw, _, _, _, _ = jax.lax.while_loop(cond, body, init)
        return bw

    return stage


def run_hard(params, C, N, A, outs, train_ns, true_strs, lo, hi, target_v, th, seeds):
    stage = make_hard_stage(params, C, N, A, train_ns, target_v)
    stage(true_strs, lo, hi, jnp.float32(1.0))  # warm-up JIT

    best_loss = float("inf")
    best_w    = true_strs
    lr_schedule = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2]

    t0 = time.perf_counter()
    for seed in seeds:
        rng = np.random.default_rng(seed)
        w = true_strs * jnp.array(rng.uniform(0.5, 1.5, len(true_strs)), jnp.float32)
        for lr in lr_schedule:
            w = stage(w, lo, hi, jnp.float32(lr))
        v_found = np.array(_hard_sim(w, params, C, N, A))
        hl = float(sum(np.sum((np.array(target_v)[:, n] - v_found[:, n]) ** 2) for n in outs))
        if hl < best_loss:
            best_loss = hl
            best_w    = w
        if best_loss < TOL:
            break

    return best_loss, best_w, time.perf_counter() - t0


def spike_counts(V_np, outs, th):
    return {n: int(np.sum(V_np[:, n] >= th)) for n in outs}


# ── per-(case, method) worker ─────────────────────────────────────────────────

def run_case_method(case_idx, method):
    """Run all restarts for one (case, method) pair. Returns a result dict."""
    tc = RECURRENT_CASES[case_idx]
    conns, tw = _make_recurrent_weights(
        tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
        tc["num_neurons"], tc["output_neurons"])

    params   = dataclasses.replace(sim.default_params, steps=1000)
    C        = jnp.array(conns)
    N        = tc["num_neurons"]
    A        = jnp.array([0])
    outs     = tc["output_neurons"]
    th       = params.threshold
    n_train  = _N_TRAIN_ENV if _N_TRAIN_ENV > 0 else N
    n_train  = min(n_train, N)
    train_ns = list(range(N - n_train, N))

    true_strs = jnp.array(tw, jnp.float32)
    lo        = true_strs * 0.1
    hi        = true_strs * 5.0
    target_v  = jnp.array(_hard_sim(true_strs, params, C, N, A))
    seeds     = list(range(42, 42 + N_RESTARTS))

    if method == "soft":
        loss, w, wall = run_soft(params, C, N, A, outs, train_ns, true_strs,
                                  lo, hi, target_v, th, seeds)
    else:
        loss, w, wall = run_hard(params, C, N, A, outs, train_ns, true_strs,
                                  lo, hi, target_v, th, seeds)

    v_found = np.array(_hard_sim(w, params, C, N, A))
    sp      = spike_counts(v_found, outs, th)
    sp_true = spike_counts(np.array(target_v), outs, th)

    return {
        "case_idx": case_idx,
        "method":   method,
        "loss":     loss,
        "conv":     loss < TOL,
        "wall":     wall,
        "sp":       sp,
        "sp_true":  sp_true,
        "name":     tc["name"],
        "n_syn":    len(tw),
        "n_train":  len(train_ns),
        "n":        N,
        "outs":     outs,
    }


# ── gradient cosine section ───────────────────────────────────────────────────

def grad_cosine_section():
    tc = RECURRENT_CASES[0]
    conns, tw = _make_recurrent_weights(tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
                                         tc["num_neurons"], tc["output_neurons"])
    params   = dataclasses.replace(sim.default_params, steps=1000)
    C = jnp.array(conns); N = tc["num_neurons"]; A = jnp.array([0])
    th = params.threshold
    n_train_g = _N_TRAIN_ENV if _N_TRAIN_ENV > 0 else N
    n_train_g = min(n_train_g, N)
    train_ns  = list(range(N - n_train_g, N))
    w_true   = jnp.array(tw, jnp.float32)
    w_eval   = w_true * 1.05
    target_v = jnp.array(_hard_sim(w_true, params, C, N, A))

    V_hard, _, refs = manual._forward_with_refractory(params, C, N, w_eval, A)
    aV_seed = jnp.zeros_like(V_hard)
    for n in train_ns:
        aV_seed = aV_seed.at[:, n].set(2.0 * (V_hard[:, n] - target_v[:, n]))

    def act_slope_narrow(pre_vals):
        return (pre_vals >= th) * 1.0, manual._narrow_surrogate_slope(pre_vals, th, 30.0)

    g_hard, _ = _bptt_backward(V_hard, refs, aV_seed, w_eval, params, C, N, A, act_slope_narrow)
    g_hard_nn = jnp.nan_to_num(g_hard)

    print(f"Gradient cosine (soft@beta vs hard narrow-surrogate b=30) at w_true*1.05, {len(train_ns)} train neurons:")
    for beta in [1.0, 5.0, 13.0, 34.0]:
        b = jnp.float32(beta)
        soft_tgt = soft_sim(w_true, b, params, C, N, A)
        def loss_soft(w, b=b):
            v = soft_sim(w, b, params, C, N, A)
            return sum(jnp.sum((soft_tgt[:, n] - v[:, n]) ** 2) for n in train_ns)
        g_soft = jax.grad(loss_soft)(w_eval)
        cos = float(jnp.dot(g_soft, g_hard_nn) /
                    (jnp.linalg.norm(g_soft) * jnp.linalg.norm(g_hard_nn) + 1e-30))
        print(f"  beta={beta:5.1f}  cos(soft, hard narrow b=30) = {cos:+.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    sys.stdout.reconfigure(line_buffering=True)

    n_cases  = len(RECURRENT_CASES)
    work     = [(ci, m) for ci in range(n_cases) for m in ("soft", "hard")]
    n_procs  = min(N_WORKERS, len(work))

    n_train_disp = _N_TRAIN_ENV if _N_TRAIN_ENV > 0 else "N"
    if LOSS_MODE == "st":
        if TAU_START != TAU_END:
            tau_disp = f"{TAU_START:.0f}→{TAU_END:.0f}"
        else:
            tau_disp = str(TAU)
        loss_disp = f"{LOSS_MODE}(tau={tau_disp},seed_b={SEED_BETA})"
    else:
        loss_disp = LOSS_MODE
    print(f"NOPT={NOPT}  NR={N_RESTARTS}  PATIENCE={PATIENCE}  RTOL={RTOL}  N_TRAIN={n_train_disp}  LOSS={loss_disp}")
    print(f"Total steps per restart: 9×{NOPT}={9*NOPT}  workers={n_procs}/{len(work)}")
    print()

    ctx     = multiprocessing.get_context("spawn")
    results = {}

    with ProcessPoolExecutor(max_workers=n_procs, mp_context=ctx) as ex:
        futs = {ex.submit(run_case_method, ci, m): (ci, m) for ci, m in work}
        for f in as_completed(futs):
            ci, m = futs[f]
            try:
                results[(ci, m)] = f.result()
            except Exception as e:
                results[(ci, m)] = {"error": str(e), "case_idx": ci, "method": m}
            print(f"  [case {ci} {m} done]", flush=True)

    print()
    for ci in range(n_cases):
        rs = results.get((ci, "soft"), {})
        rh = results.get((ci, "hard"), {})
        if "error" in rs or "error" in rh:
            print(f"ERROR: {rs.get('error', '')} {rh.get('error', '')}")
            continue
        outs     = rs["outs"]
        sp_true  = rs["sp_true"]
        n_train  = rs["n_train"]
        print(f"{'─'*70}")
        print(f"{rs['name']}  ({rs['n_syn']} syn, {rs['n']} neurons, train on {n_train} neurons)")
        print(f"  Target spikes: " + "  ".join(f"N{n}={sp_true[n]}sp" for n in outs))
        for r, label in [(rs, "Soft homotopy"), (rh, "Hard surrogate")]:
            sp   = r["sp"]
            conv = "YES" if r["conv"] else "NO "
            print(f"  {label}: loss={r['loss']:.3e}  conv={conv}  {r['wall']:.1f}s  "
                  + "  ".join(f"N{n}:{sp[n]}/{sp_true[n]}sp" for n in outs))

    print()
    grad_cosine_section()
