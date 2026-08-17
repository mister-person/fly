"""Benchmark soft homotopy, hard surrogate, and target propagation.

Measures wall-clock per optimizer iteration for each (N, steps) combo,
fits power laws, and extrapolates to 20k / 150k neurons.

Usage:
    python3 performance/bench_methods.py
"""

import os, sys, time, json, dataclasses
from pathlib import Path

import types
for _n, _attrs in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        for _k, _v in _attrs.items(): setattr(_m, _k, _v)
        sys.modules[_n] = _m

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import jax
import jax.numpy as jnp

import jax_spiking_model as sim
from homotopy_core import (
    lr_for_beta, homotopy_stage, _bptt_backward, hard_sim as _hard_sim,
)
from jax_model_grads_hack import (
    _forward_with_refractory, _narrow_surrogate_slope,
)

P_CONNECT = 0.10
SIZES     = [10, 25, 50, 100, 200, 300, 500, 750]
STEPS_LIST = [500, 2000]
NOPT      = 20
BETA      = 5.0
SEED      = 42

OUT_DIR   = Path(__file__).parent


# ── helpers ──────────────────────────────────────────────────────────────────

def make_topology(N, p=P_CONNECT):
    rng = np.random.default_rng(SEED)
    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    mask  = rng.random(len(pairs)) < p
    conns = np.array([pp for pp, m in zip(pairs, mask) if m], dtype=np.int32)
    if conns.shape[0] == 0:
        conns = np.array([(i, (i+1) % N) for i in range(N)], dtype=np.int32)
    return conns

def make_weights(n_syn):
    rng = np.random.default_rng(SEED * 7)
    return rng.uniform(100, 800, n_syn).astype(np.float32)

def ft(t):
    if t is None: return "N/A"
    if t < 1e-3:    return f"{t*1e6:.0f} us"
    if t < 1:       return f"{t*1e3:.1f} ms"
    if t < 60:      return f"{t:.2f} s"
    if t < 3600:    return f"{t/60:.1f} min"
    if t < 86400:   return f"{t/3600:.1f} h"
    return f"{t/86400:.1f} d"

def fit_pl(xs, ys):
    """Power-law fit y = a * x^p via OLS in log-space."""
    lx = np.log(np.array(xs, dtype=np.float64))
    ly = np.log(np.array(ys, dtype=np.float64))
    n = len(lx)
    sx, sy = lx.sum(), ly.sum()
    sxx, sxy = (lx**2).sum(), (lx*ly).sum()
    denom = n * sxx - sx**2
    if abs(denom) < 1e-30: return None, None, None
    p  = (n * sxy - sx * sy) / denom
    ln_a = (sy - p * sx) / n
    a  = float(np.exp(ln_a))
    yhat = ln_a + p * lx
    ss_res = ((ly - yhat)**2).sum()
    ss_tot = ((ly - ly.mean())**2).sum()
    r2 = 1.0 - ss_res / (ss_tot + 1e-30)
    return a, p, float(r2)

def pred_syn(N):
    return int(N * (N - 1) * P_CONNECT)


# ════════════════════════════════════════════════════════════════════════════
# BENCHMARK FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def bench_soft(N, steps, conns, true_w):
    params = dataclasses.replace(sim.default_params, steps=steps)
    C = jnp.array(conns); A = jnp.array([0])
    w  = jnp.array(true_w * 1.1, jnp.float32)
    beta = jnp.float32(BETA)
    lr_v = jnp.float32(lr_for_beta(BETA))
    lo = jnp.array(true_w * 0.1, jnp.float32)
    hi = jnp.array(true_w * 5.0, jnp.float32)
    out = homotopy_stage(w, w, lo, hi, beta, lr_v, params, C, N, A, nopt=2, patience=0)
    jax.block_until_ready(out)
    reps = 3; t0 = time.perf_counter()
    for _ in range(reps):
        out = homotopy_stage(w, w, lo, hi, beta, lr_v, params, C, N, A, nopt=NOPT, patience=0)
        jax.block_until_ready(out)
    return (time.perf_counter() - t0) / reps, NOPT


def bench_hard(N, steps, conns, true_w):
    params = dataclasses.replace(sim.default_params, steps=steps)
    C = jnp.array(conns); A = jnp.array([0])
    w2 = jnp.array(true_w * 1.05, jnp.float32)
    th = params.threshold
    def act_slope(pre_vals):
        return (pre_vals >= th) * 1.0, _narrow_surrogate_slope(pre_vals, th, 30.0)
    def fwd_bwd():
        V, _, refs = _forward_with_refractory(params, C, N, w2, A)
        aV = jnp.zeros_like(V).at[:, :max(1, N // 10)].set(1e-3)
        g, _ = _bptt_backward(V, refs, aV, w2, params, C, N, A, act_slope)
        return g
    out = fwd_bwd(); jax.block_until_ready(out)
    reps = 3; t0 = time.perf_counter()
    for _ in range(reps):
        out = fwd_bwd(); jax.block_until_ready(out)
    return (time.perf_counter() - t0) / reps, 1


def bench_tp(N, steps, conns, true_w):
    params = dataclasses.replace(sim.default_params, steps=steps)
    C = jnp.array(conns); A = jnp.array([0])
    w3 = jnp.array(true_w, jnp.float32)
    th = params.threshold; gsw = params.global_synapse_weight
    nd = float(params.neuron_decay); rd = float(params.rise_decay)
    delay = params.delay_iters
    MAX_H = min(steps, 400)
    h_arr = np.zeros(MAX_H); R, V = 0.0, 0.0
    for ti in range(MAX_H):
        upd = 1.0 if ti == delay else 0.0
        R = (R + upd) * rd; V = (V - R) * nd + R; h_arr[ti] = V
    v_hard = np.array(_hard_sim(w3, params, C, N, A))
    outs = list(range(max(0, N - 3), N))
    conn_arr = np.array(conns)
    fanin = {n: conn_arr[conn_arr[:, 1] == n, 0].tolist() for n in outs}
    def tp_solve():
        for n_out in outs:
            pre_list = fanin.get(n_out, [])
            if len(pre_list) < 1: continue
            spike_times = {pn: np.where(v_hard[:, pn] >= th)[0].tolist() for pn in pre_list}
            tgt_spikes = np.where(v_hard[:, n_out] >= th)[0].tolist()
            if not tgt_spikes: continue
            A_mat = np.zeros((len(tgt_spikes), len(pre_list)))
            for i, Tj in enumerate(tgt_spikes):
                for j, pn in enumerate(pre_list):
                    c = sum(h_arr[Tj - tk] for tk in spike_times.get(pn, [])
                            if 0 < Tj - tk < MAX_H)
                    A_mat[i, j] = c
            b = np.full(len(tgt_spikes), th / gsw)
            try:
                np.linalg.solve(A_mat.T @ A_mat + 1e-4 * np.eye(len(pre_list)), A_mat.T @ b)
            except np.linalg.LinAlgError:
                np.linalg.lstsq(A_mat, b, rcond=None)
        return 0
    tp_solve()
    reps = 5; t0 = time.perf_counter()
    for _ in range(reps): tp_solve()
    return (time.perf_counter() - t0) / reps, 1


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("SNN PERFORMANCE BENCHMARK")
    print(f"  JAX: {jax.default_backend()} / {jax.local_devices()[0].device_kind}")
    print(f"  Sizes: {SIZES}  Steps: {STEPS_LIST}  NOPT: {NOPT}")
    print()

    benchers = [
        ("soft", "Soft per Adam step", bench_soft),
        ("hard", "Hard fwd+bwd",       bench_hard),
        ("tp",   "Target prop",        bench_tp),
    ]

    results = {}

    for N in SIZES:
        conns = make_topology(N)
        n_syn = len(conns)
        true_w = make_weights(n_syn)
        print(f"N={N:>5}  syn={n_syn:>5}  fan_in={n_syn/N:.1f}", flush=True)

        for steps in STEPS_LIST:
            key = (N, steps)
            r = {"N": N, "steps": steps, "n_syn": n_syn}
            print(f"  steps={steps}: ", end="", flush=True)

            parts = []
            for field, label, bfn in benchers:
                try:
                    t_total, n_ops = bfn(N, steps, conns, true_w)
                    r[f"{field}_total"] = t_total
                    r[f"{field}_per"]   = t_total / n_ops
                    parts.append(f"{field}={ft(r[f'{field}_per'])}/{n_ops}")
                except Exception as e:
                    parts.append(f"{field}=ERR")
            print(" ".join(parts), flush=True)
            results[key] = r

    # ═══════════════════════════════════════════════════════════════
    # SCALING FITS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("SCALING FITS (normalized to steps=1000)\n")

    methods = [("soft", "Soft per Adam step"),
               ("hard", "Hard fwd+bwd"),
               ("tp",   "Target prop")]

    print("  vs Neurons (N):")
    fits_N = {}
    for k, label in methods:
        xs = [r["N"] for r in results.values() if f"{k}_per" in r]
        ys = [r[f"{k}_per"] / (r["steps"] / 1000.0) for r in results.values() if f"{k}_per" in r]
        if len(xs) >= 3:
            a, p, r2 = fit_pl(xs, ys)
            fits_N[k] = (a, p, r2)
            print(f"    {label:22s}  t = {a:.2e} * N^{p:.3f}  R2={r2:.3f}")

    print("\n  vs Synapses (n_syn):")
    fits_syn = {}
    for k, label in methods:
        xs = [r["n_syn"] for r in results.values() if f"{k}_per" in r]
        ys = [r[f"{k}_per"] / (r["steps"] / 1000.0) for r in results.values() if f"{k}_per" in r]
        if len(xs) >= 3:
            a, p, r2 = fit_pl(xs, ys)
            fits_syn[k] = (a, p, r2)
            print(f"    {label:22s}  t = {a:.2e} * n_syn^{p:.3f}  R2={r2:.3f}")

    max_N = max(SIZES)
    print(f"\n  Steps scaling (N={max_N}):")
    q_vals = {}
    for k, label in methods:
        xs = [r["steps"] for r in results.values() if r["N"] == max_N and f"{k}_per" in r]
        ys = [r[f"{k}_per"] for r in results.values() if r["N"] == max_N and f"{k}_per" in r]
        if len(xs) >= 2:
            _, q, r2 = fit_pl(xs, ys)
            q_vals[k] = q
            print(f"    {label:22s}  t ~ steps^{q:.3f}  R2={r2:.3f}")
        else:
            q_vals[k] = 1.0
            print(f"    {label:22s}  t ~ steps^1.0  (assumed)")

    # Best fit var
    best = {}
    for k, label in methods:
        r2n = fits_N.get(k, (None, None, -1))[2]
        r2s = fits_syn.get(k, (None, None, -1))[2]
        best[k] = ("syn", fits_syn[k]) if r2s > r2n else ("N", fits_N[k])

    # ═══════════════════════════════════════════════════════════════
    # EXTRAPOLATIONS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("EXTRAPOLATED TIMES\n")

    for steps_ext in [1000, 5000]:
        for N_ext in [20_000, 150_000]:
            ns_ext = pred_syn(N_ext)
            print(f"  N={N_ext:>8,}  syn={ns_ext:>10,}  steps={steps_ext:,}")
            print(f"  {'-'*55}")
            for k, label in methods:
                var, (a, p, r2) = best[k]
                q = q_vals.get(k, 1.0)
                var_ext = ns_ext if var == "syn" else N_ext
                t_pred = a * (var_ext ** p) * ((steps_ext / 1000.0) ** q)
                print(f"    {label:22s}  {ft(t_pred):>14}  (R2={r2:.2f}, p={p:.2f})")

            sv, (sa, sp, sr) = best["soft"]
            sq = q_vals.get("soft", 1.0)
            sv_ext = ns_ext if sv == "syn" else N_ext
            t_per_step = sa * (sv_ext ** sp) * ((steps_ext / 1000.0) ** sq)
            t_full = t_per_step * 9 * NOPT * 4
            print(f"    {'Full soft run':22s}  {ft(t_full):>14}  "
                  f"(9stg x {NOPT} x 4rst)")
            print()

    # Save JSON
    json_path = OUT_DIR / "bench_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "config": {"sizes": SIZES, "steps": STEPS_LIST, "nopt": NOPT,
                       "p_connect": P_CONNECT,
                       "backend": jax.default_backend(),
                       "device": jax.local_devices()[0].device_kind},
            "measured": {f"N={N}_S={S}": v for (N, S), v in sorted(results.items())},
            "fits_N":   {k: {"a": v[0], "p": v[1], "r2": v[2]} for k, v in fits_N.items()},
            "fits_syn": {k: {"a": v[0], "p": v[1], "r2": v[2]} for k, v in fits_syn.items()},
            "q_steps":  q_vals,
        }, f, indent=2, default=str)
    print(f"JSON saved: {json_path}")


if __name__ == "__main__":
    main()
