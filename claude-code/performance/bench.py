"""Performance scaling benchmark for the 5 weight-recovery methods.

Measures the *per-call cost of each method's core computational primitive* as a
function of network size (number of neurons) and simulation length (steps), on
synthetic random-recurrent networks with a fixed average in-degree.

Primitives measured (one Adam step / one objective eval / one weight-recovery
pass — whichever is the inner loop of the method):

  forward     one hard_sim forward pass        (black-box obj eval + final eval)
  soft_grad   one soft value_and_grad step     (soft homotopy inner loop)
  hard_grad   one manual-BPTT grad step        (hard surrogate inner loop)
  tp_pass     one full per-neuron TP weight recovery (target prop inner loop)

The total wall time of a real run = (algorithmic step budget) x (per-call cost)
+ compile.  bench.py measures the per-call cost and compile; analyze.py combines
it with each method's step budget and extrapolates.

Two sweeps:
  * neuron sweep   vary N, fixed steps=1000, fixed avg in-degree
  * step   sweep   vary steps, fixed N

Env vars:
  DEGREE   avg in-degree (synapses ~ DEGREE * N).  default 5
  REPS     timed repetitions per primitive (median reported).  default 5
  OUT      output json path.  default performance/results.json
"""

import sys, os, types, dataclasses, time, json, statistics

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

from homotopy_core import soft_sim, hard_sim, _bptt_backward
import jax_spiking_model as sim
import jax_model_grads_hack as manual

DEGREE = int(os.environ.get("DEGREE", "5"))
REPS   = int(os.environ.get("REPS", "5"))
OUT    = os.environ.get("OUT", "/workspace/project/performance/results.json")

BETA_SURR = 30.0


# ── network construction ──────────────────────────────────────────────────────

def build_network(N, degree, steps, seed=0):
    """Random recurrent network with ~degree incoming synapses per neuron.

    Input is injected at a fixed *fraction* of neurons (N//50, >=1) so that
    activity density is comparable across sizes.  Weights are scaled so a
    meaningful fraction of neurons spike (needed to give target prop real work).
    """
    rng = np.random.default_rng(seed)
    n_edges = degree * N
    src = rng.integers(0, N, size=n_edges)
    dst = rng.integers(0, N, size=n_edges)
    keep = src != dst
    conns = np.stack([src[keep], dst[keep]], axis=1).astype(np.int32)
    w = rng.uniform(150, 450, size=len(conns)).astype(np.float32)
    params = dataclasses.replace(sim.default_params, steps=steps)
    n_input = max(1, N // 50)
    A = jnp.array(np.arange(n_input))
    return params, jnp.array(conns), N, A, jnp.array(w), conns, np.array(w)


# ── timing helper ─────────────────────────────────────────────────────────────

def time_call(fn, reps=REPS):
    """Return (compile_plus_first_s, median_run_s).  fn() must block internally."""
    t0 = time.perf_counter()
    fn()
    compile_s = time.perf_counter() - t0
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return compile_s, statistics.median(ts)


def block(x):
    if isinstance(x, (tuple, list)):
        for e in x:
            block(e)
    elif hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


# ── target-prop primitive (self-contained, mirrors tp_50neuron.py) ────────────

def make_tp(params, conns_np, N, th, gsw, delay, nd, rd, lo_np, hi_np):
    MAX_H = 600
    h = np.zeros(MAX_H)
    R = V = 0.0
    for t in range(MAX_H):
        upd = 1.0 if t == delay else 0.0
        R = (R + upd) * rd
        V = (V - R) * nd + R
        h[t] = V
    edges = conns_np

    # precompute pre-synapse index/id lists per neuron
    pre_map = {}
    for n in range(N):
        idx = np.where(edges[:, 1] == n)[0]
        pre_map[n] = (idx, edges[idx, 0])

    from scipy.optimize import nnls, minimize

    def build_contrib(pres, spikes_by_neuron, eval_times, epoch_starts):
        rows = []
        for j, Tj in enumerate(eval_times):
            T_prev = epoch_starts[j]
            row = np.zeros(len(pres))
            for ci, p in enumerate(pres):
                for t_k in spikes_by_neuron.get(int(p), ()):
                    if T_prev < t_k < Tj:
                        dt = Tj - t_k
                        if 0 < dt < MAX_H:
                            row[ci] += h[dt]
            rows.append(row)
        return np.array(rows)

    def tp_neuron(n, spikes_by_neuron, tgt_times, margin):
        syn_idxs, pres = pre_map[n]
        if len(pres) == 0 or len(tgt_times) == 0:
            return None
        refr = params.refractory_iters
        tgt = list(tgt_times)
        epoch_lo = [0] + tgt[:-1]
        A_lo = build_contrib(pres, spikes_by_neuron, tgt, epoch_lo)
        b = np.full(len(tgt), th / gsw)
        valid = A_lo.max(axis=1) > 1e-12 if A_lo.size else np.zeros(0, bool)
        if A_lo.size == 0 or not valid.any():
            return None
        A_lo, b = A_lo[valid], b[valid]
        w_nnls, _ = nnls(A_lo, b)
        if margin <= 0:
            return syn_idxs, np.clip(w_nnls, lo_np[syn_idxs], hi_np[syn_idxs])
        # upper-bound rows at sampled non-target times
        nt_times, nt_epochs = [], []
        for j in range(len(tgt) - 1):
            T_ref_end = tgt[j] + refr + 2
            T_next = tgt[j + 1]
            if T_ref_end >= T_next:
                continue
            for frac in (0.25, 0.45, 0.65, 0.85):
                nt_times.append(int(T_ref_end + frac * (T_next - T_ref_end)))
                nt_epochs.append(tgt[j])
        cons = [{'type': 'ineq', 'fun': lambda w, A=A_lo, bb=b: A @ w - bb}]
        if nt_times:
            A_up = build_contrib(pres, spikes_by_neuron, nt_times, nt_epochs)
            b_up = np.full(len(nt_times), (1.0 - margin) * th / gsw)
            vu = A_up.max(axis=1) > 1e-12
            A_up, b_up = A_up[vu], b_up[vu]
            if len(A_up):
                cons.append({'type': 'ineq', 'fun': lambda w, A=A_up, bb=b_up: bb - A @ w})
        bounds = [(float(lo_np[si]), float(hi_np[si])) for si in syn_idxs]
        res = minimize(lambda w: 0.5 * float(np.dot(w - w_nnls, w - w_nnls)),
                       w_nnls.copy(), jac=lambda w: (w - w_nnls),
                       method='SLSQP', constraints=cons, bounds=bounds,
                       options={'ftol': 1e-10, 'maxiter': 2000, 'disp': False})
        w_sol = res.x if (res.success or res.status in (0, 4)) else w_nnls
        return syn_idxs, np.clip(w_sol, lo_np[syn_idxs], hi_np[syn_idxs])

    def tp_pass(spikes_by_neuron, w_base, margin=0.10):
        w_new = w_base.copy()
        n_solved = 0
        for n in range(1, N):
            tgt = spikes_by_neuron.get(n, ())
            if len(tgt) == 0:
                continue
            r = tp_neuron(n, spikes_by_neuron, tgt, margin)
            if r is None:
                continue
            syn_idxs, w_sol = r
            w_new[syn_idxs] = w_sol
            n_solved += 1
        return w_new, n_solved

    return tp_pass


# ── one measurement point ─────────────────────────────────────────────────────

def measure_point(N, steps, seed=0):
    params, C, Nn, A, w, conns_np, w_np = build_network(N, DEGREE, steps, seed)
    th = params.threshold
    gsw = params.global_synapse_weight
    n_syn = len(conns_np)

    rec = {"N": N, "steps": steps, "n_syn": n_syn, "degree": DEGREE,
           "n_input": int(A.shape[0])}

    # ---- forward (hard_sim) ----
    fwd = lambda: block(hard_sim(w, params, C, Nn, A))
    c, t = time_call(fwd)
    rec["forward"] = {"compile": c, "call": t}

    # target voltages / spikes for grad seeds + TP targets
    target_v = np.array(block(hard_sim(w, params, C, Nn, A)))
    spikes_by_neuron = {n: np.where(target_v[:, n] >= th)[0].tolist() for n in range(Nn)}
    total_spikes = int(sum(len(v) for v in spikes_by_neuron.values()))
    alive = int(sum(1 for v in spikes_by_neuron.values() if v))
    rec["total_spikes"] = total_spikes
    rec["alive"] = alive

    # ---- soft grad step ----
    beta = jnp.float32(5.0)
    tgt_soft = soft_sim(w, beta, params, C, Nn, A)
    def soft_loss(ww):
        v = soft_sim(ww, beta, params, C, Nn, A)
        return jnp.sum((tgt_soft - v) ** 2)
    soft_vg = jax.jit(jax.value_and_grad(soft_loss))
    w_off = w * 1.1
    sg = lambda: block(soft_vg(w_off))
    c, t = time_call(sg)
    rec["soft_grad"] = {"compile": c, "call": t}

    # ---- hard grad step ----
    def act_slope(pv):
        return (pv >= th) * 1.0, manual._narrow_surrogate_slope(pv, th, BETA_SURR)
    tv = jnp.array(target_v)
    @jax.jit
    def hard_vg(ww):
        V, _, refs = manual._forward_with_refractory(params, C, Nn, ww, A)
        aV = jnp.zeros_like(V).at[:, Nn - 1].set(2.0 * (V[:, Nn - 1] - tv[:, Nn - 1]))
        g, _ = _bptt_backward(V, refs, aV, ww, params, C, Nn, A, act_slope)
        return g
    hg = lambda: block(hard_vg(w_off))
    c, t = time_call(hg)
    rec["hard_grad"] = {"compile": c, "call": t}

    # ---- batched forward (black-box population eval) ----
    # CMA-ES/DE evaluate a whole population at once via vmap.  Measure per-member
    # cost at a representative population size so amortized batching is captured.
    POP = 16
    single = lambda ww: jnp.sum((tv[:, Nn - 1] - hard_sim(ww, params, C, Nn, A)[:, Nn - 1]) ** 2)
    batch = jax.jit(jax.vmap(single))
    Wpop = jnp.array(np.stack([w_np * (0.9 + 0.2 * np.random.rand(n_syn)) for _ in range(POP)]), jnp.float32)
    bf = lambda: block(batch(Wpop))
    c, t = time_call(bf)
    rec["batch_forward"] = {"compile": c, "call": t, "pop": POP, "per_member": t / POP}

    # ---- target-prop weight-recovery pass ----
    tp_pass = make_tp(params, conns_np, Nn, th, gsw, params.delay_iters,
                      float(params.neuron_decay), float(params.rise_decay),
                      np.array(w_np * 0.1), np.array(w_np * 5.0))
    # margin=0.10 realistic pass (NNLS + SLSQP QP per neuron)
    tp = lambda: tp_pass(spikes_by_neuron, w_np, margin=0.10)
    c, t = time_call(tp, reps=max(2, REPS // 2))
    rec["tp_pass"] = {"compile": c, "call": t}

    return rec


# ── sweeps ────────────────────────────────────────────────────────────────────

def main():
    neuron_sizes = [int(x) for x in os.environ.get(
        "N_SWEEP", "50,100,200,400,800,1600").split(",")]
    step_sizes = [int(x) for x in os.environ.get(
        "STEP_SWEEP", "250,500,1000,2000,4000").split(",")]
    step_fixed_N = int(os.environ.get("STEP_N", "200"))

    out = {"degree": DEGREE, "reps": REPS, "neuron_sweep": [], "step_sweep": [],
           "jax_version": jax.__version__, "step_fixed_N": step_fixed_N}

    print(f"=== neuron sweep (steps=1000, degree={DEGREE}) ===", flush=True)
    for N in neuron_sizes:
        rec = measure_point(N, 1000)
        out["neuron_sweep"].append(rec)
        print(f"  N={N:5d} n_syn={rec['n_syn']:6d} spikes={rec['total_spikes']:5d} "
              f"alive={rec['alive']:4d}/{N}  "
              f"fwd={rec['forward']['call']*1e3:8.2f}ms  "
              f"soft_g={rec['soft_grad']['call']*1e3:8.2f}ms  "
              f"hard_g={rec['hard_grad']['call']*1e3:8.2f}ms  "
              f"tp={rec['tp_pass']['call']*1e3:9.2f}ms", flush=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2)

    print(f"\n=== step sweep (N={step_fixed_N}, degree={DEGREE}) ===", flush=True)
    for steps in step_sizes:
        rec = measure_point(step_fixed_N, steps)
        out["step_sweep"].append(rec)
        print(f"  steps={steps:5d} spikes={rec['total_spikes']:5d}  "
              f"fwd={rec['forward']['call']*1e3:8.2f}ms  "
              f"soft_g={rec['soft_grad']['call']*1e3:8.2f}ms  "
              f"hard_g={rec['hard_grad']['call']*1e3:8.2f}ms  "
              f"tp={rec['tp_pass']['call']*1e3:9.2f}ms", flush=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2)

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
