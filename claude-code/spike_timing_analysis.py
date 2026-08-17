"""Compare spike times (not just counts) between target and found solution.

Runs soft homotopy + spike-timing loss on the two cases that previously achieved
exact spike counts, then reports the timestep of each spike and the timing error.
"""

import sys, os, types, dataclasses, time

# Must be set before recurrent_compare is imported (LOSS_MODE is captured at import).
os.environ.setdefault("LOSS", "st")

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

from homotopy_core import hard_sim as _hard_sim
import jax_spiking_model as sim
from test_cases import RECURRENT_CASES, _make_recurrent_weights, BETAS
from recurrent_compare import make_soft_stage, TAU_SCHEDULE   # imports with LOSS=st already set

NR       = int(os.environ.get("NR",       "8"))
NOPT     = int(os.environ.get("NOPT",   "600"))
TOL      = 1e-6
CASES    = [0, 1]   # the two that reached exact counts with soft+ST


def spike_times(V_np, neuron, th):
    return np.where(V_np[:, neuron] >= th)[0].tolist()


def run_case(case_idx):
    tc = RECURRENT_CASES[case_idx]
    conns, tw = _make_recurrent_weights(
        tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
        tc["num_neurons"], tc["output_neurons"])

    params    = dataclasses.replace(sim.default_params, steps=1000)
    C         = jnp.array(conns)
    N         = tc["num_neurons"]
    A         = jnp.array([0])
    outs      = tc["output_neurons"]
    th        = params.threshold
    train_ns  = list(range(N))

    true_strs = jnp.array(tw, jnp.float32)
    lo        = true_strs * 0.1
    hi        = true_strs * 5.0
    target_v  = jnp.array(_hard_sim(true_strs, params, C, N, A))
    target_np = np.array(target_v)

    stage = make_soft_stage(params, C, N, A, train_ns)
    _d0 = jnp.float32(np.exp(-1.0 / TAU_SCHEDULE[0]))
    stage(true_strs, true_strs, lo, hi, jnp.float32(1.0), jnp.float32(1.0), _d0)  # warm-up JIT

    best_loss = float("inf")
    best_w    = true_strs
    t0 = time.perf_counter()

    for seed in range(42, 42 + NR):
        rng = np.random.default_rng(seed)
        w = true_strs * jnp.array(rng.uniform(0.5, 1.5, len(true_strs)), jnp.float32)
        for beta, tau_i in zip(BETAS, TAU_SCHEDULE):
            lr    = 1.0 if beta <= 2 else (0.5 if beta <= 8 else 0.2)
            decay = jnp.float32(np.exp(-1.0 / tau_i))
            w     = stage(w, true_strs, lo, hi, jnp.float32(beta), jnp.float32(lr), decay)
        v_found = np.array(_hard_sim(w, params, C, N, A))
        hl = float(sum(np.sum((target_np[:, n] - v_found[:, n]) ** 2) for n in outs))
        if hl < best_loss:
            best_loss = hl
            best_w    = w
        print(f"  restart {seed}  loss={hl:.3e}  best={best_loss:.3e}", flush=True)
        if best_loss < TOL:
            break

    wall = time.perf_counter() - t0

    save_path = f"best_weights_case{case_idx}.npy"
    np.save(save_path, np.array(best_w))
    print(f"\nBest weights saved to {save_path}")

    v_found = np.array(_hard_sim(best_w, params, C, N, A))

    print(f"\n{'═'*62}")
    print(f"{tc['name']}  ({len(tw)} syn)  wall={wall:.1f}s  eval_loss={best_loss:.3e}")
    print(f"{'═'*62}")

    for n in outs:
        t_true  = spike_times(target_np, n, th)
        t_found = spike_times(v_found,   n, th)
        n_true, n_found = len(t_true), len(t_found)

        match = "✓" if n_true == n_found else "✗"
        print(f"\n  Neuron {n}:  target={n_true}sp  found={n_found}sp  {match}")

        if n_true == 0 and n_found == 0:
            print(f"    (both silent)")
            continue

        if n_true != n_found:
            print(f"    target spikes @ {t_true}")
            print(f"    found  spikes @ {t_found}")
            continue

        diffs = [tf - tt for tt, tf in zip(t_true, t_found)]
        print(f"    {'#':>3}  {'target_t':>9}  {'found_t':>9}  {'diff (steps)':>13}")
        for i, (tt, tf, d) in enumerate(zip(t_true, t_found, diffs)):
            flag = "  ←" if abs(d) > 10 else ""
            print(f"    {i+1:>3}  {tt:>9}  {tf:>9}  {d:>+13}{flag}")
        abs_diffs = [abs(d) for d in diffs]
        print(f"    max |error| = {max(abs_diffs)} steps   "
              f"mean |error| = {np.mean(abs_diffs):.1f} steps   "
              f"rms = {np.sqrt(np.mean(np.array(abs_diffs)**2)):.1f} steps")


if __name__ == "__main__":
    for ci in CASES:
        print(f"\n{'─'*62}")
        print(f"Case {ci}: {RECURRENT_CASES[ci]['name']}")
        print(f"{'─'*62}")
        run_case(ci)
