import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

"""Detailed per-stage diagnostics for observe_last failures.

For each test case, shows at every beta stage:
  - current weight mods
  - soft loss (what the optimizer sees, N2-only)
  - hard loss (true spiking, all neurons)
  - gradient magnitude for w0 and w1
  - whether N1 and N2 fire in the current hard sim

Run:  python3 diagnose_observe_last.py [case_name ...]
      (no args = run all cases from test_cases.py that fail under observe_last)
"""
import sys, os, dataclasses
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import jax
import jax.numpy as jnp

from homotopy_core import soft_sim, hard_sim, homotopy_stage, lr_for_beta
import jax_spiking_model as sim
import test_cases as tc_module

BETAS   = [0.5, 1, 2, 3, 5, 8, 13, 21, 34]
NOPT    = 300
TOL     = 1e-6

# fixed 3-neuron chain topology (matches visualize_training.py)
CONNECTIONS      = jnp.array([[0, 1], [1, 2]])
NEURONS_ACTIVATE = jnp.array([0])
NUM_NEURONS      = 10


def fires(v, th):
    """Which neurons fire at least once in this voltage trace?"""
    return [bool(jnp.any(v[:, i] >= th)) for i in range(3)]


def run_case(tc, seed=42):
    params = dataclasses.replace(sim.default_params, steps=1000)
    th = params.threshold
    rng = np.random.default_rng(seed)

    true_strs = jnp.array(tc["true_strs"], dtype=jnp.float32)
    lo = true_strs * tc.get("lo_scale", 0.3)
    hi = true_strs * tc.get("hi_scale", 3.0)

    target_v    = hard_sim(true_strs, params, CONNECTIONS, NUM_NEURONS, NEURONS_ACTIVATE)
    target_fire = fires(target_v, th)

    print(f"\n{'='*70}")
    print(f"Case: {tc['name']}  —  {tc.get('desc','')}")
    print(f"  true_strs={tc['true_strs']}  target fires: N0={target_fire[0]} N1={target_fire[1]} N2={target_fire[2]}")

    # JIT the stage and gradient function once
    @jax.jit
    def stage(w, beta, lr):
        return homotopy_stage(w, true_strs, lo, hi, beta, lr,
                              params, CONNECTIONS, NUM_NEURONS, NEURONS_ACTIVATE,
                              nopt=NOPT, observe_last=True)

    @jax.jit
    def grads_and_loss(w, beta):
        target_s = soft_sim(true_strs, beta, params, CONNECTIONS, NUM_NEURONS, NEURONS_ACTIVATE)
        def loss_fn(w):
            v = soft_sim(w, beta, params, CONNECTIONS, NUM_NEURONS, NEURONS_ACTIVATE)
            return jnp.sum((target_s[:, 2] - v[:, 2]) ** 2)
        return jax.value_and_grad(loss_fn)(w)

    for restart in range(4):
        if restart == 0 and tc.get("init_mods") is not None:
            w = true_strs * jnp.array(tc["init_mods"], dtype=jnp.float32)
        else:
            w = true_strs * jnp.array(rng.uniform(0.5, 1.5, size=2), dtype=jnp.float32)

        init_mods = np.round(np.array(w / true_strs), 3)
        print(f"\n  Restart {restart+1}  init_mods={init_mods}")
        print(f"  {'beta':>5}  {'mods':>16}  {'soft_loss':>12}  {'|grad_w0|':>10}  {'|grad_w1|':>10}  {'hard_loss':>12}  fires")

        for beta in BETAS:
            lr = jnp.float32(lr_for_beta(beta))
            w  = stage(w, jnp.float32(beta), lr)
            sl, g = grads_and_loss(w, jnp.float32(beta))
            v_hard = hard_sim(w, params, CONNECTIONS, NUM_NEURONS, NEURONS_ACTIVATE)
            hl = float(jnp.sum((target_v - v_hard) ** 2))
            f  = fires(v_hard, th)
            mods = np.round(np.array(w / true_strs), 4)
            print(f"  {beta:>5.1f}  {str(mods):>16}  {float(sl):>12.3e}  "
                  f"{float(abs(g[0])):>10.3e}  {float(abs(g[1])):>10.3e}  "
                  f"{hl:>12.3e}  N1={'Y' if f[1] else 'N'} N2={'Y' if f[2] else 'N'}")

            if hl < TOL:
                print(f"  → CONVERGED")
                break


def main():
    names = sys.argv[1:]

    # default: run the cases we know fail under observe_last
    known_fail = [
        "s0_way_low", "s1_way_low", "both_way_low",
        "s0_way_high", "s1_way_high", "both_way_high",
        "s0_hi_s1_lo", "s0_lo_s1_hi",
        "true_300_600", "true_600_300",
        "true_800_100", "true_200_200", "true_1000_1000", "true_600_50",
        "target_n2_silent", "target_n2_silent_asym",
        "start_n1_silent", "start_n1n2_silent",
        "start_n1_too_high_n2_silent", "start_n1_silent_s1_too_high",
    ]
    lookup = {t["name"]: t for t in tc_module.TEST_CASES}

    if names:
        cases = [lookup[n] for n in names if n in lookup]
    else:
        cases = [lookup[n] for n in known_fail if n in lookup]

    for tc in cases:
        run_case(tc)


if __name__ == "__main__":
    main()
