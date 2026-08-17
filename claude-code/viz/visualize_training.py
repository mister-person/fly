import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

"""Live voltage visualization of the 3-neuron chain during training.

Applies every trick from learn_homotopy.py:

1. Soft forward pass  -- replace every hard step ``(x >= threshold)`` (both
   synaptic activation and refractory trigger) with a sigmoid of sharpness
   beta.  The whole forward is now smooth and jax.grad works cleanly.

2. Soft target at same beta  -- the optimisation target is soft_sim(true_w,
   beta), NOT hard_sim(true_w).  This guarantees the global minimum is always
   at w = true_weights, whatever beta is.

3. Beta homotopy (continuation)  -- anneal beta from 0.5 → 34.  Low beta is
   a smooth, easy landscape; high beta approaches the hard (spike) model.
   Weights are warm-started at each stage so they track the sharpening minimum.

4. Adam with best-iterate tracking  -- inside each stage we run Adam and
   keep the iterate with the lowest soft loss seen.

5. Multiple random restarts  -- a few restarts cover the rare bistable seeds
   where the easy basin is tiny; we display the best result after each stage.

Network: neuron 0 (driven) -> synapse[0] -> neuron 1 -> synapse[1] -> neuron 2

Usage:
  python visualize_training.py [options]

  --true-strs W0 W1     true synapse weights  (default: 420 420)
  --init-mods M0 M1     starting weight multipliers for restart 0; subsequent
                        restarts are always random  (default: random)
  --n-restarts N        max random restarts before stopping  (default: 8)
  --nopt N              Adam steps per beta stage  (default: 300)
  --betas B [B ...]     beta annealing schedule  (default: 0.5 1 2 3 5 8 13 21 34)
  --runtime N           simulation length in ms-equivalent units (×10 = steps)
                        (default: 100)
  --seed N              RNG seed for random restarts  (default: 42)

Examples:
  python visualize_training.py --true-strs 300 500 --init-mods 1.3 1.3
  python visualize_training.py --true-strs 600 400 --n-restarts 3 --nopt 500
  python visualize_training.py --init-mods 0.5 0.5 --betas 0.5 1 2 5 13 34
"""
import argparse
import dataclasses
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# homotopy_core installs the brian2/neuron_model stubs and imports jax_spiking_model
from homotopy_core import (
    soft_sim as _soft_sim_core,
    hard_sim as _hard_sim_core,
    homotopy_stage as _homotopy_stage_core,
    lr_for_beta,
)
import jax_spiking_model as sim

# ── fixed 3-neuron chain topology ────────────────────────────────────────────
CONNECTIONS      = jnp.array([[0, 1], [1, 2]])
NEURONS_ACTIVATE = jnp.array([0])
NUM_NEURONS      = 10

COLORS = ["tab:blue", "tab:orange", "tab:green"]


# ── thin wrappers binding the fixed topology ──────────────────────────────────
# These keep the same API that test_cases.py imports.

def soft_sim(w, beta, params):
    return _soft_sim_core(w, beta, params, CONNECTIONS, NUM_NEURONS, NEURONS_ACTIVATE)


def hard_sim(w, params):
    return _hard_sim_core(w, params, CONNECTIONS, NUM_NEURONS, NEURONS_ACTIVATE)


@partial(jax.jit, static_argnames=["params", "nopt", "observe_last"])
def homotopy_stage(w0, base, lo, hi, beta, lr, params, nopt=300, observe_last=False):
    return _homotopy_stage_core(
        w0, base, lo, hi, beta, lr,
        params, CONNECTIONS, NUM_NEURONS, NEURONS_ACTIVATE,
        nopt=nopt, observe_last=observe_last,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="3-neuron homotopy training visualizer",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-case", default=None,
                   help="load config from a named entry in test_cases.py "
                        "(true-strs, init-mods, lo-scale, hi-scale); "
                        "any explicit CLI args override the test-case values")
    p.add_argument("--true-strs", type=float, nargs=2, default=None,
                   metavar=("W0", "W1"),
                   help="true synapse weights (default: 420 420)")
    p.add_argument("--init-mods", type=float, nargs=2, default=None,
                   metavar=("M0", "M1"),
                   help="starting multipliers for restart 0 (default: random)")
    p.add_argument("--lo-scale", type=float, default=None,
                   help="lower bound = true_strs * lo_scale (default: 0.3)")
    p.add_argument("--hi-scale", type=float, default=None,
                   help="upper bound = true_strs * hi_scale (default: 3.0)")
    p.add_argument("--n-restarts", type=int, default=8,
                   help="max random restarts (default: 8)")
    p.add_argument("--nopt", type=int, default=300,
                   help="Adam steps per beta stage (default: 300)")
    p.add_argument("--betas", type=float, nargs="+",
                   default=[0.5, 1, 2, 3, 5, 8, 13, 21, 34],
                   help="beta annealing schedule (default: 0.5 1 2 3 5 8 13 21 34)")
    p.add_argument("--runtime", type=int, default=100,
                   help="simulation length in ms-equivalent units ×10=steps (default: 100)")
    p.add_argument("--lr-scale", type=float, default=1.0,
                   help="multiply the LR schedule by this factor — use <1 to "
                        "slow training so you can watch it (default: 1.0)")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="stop when hard loss drops below this threshold (default: 1e-6)")
    p.add_argument("--observe-last", action="store_true",
                   help="restrict training loss to neuron 2 only (partial observation); "
                        "convergence is still judged on all neurons")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for random restarts (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── resolve test case (values become defaults; explicit CLI args override) ──
    tc_true_strs = [420.0, 420.0]
    tc_init_mods = None
    tc_lo_scale  = 0.3
    tc_hi_scale  = 3.0

    if args.test_case:
        import test_cases as _tc
        tc_lookup = {tc["name"]: tc for tc in _tc.TEST_CASES}
        if args.test_case not in tc_lookup:
            names = list(tc_lookup)
            raise SystemExit(f"Unknown test case '{args.test_case}'.\nAvailable: {names}")
        tc = tc_lookup[args.test_case]
        tc_true_strs = tc["true_strs"]
        tc_init_mods = tc.get("init_mods")
        tc_lo_scale  = tc.get("lo_scale", 0.3)
        tc_hi_scale  = tc.get("hi_scale", 3.0)
        print(f"Test case: {args.test_case!r} — {tc['desc']}")

    true_strs_list = args.true_strs  if args.true_strs is not None else tc_true_strs
    init_mods      = args.init_mods  if args.init_mods is not None else tc_init_mods
    lo_scale       = args.lo_scale   if args.lo_scale  is not None else tc_lo_scale
    hi_scale       = args.hi_scale   if args.hi_scale  is not None else tc_hi_scale

    true_strs    = jnp.array(true_strs_list, dtype=jnp.float32)
    betas        = args.betas
    n_restarts   = args.n_restarts
    nopt         = args.nopt
    lr_scale      = args.lr_scale
    tol           = args.tol
    observe_last  = args.observe_last
    params = dataclasses.replace(sim.default_params, steps=args.runtime * 10)
    th     = params.threshold

    obs_str = "N2 only" if observe_last else "all neurons"
    print(f"true_strs={list(np.array(true_strs))}  "
          f"init_mods={init_mods}  lo_scale={lo_scale}  hi_scale={hi_scale}  "
          f"n_restarts={n_restarts}  nopt={nopt}  lr_scale={lr_scale}  "
          f"observe={obs_str}  betas={betas}  steps={params.steps}")
    print("Computing hard target voltages (JIT compiles here, ~5 s)…")
    target_v  = hard_sim(true_strs, params)
    target_np = np.array(target_v)

    lo = true_strs * lo_scale
    hi = true_strs * hi_scale

    rng = np.random.default_rng(args.seed)

    # ── figure setup ─────────────────────────────────────────────────────────
    plt.ion()
    fig = plt.figure(figsize=(13, 11))
    gs  = gridspec.GridSpec(5, 1, figure=fig, hspace=0.65,
                            height_ratios=[2, 2, 2, 1.5, 1.5])

    ax_v    = [fig.add_subplot(gs[i]) for i in range(3)]
    ax_loss = fig.add_subplot(gs[3])
    ax_wts  = fig.add_subplot(gs[4])

    T = np.arange(params.steps)

    lines_tgt, lines_cur = [], []
    for i, ax in enumerate(ax_v):
        lt, = ax.plot(T, target_np[:, i], "--", color=COLORS[i],
                      alpha=0.55, linewidth=1, label="target")
        lc, = ax.plot(T, np.zeros(params.steps), color=COLORS[i],
                      linewidth=1.3, label="current")
        ax.axhline(th, color="gray", lw=0.7, ls=":", alpha=0.7)
        ax.set_title(f"Neuron {i}", fontsize=9)
        ax.set_ylabel("voltage", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, params.steps)
        yhi = max(float(target_np[:, i].max()), th) * 1.7 + 1e-5
        ax.set_ylim(-yhi * 0.05, yhi)
        lines_tgt.append(lt)
        lines_cur.append(lc)
    ax_v[-1].set_xlabel("timestep", fontsize=8)

    loss_ln, = ax_loss.plot([], [], color="red",    lw=1.2, marker="o", ms=4)
    ax_loss.set_ylabel("hard loss", fontsize=8)
    ax_loss.set_xlabel("beta stage", fontsize=8)
    ax_loss.set_xlim(-0.5, len(betas) - 0.5)
    ax_loss.set_xticks(range(len(betas)))
    ax_loss.set_xticklabels([str(b) for b in betas], fontsize=7)

    w0_ln, = ax_wts.plot([], [], color=COLORS[0], lw=1.2, marker="o", ms=4, label="w[0]")
    w1_ln, = ax_wts.plot([], [], color=COLORS[1], lw=1.2, marker="o", ms=4, label="w[1]")
    ax_wts.axhline(1.0, color="gray", lw=0.7, ls="--", alpha=0.7, label="target (1.0)")
    ax_wts.set_ylabel("weight mult", fontsize=8)
    ax_wts.set_xlabel("beta stage",  fontsize=8)
    ax_wts.set_xlim(-0.5, len(betas) - 0.5)
    ax_wts.set_xticks(range(len(betas)))
    ax_wts.set_xticklabels([str(b) for b in betas], fontsize=7)
    ax_wts.legend(fontsize=7, loc="upper right")

    obs_label = " [observe N2 only]" if observe_last else ""
    fig.suptitle(f"3-neuron chain — homotopy training{obs_label}", fontsize=11)

    def redraw(v_np, stage_losses, stage_w0s, stage_w1s, beta, restart, label=""):
        for i in range(3):
            lines_cur[i].set_ydata(v_np[:, i])
        xi = np.arange(len(stage_losses))
        loss_ln.set_data(xi, stage_losses)
        w0_ln.set_data(xi, stage_w0s)
        w1_ln.set_data(xi, stage_w1s)
        ax_loss.set_ylim(0, max(stage_losses) * 1.1 + 1e-10)
        all_w = stage_w0s + stage_w1s
        wlo, whi = min(all_w), max(all_w)
        margin = max((whi - wlo) * 0.15, 0.05)
        ax_wts.set_ylim(wlo - margin, whi + margin)
        fig.suptitle(f"3-neuron chain — beta={beta:.1f}  restart {restart+1}/{n_restarts}"
                     + (f"  {label}" if label else ""), fontsize=11)
        fig.canvas.draw_idle()
        plt.pause(0.05)

    print("Training…  close window to stop.")

    # ── outer restart loop (trick 5) ─────────────────────────────────────────
    best_w    = true_strs * rng.uniform(0.5, 1.5, size=2).astype(np.float32)
    best_loss = float("inf")

    for restart in range(n_restarts):
        if not plt.fignum_exists(fig.number):
            break

        if restart == 0 and init_mods is not None:
            w0 = true_strs * jnp.array(init_mods, dtype=jnp.float32)
        else:
            w0 = true_strs * rng.uniform(0.5, 1.5, size=2).astype(np.float32)
        w   = jnp.array(w0)
        stage_losses, stage_w0s, stage_w1s = [], [], []

        print(f"\nRestart {restart+1}/{n_restarts}  init mods="
              f"{np.array(w/true_strs).round(3)}")

        # ── beta annealing loop (trick 3) ────────────────────────────────────
        for stage_idx, beta in enumerate(betas):
            if not plt.fignum_exists(fig.number):
                break

            lr = lr_for_beta(beta, lr_scale)
            w  = homotopy_stage(w, true_strs, lo, hi, jnp.float32(beta), jnp.float32(lr),
                                params, nopt=nopt, observe_last=observe_last)

            v_np   = np.array(hard_sim(w, params))
            hl     = float(jnp.sum((target_v - jnp.array(v_np)) ** 2))
            mods   = np.array(w / true_strs)
            stage_losses.append(hl)
            stage_w0s.append(float(mods[0]))
            stage_w1s.append(float(mods[1]))

            if hl < best_loss:
                best_loss = hl
                best_w    = w

            print(f"  beta={beta:5.1f}  hard_loss={hl:.4e}  mods={mods.round(4)}")
            redraw(v_np, stage_losses, stage_w0s, stage_w1s, float(beta), restart)

            if best_loss < tol:
                print(f"Converged (hard loss={best_loss:.2e} < tol={tol:.0e}) "
                      f"— pausing. Close window to exit.")
                redraw(v_np, stage_losses, stage_w0s, stage_w1s, float(beta), restart,
                       label="CONVERGED")
                while plt.fignum_exists(fig.number):
                    plt.pause(0.2)
                return

    print(f"\nAll restarts done. Best hard loss: {best_loss:.4e}")
    while plt.fignum_exists(fig.number):
        plt.pause(0.2)


if __name__ == "__main__":
    main()
