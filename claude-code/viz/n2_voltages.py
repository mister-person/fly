"""Plot N2 (and N1) voltage traces for hard and soft models.

Shows how the hard model distinguishes true vs init weights clearly,
while the soft model at low beta cannot (explaining the zero-gradient problem).

Usage:
  python viz/n2_voltages.py                    # default: s0_way_low
  python viz/n2_voltages.py --case s1_way_low
  python viz/n2_voltages.py --out my_plot.png
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import dataclasses

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax.numpy as jnp
from homotopy_core import soft_sim, hard_sim
import jax_spiking_model as sim
import test_cases as tc_module

CONNECTIONS      = jnp.array([[0, 1], [1, 2]])
NEURONS_ACTIVATE = jnp.array([0])
NUM_NEURONS      = 10

BETAS_SHOW = [0.5, 1, 2, 5, 13, 34]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="s0_way_low",
                   help="test case name (default: s0_way_low)")
    p.add_argument("--steps", type=int, default=300,
                   help="simulation steps (default: 300)")
    p.add_argument("--out", default=None,
                   help="output file path (default: viz/<case>_n2.png)")
    return p.parse_args()


def run():
    args = parse_args()
    lookup = {t["name"]: t for t in tc_module.TEST_CASES}
    if args.case not in lookup:
        raise SystemExit(f"Unknown case '{args.case}'. Available: {list(lookup)}")
    tc = lookup[args.case]

    params = dataclasses.replace(sim.default_params, steps=args.steps)
    th = params.threshold

    true_strs = jnp.array(tc["true_strs"], dtype=jnp.float32)
    init_mods = tc.get("init_mods")
    if init_mods is None:
        init_mods = [1.0, 1.0]
    init_w = true_strs * jnp.array(init_mods, dtype=jnp.float32)

    T = np.arange(args.steps)

    # hard model traces
    v_hard_true = np.array(hard_sim(true_strs, params, CONNECTIONS, NUM_NEURONS, NEURONS_ACTIVATE))
    v_hard_init = np.array(hard_sim(init_w,    params, CONNECTIONS, NUM_NEURONS, NEURONS_ACTIVATE))

    # soft model traces at each beta
    soft_true = {}
    soft_init = {}
    for beta in BETAS_SHOW:
        soft_true[beta] = np.array(soft_sim(true_strs, float(beta), params, CONNECTIONS, NUM_NEURONS, NEURONS_ACTIVATE))
        soft_init[beta] = np.array(soft_sim(init_w,    float(beta), params, CONNECTIONS, NUM_NEURONS, NEURONS_ACTIVATE))

    # ── layout: 2 rows (N1 / N2), columns = hard + one per beta ─────────────
    ncols = 1 + len(BETAS_SHOW)
    fig, axes = plt.subplots(2, ncols, figsize=(3.5 * ncols, 5), sharex=True)
    fig.suptitle(
        f"{tc['name']}  —  {tc['desc']}\n"
        f"true={tc['true_strs']}  init_mods={init_mods}",
        fontsize=10,
    )

    row_labels = ["N1", "N2"]
    neuron_idx = [1, 2]

    col_titles = ["hard model"] + [f"soft β={b}" for b in BETAS_SHOW]

    for col, title in enumerate(col_titles):
        for row, (lbl, ni) in enumerate(zip(row_labels, neuron_idx)):
            ax = axes[row, col]
            if col == 0:
                v_t = v_hard_true[:, ni]
                v_i = v_hard_init[:, ni]
            else:
                beta = BETAS_SHOW[col - 1]
                v_t = soft_true[beta][:, ni]
                v_i = soft_init[beta][:, ni]

            ax.plot(T, v_t, color="steelblue",  lw=1.2, label="true w")
            ax.plot(T, v_i, color="tomato",      lw=1.2, label="init w", alpha=0.85)
            ax.axhline(th, color="gray", lw=0.7, ls=":", alpha=0.6)

            # annotate max difference
            diff = float(np.max(np.abs(v_t - v_i)))
            ax.set_title(f"{title}\n{lbl}  Δmax={diff:.2e}", fontsize=7.5)

            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")
            if col == 0:
                ax.set_ylabel("voltage", fontsize=8)
            if row == len(neuron_idx) - 1:
                ax.set_xlabel("timestep", fontsize=8)

            ax.tick_params(labelsize=7)

    fig.tight_layout()

    out = args.out or os.path.join(os.path.dirname(__file__), f"{args.case}_n2.png")
    fig.savefig(out, dpi=120)
    print(f"Saved → {out}")


if __name__ == "__main__":
    run()
