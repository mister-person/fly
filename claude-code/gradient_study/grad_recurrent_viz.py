"""Before/after visualisation of a recurrent (chain) training run: for each
trainable neuron, its VOLTAGE and gradient FORCE at the random init vs after
layer-local voltage-target learning.

Rows: N1, N2, N3 (N0 is the periodic input driver).
Each row: init voltage (grey dashed) and final voltage (blue); threshold; target
spike times (green dotted); final spikes (markers); and the gradient force
-dL/dV at init (bold stems, green=push up / red=push down) and at the end
(faint stems, ~0 once learned).
"""

import sys, os
sys.path.insert(0, "/workspace/project/gradient_study")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib.pyplot as plt
from grad_recurrent import (NETS, full_sim, spikes_of, incoming, local_inputs,
                            train_recurrent, count_err, T, OUTPUT)
from grad_multi_neuron import voltage_grad
from grad_method import TH, DIR


def main():
    net = NETS["chain"]
    Vt = full_sim(net, net["w"])
    T_true = {n: spikes_of(Vt, n) for n in range(4)}
    targets = {n: T_true[n] for n in [1, 2, 3]}

    # pick the seed whose trained output best matches, for a clear before/after
    best = None
    for seed in range(6):
        w_final = train_recurrent(net, targets, seed=seed, iters=60, inner=30)
        V = full_sim(net, w_final)
        ce = count_err(V, targets)
        if best is None or ce < best[0]:
            rng = np.random.default_rng(seed)
            w_init = (net["w"] * rng.uniform(0.5, 1.5, len(net["w"]))).astype(np.float64)
            best = (ce, seed, w_init, w_final)
    ce, seed, w_init, w_final = best
    print(f"using seed {seed} (count-err {ce})")

    V_init = full_sim(net, w_init)
    V_final = full_sim(net, w_final)
    print("init  output N3:", spikes_of(V_init, OUTPUT))
    print("final output N3:", spikes_of(V_final, OUTPUT), " target", T_true[OUTPUT])

    neurons = [1, 2, 3]
    fig, axes = plt.subplots(len(neurons), 1, figsize=(12, 9), sharex=True)
    for row, n in enumerate(neurons):
        syn, pres = incoming(net, n)
        ia_i = local_inputs(V_init, pres)
        ia_f = local_inputs(V_final, pres)
        _, _, _, force_i = voltage_grad(w_init[syn], ia_i, targets[n], T, suppress=True)
        _, _, _, force_f = voltage_grad(w_final[syn], ia_f, targets[n], T, suppress=True)

        ax = axes[row]
        ax.plot(V_init[:, n], color="0.6", ls="--", lw=1.2, label="V init")
        ax.plot(V_final[:, n], color="C0", lw=1.6, label="V final")
        ax.axhline(TH, color="k", ls="--", lw=1)
        for tt in targets[n]:
            ax.axvline(tt, color="green", ls=":", lw=1)
        fsp = spikes_of(V_final, n)
        ax.plot(fsp, [TH] * len(fsp), "v", color="C0", ms=8)
        ax.set_ylim(0, TH * 1.5); ax.set_ylabel(f"N{n}  V")
        ax.set_title(f"N{n}  (inputs {pres.tolist()}): init spikes "
                     f"{spikes_of(V_init,n)} -> final {fsp}   target {targets[n]}",
                     fontsize=9)
        if row == 0:
            ax.legend(fontsize=8, loc="upper right")

        axg = ax.twinx()
        for f, alpha, lw in [(force_i, 0.7, 3), (force_f, 0.9, 1.2)]:
            up = np.where(f > 1e-9)[0]; dn = np.where(f < -1e-9)[0]
            axg.vlines(up, 0, f[up], color="green", lw=lw, alpha=alpha)
            axg.vlines(dn, 0, f[dn], color="red", lw=lw, alpha=alpha)
        m = max(np.abs(force_i).max(), 1e-9)
        axg.set_ylim(-1.3 * m, 1.3 * m)
        axg.set_ylabel("force -dL/dV", color="gray", fontsize=8)

    axes[-1].set_xlabel("t")
    fig.suptitle("Recurrent chain N0→N1→N2→N3: per-neuron voltage & gradient, "
                 "INIT vs AFTER learning\n(grey dashed=init V, blue=final V; bold "
                 "stems=init gradient force, faint=final; green ⋮=targets)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{DIR}/grad_recurrent_viz.png", dpi=120)
    print(f"wrote {DIR}/grad_recurrent_viz.png")


if __name__ == "__main__":
    main()
