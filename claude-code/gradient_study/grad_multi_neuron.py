"""Per-neuron voltage + gradient visualisation, with EXTRA-SPIKE SUPPRESSION.

The voltage-target objective, now with both directions of force:

  create   : at a target time with no spike, push V UP toward threshold
             grad += (V(t*) - th) * dV/dw            (dead/silent -> revive)
  suppress : at a spike with no nearby target, push V DOWN below threshold
             grad += lam * (V(t_s) - (1-m)*th) * dV/dw   (kill the extra spike)

We show, for each of several output neurons (each with its own weights into a
shared set of 5 independent pulse-inputs), the voltage trace and the per-timestep
"force on the voltage" (-dL/dV): green = pushed up to make a spike, red = pushed
down to remove one.  We also compare fire-only vs fire+suppress to show the
extra-spike problem and its fix.
"""

import sys, os, types
sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")
for _n, _a in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        for _k, _v in _a.items():
            setattr(_m, _k, _v)
        sys.modules[_n] = _m

import numpy as np
import matplotlib.pyplot as plt
from grad_method import lif_tangent, TH, DIR

MARGIN = 0.12
WINDOW = 20          # a spike within WINDOW of a target counts as "that" target
LAM    = 1.0         # suppression strength


def voltage_grad(w, ia, targets, T, suppress=True):
    """Return (grad, spikes, V, force) for the create(+suppress) objective.
    force[t] = -dL/dV[t]  (up>0 to make a spike, down<0 to remove one)."""
    V, spikes, dV = lif_tangent(w, ia, T)
    g = np.zeros(len(w)); force = np.zeros(T)
    cap = (1.0 - MARGIN) * TH
    # create: targets with no spike nearby -> push V up at t*
    for tt in targets:
        if not any(abs(s - tt) <= WINDOW for s in spikes):
            if V[tt] < TH:
                g += (V[tt] - TH) * dV[tt]
                force[tt] += (TH - V[tt])
    # suppress: spikes with no target nearby -> push V down at t_s
    if suppress:
        for ts in spikes:
            if not any(abs(ts - tt) <= WINDOW for tt in targets):
                if V[ts] > cap:
                    g += LAM * (V[ts] - cap) * dV[ts]
                    force[ts] -= (V[ts] - cap)
    return g, spikes, V, force


def train(w0, ia, targets, T, suppress=True, step=4.0, iters=300):
    w = np.array(w0, float)
    for _ in range(iters):
        g = voltage_grad(w, ia, targets, T, suppress)[0]
        gn = np.linalg.norm(g)
        if gn > 1e-30:
            w = np.clip(w - step * g / gn, 20, 3000)
    return w


def pulse_inputs(onsets, T):
    ia = np.zeros((len(onsets), T), bool)
    for i, o in enumerate(onsets):
        ia[i, o] = True
    return ia


def main():
    onsets = [15, 115, 215, 315]           # 4 pulses spaced >> refractory (one input each)
    T = 440
    ia = pulse_inputs(onsets, T)
    n_in = len(onsets)

    # natural crossing time of each pulse at a strong weight -> use as target times
    cross = []
    for i in range(n_in):
        wsolo = np.zeros(n_in); wsolo[i] = 800.0
        sp = lif_tangent(wsolo, ia, T)[1]
        cross.append(sp[0] if sp else onsets[i] + 80)
    cross = np.array(cross)
    print("pulse crossing times (strong weight):", cross.tolist())

    # three output neurons, each: target = fire on a chosen SUBSET of pulses
    neurons = [
        ("A: target pulses {1,3}", [1, 3]),
        ("B: target pulses {0,1,2,3}", [0, 1, 2, 3]),
        ("C: target pulse {2} only", [2]),
    ]
    # messy init: fires pulses 0 and 2 only (weights 1,3 are sub-threshold).
    # So neuron A must CREATE 1,3 and SUPPRESS 0,2 (both forces); B only creates;
    # C only suppresses -> all three cases visible.
    w_init = np.array([700.0, 100.0, 700.0, 100.0])

    fig, axes = plt.subplots(len(neurons), 1, figsize=(11, 9), sharex=True)
    for row, (name, tgt_pulses) in enumerate(neurons):
        targets = [int(cross[i]) for i in tgt_pulses]
        w = train(w_init, ia, targets, T, suppress=True)
        g, spikes, V, force = voltage_grad(w, ia, targets, T, suppress=True)
        sp0 = lif_tangent(w_init, ia, T)[1]
        print(f"{name}: init spikes={sp0} -> trained spikes={spikes}  targets={targets}")

        ax = axes[row]
        ax.plot(V, color="C0", lw=1.5, label="V(t)")
        ax.axhline(TH, color="k", ls="--", lw=1)
        for tt in targets:
            ax.axvline(tt, color="green", ls=":", lw=1)
        ax.plot(spikes, [TH] * len(spikes), "v", color="C0", ms=8)
        ax.set_ylim(0, TH * 1.5); ax.set_ylabel("V")
        ax.set_title(f"{name}   (green ⋮ = target times, ▼ = actual spikes)", fontsize=10)
        # gradient force on twin axis
        axg = ax.twinx()
        # recompute force at the INITIAL weights to show what the gradient did
        _, _, _, force0 = voltage_grad(w_init, ia, targets, T, suppress=True)
        up = np.where(force0 > 0)[0]; dn = np.where(force0 < 0)[0]
        axg.vlines(up, 0, force0[up], color="green", lw=3, alpha=.6)
        axg.vlines(dn, 0, force0[dn], color="red", lw=3, alpha=.6)
        axg.axhline(0, color="gray", lw=.5)
        m = np.abs(force0).max() or 1
        axg.set_ylim(-1.3 * m, 1.3 * m); axg.set_ylabel("force -dL/dV", color="gray")
    axes[-1].set_xlabel("t")
    fig.suptitle("Per-neuron voltage and gradient force (green=push up to spike, "
                 "red=push down to suppress).\nForce shown at the INITIAL over-active "
                 "weights; each neuron trained to its target spike count.", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{DIR}/grad_multi_neuron.png", dpi=120)
    print(f"wrote {DIR}/grad_multi_neuron.png")

    # ── with vs without suppression, on neuron C (target = 1 spike) ──────────
    print("\nExtra-spike suppression on/off (neuron C, target = pulse 2 only):")
    targets = [int(cross[2])]
    fig2, ax2 = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for k, sup in enumerate([False, True]):
        w = train(w_init, ia, targets, T, suppress=sup)
        V, spikes, _ = lif_tangent(w, ia, T)
        lab = "fire+SUPPRESS" if sup else "fire only (no suppression)"
        print(f"  {lab:28s}: spikes={spikes}  (target={targets})")
        a = ax2[k]
        a.plot(V, color="C0"); a.axhline(TH, color="k", ls="--", lw=1)
        for tt in targets:
            a.axvline(tt, color="green", ls=":", lw=1)
        a.plot(spikes, [TH] * len(spikes), "v", color=("C3" if not sup else "C2"), ms=9)
        a.set_ylim(0, TH * 1.5); a.set_ylabel("V")
        a.set_title(f"{lab}: {len(spikes)} spike(s)  {spikes}", fontsize=10)
    ax2[-1].set_xlabel("t")
    fig2.suptitle("Without a suppression term the extra spikes survive; with it, only "
                  "the target remains", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(f"{DIR}/grad_suppression.png", dpi=120)
    print(f"wrote {DIR}/grad_suppression.png")


if __name__ == "__main__":
    main()
