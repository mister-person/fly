"""Voltages + gradient force for the unified sub-threshold-voltage objective.

For a single input the objective acts on the SUB-THRESHOLD voltage
V_sub(t) = w * hk(t - t_in) (no reset), pulling V_sub(t*) -> th at each target.
The gradient force at a target is  (th - V_sub(t*))  (up if below th, down if
above) — this is monotonic in w, so it points the right way whether the spike
must move earlier or later, or be revived from dead.

Each panel: init V_sub (grey dashed) and final V_sub (blue); the real forward
voltage after learning (thin, with the actual spike); threshold; target time;
and the INITIAL gradient force at the target (green up / red down arrow).
"""

import sys, os
sys.path.insert(0, "/workspace/project/gradient_study")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib.pyplot as plt
from grad_method import lif_tangent, TH, DIR
import grad_unified as U

T = U.T
TIN = 15


def vsub(w):
    return np.array([w * U.hk(t - TIN) for t in range(T)])


def realV(w):
    return lif_tangent(np.array([float(w)]), U.make_input([TIN]), T)[0]


CASES = [
    ("dead-start  (revive)",       [80],  100),
    ("move-earlier (late init)",   [80],  460),   # fires ~102 -> must move earlier
    ("move-later  (early init)",   [90],  2000),  # fires ~41  -> must move later
    ("refractory (impossible->graceful)", [60, 68], 500),
]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (name, targets, w0) in zip(axes.ravel(), CASES):
        wf = U.train(w0, [TIN], targets)[0]
        Vsi, Vsf = vsub(w0), vsub(wf)
        Vr = realV(wf)
        ax.plot(Vsi / TH, color="0.6", ls="--", lw=1.3, label="V_sub init")
        ax.plot(Vsf / TH, color="C0", lw=1.6, label="V_sub final")
        ax.plot(Vr / TH, color="C0", lw=0.9, alpha=0.5, label="real V (final)")
        ax.axhline(1.0, color="k", ls="--", lw=1)
        sp = lif_tangent(np.array([wf]), U.make_input([TIN]), T)[1]
        ax.plot(sp, [1.05] * len(sp), "v", color="C0", ms=9)
        for tt in targets:
            ax.axvline(tt, color="green", ls=":", lw=1.2)
            # initial gradient force at the target: (th - V_sub_init(t*))/th
            f = (TH - Vsi[tt]) / TH
            col = "green" if f > 0 else "red"
            ax.annotate("", xy=(tt, 1.0), xytext=(tt, Vsi[tt] / TH),
                        arrowprops=dict(arrowstyle="->", color=col, lw=2.5))
        ax.set_title(f"{name}: init w={w0:.0f} -> {wf:.0f}, spikes {sp} (target {targets})",
                     fontsize=9)
        ax.set_ylabel("V / threshold"); ax.set_ylim(0, 2.2); ax.set_xlabel("t")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Unified objective: the gradient force (arrow) pulls the sub-threshold "
                 "voltage V_sub(t*) to threshold at each target —\nup or down as needed, "
                 "no direction switch, no barrier (green ⋮ = target, ▼ = actual spike)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{DIR}/grad_unified_viz.png", dpi=120)
    print("wrote", f"{DIR}/grad_unified_viz.png")
    for name, targets, w0 in CASES:
        wf = U.train(w0, [TIN], targets)[0]
        sp = lif_tangent(np.array([wf]), U.make_input([TIN]), T)[1]
        print(f"  {name:34s} w {w0}->{wf:.0f}  spikes {sp}  target {targets}")


if __name__ == "__main__":
    main()
