"""EXACTLY why the correlated-input reconstruction fails.

Topology:  A -> O,  B -> O   (two sources into one LIF output).
A fires 30,130,230 ; B fires 30+d,130+d,230+d ; true weights balanced [300,300].

The failure has one cause: the reconstruction is UNDER-DETERMINED.  The output
fires periodically, so every output-spike constraint is the SAME equation
    w_A·a1 + w_B·a2 = th
(a1,a2 = the two inputs' PSP contributions at the crossing).  That is ONE equation
in TWO unknowns -> a whole LINE of weight pairs reproduces the output on the true
inputs.  NNLS picks a VERTEX of that line (all weight on A), which makes the output
track A alone -> when the sim drifts the inputs, the crossing moves the full drift.
The balanced interior point follows the average and barely moves.
"""

import sys, os
sys.path.insert(0, "/workspace/project/gradient_study")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from grad_method import lif_tangent, TH, DIR
import grad_unified as U

T = 360
hk = U.hk
DELTA = 7


def inputs(shift=0):
    ia = np.zeros((2, T), bool)
    for base in (30, 130, 230):
        ia[0, base + shift] = True
        ia[1, base + DELTA - shift] = True
    return ia


def out(wA, wB, shift=0):
    ia = inputs(shift)
    V, sp, _ = lif_tangent(np.array([wA, wB], float), ia, T)
    return V, sp


def main():
    ia = inputs()
    Vt, sp_true, _ = lif_tangent(np.array([300., 300.]), ia, T)
    print(f"true [300,300] output spikes: {sp_true}")

    # the reconstruction matrix A (one row per output spike)
    preA = [30, 130, 230]; preB = [30 + DELTA, 130 + DELTA, 230 + DELTA]
    prev = 0; rows = []
    for Tj in sp_true:
        a1 = sum(hk(Tj - t) for t in preA if prev < t < Tj)
        a2 = sum(hk(Tj - t) for t in preB if prev < t < Tj)
        rows.append([a1, a2]); prev = Tj
    A = np.array(rows)
    print(f"\nA-matrix rows (one per output spike):")
    for r in A:
        print(f"   [{r[0]:.5f}, {r[1]:.5f}]")
    print(f"rank(A) = {np.linalg.matrix_rank(A)}  (rows identical -> ONE equation, TWO unknowns)")
    a1, a2 = A[0]
    print(f"\nthe single equation:  w_A·{a1:.4f} + w_B·{a2:.4f} = th={TH}")
    print(f"  -> a LINE of solutions.  Any of these reproduce the output on true inputs:")

    w_nnls, _ = nnls(A, np.full(len(A), TH))
    splits = {
        "NNLS  (vertex, all on A)": w_nnls,
        "balanced (interior)":      np.array([TH / (a1 + a2)] * 2),
        "true":                     np.array([300., 300.]),
        "all on B (other vertex)":  np.array([0.0, TH / a2]),
    }
    print(f"\n  {'split':26s} {'V(t*) true':>11s} {'out spikes':>16s} "
          f"{'drift dt/dd':>12s}")
    for name, w in splits.items():
        _, sp0 = out(*w, shift=0)
        _, spp = out(*w, shift=+4)
        _, spm = out(*w, shift=-4)
        dvdd = ((spp[0] - spm[0]) / 8.0) if (spp and spm) else float("nan")
        vts = a1 * w[0] + a2 * w[1]                # V_sub at the crossing (per weight sum)
        print(f"  {name:26s} {vts/TH:>10.3f}x {str(sp0[:3]):>16s} {dvdd:>12.2f}")

    # ── figure: output voltage under NNLS vs balanced, true vs drifted inputs ──
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for a, shift, title in [(ax[0], 0, "true inputs"), (ax[1], 6, "inputs drifted (A+6, B-6)")]:
        for name, w, c in [("NNLS [%.0f,%.0f]" % tuple(w_nnls), w_nnls, "C3"),
                           ("balanced [%.0f,%.0f]" % (TH/(a1+a2), TH/(a1+a2)),
                            np.array([TH/(a1+a2)]*2), "C0")]:
            V, sp = out(*w, shift=shift)
            a.plot(V[:120] / TH, color=c, label=name)
            if sp:
                a.plot(sp[0], 1.0, "v", color=c, ms=9)
        # mark input arrival times
        a.axvline(30 + shift, color="gray", ls=":", lw=1)
        a.axvline(30 + DELTA - shift, color="gray", ls="--", lw=1)
        a.axhline(1.0, color="k", ls="--", lw=1)
        a.set_title(f"{title}: A at {30+shift}, B at {30+DELTA-shift}", fontsize=9)
        a.set_xlabel("t"); a.set_ylabel("V(O)/th"); a.legend(fontsize=8); a.set_ylim(0, 1.3)
    fig.suptitle("Both splits fire identically on TRUE inputs; under input drift the NNLS "
                 "vertex tracks A (moves), the balanced split follows the average (stays)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(f"{DIR}/grad_fragility_why.png", dpi=120)
    print(f"\nwrote {DIR}/grad_fragility_why.png")


if __name__ == "__main__":
    main()
