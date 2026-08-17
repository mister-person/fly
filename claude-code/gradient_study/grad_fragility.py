"""Minimal examples of the binding failure: correlated-input weight-split FRAGILITY.

One output neuron, TWO inputs A,B firing a gap delta apart, periodically (so the
2 weights are over-determined by several output spikes).  True weights are
BALANCED.  We recover the weights from the true inputs+targets (NNLS, as
tp_50neuron does) and ask two things:
  1. does the recovered split reproduce the output on the TRUE inputs?  (usually yes)
  2. how much does the output move when the inputs DRIFT by d in an average-
     preserving way (A+d, B-d)?  -> the recurrent sim always drifts inputs, so a
     split that only works on the exact true inputs is fragile.

We sweep the correlation (gap delta) to find where NNLS picks a non-generalising
(vertex) split, and compare to the balanced / min-norm / ridge splits.
"""

import sys, os
sys.path.insert(0, "/workspace/project/gradient_study")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from scipy.optimize import nnls
from grad_method import lif_tangent, TH
import grad_unified as U

T = 360
hk = U.hk


def build(delta, wA, wB):
    """2 input lines: A at 30,130,230 ; B at 30+delta,... ; return input_active."""
    ia = np.zeros((2, T), bool)
    for base in (30, 130, 230):
        ia[0, base] = True
        if base + delta < T:
            ia[1, base + delta] = True
    return ia


def out_spikes(wA, wB, ia):
    return lif_tangent(np.array([wA, wB], float), ia, T)[1]


def solve_nnls(preA, preB, targets):
    prev = 0; rows = []; b = []
    for Tj in targets:
        a1 = sum(hk(Tj - t) for t in preA if prev < t < Tj)
        a2 = sum(hk(Tj - t) for t in preB if prev < t < Tj)
        if max(a1, a2) > 1e-12:
            rows.append([a1, a2]); b.append(TH)
        prev = Tj
    A = np.array(rows); bb = np.array(b)
    w_nnls, _ = nnls(A, bb)
    lam = 1e-3 * np.trace(A.T @ A) / 2
    w_ridge, _ = nnls(np.vstack([A, np.sqrt(lam) * np.eye(2)]), np.concatenate([bb, np.zeros(2)]))
    return w_nnls, w_ridge, A


def drift_sensitivity(wA, wB, delta, d=4):
    """First-output-spike shift per unit average-preserving input drift (A+d, B-d)."""
    def first_sp(shift):
        ia = np.zeros((2, T), bool)
        for base in (30, 130, 230):
            if 0 <= base + shift < T:
                ia[0, base + shift] = True
            if 0 <= base + delta - shift < T:
                ia[1, base + delta - shift] = True
        sp = out_spikes(wA, wB, ia)
        return sp[0] if sp else None
    sp_p, sp_m = first_sp(+d), first_sp(-d)
    if sp_p is None or sp_m is None:
        return np.nan
    return (sp_p - sp_m) / (2 * d)


def main():
    wt = 300.0                          # true BALANCED weights [300,300]
    print("Correlated-input fragility: true weights balanced [300,300]")
    print(f"{'gap':>4} {'out_sp':>18} {'NNLS split':>14} {'ridge split':>14} "
          f"{'drift dt/dd: NNLS':>17} {'bal':>6} {'ridge':>7}")
    for delta in [0, 2, 4, 7, 12, 20, 35]:
        ia = build(delta, wt, wt)
        sp_true = out_spikes(wt, wt, ia)
        if len(sp_true) < 2:
            print(f"{delta:>4}  output too quiet"); continue
        preA = [b for b in (30, 130, 230)]
        preB = [b + delta for b in (30, 130, 230) if b + delta < T]
        w_nnls, w_ridge, A = solve_nnls(preA, preB, sp_true)
        s_nnls = drift_sensitivity(*w_nnls, delta)
        s_bal = drift_sensitivity(wt, wt, delta)
        s_ridge = drift_sensitivity(*w_ridge, delta)
        print(f"{delta:>4} {str(sp_true[:3]):>18} "
              f"[{w_nnls[0]:>4.0f},{w_nnls[1]:>4.0f}] [{w_ridge[0]:>4.0f},{w_ridge[1]:>4.0f}] "
              f"{s_nnls:>17.2f} {s_bal:>6.2f} {s_ridge:>7.2f}")
    print("\ndrift dt/dd near 0 = robust (follows the average like the true neuron);")
    print("near +1 = fragile (tracks input A fully) -> breaks when the sim drifts inputs.")


if __name__ == "__main__":
    main()
