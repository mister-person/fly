"""Drift-robust split selection: among the weight splits that FIT the targets,
pick the one whose output is least sensitive to input jitter.

For input i, a jitter of its spike time shifts V_sub(t*) by w_i·h'(t*-t_i); the
output crossing shifts by that / slope.  So the output timing variance under
independent input jitter is proportional to  Σ_i (w_i·d_i)^2 ,  d_i = Σ h'(t*-t_i).
We minimise that subject to the fit  Σ_i w_i·h(t*-t_i)=th :

    w = argmin  K·||A w - th||^2  +  ||diag(d)·w||^2        (K large = fit first)

and compare its drift-sensitivity to NNLS, ridge, and the true/balanced split on
the minimal correlated-input examples.
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
def hkp(dt): return hk(dt) - hk(dt - 1)     # discrete kernel derivative h'


def inputs(delta, shift=0):
    ia = np.zeros((2, T), bool)
    for base in (30, 130, 230):
        ia[0, base + shift] = True
        ia[1, base + delta - shift] = True
    return ia


def out(w, delta, shift=0):
    return lif_tangent(np.asarray(w, float), inputs(delta, shift), T)[1]


def matrices(delta, targets):
    preA = [30, 130, 230]; preB = [b + delta for b in (30, 130, 230)]
    prev = 0; A = []; D = []
    for Tj in targets:
        A.append([sum(hk(Tj - t) for t in preA if prev < t < Tj),
                  sum(hk(Tj - t) for t in preB if prev < t < Tj)])
        D.append([sum(hkp(Tj - t) for t in preA if prev < t < Tj),
                  sum(hkp(Tj - t) for t in preB if prev < t < Tj)])
        prev = Tj
    return np.array(A), np.array(D)


def solve_robust(A, D, K=1e6):
    """Minimise sensitivity to the FRAGILE drift mode (average-preserving: inputs
    shift oppositely) subject to fit.  That mode's output sensitivity per epoch is
    w_A·d_A - w_B·d_B, so penalise ||P w||^2 with P = D but the 2nd input negated."""
    b = np.full(len(A), TH)
    P = D.copy()
    P[:, 1] = -P[:, 1]                       # differential (average-preserving) mode
    M = K * (A.T @ A) + (P.T @ P)
    w = np.linalg.solve(M, K * (A.T @ b))
    return np.clip(w, 0.0, 5000.0)


def solve_ridge(A):
    b = np.full(len(A), TH)
    lam = 1e-3 * np.trace(A.T @ A) / 2
    w, _ = nnls(np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])]),
                np.concatenate([b, np.zeros(A.shape[1])]))
    return w


def drift(w, delta, d=4):
    sp_p, sp_m = out(w, delta, +d), out(w, delta, -d)
    if not sp_p or not sp_m:
        return np.nan
    return (sp_p[0] - sp_m[0]) / (2 * d)


def main():
    wt = 300.0
    print("Drift-robust split vs NNLS / ridge / true.  |drift| near 0 = robust.")
    print(f"{'gap':>4}  {'NNLS':>10} {'ridge':>10} {'robust':>10} {'true':>10}   "
          f"{'drift: NNLS':>11} {'ridge':>6} {'robust':>7} {'true':>6}")
    for delta in [0, 4, 7, 12, 20, 35]:
        tgt = out([wt, wt], delta)
        if len(tgt) < 2:
            print(f"{delta:>4} quiet"); continue
        A, D = matrices(delta, tgt)
        w_nnls, _ = nnls(A, np.full(len(A), TH))
        w_ridge = solve_ridge(A)
        w_rob = solve_robust(A, D)
        def sp(w): return f"[{w[0]:.0f},{w[1]:.0f}]"
        print(f"{delta:>4}  {sp(w_nnls):>10} {sp(w_ridge):>10} {sp(w_rob):>10} "
              f"{sp([wt,wt]):>10}   {drift(w_nnls,delta):>11.2f} {drift(w_ridge,delta):>6.2f} "
              f"{drift(w_rob,delta):>7.2f} {drift([wt,wt],delta):>6.2f}")
    print("\nrobust split should match the true/balanced drift (the achievable minimum),")
    print("without being told the true weights — only the fit + jitter-sensitivity.")


if __name__ == "__main__":
    main()
