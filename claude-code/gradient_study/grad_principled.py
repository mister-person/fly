"""Schedule-free revival via a non-saturating surrogate, and its stress tests.

The van-Rossum gradient uses  dL/dV = d_found · surrogate_slope(V),  and the
surrogate slope a(1-a)·beta/th -> 0 far below threshold, so a dead neuron gets no
gradient.  Fix WITHOUT a schedule: add a non-saturating, margin-based term to the
slope:

    effective_slope(V) = surrogate_slope(V) + REV · relu((th - V)/th)

- margin term is nonzero exactly where the surrogate vanishes (sub-threshold),
- zero at/above threshold (timing precision untouched),
- auto-gated by the residual d_found: it only fires where the loss says the neuron
  is UNDER-representing a target spike (deficit), so it is not a bolted-on
  keep-alive — it follows from the loss.  With an empty target there is no
  deficit, so the neuron is driven silent (no false keep-alive).
"""

import sys, os
sys.path.insert(0, "/workspace/project/gradient_study")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from grad_method import lif_tangent, TH, REFRAC
from grad_continuous import expconv, revconv

T = 340
REV = 60.0          # revival rate of the non-saturating margin term


def principled_grad(w, ia, targets, T, beta_surr=30.0, tau=10.0):
    V, spikes, dV = lif_tangent(w, ia, T)
    decay = np.exp(-1.0 / tau)
    found = (V >= TH).astype(float)
    S = expconv(found, decay)
    tgt = np.zeros(T)
    for t in targets:
        if 0 <= t < T:
            tgt[t] = 1.0
    Stgt = expconv(tgt, decay)
    err = S - Stgt
    d_found = revconv(2.0 * err, decay)
    a = 1.0 / (1.0 + np.exp(-beta_surr * (V / TH - 1.0)))
    surrogate = a * (1.0 - a) * (beta_surr / TH)
    margin = REV * np.clip((TH - V) / TH, 0.0, 1.0)     # non-saturating, sub-th only
    dLdV = d_found * (surrogate + margin)
    return dLdV @ dV, spikes


def train(w0, ia, targets, iters=400, lr=4.0):
    w = np.array([float(w0)]); m = np.zeros(1); v = np.zeros(1)
    for t in range(1, iters + 1):
        g = np.nan_to_num(principled_grad(w, ia, targets, T)[0])
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
        w = np.clip(w - lr * mh / (np.sqrt(vh) + 1e-12), 20, 5000)
    return w


def make_input(times):
    ia = np.zeros((1, T), bool)
    for t in times:
        ia[0, t] = True
    return ia


def spikes(w, ia):
    return lif_tangent(w, ia, T)[1]


def ok(sp, targets, tol=4):
    return len(sp) == len(targets) and all(abs(a - b) <= tol for a, b in zip(sorted(sp), sorted(targets)))


CASES = [
    ("reachable",        [15],           [80],           500, "PASS"),
    ("dead-start",       [15],           [80],           100, "PASS (revive, no schedule)"),
    ("silent-target",    [15],           [],             500, "PASS (go silent, no false keep-alive)"),
    ("unreachable-late", [15],           [150],          500, "graceful FAIL (closest reachable)"),
    ("over-count",       [15],           [60, 90],       500, "graceful FAIL"),
    ("refractory",       [15],           [60, 68],       500, "graceful FAIL (stay ALIVE, not dead)"),
    ("count-3-match",    [15, 115, 215], [80, 180, 280], 100, "PASS from dead"),
    ("near-peak",        [15],           [120],          500, "PASS"),
]


def main():
    print(f"REV={REV}  REFRAC={REFRAC}")
    print(f"{'case':16s} {'target':>14s} {'init':>5s} {'-> achieved':>16s}  {'res':>5s}  expectation")
    for name, itimes, target, w0, exp in CASES:
        ia = make_input(itimes)
        w = train(w0, ia, target)
        sp = spikes(w, ia)
        # for impossible cases, "alive" (>=1 spike) is the graceful outcome
        passed = ok(sp, target)
        if name in ("unreachable-late", "over-count", "refractory"):
            verdict = "GRACE" if len(sp) >= 1 else "DEAD"
        elif name == "silent-target":
            verdict = "PASS" if len(sp) == 0 else "FAIL"
        else:
            verdict = "PASS" if passed else "FAIL"
        print(f"{name:16s} {str(target):>14s} {w0:>5d} {str(sp):>16s}  {verdict:>5s}  {exp}")


if __name__ == "__main__":
    main()
