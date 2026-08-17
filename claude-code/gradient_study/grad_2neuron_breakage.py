"""Stress-test the linear-inspired (van-Rossum) gradient on the minimal 2-neuron
system (one input neuron -> one output neuron, single weight), to find where it
breaks BEFORE scaling up.

Each case: an input spike pattern + a target output spike pattern + a prediction.
We train with the continuous van-Rossum gradient (Adam) and report achieved vs
target, PASS/FAIL, and whether it matches the prediction.  "Impossible" cases are
expected to fail — the question is whether it fails *gracefully*.
"""

import sys, os
sys.path.insert(0, "/workspace/project/gradient_study")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from grad_method import lif_tangent, TH, DIR, REFRAC
from grad_continuous import continuous_grad

T = 340


def make_input(times):
    ia = np.zeros((1, T), bool)
    for t in times:
        ia[0, t] = True
    return ia


def train(w0, ia, targets, iters=400, lr=4.0, beta_surr=30.0):
    w = np.array([float(w0)]); m = np.zeros(1); v = np.zeros(1)
    for t in range(1, iters + 1):
        g = np.nan_to_num(continuous_grad(w, ia, targets, T, beta_surr=beta_surr)[0])
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
        w = np.clip(w - lr * mh / (np.sqrt(vh) + 1e-12), 20, 5000)
    return w


def spikes(w, ia):
    return lif_tangent(w, ia, T)[1]


def evaluate(sp, targets, tol=4):
    if len(sp) != len(targets):
        return False
    return all(abs(a - b) <= tol for a, b in zip(sorted(sp), sorted(targets)))


CASES = [
    # name, input_times, target_out, init_w, prediction
    ("reachable",        [15],            [80],       500, "PASS (rising phase)"),
    ("dead-start",       [15],            [80],       100, "FAIL: sharp surrogate can't revive"),
    ("unreachable-late", [15],            [150],      500, "FAIL: past PSP peak, crossing stays on rising edge"),
    ("over-count",       [15],            [60, 90],   500, "FAIL: one input pulse -> one crossing"),
    ("refractory",       [15],            [60, 68],   500, "FAIL: targets closer than refractory"),
    ("selective-sub",    [15, 115, 215],  [80, 280],  500, "FAIL: single weight is all-or-none across pulses"),
    ("count-3-match",    [15, 115, 215],  [80, 180, 280], 500, "PASS: consistent per-pulse timing"),
    ("near-peak",        [15],            [120],      500, "risky: low slope near PSP peak"),
]


def main():
    print(f"REFRAC={REFRAC}  T={T}")
    print(f"{'case':16s} {'input':>14s} {'target':>14s} {'-> achieved':>16s}  "
          f"{'result':>6s}  prediction")
    for name, itimes, target, w0, pred in CASES:
        ia = make_input(itimes)
        w = train(w0, ia, target)
        sp = spikes(w, ia)
        ok = evaluate(sp, target)
        # for dead-start also try a WIDE surrogate (revival fix)
        extra = ""
        if name == "dead-start" and not ok:
            w2 = train(w0, ia, target, beta_surr=4.0)
            sp2 = spikes(w2, ia)
            extra = f"   [wide-surrogate beta=4 -> {sp2} {'PASS' if evaluate(sp2,target) else 'FAIL'}]"
        res = "PASS" if ok else "FAIL"
        print(f"{name:16s} {str(itimes):>14s} {str(target):>14s} {str(sp):>16s}  "
              f"{res:>6s}  {pred}{extra}")


if __name__ == "__main__":
    main()
