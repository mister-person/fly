"""Unified, direction-free objective for the 2-neuron case.

The move-earlier vs move-later split came from reading the POST-RESET voltage at
the target.  The sub-threshold voltage  V_sub(t) = w * gsw * h(t - t_in)  (no
reset) is MONOTONIC in w, so descending (V_sub(t*) - th)^2 reaches the target from
either side — no barrier, no direction switch.  This is gradient descent on the
direct-solve residual, done per epoch (reset accumulation at each target), plus a
suppression term for input pulses no target claims.

  L(w) = Σ_j (w·A_j - th)^2            A_j = Σ_{t_in in epoch j} gsw·h(t_j* - t_in)
       + λ Σ_{unclaimed pulses i} relu(w·gsw·h_peak - th)^2

We test it on ALL the 2-neuron breakage cases.
"""

import sys, os
sys.path.insert(0, "/workspace/project/gradient_study")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from grad_method import lif_tangent, TH, GSW, DELAY, ND, RD, REFRAC

T = 340
LAM = 1.0
CAP = TH

# EXACT kernel from a sub-threshold probe of the real forward: hk(dt) = voltage
# per unit weight, dt steps after a single input spike (matches lif_tangent).
_PROBE_TIN = 15
# Kernel probe must be LONG enough to capture the slow-decaying membrane tail
# (time constant ~200 steps): truncating it under-estimates V for widely-spaced
# inputs and corrupts multi-input reconstruction.
_TKERNEL = 1100
_ia_probe = np.zeros((1, _TKERNEL), bool); _ia_probe[0, _PROBE_TIN] = True
_Vp = lif_tangent(np.array([50.0]), _ia_probe, _TKERNEL)[0] / 50.0    # V per unit weight


def hk(dt):
    idx = _PROBE_TIN + dt
    return _Vp[idx] if 0 <= idx < _TKERNEL else 0.0


H_PEAK = _Vp.max()


def A_epoch(target, epoch_start, inputs):
    """dV_sub(target)/dw accumulated from inputs within (epoch_start, target]."""
    return sum(hk(target - ti) for ti in inputs if epoch_start < ti <= target)


def grad(w, inputs, targets):
    w = float(w); g = 0.0
    tg = sorted(targets)
    prev = 0
    claimed = set()
    for t_star in tg:
        A = A_epoch(t_star, prev, inputs)
        g += 2.0 * (w * A - TH) * A                 # V_sub(t*) -> th (monotonic both ways)
        for i, ti in enumerate(inputs):             # mark the pulse that serves this target
            if prev < ti <= t_star:
                claimed.add(i)
        prev = t_star
    # suppression: any input pulse no target claims must stay sub-threshold
    for i, ti in enumerate(inputs):
        if i in claimed:
            continue
        Ai = H_PEAK
        if w * Ai > CAP:
            g += 2.0 * LAM * (w * Ai - CAP) * Ai
    return np.array([g])


def train(w0, inputs, targets, iters=1500, lr=4.0):
    w = np.array([float(w0)]); m = v = np.zeros(1)
    for t in range(1, iters + 1):
        gg = np.nan_to_num(grad(w[0], inputs, targets))
        m = 0.9 * m + 0.1 * gg; v = 0.999 * v + 0.001 * gg * gg
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


CASES = [
    ("reachable",        [15],           [80],           500),
    ("dead-start",       [15],           [80],           100),
    ("silent-target",    [15],           [],             500),
    ("unreachable-late", [15],           [150],          500),
    ("over-count",       [15],           [60, 90],       500),
    ("refractory",       [15],           [60, 68],       500),
    ("count-3-match",    [15, 115, 215], [80, 180, 280], 100),
    ("near-peak",        [15],           [120],          500),
    ("move-later",       [15],           [90],           2000),   # spike starts EARLY (~41) -> must move later
    ("move-earlier",     [15],           [70],           300),    # spike starts LATE -> must move earlier
]


def verdict(name, sp, target):
    if name == "silent-target":
        return "PASS" if len(sp) == 0 else "FAIL"
    if name in ("unreachable-late", "over-count", "refractory"):
        return "GRACE(alive)" if len(sp) >= 1 else "DEAD"
    if len(sp) == len(target) and all(abs(a - b) <= 4 for a, b in zip(sorted(sp), sorted(target))):
        return "PASS"
    return "FAIL"


def main():
    print(f"single-input w* (fires) = {TH/H_PEAK:.0f}   "
          f"(objective: gradient descent on the per-epoch V_sub(t*)=th residual)")
    print(f"{'case':16s} {'input':>14s} {'target':>14s} {'init':>5s} {'-> achieved':>16s}  result")
    for name, itimes, target, w0 in CASES:
        ia = make_input(itimes)
        w = train(w0, itimes, target)
        sp = spikes(w, ia)
        print(f"{name:16s} {str(itimes):>14s} {str(target):>14s} {w0:>5d} "
              f"{str(sp):>16s}  {verdict(name, sp, target)}")


if __name__ == "__main__":
    main()
