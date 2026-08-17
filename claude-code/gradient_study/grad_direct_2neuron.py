"""The direct way (no gradient, no annealing) for the 2-neuron case.

For a single input, the sub-threshold output voltage is LINEAR in the weight:
    V(t) = w * gsw * h(t - t_in)
so to make the crossing land at t*, solve one equation:
    w* = th / (gsw * h(t* - t_in))
which we read off empirically as  w* = th * w_test / V_test(t*)  from any one
sub-threshold probe run.  One shot, exact — this is target propagation's 1-hop
solve.  Iteration/annealing is only needed when the INPUT spike times themselves
depend on the weights (recurrence); for a fixed-input readout it's a linear solve.
"""

import sys, os
sys.path.insert(0, "/workspace/project/gradient_study")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib.pyplot as plt
from grad_method import lif_tangent, TH, DIR

T = 220
T_IN = 15
ia = np.zeros((1, T), bool); ia[0, T_IN] = True


def crossing(w):
    sp = lif_tangent(np.array([float(w)]), ia, T)[1]
    return sp[0] if sp else None


def main():
    # one sub-threshold probe gives the whole PSP shape V_test(t) (linear in w)
    w_test = 100.0
    V_test, sp_test, _ = lif_tangent(np.array([w_test]), ia, T)
    assert not sp_test, "probe must stay sub-threshold"

    def solve_w(t_star):
        vt = V_test[t_star]
        if vt <= 0:
            return None
        return TH * w_test / vt          # w* = th / (gsw*h) , one shot

    c0 = crossing(700.0)
    targets = list(range(c0 - 40, c0 + 45, 2))
    achieved, reachable = [], []
    for tstar in targets:
        w = solve_w(tstar)
        # rising phase only: h increasing up to its peak -> V_test increasing
        on_rising = tstar >= 1 and V_test[tstar] >= V_test[tstar - 1]
        reachable.append(on_rising and w is not None and 20 <= w <= 3000)
        achieved.append(crossing(w) if (w and 20 <= w <= 3000) else np.nan)

    errs = [abs(a - t) for a, t, r in zip(achieved, targets, reachable) if r and a == a]
    print(f"probe w={w_test}; natural crossing at w=700 is {c0}")
    print(f"reachable targets: {sum(reachable)}/{len(targets)}  "
          f"(rising phase of the PSP)")
    print(f"one-shot direct solve — mean |achieved-target| over reachable: "
          f"{np.mean(errs):.2f} steps (max {np.max(errs):.0f})")
    # examples
    for tstar in [c0 - 20, c0, c0 + 20, c0 + 40]:
        w = solve_w(tstar)
        print(f"  target {tstar}: w*={w:.0f} -> crossing {crossing(w)}")

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    ta = [t for t, r in zip(targets, reachable) if r]
    aa = [a for a, r in zip(achieved, reachable) if r]
    ax.plot(ta, aa, "s-", color="C2", ms=5, label="direct solve  w*=th/(gsw·h)")
    ax.plot(targets, targets, color="gray", ls="--", label="ideal (achieved=target)")
    ax.set_title("2-neuron: direct closed-form weight hits every reachable target\n"
                 "(one shot — no gradient, no window, no annealing)")
    ax.set_xlabel("requested target time"); ax.set_ylabel("achieved spike time")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{DIR}/grad_direct_2neuron.png", dpi=120)
    print(f"wrote {DIR}/grad_direct_2neuron.png")


if __name__ == "__main__":
    main()
