"""Minimal example where the SILENCE VETO is too strong (mirror of grad_overfire_minimal).

Net (4 neurons):
    N0 -> N1  (w500): N1 must FIRE
    N1 -> N2  (w500): N2 is driven ONLY by N1        (output, must fire)
    N1 -> N3  (w50) : too weak to reach threshold     (output, must stay SILENT)

N3's silence does NOT mean "N1 must be silent" -- it means "the N1->N3 WEIGHT must stay
small".  But the veto charges it against N1's FIRING: N1's credit for N2 is 1.0 and its
structural influence on silent N3 is also 1.0 (N1 is N3's only input), so veto >= credit
and N1 is inferred silent -> N2 loses its only driver -> both outputs wrong.

The veto is applied to the wrong variable: silence constrains the EDGE, not the SOURCE.
"""
import sys, os, types
sys.path.insert(0, "/workspace/project/gradient_study")
sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")
for _n, _a in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        for _k, _v in _a.items():
            setattr(_m, _k, _v)
        sys.modules[_n] = _m
import numpy as np
import grad_credit as G
from grad_infer_relax import run as run_relax

C = [[0, 1], [1, 2], [1, 3]]
N = 4
OUTS = [2, 3]
W = [500., 500., 50.]


def main():
    params = G.mkparams(520)
    Cn = np.array(C, np.int32)
    tv = G.fsim(Cn, N, np.array(W, np.float32), params)
    T = {n: G.sp(tv, n) for n in range(N)}
    print("true spikes:", {n: T[n] for n in range(N)})
    print("  N1 MUST fire (sole driver of N2); N3 must stay silent (w=50 too weak).\n")

    print("credit + silence veto:")
    G.run("  veto version", C, N, OUTS, W, verbose=True)
    print("\nrelaxation loop (no veto) on the same net:")
    run_relax("  no-veto version", C, N, OUTS, W, rounds=30)


if __name__ == "__main__":
    main()
