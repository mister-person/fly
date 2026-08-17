"""Minimal example of the 50-neuron OVER-FIRING failure of output-only inference.

Same root cause isolated on the 50-neuron cases (grad_relax_50, true_pre run): the
failure is TARGET ASSIGNMENT, not presynaptic timing.  A hidden neuron with an edge
into an output that is really driven by a DIFFERENT source gets CREDITED for that
output's spikes, so output-only inference demands it fire when it shouldn't.

Minimal net (4 neurons):
    N0 -> N1   weak  (w150): N1 truly NEVER fires
    N0 -> N2   strong(w500): N2 fires 5x on its own  (output)
    N1 -> N2   weak  (w50) : real but irrelevant
    N1 -> N3   (w500)      : N3 truly silent          (output)
Because N1 feeds output N2, the inference credits N1 with ALL of N2's spikes and infers
N1 must fire ~5x.  That over-firing then drives N3 (fed only by N1) to spike, though its
target is empty -> outputs cannot be recovered.
"""
import sys, os, dataclasses, types
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
from grad_infer_relax import fsim, sp, infer_relax, solve_vsub

C = np.array([[0, 1], [0, 2], [1, 2], [1, 3]], np.int32)
N = 4
OUTS = [2, 3]
W_TRUE = np.array([150., 500., 50., 500.], np.float32)


def main():
    tv = fsim(C, N, W_TRUE); T = {n: sp(tv, n) for n in range(N)}
    print("true spikes:", {n: T[n] for n in range(N)})
    print("  N1 should be SILENT; N3 should be SILENT; only N2 fires (from N0 directly).\n")

    inc = {n: (np.where(C[:, 1] == n)[0], C[np.where(C[:, 1] == n)[0], 0]) for n in range(N)}
    out_t = {o: T[o] for o in OUTS}
    for seed in range(4):
        w = (W_TRUE * np.random.default_rng(seed).uniform(0.5, 1.5, len(W_TRUE))).astype(float)
        tgt = {}
        for _ in range(30):
            V = fsim(C, N, w); spall = {p: sp(V, p) for p in range(N)}
            tgt = infer_relax(C, N, out_t, spall, tgt)
            for n in range(1, N):
                if n not in tgt or not tgt[n]:
                    continue
                syn, pres = inc[n]
                if len(syn) == 0:
                    continue
                sol = solve_vsub([spall[int(p)] for p in pres], tgt[n], robust=(len(pres) > 1))
                w[syn] = (1 - 0.5) * w[syn] + 0.5 * sol
        V = fsim(C, N, w)
        if seed == 0:
            print(f"inferred N1 target = {tgt.get(1)}  (true N1 = {T[1]})  <-- over-attributed")
        print(f"seed{seed}: outputs found N2={sp(V,2)} N3={sp(V,3)}  "
              f"(target N2={T[2]} N3={T[3]})  "
              f"{'OK' if all(sp(V,o)==T[o] for o in OUTS) else 'FAIL (N3 over-fires)'}")


if __name__ == "__main__":
    main()
