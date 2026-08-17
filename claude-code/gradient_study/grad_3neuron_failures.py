"""Failing 3-NEURON cases -- small enough to debug by hand.

Found by sweeping the whole 3-neuron edge space {0->1, 0->2, 1->2, 2->1} against a weight
grid [100,200,300,500,700,900,1200], output = N2, keeping configs the trace gradient fails
on (2 seeds, 150 rounds).  98 failing configs; the four below are the cleanest.

Case A is the most useful: FEEDFORWARD (no recurrence at all), 3 edges, and it fails the
same way on every seed -- which kills the idea that recurrence is what breaks the method.
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
import grad_trace as G

CASES = {
    # A: FEEDFORWARD.  N2 is a weak accumulator firing 3x across 5 input periods, so its
    #    gaps (153,147) come from cross-period integration.  Every seed collapses w(0->2)
    #    200 -> ~60 and raises w(1->2) 300 -> ~420, landing on gaps ~124/173.
    "A feedforward accumulator": ([[0, 1], [0, 2], [1, 2]], [900., 200., 300.]),
    # B: the truth has ONE perturbed spike (341 breaks the 100-step period); the method
    #    returns a perfectly regular train and misses it entirely.
    "B one perturbed spike":     ([[0, 2], [1, 2], [2, 1]], [500., 1200., 200.]),
    # C: largest timing error found in the sweep.
    "C large timing error":      ([[0, 2], [1, 2], [2, 1]], [200., 1200., 500.]),
    # D: count mismatch -- the method drops a spike entirely.
    "D dropped spike":           ([[0, 1], [0, 2], [1, 2]], [200., 1200., 700.]),
}


def main():
    params = G.mkparams(520)
    for name, (E, Wl) in CASES.items():
        C = np.array(E, np.int32); W = np.array(Wl, np.float32)
        tv = G.fsim(C, 3, W, params)
        T = {n: G.sp(tv, n) for n in range(3)}
        print(f"=== {name} ===")
        print(f"    edges {E}  weights {Wl}")
        for n in range(3):
            print(f"    TRUE N{n}: {T[n]}")
        print(f"    output gaps {np.diff(T[2]).tolist()}")
        for seed in range(3):
            w = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(W))).astype(float)
            w = G.train(C, 3, [2], w, T, params, rounds=300, lr=10.0)
            f = G.sp(G.fsim(C, 3, w, params), 2); t = T[2]
            if len(f) == len(t):
                off = [a - b for a, b in zip(f, t)]
                print(f"      seed{seed}: w={np.round(w,0).tolist()} out={f} offsets={off}")
            else:
                print(f"      seed{seed}: w={np.round(w,0).tolist()} out={f} COUNT {len(f)} vs {len(t)}")
        print()


if __name__ == "__main__":
    main()
