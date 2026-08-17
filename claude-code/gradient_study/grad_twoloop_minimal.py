"""Minimal example failing the way the 50-NEURON cases do: right spike COUNT, wrong TIMES.

The 50-neuron recurrent cases now recover the output spike COUNT exactly (3/3 on all
three) but the TIMES are off by tens of steps -- e.g. case0 N47 target
[140,340,540,633,722,846,936] vs found [222,426,548,668,787,883,989].  This is the
smallest net found (by random search over 4-6 neuron nets, then edge pruning) that fails
with the same signature.

Net (5 neurons, 10 edges, densely recurrent, output N4):
    N0->N2                            input drive
    N1->N2  N3->N2                    N2's other inputs
    N2->N1  N2->N3                    forward
    N1->N4  N2->N4  N3->N4            output fan-in 3
    N4->N1  N4->N3                    OUTPUT FEEDS BACK into its own inputs

Result: counts correct on ALL seeds, times wrong by up to 23 steps, 0/4 exact --
    seed0 offsets [22, 2,-12, -2,-15,-20,-14,-23]  max 23  mean 13.8
    seed2 offsets [10,-6,-23, -4,-18,-21, -3, -3]  max 23  mean 11.0
which is the 50-neuron signature (case0 N47 offsets [82,86,8,35,65,37,53], max 86).

NOTE -- an earlier 4-neuron version of this file was NOT a valid reproduction.  Pruning
down to two loops dropped the timing error to max 1-10 steps (best seed offsets
[-1,0,0,0,0,1]), i.e. a near-miss like the 2-cycle rather than the gross mistiming seen at
50 neurons.  The property being selected for did not survive the pruning and had to be
re-checked.  What the surviving net has and the pruned one lacks is DENSE recurrence with
the output feeding back into its own multiple input paths -- matching the 50-neuron
fan-in of 6-11.
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

C = np.array([[2, 1], [4, 1], [0, 2], [1, 2], [3, 2],
              [2, 3], [4, 3], [1, 4], [2, 4], [3, 4]], np.int32)
W = np.array([200., 500., 1200., 200., 200., 500., 1200., 200., 900., 900.], np.float32)
N = 5
OUTS = [4]


def main():
    params = G.mkparams(520)
    tv = G.fsim(C, N, W, params)
    T = {n: G.sp(tv, n) for n in range(N)}
    print("5 neurons, output N4; N4 feeds back into N1 and N3, which feed N4 again")
    for n in range(N):
        print(f"   TRUE N{n}: {T[n]}")
    print(f"   output gaps {np.diff(T[OUTS[0]]).tolist()}\n")

    ok = 0
    for seed in range(4):
        w = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(W))).astype(float)
        w = G.train(C, N, OUTS, w, T, params, rounds=300, lr=10.0)
        f = G.sp(G.fsim(C, N, w, params), OUTS[0])
        t = T[OUTS[0]]
        cnt = len(f) == len(t)
        ok += f == t
        if cnt:
            off = [a - b for a, b in zip(f, t)]
            print(f"seed{seed}: found={f}")
            print(f"         offsets {off}  max={max(abs(x) for x in off)} "
                  f"mean={np.mean([abs(x) for x in off]):.1f}")
        else:
            print(f"seed{seed}: COUNT MISMATCH ({len(f)} vs {len(t)})")
    print(f"\nrecovered {ok}/4   (target {T[OUTS[0]]})")


if __name__ == "__main__":
    main()
