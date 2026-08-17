"""Minimal example where COMPETITIVE credit breaks: a spike with multiple legitimate sources.

Competitive credit gives each downstream spike to the presynaptic with the top share.
But a spike can legitimately have SEVERAL sources that all deserve credit.  Handing it to
the majority driver alone starves the others.

Net (4 neurons):
    N0->N1 (500): N1 fires PERIODICALLY          [72, 172, 272, 372, 472]
    N0->N2 (300): N2 is a weak ACCUMULATOR that fires SPARSELY  [140, 340]
    N1->N3 (350), N2->N3 (400):  output N3       [167, 301, 403]   ISIs [134, 102]

N1 supplies the larger share of N3's drive, so competitive credit awards it every one of
N3's spikes and N2 receives ZERO credit -> N2 is inferred silent on 4/4 seeds -> the
output collapses to N1's REGULAR period-100 pattern [175,275,375,475] (4 spikes, wrong
times) instead of the irregular target (3 spikes).

Why this example is sound (not just a weight degeneracy): N2's sparse, off-phase firing
is what makes N3's output IRREGULAR (ISIs 134 vs 102).  N1 alone is periodic, so no
setting of w(1->3) can reproduce the target -- verified by scanning the whole range
20..3000: NO match exists, closest is [231,431].  The second source is genuinely
necessary, so this is an OUTPUT failure, not an unidentifiable-weights artifact.
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

C = np.array([[0, 1], [0, 2], [1, 3], [2, 3]], np.int32)
N = 4
OUTS = [3]
W = np.array([500., 300., 350., 400.], np.float32)


def main():
    params = G.mkparams(520)
    tv = G.fsim(C, N, W, params)
    T = {n: G.sp(tv, n) for n in range(N)}
    print("true spikes:", {n: T[n] for n in range(N)})
    print(f"  N1 periodic ({len(T[1])}x), N2 sparse ({len(T[2])}x); "
          f"output ISIs {np.diff(T[3]).tolist()} are IRREGULAR -- only N2 can cause that.\n")

    ok = 0
    for seed in range(4):
        w = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(W))).astype(float)
        w, tgt = G.train(C, N, OUTS, w, T, params, rounds=30)
        V = G.fsim(C, N, w, params)
        out = G.sp(V, 3)
        ok += out == T[3]
        print(f"seed{seed}: N2 target={tgt.get(2)} -> N2 fires {len(G.sp(V,2))}x (true {len(T[2])}x)"
              f"   OUT={out}  (target {T[3]})  {'OK' if out == T[3] else 'FAIL'}")
    print(f"\nrecovered {ok}/4")

    print("\nnon-compensability check -- scan w(1->3) with N2 silenced:")
    hits = [x for x in range(20, 3001, 20)
            if G.sp(G.fsim(C, N, np.array([500., 300., float(x), 0.], np.float32), params), 3) == T[3]]
    print(f"  w(1->3) values reproducing the target output: "
          f"{hits if hits else 'NONE -- N2 is genuinely required'}")


if __name__ == "__main__":
    main()
