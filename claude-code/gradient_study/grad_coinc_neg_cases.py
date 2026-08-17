"""Two new case families: 4-WAY COINCIDENCE and NEGATIVE weights.

  8n K  COINCIDENCE.  N0 fans out to N1..N4 at four different weights (so the hidden spikes
        are spread in time), and N1..N4 each feed all three outputs N5,N6,N7.  Every fan-in
        weight is ~0.27x threshold, so NO single connection can fire anything and roughly
        four coincident arrivals are needed.  Three outputs with 16 edges keeps it from
        being badly underdetermined.  The fan-in weights differ per output so the three
        output trains are distinct -- with identical weights all three outputs are the same
        train, which wastes two thirds of the supervision.

  3n L  NEGATIVE.  One inhibitory edge.  Expected to break things: every deficit, hinge and
        creation-request argument in the method assumes more weight => more drive => earlier
        spike, and that reverses for w < 0.  The suppression path assumes the mirror.
        w(1->2) = -700 turns N2's regular period-100 train into [33, 237, 437].
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

E_K = [[0, 1], [0, 2], [0, 3], [0, 4]] + [[h, o] for o in (5, 6, 7) for h in (1, 2, 3, 4)]
W_K = [900., 700., 550., 460.,
       120., 120., 120., 120.,          # -> N5
       150., 130., 110., 140.,          # -> N6
       100., 160., 140., 120.]          # -> N7

CASES = {
    "8n K": (E_K, 8, [5, 6, 7], W_K),
    "3n L": ([[0, 1], [0, 2], [1, 2]], 3, [2], [700., 1200., -700.]),
}


def main():
    params = G.mkparams(520)
    print(f"W_CRIT {G.W_CRIT:.1f}   th {G.TH:.3e}\n")
    for name, (E, N, outs, Wl) in CASES.items():
        C = np.array(E, np.int32)
        T = {n: G.sp(G.fsim(C, N, np.array(Wl, np.float32), params), n) for n in range(N)}
        print(f"=== {name}   {len(E)} edges, {len(outs)} output(s) ===")
        for n in range(N):
            tag = " <- OUT" if n in outs else (" (input)" if n == 0 else " hidden")
            print(f"    N{n}: {T[n]}{tag}")
        strongest = max(abs(x) for x in Wl)
        print(f"    strongest |w| = {strongest:.0f} vs W_CRIT {G.W_CRIT:.1f}"
              f" -> {'NO single edge can fire a neuron' if strongest < G.W_CRIT else 'some edge is supra-critical'}")
        if len(outs) > 1:
            same = all(T[outs[0]] == T[o] for o in outs[1:])
            print(f"    outputs distinct: {'NO -- all identical' if same else 'yes'}")
        # every incoming edge of every output must matter
        dead = []
        for si in range(len(Wl)):
            if int(C[si, 1]) not in outs:
                continue
            w2 = list(Wl); w2[si] = 0.0
            for o in outs:
                if G.sp(G.fsim(C, N, np.array(w2, np.float32), params), o) != T[o]:
                    break
            else:
                dead.append(f"w({C[si,0]}->{C[si,1]})")
        print(f"    redundant output edges: {dead if dead else 'none'}")
        print()


if __name__ == "__main__":
    main()
