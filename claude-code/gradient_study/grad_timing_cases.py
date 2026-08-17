"""Three cases matching the 50-NEURON failure signature: spike COUNT easy, TIMING hard.

At 50 neurons the method recovers the correct spike COUNT on 17 of 18 outputs and gets the
TIMES wrong by 6-44 steps, and it plateaus there (800 rounds is bit-identical to 300).  The
existing small suite does not contain a case with that shape -- its failures are count
failures (a relay dies, an accumulator fires once instead of twice).  These three isolate
timing instead:

  5n H  DEEP CHAIN    N0->N1->N2->N3->N4, four distinct weights.  Timing has to survive
                      three hidden hops, so per-hop error compounds.
  4n I  CONVERGENT    N0->N1->N3 and N0->N2->N3.  Each path ALONE produces only 2 output
                      spikes; together they produce 5.  The output exists only where the
                      two arrivals coincide, so it is pure timing alignment.
  3n J  TIGHT ISI     output ISIs alternate 47/53 against a refractory period of 22, so
                      there is little slack for a mistimed spike.

Each is checked for the same non-degeneracy the 3n D variants were: severing a hidden path
must change the output, and no single weight may reproduce the target with a path severed.
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
    "5n H": ([[0, 1], [1, 2], [2, 3], [3, 4]], 5, [4], [500., 600., 700., 800.]),
    # 4n I (convergent) DROPPED AS DEGENERATE.  Both [500,500,400,400] (symmetric paths,
    # N1 and N2 firing identically) and [900,500,300,500] (asymmetric) can be reproduced
    # with one path SEVERED by retuning a single remaining weight -- 800 and 1140
    # respectively.  The cause is general: the output is a REGULAR period-100 train, and a
    # regular train is reproducible by many single paths at the right phase.  A convergent
    # case is only well posed if the two paths jointly produce something IRREGULAR that
    # neither can alone, i.e. the coincidence structure of grad_coincidence_minimal.
    "3n J": ([[0, 1], [0, 2], [1, 2]],         3, [2], [700., 900., 900.]),
}


def main():
    params = G.mkparams(520)
    print(f"critical single-spike weight {G.W_CRIT:.1f}\n")
    for name, (E, N, outs, Wl) in CASES.items():
        C = np.array(E, np.int32)
        T = {n: G.sp(G.fsim(C, N, np.array(Wl, np.float32), params), n) for n in range(N)}
        o = outs[0]
        print(f"=== {name}   edges {E}   w {[int(v) for v in Wl]} ===")
        for n in range(N):
            tag = " <- OUT" if n == o else (" (input)" if n == 0 else " hidden")
            print(f"    N{n}: {T[n]}{tag}")
        if len(T[o]) > 1:
            print(f"    output ISIs {np.diff(T[o]).tolist()}   (refractory {G.REFRAC_ITERS})")
        # severing each incoming edge of the output must change the train
        for si in np.where(C[:, 1] == o)[0]:
            w2 = list(Wl); w2[si] = 0.0
            b = G.sp(G.fsim(C, N, np.array(w2, np.float32), params), o)
            print(f"    without w({C[si,0]}->{C[si,1]}): {b}"
                  f"   {'CHANGES' if b != T[o] else 'NO CHANGE -- edge is redundant!'}")
        # can any single weight reproduce the target with one path severed?
        deg = []
        for si in np.where(C[:, 1] == o)[0]:
            for oth in range(len(Wl)):
                if oth == si:
                    continue
                for x in range(20, 3001, 20):
                    w3 = list(Wl); w3[si] = 0.0; w3[oth] = float(x)
                    if G.sp(G.fsim(C, N, np.array(w3, np.float32), params), o) == T[o]:
                        deg.append((si, oth, x))
                        break
        print(f"    degenerate reproductions with a path severed: "
              f"{deg if deg else 'NONE -- well posed'}")
        print()


if __name__ == "__main__":
    main()
