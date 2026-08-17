"""3n D variants: a RARE sub-critical accumulator whose spikes each mark the output.

3n D's whole content is that w(0->1)=200 is BELOW the critical single-spike weight 444.5,
so N1 cannot fire from one input spike -- it accumulates and fires ONCE, and that single
spike leaves exactly one extra spike on the output.  Everything the method finds hard about
it (the request plateau, the fire-once assumption in SHARP_FLIP, occlusion) comes from that
one rare spike carrying the entire signature of two weights.

These three vary that structure along the two axes that matter:

  E  the accumulator fires TWICE, so the fire-once reading is simply false.  This is the
     shape SHARP_FLIP mis-signs: with one requested time tau, the SECOND correct spike is
     "later than tau" and gets pushed up until the neuron goes supercritical.
  F  the rare spike is RELAYED through a second hidden neuron before reaching the output,
     so the demand has to survive two backward hops and two occlusion windows.
  G  both: two rare spikes, each relayed.

CONSTRUCTION.  Each case is checked to be non-vacuous:
  * the accumulator fires the intended number of times (not once per input cycle),
  * every hidden spike CREATES an output spike rather than shifting an existing one --
    verified by severing the hidden edge and comparing trains, since a hidden spike whose
    arrival lands in the output's refractory shadow merely drags a spike earlier and tests
    nothing (that is what killed the first several candidates),
  * the target is NOT reproducible with the hidden path silent for ANY value of the direct
    input->output weight, scanned 20..3000 -- so the hidden weights are identifiable and
    the case is not a weight degeneracy.
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

# name: (edges, N, outs, true weights, hidden->output edge, direct input->output edge)
CASES = {
    "3n D": ([[0, 1], [0, 2], [1, 2]], 3, [2], [200., 1200., 700.], 2, 1),
    "3n E": ([[0, 1], [0, 2], [1, 2]], 3, [2], [260., 1200., 950.], 2, 1),
    "4n F": ([[0, 1], [1, 2], [0, 3], [2, 3]], 4, [3], [240., 1200., 1200., 1100.], 3, 2),
    "4n G": ([[0, 1], [1, 2], [0, 3], [2, 3]], 4, [3], [250., 500., 1200., 700.], 3, 2),
}


def main():
    params = G.mkparams(520)
    print(f"critical single-spike weight = {G.TH / float(G.HK.max()):.1f}"
          f"   (accumulators sit BELOW it)\n")
    for name, (E, N, outs, Wl, hid_e, dir_e) in CASES.items():
        C = np.array(E, np.int32)
        T = {n: G.sp(G.fsim(C, N, np.array(Wl, np.float32), params), n) for n in range(N)}
        o = outs[0]
        w2 = list(Wl); w2[hid_e] = 0.0
        base = G.sp(G.fsim(C, N, np.array(w2, np.float32), params), o)
        print(f"=== {name}   edges {E}   w {[int(v) for v in Wl]} ===")
        for n in range(N):
            tag = " <- OUT" if n == o else (" (input)" if n == 0 else " hidden")
            print(f"    N{n}: {T[n]}{tag}")
        created = [t for t in T[o] if t not in base]
        print(f"    hidden path severed -> {base}")
        print(f"    spikes CREATED by the hidden path: {created}")
        hits = [x for x in range(20, 3001, 10)
                if G.sp(G.fsim(C, N, np.array(
                    [0.0 if i == hid_e else (float(x) if i == dir_e else v)
                     for i, v in enumerate(Wl)], np.float32), params), o) == T[o]]
        print(f"    direct-edge-only reproductions: "
              f"{hits if hits else 'NONE -- hidden path required'}")
        print()


if __name__ == "__main__":
    main()
