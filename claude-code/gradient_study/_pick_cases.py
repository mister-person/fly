"""Choose the three variants and check they are WELL-POSED.

Non-degeneracy check (as grad_coincidence_minimal does): with the hidden path severed,
scan the DIRECT input->output weight over its whole range.  If some value reproduces the
target train, the hidden weights are unidentifiable and the case tests nothing.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

params = G.mkparams(520)


def sim(E, N, w):
    V = G.fsim(np.array(E, np.int32), N, np.array(w, np.float32), params)
    return {n: G.sp(V, n) for n in range(N)}


CAND = {
    # name: (edges, N, out, weights, index of the hidden->output edge, index of direct edge)
    "3n D  (baseline, N1 x1)": ([[0, 1], [0, 2], [1, 2]], 3, 2, [200., 1200., 700.], 2, 1),
    "3n E  (N1 x2)":           ([[0, 1], [0, 2], [1, 2]], 3, 2, [260., 1200., 950.], 2, 1),
    "4n F  (split chain, x1)": ([[0, 1], [1, 2], [0, 3], [2, 3]], 4, 3,
                                [243., 1200., 1200., 700.], 3, 2),
    "4n G  (split chain, x2)": ([[0, 1], [1, 2], [0, 3], [2, 3]], 4, 3,
                                [250., 500., 1200., 700.], 3, 2),
}

for name, (E, N, out, w, hid_e, dir_e) in CAND.items():
    s = sim(E, N, w)
    w2 = list(w); w2[hid_e] = 0.0
    b = sim(E, N, w2)
    print(f"=== {name} ===")
    print(f"    edges {E}   w {[int(v) for v in w]}")
    for n in range(N):
        tag = "  <- OUT" if n == out else ("  (input)" if n == 0 else "  hidden")
        print(f"    N{n}: {s[n]}{tag}")
    print(f"    hidden path severed -> {b[out]}   ({len(s[out])-len(b[out])} spikes created)")

    # non-degeneracy: can the DIRECT edge alone reproduce the target?
    hits = []
    for x in range(20, 3001, 10):
        w3 = list(w); w3[hid_e] = 0.0; w3[dir_e] = float(x)
        if sim(E, N, w3)[out] == s[out]:
            hits.append(x)
    print(f"    direct-edge-only reproductions: "
          f"{hits if hits else 'NONE -- hidden path genuinely required'}")
    # and can it be reproduced with the hidden path present but any direct weight?
    print()
