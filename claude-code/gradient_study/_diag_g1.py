"""Decompose the gradient on N1's only input edge (N0->N1) across 3n D's situations.

N1 has ONE incoming synapse, so g(0->1) = dot(L[1], eps[(0,1)]) is the entire story.
L[1] is built from the downstream creation request (deficit at N2's missing 293 spike,
propagated back through w(1->2)) times N1's local near-threshold sensitivity g_n(t), plus
any timing term.  We report, at a chosen weight point:
   - what N1 and N2 currently do
   - g on all three edges
   - L[1]: how many nonzero, sign split, where the mass sits
   - the request R[1] separately (REQ_GAIN sweep isolates it)
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

C = np.array([[0, 1], [0, 2], [1, 2]], np.int32)
N, OUTS, Wl = 3, [2], [200., 1200., 700.]
params = G.mkparams(520)
T = {n: G.sp(G.fsim(C, N, np.array(Wl, np.float32), params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}

# situations, taken from the per-seed dump
SITU = {
    "TRUE          (N1@246, out ok)":        [200., 1200., 700.],
    "seed6 recovered (N1@243)":              [204., 1191., 655.],
    "seed2 N1 LATE   (N1@438 -> 483)":       [159., 1139., 738.],
    "seed0 N1 WEAK   (w12 sub-crit, no 293)":[243., 940., 379.],
    "seed4 N1 TWICE  (N1@151,351)":          [274., 1199., 1019.],
}


def analyse(label, w):
    w = np.array(w, float)
    V = G.fsim(C, N, w, params); spall = {p: G.sp(V, p) for p in range(N)}
    eps, L, vsub, wreq = G.traces(C, N, w, spall, params.steps, {o: T[o] for o in OUTS}, V)
    g = np.zeros(len(w))
    for n in range(N):
        for si in inc[n]:
            g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
    L1 = L[1]; nz = np.nonzero(L1)[0]
    print(f"\n=== {label} ===")
    print(f"    w={w.tolist()}   N1={spall[1]}  N2={spall[2]}")
    print(f"    N2 target {T[2]}   (293 driven by N1)")
    print(f"    g(0->1)={g[0]:+.3e}   g(0->2)={g[1]:+.3e}   g(1->2)={g[2]:+.3e}")
    if len(nz):
        pos = int((L1[nz] > 0).sum()); neg = int((L1[nz] < 0).sum())
        cont = eps[(0, 1)]
        # where along L[1] is the g(0->1) mass coming from
        prod = L1 * cont
        pnz = np.nonzero(prod)[0]
        print(f"    L[1]: {len(nz)} nonzero over [{nz.min()}..{nz.max()}], "
              f"{pos} pos / {neg} neg, sum|L1|={np.abs(L1).sum():.3e}")
        top = pnz[np.argsort(-np.abs(prod[pnz]))[:6]]
        print("    g(0->1) contributions (t: L1*eps): " +
              ", ".join(f"{t}:{prod[t]:+.2e}" for t in sorted(top)))
    else:
        print(f"    L[1]: EMPTY -> g(0->1)={g[0]:+.3e} (no demand on N1)")
    return g


print(f"SHARP_GAIN={G.SHARP_GAIN} SHARP_FLIP={G.SHARP_FLIP} REQ_GAIN={G.REQ_GAIN}  "
      f"critical w = {G.TH/float(G.HK.max()):.1f}")
for lab, w in SITU.items():
    analyse(lab, w)
