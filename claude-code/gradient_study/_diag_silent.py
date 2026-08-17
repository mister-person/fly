"""Why is g(1->2) exactly 0 when N2 is silent, given that eps[(1,2)] depends only on N1?

eligibility() is built from the PRESYNAPTIC spike train, so a silent postsynaptic neuron
should NOT zero it -- with no resets the epoch is unbounded and eps is just the raw PSP.
And L[2] was measured NONZERO (max 1.14e-01).  So the dot product must be vanishing because
the two are nonzero in DISJOINT time ranges.  Locate that.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

E = [[0, 1], [1, 2], [0, 3], [2, 3]]
C = np.array(E, np.int32)
N, TRUE = 4, [250., 500., 1200., 700.]
params = G.mkparams(520)
T = {n: G.sp(G.fsim(C, N, np.array(TRUE, np.float32), params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}

for lab, w in [("seed5 LIVE (relay dead)", [187., 440., 1232., 532.]),
               ("seed7 LIVE", [185., 496., 1244., 495.])]:
    w = np.array(w)
    V = G.fsim(C, N, w, params); s = {p: G.sp(V, p) for p in range(N)}
    eps, L, vsub, wreq = G.traces(C, N, w, s, params.steps, {3: T[3]}, V)
    print(f"=== {lab}  w={w.tolist()} ===")
    print(f"    N1={s[1]}  N2={s[2]}  N3={s[3]}   target N3={T[3]}")
    e12 = eps[(1, 2)]
    nzE = np.nonzero(e12)[0]
    nzL = np.nonzero(L[2])[0]
    print(f"    eps[(1,2)] nonzero on [{nzE.min() if len(nzE) else '-'}"
          f"..{nzE.max() if len(nzE) else '-'}]  ({len(nzE)} pts, max {e12.max():.3e})")
    print(f"    L[2]       nonzero at {nzL.tolist()[:10]}  (max|L| {np.abs(L[2]).max():.3e})")
    ov = sorted(set(nzE.tolist()) & set(nzL.tolist()))
    print(f"    OVERLAP: {ov if ov else 'NONE -> dot product is exactly 0'}")
    print(f"    g(1->2) = {float(np.dot(L[2], e12)):.3e}")
    # where WOULD a request have to sit to be actionable?
    if len(nzE):
        print(f"    N1's spikes {s[1]} -> eps[(1,2)] can only act from t={nzE.min()} onward;")
        print(f"    the demand sits at {nzL.tolist()[:5]}, i.e. "
              f"{'BEFORE' if len(nzL) and nzL.min() < nzE.min() else 'after'} anything N1 can supply.")
    print()
