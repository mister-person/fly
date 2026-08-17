"""Are the "need a spike" and "bad spike" signals reconciled on a HIDDEN neuron?

At the OUTPUT they are: a one-to-one closest-pair matching (SHARED_MATCH/PAIR_MATCH)
decides which actual spike claims which target, so a LATE spike is one late spike rather
than "spurious + missing".

A hidden neuron has no target, so there is no such matching.  It receives two things:
  propagated  = the backward relaxation of downstream L (carries BOTH signs)
  R[n]        = the creation request (deficit-seeded, strictly POSITIVE)
which are combined as  L[n] = propagated + R[n] - SUPP_GAIN*S[n]   (SUPP_GAIN defaults 0).
Isolate them by differencing REQ_GAIN, and check whether they land at the same times --
i.e. whether anything notices that an EXISTING spike could BE the requested one.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

C = np.array([[0, 1], [0, 2], [1, 2]], np.int32)
N, OUTS = 3, [2]
params = G.mkparams(520)
T = {n: G.sp(G.fsim(C, N, np.array([200., 1200., 700.], np.float32), params), n) for n in range(N)}

SITU = {
    "seed4 N1 TWICE (151,351)": [274., 1199., 1019.],
    "seed2 N1 LATE  (437)":     [159., 1139., 738.],
    "seed2 START    (453)":     None,   # filled below
}
SITU["seed2 START    (453)"] = (np.array([200., 1200., 700.], np.float32) *
                                np.random.default_rng(2).uniform(0.5, 1.5, 3)).astype(float)

print(f"SUPP_GAIN={G.SUPP_GAIN}  (explicit upstream suppression term S[n])")
print(f"SHARED_MATCH={G.SHARED_MATCH} PAIR_MATCH={G.PAIR_MATCH}  (OUTPUT-side matching)\n")


def parts(w):
    w = np.array(w, float)
    V = G.fsim(C, N, w, params); s = {p: G.sp(V, p) for p in range(N)}
    out = {}
    for rg in (0.0, G.REQ_GAIN if G.REQ_GAIN > 0 else 3.0):
        G.REQ_GAIN = rg if rg > 0 else G.REQ_GAIN
        old = G.REQ_GAIN; G.REQ_GAIN = rg
        eps, L, vsub, wreq = G.traces(C, N, w, s, params.steps, {o: T[o] for o in OUTS}, V)
        out[rg] = (L[1].copy(), eps[(0, 1)].copy())
        G.REQ_GAIN = old
    return s, out


for lab, w in SITU.items():
    G.REQ_GAIN = 3.0
    s, out = parts(w)
    prop = out[0.0][0]                    # propagated only (both signs)
    both = out[3.0][0]
    req = both - prop                     # the creation request
    e01 = out[0.0][1]
    print(f"=== {lab}   w={np.round(np.array(w,float),0).tolist()} ===")
    print(f"    N1 actual spikes {s[1]}   (true {T[1]})")
    pz = np.nonzero(prop)[0]; rz = np.nonzero(req)[0]
    print(f"    propagated: {len(pz)} nonzero"
          + (f" over [{pz.min()}..{pz.max()}]" if len(pz) else "")
          + f"   at N1's spikes: {[(int(q), round(float(prop[q]),5)) for q in s[1]]}")
    print(f"    request   : {len(rz)} nonzero"
          + (f" over [{rz.min()}..{rz.max()}]" if len(rz) else "")
          + f"   peak at t={int(rz[np.argmax(req[rz])]) if len(rz) else '-'}"
          + f"   at N1's spikes: {[(int(q), round(float(req[q]),5)) for q in s[1]]}")
    for q in s[1]:
        d = min((abs(int(q) - int(t)) for t in rz), default=None)
        print(f"    -> spike {q}: nearest requested time is {d} steps away"
              f"   (SHARP_WIN={G.SHARP_WIN})")
    print(f"    g(0->1) propagated-only = {float(np.dot(prop, e01)):+.3e}"
          f"   with request = {float(np.dot(both, e01)):+.3e}")
    print()
