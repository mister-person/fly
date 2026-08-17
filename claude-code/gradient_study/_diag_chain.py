"""The full causal chain from N3's missed targets to g(0->1) and g(1->2).

Walks 4n G seed4's resting point stage by stage and reports what each stage actually
contributes, so the dead stages are visible.  Each stage is toggled by a flag, so "does
this stage matter here" is measured by turning it off, not asserted.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

E = [[0, 1], [1, 2], [0, 3], [2, 3]]
C = np.array(E, np.int32)
N, TRUE = 4, [250., 500., 1200., 700.]
G.REQ_GAIN, G.TRUST = 0.3, 5.0
params = G.mkparams(520)
T = {n: G.sp(G.fsim(C, N, np.array(TRUE, np.float32), params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
w = np.array([289., 420., 1262., 538.])
V = G.fsim(C, N, w, params)
s = {p: G.sp(V, p) for p in range(N)}


def grads():
    eps, L, vsub, wr = G.traces(C, N, w, s, params.steps, {3: T[3]}, V)
    g = np.zeros(4)
    for n in range(N):
        for si in inc[n]:
            g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
    return g, L, eps, vsub


print(f"w {w.tolist()}   N1 {s[1]}  N2 {s[2]}  N3 {s[3]}")
print(f"target N3 {T[3]}   missing {[t for t in T[3] if all(abs(t-q)>5 for q in s[3])]}\n")

g, L, eps, vsub = grads()
print("STAGE-BY-STAGE, at this point")
print(f"  1. output hinge     L[3] nonzero at {np.nonzero(L[3])[0].tolist()}")
print(f"                      values {[round(float(L[3][t]),5) for t in np.nonzero(L[3])[0]]}")
print(f"  2. request R[3]     seeded from the same deficits (REQ_GAIN={G.REQ_GAIN})")
print(f"  3. propagate to N2  back_corr(unmet, HK), self-normalised, x REQ_GAIN")
print(f"  4. feasibility mask ok_n = epoch/refractory window at N3")
print(f"  5. relaxation       Ln = w*back_corr(L[3],HK) * near-th gate, + timing/slope term")
print(f"  6. LN_RELOC         rejected POSITIVE Ln carried to nearest feasible+reachable t")
print(f"  ->  L[2] nonzero at {np.nonzero(L[2])[0].tolist()}  values "
      f"{[round(float(L[2][t]),6) for t in np.nonzero(L[2])[0]]}")
print(f"  ->  L[1] nonzero at {np.nonzero(L[1])[0].tolist()[:8]}")
print(f"  ->  g = {np.array2string(g, precision=3)}\n")

print("WHICH STAGES ACTUALLY MATTER HERE (turn each off, watch g(1->2)):")
base = g[1]
flags = [("SHARP_GAIN", 0.0), ("OCCL_GAIN", 0.0), ("OCCL_MASK", 0), ("LN_RELOC", 0),
         ("MOVE_GAIN", 0.0), ("REQ_GAIN", 0.0), ("EPOCH_EXTEND", 0),
         ("SUPP_GAIN", 0.0), ("PIVOT_GAIN", 0.0), ("BLOCK_GAIN", 0.0), ("WREQ_GAIN", 0.0)]
for name, off in flags:
    old = getattr(G, name)
    setattr(G, name, off)
    g2, _, _, _ = grads()
    setattr(G, name, old)
    d = "no change" if abs(g2[1] - base) < 1e-12 else f"g(1->2) {base:+.3e} -> {g2[1]:+.3e}"
    print(f"   {name:14s} = {off!r:6}  {d}")
