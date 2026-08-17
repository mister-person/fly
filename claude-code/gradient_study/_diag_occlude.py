"""OCCLUSION: an edge is exactly inert when its presynaptic arrival lands inside the
postsynaptic neuron's own refractory shadow.  That is observable without resimulating.

Check: (a) does the TRUE solution clear the shadow, (b) how far must seed0 move to clear
it, (c) is the escape reachable by any single weight, or is it irreducibly joint?
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

C = np.array([[0, 1], [0, 2], [1, 2]], np.int32)
N = 3
params = G.mkparams(520)
D, RF = G.DELAY_ITERS, G.REFRAC_ITERS
TRUE = [200., 1200., 700.]
T = {n: G.sp(G.fsim(C, N, np.array(TRUE, np.float32), params), n) for n in range(N)}
BASE = [243., 940., 379.]
print(f"DELAY_ITERS={D}  REFRAC_ITERS={RF}   (arrival = spike + {D}; shadow = [s, s+{RF}])\n")


def occl(w, label):
    w = np.array(w, float)
    V = G.fsim(C, N, w, params); s = {p: G.sp(V, p) for p in range(N)}
    print(f"  {label}: w={w.tolist()}")
    print(f"      N1={s[1]}  N2={s[2]}")
    for q in s[1]:
        arr = q + D
        hit = [x for x in s[2] if x <= arr <= x + RF]
        tag = (f"OCCLUDED by N2@{hit[0]} (shadow [{hit[0]},{hit[0]+RF}])" if hit
               else "clear -> contributes")
        print(f"      N1@{q} -> arrives {arr}: {tag}")
    return s


occl(TRUE, "TRUE     ")
occl(BASE, "seed0 end")

print("\n=== can seed0 escape by moving ONE weight? (w12 is inert, so try w01 / w02) ===")
for e, nm in ((0, "w(0->1)"), (1, "w(0->2)")):
    esc = []
    for x in range(20, 3001, 10):
        w = np.array(BASE, float); w[e] = x
        V = G.fsim(C, N, w, params); s1 = G.sp(V, 1); s2 = G.sp(V, 2)
        if any(not any(y <= q + D <= y + RF for y in s2) for q in s1):
            esc.append(x)
    if esc:
        runs = []
        st = esc[0]; pv = esc[0]
        for x in esc[1:]:
            if x - pv > 10:
                runs.append((st, pv)); st = x
            pv = x
        runs.append((st, pv))
        print(f"   {nm}: clears the shadow for {runs}  (base {BASE[e]:.0f})")
    else:
        print(f"   {nm}: NEVER clears the shadow over 20..3000")

print("\n=== joint (w01, w02) grid: which combinations hit the target exactly? ===")
hits = []
for a in range(150, 401, 5):
    for b in range(800, 1601, 20):
        w = np.array([float(a), float(b), 379.])
        if G.sp(G.fsim(C, N, w, params), 2) == T[2]:
            hits.append((a, b))
print(f"   with w(1->2) stuck at seed0's 379: {len(hits)} exact hits {hits[:10]}")
hits2 = []
for a in range(150, 401, 5):
    for c in range(400, 1201, 25):
        w = np.array([float(a), 940., float(c)])
        if G.sp(G.fsim(C, N, w, params), 2) == T[2]:
            hits2.append((a, c))
print(f"   with w(0->2) stuck at seed0's 940: {len(hits2)} exact hits {hits2[:10]}")
