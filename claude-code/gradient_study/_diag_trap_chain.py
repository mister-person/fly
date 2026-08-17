"""Why doesn't N3's timing demand travel back to N2 and N1?

N3 fires at 264,464; targets 214,314,414,514.  264 is EXACTLY 50 from both 214 and 314, so
the closest-pair matching is a tie -- and which side it picks decides whether the demand
says "fire 50 earlier" (correct: pulls N2 and N1 earlier) or "fire 50 later" (wrong).
Trace the demand hop by hop.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

E = [[0, 1], [1, 2], [2, 3]]
C = np.array(E, np.int32)
N, OUTS = 4, [3]
TRUE = [500., 500., 500.]
STUCK = np.array([456., 415., 626.])
params = G.mkparams(520)
T = {n: G.sp(G.fsim(C, N, np.array(TRUE, np.float32), params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}

V = G.fsim(C, N, STUCK, params)
s = {p: G.sp(V, p) for p in range(N)}
eps, L, vsub, wreq = G.traces(C, N, STUCK, s, params.steps, {3: T[3]}, V)

print(f"stuck w {STUCK.tolist()}")
print(f"   N1 {s[1]}   (true {T[1]})")
print(f"   N2 {s[2]}   (true {T[2]})")
print(f"   N3 {s[3]}   target {T[3]}")
print(f"   MATCH_WIN={G.MATCH_WIN}  HIT_TOL={G.HIT_TOL}  DEAD_ZONE={G.DEAD_ZONE}\n")

print("closest-pair matching, as the code computes it:")
pairs = sorted(((abs(t - q), t, q) for t in s[3] for q in T[3]), key=lambda x: x[0])
ut, uq = set(), set()
for d, t, q in pairs:
    if t in ut or q in uq:
        continue
    ut.add(t); uq.add(q)
    print(f"   N3 spike {t} <- target {q}   (distance {d}, "
          f"{'must fire EARLIER' if t > q else 'must fire LATER' if t < q else 'exact'})")
print(f"   unmatched targets: {sorted(set(T[3]) - uq)}")
print(f"   ties at distance 50: 264 is 50 from BOTH 214 and 314; "
      f"464 is 50 from BOTH 414 and 514\n")

for n in (3, 2, 1):
    nz = np.nonzero(L[n])[0]
    print(f"L[{n}]: {len(nz)} nonzero  max|L|={np.abs(L[n]).max():.3e}")
    for t in nz[:10]:
        at = "  <- at one of its OWN spikes" if t in s[n] else ""
        print(f"     t={t:4d}  L={L[n][t]:+.4e}{at}")
    if len(nz) == 0:
        print("     (empty -- nothing reaches this neuron)")

g = np.zeros(3)
for n in range(N):
    for si in inc[n]:
        g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
print(f"\ng = {np.array2string(g, precision=3)}")
print(f"   w0 {STUCK[0]:.0f} -> true 500 needs {'UP' if TRUE[0] > STUCK[0] else 'DOWN'};"
      f"  gradient says {'UP' if g[0] > 0 else 'DOWN' if g[0] < 0 else 'NOTHING'}")
print(f"   w1 {STUCK[1]:.0f} -> true 500 needs {'UP' if TRUE[1] > STUCK[1] else 'DOWN'};"
      f"  gradient says {'UP' if g[1] > 0 else 'DOWN' if g[1] < 0 else 'NOTHING'}")
print(f"   w2 {STUCK[2]:.0f} -> true 500 needs {'UP' if TRUE[2] > STUCK[2] else 'DOWN'};"
      f"  gradient says {'UP' if g[2] > 0 else 'DOWN' if g[2] < 0 else 'NOTHING'}")
