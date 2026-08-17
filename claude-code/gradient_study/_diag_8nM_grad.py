"""8n M at 1040, stuck: the outputs fire EARLIER and MORE OFTEN than the target.
Why is the gradient not pushing away from that?

N6 found [148,248,...,948] -- 9 spikes on a period-100 rhythm -- against a target of 6
irregular ones starting at 190.  So: 3 extra spikes, and the first is 42 steps early.  Both
should be visible to the output seeding (suppression of unmatched spikes, plus the signed
timing demand at matched-but-late/early ones).  Check what the demand actually says.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G
from _diag import CASES, steps_for

name = "8n M"
E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
params = G.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}

w0 = (W * np.random.default_rng(0).uniform(0.5, 1.5, len(Wl))).astype(float)
live = {}
G.train(C, N, outs, w0.copy(), T, params, rounds=3200, lr=G.LR,
        cb=lambda it, w, *a: live.update(w=w.copy()))
w = live["w"]
V = G.fsim(C, N, w, params)
s = {n: G.sp(V, n) for n in range(N)}
eps, L, vsub, wreq = G.traces(C, N, w, s, params.steps, {o: T[o] for o in outs}, V)

print(f"MATCH_WIN={G.MATCH_WIN} HIT_TOL={G.HIT_TOL} DEAD_ZONE={G.DEAD_ZONE} "
      f"GRADE_SUPP={G.GRADE_SUPP} MOVE_GAIN={G.MOVE_GAIN} MOVE_COHERE={G.MOVE_COHERE}\n")

for o in outs:
    f, tg = s[o], T[o]
    print(f"=== N{o}: found {len(f)} {f}")
    print(f"          target {len(tg)} {tg}")
    # replicate the closest-pair matching the seeding uses
    pairs = sorted(((abs(a - b), a, b) for a in f for b in tg), key=lambda x: x[0])
    ut, uq, claim = set(), set(), {}
    for d, a, b in pairs:
        if a in ut or b in uq:
            continue
        claim[a] = b; ut.add(a); uq.add(b)
    unmatched_found = [a for a in f if a not in claim]
    print(f"    matched pairs (found<-target, offset): "
          + ", ".join(f"{a}<-{claim[a]}({a-claim[a]:+d})" for a in sorted(claim)))
    print(f"    UNMATCHED found spikes (should be suppressed): {unmatched_found}")
    print(f"    unmatched targets: {sorted(set(tg) - uq)}")
    nz = np.nonzero(L[o])[0]
    pos = [int(x) for x in nz if L[o][x] > 0]
    neg = [int(x) for x in nz if L[o][x] < 0]
    print(f"    L[{o}]: {len(pos)} POSITIVE at {pos[:12]}")
    print(f"            {len(neg)} NEGATIVE at {neg[:12]}")
    for a in unmatched_found:
        print(f"       demand at unmatched spike {a}: L={L[o][a]:+.3e}"
              f"   (suppression would be negative)")
    for a in sorted(claim):
        off = a - claim[a]
        if abs(off) > G.DEAD_ZONE:
            print(f"       demand at matched spike {a} (off {off:+d}): L={L[o][a]:+.3e}"
                  f"   (early=>needs NEGATIVE)")
    print()

g = np.zeros(len(w))
for n in range(N):
    for si in inc[n]:
        g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
print("gradient on the fan-in edges (into the outputs):")
for si in range(len(w)):
    a, b = int(C[si, 0]), int(C[si, 1])
    if b not in outs:
        continue
    need = Wl[si] - w[si]
    print(f"   w(N{a}->N{b}) {w[si]:7.1f}  true {Wl[si]:6.1f}  need {need:+7.1f}"
          f"   g={g[si]:+.2e}  {'CORRECT' if g[si]*need > 0 else ('zero' if g[si]==0 else 'WRONG')}")
