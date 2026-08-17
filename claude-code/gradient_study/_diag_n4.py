"""Is something pushing BACK against N4 reviving, or is there simply no push?

N3 came back (one spike) and N4 did not.  Track both input weights and their gradients over
the whole run: if the gradient on w(0->4) is consistently positive and the weight still does
not rise, something is opposing it; if the gradient flips sign, it is being pushed back; if
the weight rises and falls, N4 revives and is then killed again.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G
from _diag import CASES, steps_for

E, N, outs, Wl = CASES["8n M"]
C = np.array(E, np.int32)
params = G.mkparams(steps_for("8n M"))
W = np.array(Wl, np.float32)
T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
w0 = (W * np.random.default_rng(0).uniform(0.5, 1.5, len(Wl))).astype(float)

hist = []


def cb(it, w, upd, g, spall, vsub, L):
    hist.append((it, w[2], w[3], g[2], g[3], upd[2], upd[3],
                 len(spall[3]), len(spall[4])))


G.train(C, N, outs, w0.copy(), T, params, rounds=3200, lr=G.LR, cb=cb)
H = np.array([(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]) for h in hist])

print(f"true w(0->3)={Wl[2]:.0f}  w(0->4)={Wl[3]:.0f}   start "
      f"{w0[2]:.0f} / {w0[3]:.0f}\n")
print(f"{'it':>6} {'w(0->3)':>8} {'g(0->3)':>11} {'w(0->4)':>8} {'g(0->4)':>11} "
      f"{'upd4':>9}  N3 N4")
for i in range(0, len(H), 200):
    r = H[i]
    print(f"{int(r[0]):>6} {r[1]:>8.1f} {r[3]:>11.2e} {r[2]:>8.1f} {r[4]:>11.2e} "
          f"{r[5+1]:>9.4f}  {int(r[7]):>2} {int(r[8]):>2}")

g4 = H[:, 4]
w4 = H[:, 2]
print(f"\nw(0->4) over the run: start {w4[0]:.1f}  max {w4.max():.1f}"
      f"  min {w4.min():.1f}  end {w4[-1]:.1f}   (true {Wl[3]:.0f})")
print(f"g(0->4): positive on {100*(g4 > 0).mean():.1f}% of iterations, "
      f"negative on {100*(g4 < 0).mean():.1f}%, zero on {100*(g4 == 0).mean():.1f}%")
print(f"   mean g when nonzero: {g4[g4 != 0].mean():+.3e}" if (g4 != 0).any() else "   always zero")
print(f"N4 alive on {100*(H[:, 8] > 0).mean():.1f}% of iterations; "
      f"N3 alive on {100*(H[:, 7] > 0).mean():.1f}%")
ever = np.nonzero(H[:, 8] > 0)[0]
if len(ever):
    print(f"   N4 first fires at it{int(H[ever[0],0])} (w={H[ever[0],2]:.1f}), "
          f"last at it{int(H[ever[-1],0])}")
