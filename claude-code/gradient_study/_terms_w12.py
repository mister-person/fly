"""Every term contributing to g[w12] on chain's stuck seed, separated by spike where relevant.

    python3 _terms_w12.py [seed]

g[si] = sum_t L[n][t] * eps[(k,n)][t], so every contribution is a (time, demand, eligibility)
triple and the total is exactly their sum.  The demand on N2 is built from two sources under
the PUSH config:

    DENSITY   L = the field itself, at every sample
    PUSH      per spike: positive at its nearest field peak, negative at the spike if early

The push part is isolated by differencing the demand against a PUSH=0 run rather than
re-deriving it, so what is attributed is exactly what the code did.  Density is then broken
down by EPOCH -- the interval between two of N2's own spikes -- because eligibility resets
there, so each epoch is an independent additive block.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import importlib

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 4
BASE = dict(F_DENSITY="1", F_CMASK="1", F_GRADED_ERR="1", F_KNORM="1", F_CREATE_FLOOR="0")


def state(push):
    os.environ.update(BASE); os.environ["F_PUSH"] = str(push)
    import field_trace as F
    importlib.reload(F)
    from _diag import CASES, steps_for
    E, N, outs, Wl = CASES["chain"]
    C = np.array(E, np.int32); p = F.mkparams(steps_for("chain"))
    W = np.array(Wl, np.float32)
    T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
    w0 = (W * np.random.default_rng(SEED).uniform(0.5, 1.5, len(Wl))).astype(float)
    w = np.asarray(F.train(C, N, outs, w0.copy(), T, p, rounds=800, lr=F.LR), float)
    sp = {n: F.sp(F.fsim(C, N, np.asarray(w, np.float32), p), n) for n in range(N)}
    g, Fl, L, ep = F.gradient(C, N, w, sp, p.steps, {o: T[o] for o in outs})
    return F, C, N, outs, T, w, sp, g, Fl, L, ep, p


F, C, N, outs, T, w, sp, g2, Fl, L2, ep, p = state(2.0)
_, _, _, _, _, w0_, sp0, g0, Fl0, L0, ep0, _ = state(0.0)

e = ep[(1, 2)]
Lfull = np.asarray(L2[2], float)
# the PUSH=0 demand recomputed AT THE SAME WEIGHTS, so the difference is the push term alone
os.environ.update(BASE); os.environ["F_PUSH"] = "0"
import field_trace as FF
importlib.reload(FF)
spx = {n: FF.sp(FF.fsim(C, N, np.asarray(w, np.float32), p), n) for n in range(N)}
gx, Flx, Lx, epx = FF.gradient(C, N, w, spx, p.steps, {o: T[o] for o in outs})
Ldens = np.asarray(Lx[2], float)
Lpush = Lfull - Ldens

print(f"chain seed {SEED}, PUSH config.  weights {F.wstr(C, w)}  (true 500/500/500)")
print(f"N2 fires {list(sp[2])}   true {list(T[2])}")
v = np.asarray(Fl[2], float)
pk = np.nonzero((v[1:-1] > 0) & (v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:]))[0] + 1
print(f"N2 field peaks {list(pk)}   eps(1->2) support {int((e>0).sum())}/{p.steps} samples")
print()
print(f"  g[w12] total = {g2[1]:+.4e}      (= sum over t of L[2][t] * eps[t])")
print(f"    from DENSITY : {float(np.dot(Ldens, e)):+.4e}")
print(f"    from PUSH    : {float(np.dot(Lpush, e)):+.4e}")
print()

print("DENSITY, broken down by EPOCH (eligibility resets at each of N2's spikes):")
bounds = [0] + [int(s) for s in sp[2]] + [p.steps]
print(f"  {'epoch':>13} {'demand +':>10} {'demand -':>10} {'eps mass':>9} {'contribution':>13}")
for a, b in zip(bounds[:-1], bounds[1:]):
    seg = slice(a, b)
    d = Ldens[seg]; ee = e[seg]
    print(f"  {f'{a:>4}..{b:<4}':>13} {float(d[d>0].sum()):>10.2e} {float(-d[d<0].sum()):>10.2e} "
          f"{float(ee.sum()):>9.2e} {float(np.dot(d, ee)):>+13.3e}")
print()

print("PUSH, per spike of N2 (nearest peak, whether it is early, and what it contributes):")
print(f"  {'spike':>6} {'true':>6} {'peak':>6} {'verdict':>8} {'at spike':>12} {'at peak':>12} "
      f"{'contribution':>13}")
tot = 0.0
for i, f_ in enumerate(sp[2]):
    q = int(pk[int(np.argmin(np.abs(pk - f_)))]) if len(pk) else -1
    tv = T[2][i] if i < len(T[2]) else None
    at_s = float(Lpush[f_] * e[f_])
    at_p = float(Lpush[q] * e[q]) if q >= 0 else 0.0
    tot += at_s + at_p
    print(f"  {f_:>6} {str(tv):>6} {q:>6} {'EARLY' if q > f_ else 'late':>8} "
          f"{at_s:>+12.3e} {at_p:>+12.3e} {at_s + at_p:>+13.3e}")
print(f"  {'':>6} {'':>6} {'':>6} {'TOTAL':>8} {'':>12} {'':>12} {tot:>+13.3e}")
print()
print("SIGN CHECK: lowering w12 improves the output, so a NEGATIVE g[w12] is correct.")
