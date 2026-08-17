"""Where does a balanced demand turn into a suppression-dominated gradient?

    python3 _eps_support.py

g[si] = dot(L[n], eps[(k,n)]).  So a demand at time t only reaches a weight if that edge has
ELIGIBILITY at t.  eligibility() truncates each presynaptic PSP at the postsynaptic neuron's
next reset, which makes the two demand signs structurally unequal:

  - a SUPPRESS request sits ON a spike the neuron really produced, so by construction there was
    input arriving and the epoch containing it is intact -> eps > 0, always.
  - a CREATE request sits where the neuron did NOT fire.  Nothing guarantees any presynaptic
    arrival there, and if a wrongly-placed spike lies between, the PSP is cut before reaching
    it -> eps can be exactly 0, and the request multiplies out of existence.

This measures one build() at the initial weights: total demand mass by sign, and how much of it
survives the eps product.  The 50n nets are compared against small cases where the field works.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import field_trace as F
from _diag import CASES, steps_for

print(f"{'case':<8} {'seed':>4} | {'L+':>9} {'L-':>9} {'L+/L-':>6} | "
      f"{'surv+':>7} {'surv-':>7} | {'g+/g-':>6} | {'dead+':>6}")
print("-" * 82)
for name in ("50n A", "50n B", "50n C", "4n F", "3n D", "3n J"):
    for seed in (0, 1):
        E, N, outs, Wl = CASES[name]
        C = np.array(E, np.int32)
        p = F.mkparams(steps_for(name))
        W = np.array(Wl, np.float32)
        T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
        w = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
        V = F.fsim(C, N, np.asarray(w, np.float32), p)
        spall = {n: F.sp(V, n) for n in range(N)}
        Ff, L, Lc, ep, PR, Fc = F.build(C, N, w, spall, p.steps, {o: list(T[o]) for o in outs})

        inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
        Lp = Ln = sp_ = sn_ = 0.0
        dead_pos = dead_all = 0.0
        oL = {True: [0.0, 0.0], False: [0.0, 0.0]}     # is-output -> [pos mass, neg mass]
        for n in range(N):
            v = L[n]
            if not len(inc[n]):
                continue
            # total eligibility available at each time, summed over the edges into n
            tot = sum(ep[(int(C[si, 0]), n)] for si in inc[n])
            pos, neg = v > 0, v < 0
            Lp += float(v[pos].sum()); Ln += float(-v[neg].sum())
            sp_ += float(np.dot(v[pos], tot[pos]))
            sn_ += float(-np.dot(v[neg], tot[neg]))
            # demand mass that multiplies (near-)zero eligibility -- silently discarded
            thr = 1e-3 * (tot.max() if tot.max() > 0 else 1.0)
            dead_pos += float(v[pos & (tot < thr)].sum())
            dead_all += float(v[pos].sum())
            sl = oL[n in outs]
            sl[0] += float(v[pos].sum()); sl[1] += float(-v[neg].sum())
        oo, hh = oL[True], oL[False]
        print(f"{name:<8} {seed:>4} | {Lp:>9.2f} {Ln:>9.2f} {Lp/max(Ln,1e-9):>6.2f} | "
              f"{sp_:>7.2e} {sn_:>7.2e} | {sp_/max(sn_,1e-30):>6.2f} | "
              f"{100*dead_pos/max(dead_all,1e-9):>5.1f}% | "
              f"out {oo[0]/max(oo[1],1e-9):>6.2f}  hid {hh[0]/max(hh[1],1e-9):>6.2f}  "
              f"(hid share of L- {100*hh[1]/max(hh[1]+oo[1],1e-9):>4.0f}%)")
