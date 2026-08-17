"""Sweep: where does the code's eligibility DISAGREE with the true derivative?

eps is the direct term only, with reset/epoch boundaries treated as constants.  If the
omitted reset-pathway term  sum_s dV(t)/ds * ds/dw  is ever material, eps must differ from
a central finite difference of the real simulated voltage.  Sample widely and report the
worst disagreements.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from _diag import CASES
import grad_trace as G

rng = np.random.default_rng(0)
rows = []
for name, (E, Nn, outs, Wl) in CASES.items():
    C = np.array(E, np.int32)
    params = G.mkparams(520)
    T = {n: G.sp(G.fsim(C, Nn, np.array(Wl, np.float32), params), n) for n in range(Nn)}
    for trial in range(6):
        w = (np.array(Wl, float) * rng.uniform(0.5, 1.5, len(Wl)))
        V = G.fsim(C, Nn, w, params); s = {p: G.sp(V, p) for p in range(Nn)}
        eps, L, vsub, wreq = G.traces(C, Nn, w, s, params.steps, {o: T[o] for o in outs}, V)
        for si in range(len(w)):
            pre, post = int(C[si, 0]), int(C[si, 1])
            h = max(1.0, abs(w[si]) * 1e-3)
            wp = w.copy(); wp[si] += h
            wm = w.copy(); wm[si] -= h
            Vp = G.fsim(C, Nn, wp, params); Vm = G.fsim(C, Nn, wm, params)
            # A central difference that STRADDLES a spike-train change measures a jump, not
            # a derivative.  Flag those separately: there eps is arguably the correct local
            # slope and the discontinuity is the real obstacle.
            moved = any(G.sp(Vp, k) != G.sp(Vm, k) for k in range(Nn))
            # Probe at this neuron's TARGET times (outputs).  For HIDDEN neurons probing at
            # its OWN spikes is useless -- the near-spike filter removes every one of them,
            # which silently left the comparison with ZERO hidden probes.  Sample a grid
            # BETWEEN its spikes instead, which is where eps is actually consulted.
            probes = T[post] if post in outs else list(range(20, params.steps - 20, 17))
            for t in probes:
                if not (0 <= t < params.steps):
                    continue
                fd = (float(Vp[t, post]) - float(Vm[t, post])) / (2 * h)
                code = float(eps[(pre, post)][t])
                # skip points where the neuron actually spiked (V is clamped/reset there)
                if any(abs(t - q) <= 1 for q in s[post]):
                    continue
                rows.append((abs(fd - code), code, fd, name, trial, si, pre, post, t,
                             moved, post in outs))

rows.sort(reverse=True)
print(f"{len(rows)} probes compared (eps vs central finite difference of simulated V)\n")
print(f"{'|fd-eps|':>10} {'eps':>12} {'fd':>12}  case / trial / edge / probe")
for d, code, fd, name, tr, si, pre, post, t, mv, isout in rows[:12]:
    print(f"{d:10.3e} {code:12.3e} {fd:12.3e}  {name} t{tr} e{si}(N{pre}->N{post}) t={t}"
          f"  {'TRAIN MOVED' if mv else 'smooth'} {'OUT(counterfactual)' if isout else 'HIDDEN'}")

# eps for OUTPUT neurons is the derivative of the COUNTERFACTUAL epoch-reset V_sub (resets
# at TARGET times), so it is not supposed to equal dV_sim/dw.  Only HIDDEN neurons, whose
# epochs reset at their OWN spikes, are a fair comparison.
smooth = [r for r in rows if not r[9] and not r[10]]
moved  = [r for r in rows if r[9] and not r[10]]
print(f"  (restricted to HIDDEN post-neurons; outputs use a counterfactual V_sub by design)")
print(f"\ntotal probes {len(rows)}:  {len(smooth)} SMOOTH (train unchanged), "
      f"{len(moved)} straddle a train change")
for lab, sub in (("SMOOTH", smooth), ("TRAIN-CHANGE", moved)):
    if not sub:
        continue
    bad = [r for r in sub if r[0] > 1e-9]
    print(f"  {lab}: {len(bad)}/{len(sub)} disagree by >1e-9, max |fd-eps| = {max(r[0] for r in sub):.3e}")
    rel = [r[0]/max(abs(r[2]),1e-30) for r in sub if abs(r[2]) > 1e-12]
    if rel:
        print(f"      median relative error where fd != 0: {np.median(rel):.3e}")
