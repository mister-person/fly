"""How many spikes does a positive field run actually stand for?

    python3 _bumpdeficit.py ["14n Q"] [seed]

bumps_of() emits ONE request per positive run of the field, at its centroid.  So a run that is
wide enough to hold six spikes still asks for one, and the count the field hands downstream is
the number of RUNS, not the number of spikes the run represents.  This measures the gap: for
every positive run, how many of the neuron's true spikes fall inside it.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import field_trace as F
from _diag import CASES, steps_for

name = sys.argv[1] if len(sys.argv) > 1 else "14n Q"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 800

E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
p = F.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
w = np.asarray(F.train(C, N, outs, w0.copy(), T, p, rounds=ROUNDS, lr=F.LR), float)
V = F.fsim(C, N, np.asarray(w, np.float32), p)
spall = {n: F.sp(V, n) for n in range(N)}
g, Fl, L, ep = F.gradient(C, N, w, spall, p.steps, {o: T[o] for o in outs})

print(f"{name} seed {seed} — REFRAC={F.REFRAC_ITERS}, so a run of width W can hold "
      f"~1+W/{F.REFRAC_ITERS} spikes\n")
print(f"{'neuron':>7} {'runs':>5} {'asks':>5} {'true':>5} | {'run widths':<28} "
      f"{'true spikes inside each run':<28}")
tot_ask = tot_in = 0
for n in range(1, N):
    if n in outs or not Fl[n].any():
        continue
    bumps = F.bumps_of(Fl[n])
    runs = [b[2] for b in bumps]
    widths = [int(r[-1] - r[0]) for r in runs]
    inside = [sum(1 for s in T[n] if r[0] <= s <= r[-1]) for r in runs]
    tot_ask += len(bumps); tot_in += sum(inside)
    print(f"{'N'+str(n):>7} {len(runs):>5} {len(bumps):>5} {len(T[n]):>5} | "
          f"{str(widths[:6]):<28} {str(inside[:6]):<28}")
# TWO OPPOSITE ERRORS, and they must not be netted against each other:
#   under = a run holding k>1 true spikes still asks once   -> sum(k-1)
#   over  = a run holding no true spike asks anyway         -> count of k==0 runs
# The aggregate difference is near zero on some cases purely because these cancel, which
# would read as "the count is fine" when in fact both halves are wrong.
under = over = 0
for n in range(1, N):
    if n in outs or not Fl[n].any():
        continue
    for _q, _h, r in F.bumps_of(Fl[n]):
        k = sum(1 for s in T[n] if r[0] <= s <= r[-1])
        under += max(0, k - 1)
        over += (k == 0)
print(f"\nhidden total: {tot_ask} requests against {tot_in} true spikes inside the same runs")
print(f"  UNDER-ask (wide runs holding several spikes): {under} spikes never requested")
print(f"  OVER-ask  (runs holding no true spike at all): {over} spurious requests")
wide = [(n, int(b[2][-1] - b[2][0]), sum(1 for s in T[n] if b[2][0] <= s <= b[2][-1]))
        for n in range(1, N) if n not in outs and Fl[n].any() for b in F.bumps_of(Fl[n])]
wide.sort(key=lambda x: -x[2])
print("widest under-counted runs (neuron, width, true spikes inside):", wide[:6])
