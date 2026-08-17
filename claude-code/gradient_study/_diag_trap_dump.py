"""Full spike/weight dump for the chain compensation trap: truth vs [456,415,626]."""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

E = [[0, 1], [1, 2], [2, 3]]
C = np.array(E, np.int32)
N, OUT = 4, 3
TRUE = [500., 500., 500.]
STUCK = [456., 415., 626.]
params = G.mkparams(520)
CRIT = G.TH / float(G.HK.max())


def run(w):
    V = G.fsim(C, N, np.array(w, np.float32), params)
    return V, {n: G.sp(V, n) for n in range(N)}


Vt, st = run(TRUE)
Vs, ss = run(STUCK)

print(f"topology:  N0 --w0--> N1 --w1--> N2 --w2--> N3      (N3 = output)")
print(f"critical single-spike weight = {CRIT:.1f}\n")

print(f"{'':10} {'w0':>7} {'w1':>7} {'w2':>7}")
print(f"{'TRUE':10} {TRUE[0]:7.0f} {TRUE[1]:7.0f} {TRUE[2]:7.0f}")
print(f"{'STUCK':10} {STUCK[0]:7.0f} {STUCK[1]:7.0f} {STUCK[2]:7.0f}")
print(f"{'':10} {'':7} {'SUB-CRIT':>7} {'+25%':>7}\n")

for lab, s in (("TRUE", st), ("STUCK", ss)):
    print(f"{lab} spike times:")
    for n in range(N):
        tag = " <- OUTPUT" if n == OUT else (" (input)" if n == 0 else "")
        print(f"    N{n} ({len(s[n])}x): {s[n]}{tag}")
    print()

print(f"output: target {st[OUT]}  ({len(st[OUT])} spikes)")
print(f"        found  {ss[OUT]}  ({len(ss[OUT])} spikes)\n")

print("WHERE IT BREAKS -- N2 is the neuron that under-fires:")
print(f"    N2 true  ({len(st[2])}x): {st[2]}")
print(f"    N2 stuck ({len(ss[2])}x): {ss[2]}")
print(f"    w1 = {STUCK[1]:.0f} < {CRIT:.1f}, so one N1 spike cannot fire N2;")
print(f"    it must accumulate, so it fires {len(ss[2])} times instead of {len(st[2])}.\n")

print("THE COMPENSATION -- drive delivered to N3 per N2 spike:")
for lab, w, s in (("TRUE ", TRUE, st), ("STUCK", STUCK, ss)):
    peak = w[2] * float(G.HK.max())
    print(f"    {lab}: w2={w[2]:5.0f}  peak PSP at N3 = {peak:.3e}"
          f"  = {peak / G.TH:5.2f} x threshold   ({len(s[2])} N2 spikes)")
print(f"    -> the inflated w2 lets FEWER N2 spikes still drive N3, so the output error")
print(f"       stays small and nothing demands more spikes from N2.\n")

print("voltage margin at N3 for each of its spikes (how hard it is driven):")
for lab, V, s in (("TRUE ", Vt, st), ("STUCK", Vs, ss)):
    m = [f"{float(V[t, OUT]) / G.TH:.2f}" for t in s[OUT]]
    print(f"    {lab}: {m}   (x threshold, at N3's own spike times {s[OUT]})")
