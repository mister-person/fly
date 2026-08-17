"""Two new case families:
  (A) COINCIDENCE -- no single connection can fire anything (all weights far below
      W_CRIT=444.5), so ~4 presynaptic spikes must arrive together.  Several outputs so the
      system is not badly underdetermined.
  (B) NEGATIVE weights -- inhibitory edges.  Expected to break things: every deficit /
      hinge / request argument in the method assumes more weight => more drive => earlier
      spike, which reverses in sign for w < 0.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

params = G.mkparams(520)
print(f"W_CRIT = {G.W_CRIT:.1f}   th = {G.TH:.3e}   peak HK = {G.HK.max():.3e}\n")


def sim(E, N, w):
    V = G.fsim(np.array(E, np.int32), N, np.array(w, np.float32), params)
    return {n: G.sp(V, n) for n in range(N)}


# ---------- (A) coincidence: N0 -> N1..N4 (fan-out), N1..N4 -> N5..N7 (3 outputs) --------
print("=== (A) 4-way coincidence, 8 neurons, 3 outputs ===")
E_A = [[0, 1], [0, 2], [0, 3], [0, 4]] + [[h, o] for o in (5, 6, 7) for h in (1, 2, 3, 4)]
print(f"    edges: {len(E_A)}  (4 fan-out + 12 fan-in)")
# fan-out weights spread the hidden spikes in time; fan-in weights all sub-critical
for fan in ([900., 700., 550., 460.], [1200., 800., 600., 480.]):
    for k in (120., 150., 180.):
        w = fan + [k] * 12
        s = sim(E_A, 8, w)
        hid = [len(s[n]) for n in (1, 2, 3, 4)]
        outs = [len(s[n]) for n in (5, 6, 7)]
        share = k * float(G.HK.max()) / G.TH
        print(f"    fan={[int(x) for x in fan]} k={int(k)}  one edge = {share:.2f}x th"
              f"  hidden spikes {hid}  outputs {outs}")
        if all(o > 0 for o in outs):
            print(f"        N1={s[1]}\n        N5={s[5]}  N6={s[6]}  N7={s[7]}")

# ---------- (B) negative weights ---------------------------------------------------
print("\n=== (B) negative weights: does the simulator even accept them? ===")
E_B = [[0, 1], [0, 2], [1, 2]]
for w12 in (700., 0., -300., -700.):
    s = sim(E_B, 3, [700., 1200., w12])
    print(f"    w(1->2)={w12:7.0f}  N1={s[1]}  N2={s[2]}")
