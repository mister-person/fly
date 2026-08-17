"""Search small cases matching the 50-neuron failure signature: COUNT easy, TIMING hard.

At 50 neurons the method gets the spike count right on 17/18 outputs and the times wrong by
6-44 steps, plateauing early.  The small suite does not currently contain a case with that
shape -- its failures are count failures.  Three structures that should isolate timing:

  H  DEEP CHAIN      N0->N1->N2->N3->N4      timing must survive three hidden hops
  I  CONVERGENT      N0->N1->N3, N0->N2->N3  two hidden paths whose arrivals must ALIGN
  J  TIGHT ISI       output spikes close together, so small timing errors are fatal
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

params = G.mkparams(520)


def sim(E, N, w):
    V = G.fsim(np.array(E, np.int32), N, np.array(w, np.float32), params)
    return {n: G.sp(V, n) for n in range(N)}


print(f"critical weight {G.W_CRIT:.1f}\n")

print("=== H: deep chain N0->N1->N2->N3->N4 (out N4) ===")
EH = [[0, 1], [1, 2], [2, 3], [3, 4]]
for w in ([500.] * 4, [600., 500., 500., 500.], [500., 600., 700., 800.]):
    s = sim(EH, 5, w)
    print(f"   w={[int(x) for x in w]}  N1={s[1]} N2={s[2]} N3={s[3]}")
    print(f"      OUT N4={s[4]}")

print("\n=== I: convergent, two hidden paths of different length (out N3) ===")
EI = [[0, 1], [0, 2], [1, 3], [2, 3]]
for w in ([500., 900., 500., 500.], [500., 500., 400., 400.], [900., 500., 300., 500.]):
    s = sim(EI, 4, w)
    b1 = sim(EI, 4, [w[0], w[1], 0.0, w[3]])[3]
    b2 = sim(EI, 4, [w[0], w[1], w[2], 0.0])[3]
    print(f"   w={[int(x) for x in w]}  N1={s[1]} N2={s[2]}")
    print(f"      OUT N3={s[3]}   without path1 {b1}   without path2 {b2}")

print("\n=== J: tight ISI on the output (out N2) ===")
EJ = [[0, 1], [0, 2], [1, 2]]
for w in ([700., 900., 900.], [800., 800., 1200.], [600., 1000., 1100.]):
    s = sim(EJ, 3, w)
    d = np.diff(s[2]).tolist() if len(s[2]) > 1 else []
    print(f"   w={[int(x) for x in w]}  N1={s[1]}  OUT N2={s[2]}  ISIs={d}"
          f"   (refractory {G.REFRAC_ITERS})")
