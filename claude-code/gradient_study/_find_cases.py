"""Search for 3n D variants: N1 firing TWICE, N1 SPLIT into a chain, and both.

3n D's content is that a SUB-CRITICAL accumulator (w below the 444.5 single-spike
threshold) fires rarely, and each of its spikes leaves exactly one extra mark on the
output.  These variants stress the same mechanism with (a) two marks instead of one,
(b) the rare spike relayed through a second hidden neuron, (c) both.

Selection criteria, so the cases actually TEST something:
  * the accumulator fires the intended number of times (not once per input cycle)
  * every hidden spike leaves a visible extra spike on the output
  * the output is NOT reproducible with the hidden path silent -- otherwise the hidden
    weights are unidentifiable and the case is vacuous
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

params = G.mkparams(520)
CRIT = G.TH / float(G.HK.max())


def sim(E, N, w):
    C = np.array(E, np.int32)
    V = G.fsim(C, N, np.array(w, np.float32), params)
    return {n: G.sp(V, n) for n in range(N)}


def base_out(E, N, w, kill):
    """output train with the hidden path severed (weight -> 0)"""
    w2 = list(w); w2[kill] = 0.0
    return sim(E, N, w2)


print(f"critical single-spike weight = {CRIT:.1f}\n")

# ---------- E: 3 neurons, N1 fires TWICE -------------------------------------------
print("=== E: N1 must fire TWICE (edges [[0,1],[0,2],[1,2]], out N2) ===")
E_E = [[0, 1], [0, 2], [1, 2]]
found = []
for w01 in range(220, 445, 5):
    s = sim(E_E, 3, [w01, 1200., 700.])
    if len(s[1]) == 2:
        found.append((w01, s[1]))
for w01, n1 in found[:6]:
    print(f"   w(0->1)={w01}  N1={n1}")
if found:
    w01 = found[len(found) // 2][0]
    for w12 in (500., 700., 900.):
        s = sim(E_E, 3, [w01, 1200., w12])
        b = base_out(E_E, 3, [w01, 1200., w12], 2)
        print(f"   -> w=[{w01},1200,{w12:.0f}]  N1={s[1]}  N2={s[2]}"
              f"  (silent-N1 output {b[2]}, extra={len(s[2])-len(b[2])})")

# ---------- F: 4 neurons, accumulator relayed through a second hidden --------------
print("\n=== F: split chain N0->N1->N2->N3, N1 rare, ONE mark (out N3) ===")
E_F = [[0, 1], [1, 2], [0, 3], [2, 3]]
for w01 in (200., 250.):
    for w12 in (600., 900., 1200.):
        s = sim(E_F, 4, [w01, w12, 1200., 700.])
        if len(s[1]) == 1 and len(s[2]) == 1:
            b = base_out(E_F, 4, [w01, w12, 1200., 700.], 3)
            print(f"   w=[{w01:.0f},{w12:.0f},1200,700]  N1={s[1]} N2={s[2]} N3={s[3]}"
                  f"  (silent {b[3]}, extra={len(s[3])-len(b[3])})")

# ---------- G: split chain AND two marks ------------------------------------------
print("\n=== G: split chain, N1 fires TWICE (out N3) ===")
for w01 in [f[0] for f in found[:8]]:
    for w12 in (600., 900., 1200.):
        s = sim(E_F, 4, [float(w01), w12, 1200., 700.])
        if len(s[1]) == 2 and len(s[2]) == 2:
            b = base_out(E_F, 4, [float(w01), w12, 1200., 700.], 3)
            print(f"   w=[{w01},{w12:.0f},1200,700]  N1={s[1]} N2={s[2]} N3={s[3]}"
                  f"  (silent {b[3]}, extra={len(s[3])-len(b[3])})")
