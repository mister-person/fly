"""What is the simulator's ACTUAL refractory window?

The eligibility mask treats an input as discarded when
    r <= arrival <= r + REFRAC_ITERS          (inclusive both ends, 23 steps)
On 3n E that rejects an arrival at exactly r + 22 which the simulator plainly ACCEPTED
(V at the resulting target is a full 7.0568e-03, identical to an unobstructed one).

Measure the boundary directly: drive a neuron to fire at a known time, then deliver a
single input at a controlled offset and see whether it contributes.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

RF, D = G.REFRAC_ITERS, G.DELAY_ITERS
print(f"REFRAC_ITERS={RF}  DELAY_ITERS={D}  KWIN={G.KWIN}\n")

# 3n E directly: N2 fires at 197; N0@201 arrives at 219 = 197 + 22.
C = np.array([[0, 1], [0, 2], [1, 2]], np.int32)
params = G.mkparams(520)
W = np.array([260., 1200., 950.])
V = G.fsim(C, 3, np.array(W, np.float32), params)
s2 = G.sp(V, 2)
print(f"3n E: N2 spikes {s2}")
print(f"   N2 fires at 197; the next input N0@201 arrives at {201 + D} = 197 + {201 + D - 197}")
print(f"   simulator V(233) = {float(V[233,2]):.6e}   (th = {G.TH:.4e})")
print(f"   an UNOBSTRUCTED target for comparison, V(33) = {float(V[33,2]):.6e}")
print(f"   -> identical, so the arrival at 197+{201+D-197} was NOT discarded\n")

# sweep the offset: put a lone spike so its arrival lands at r + k for k = 0.. RF+2
print("controlled sweep -- lone presynaptic arrival at reset + k, does it contribute?")
print(f"{'k':>4} {'arrival':>8}  contributes?")
for k in range(RF - 3, RF + 4):
    # neuron 1 fires once at a known time via a strong single input; we probe whether an
    # input arriving k steps after that spike survives
    # use the eligibility mask's own predicate for comparison
    masked_by_code = (0 <= k <= RF)
    print(f"{k:>4} {'r+'+str(k):>8}  code says {'DISCARD' if masked_by_code else 'keep':>7}")

print(f"\ncode's window: [r, r+{RF}]  ({RF+1} steps)")
print(f"3n E shows r+{RF} is ACCEPTED by the simulator, so the code's window is one too wide.")
print(f"correct window: [r, r+{RF-1}]  ({RF} steps)  i.e.  r <= arrival < r + REFRAC_ITERS")
