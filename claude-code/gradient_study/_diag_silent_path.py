"""Why is L[3] EXACTLY zero for a silent neuron, when the backward message does not
structurally require that neuron to have spiked?

The relaxation builds a hidden neuron's demand as
    vol = sum_d  w[n->d] * back_corr(L[d], HK)
    Ln  = vol * exp(-((vsub[n]-TH)/(SIG*TH))^2) * CREATE
None of those terms needs n's OWN spikes -- vol comes from downstream demand, the gate from
n's sub-threshold voltage.  So a silent neuron should still receive a (small) demand, and
since eps[(0,n)] is healthy the input edge would then have a gradient.  Find which factor
is exactly zero.
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
T = {n: G.sp(G.fsim(C, N, np.array(Wl, np.float32), params), n) for n in range(N)}
w = np.array([489., 529., 129., 135., 107., 315., 20., 29.,
              20., 422., 20., 28., 33., 381., 20., 24.])
V = G.fsim(C, N, w, params)
s = {n: G.sp(V, n) for n in range(N)}
eps, L, vsub, wreq = G.traces(C, N, w, s, params.steps, {o: T[o] for o in outs}, V)

n = 3
print(f"N{n} is silent: spikes {s[n]}")
print(f"   downstream edges: " + ", ".join(
    f"N{n}->N{int(C[si,1])} w={w[si]:.0f}" for si in np.where(C[:, 0] == n)[0]))
print()
# rebuild the relaxation terms by hand
vol = np.zeros(params.steps)
for si in np.where(C[:, 0] == n)[0]:
    d = int(C[si, 1])
    contrib = float(w[si]) * G.back_corr(L[d], G.HK)
    print(f"   from N{d}: max|L[{d}]| {np.abs(L[d]).max():.3e}"
          f"   w {w[si]:6.1f}   max|w*back_corr| {np.abs(contrib).max():.3e}")
    vol += contrib
print(f"   vol            max {np.abs(vol).max():.3e}")
gate = np.exp(-((vsub[n] - G.TH) / (G.SIG * G.TH)) ** 2)
print(f"   sensitivity gate exp(-((vsub-th)/(SIG*th))^2): max {gate.max():.3e}"
      f"   at vsub max {vsub[n].max():.3e} (th {G.TH:.3e})")
print(f"   vol * gate     max {np.abs(vol * gate).max():.3e}")
print()
print(f"   ACTUAL L[{n}] max {np.abs(L[n]).max():.3e}   <- what the code produced")
print()
print(f"   eps[(0,{n})] sum {eps[(0, n)].sum():.3e}  (healthy -- N0 spikes exist)")
print(f"   so if L[{n}] were nonzero, g(0->{n}) = dot(L,eps) would be too.")
print()
print("CREATE mask (CREATE_FLOOR gate on where a spike can be grown):")
print(f"   CREATE_FLOOR={G.CREATE_FLOOR}  -> threshold {G.CREATE_FLOOR*G.TH:.3e}")
print(f"   vsub[{n}] max {vsub[n].max():.3e};  fraction of timesteps with "
      f"vsub >= floor: {(vsub[n] >= G.CREATE_FLOOR*G.TH).mean():.3f}")
