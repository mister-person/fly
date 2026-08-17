"""Is the missing signal the derivative THROUGH the neuron's own spike times?

eligibility() truncates at reset times and treats them as CONSTANTS:
    e[s:hi] += HK[:hi-s],  hi = min(s+KWIN, next_reset+1, T)
But a reset time IS a spike time, hence a function of w.  The true derivative is

    dV_n(t)/dw = eps_k(t)                             <- direct, what the code has
               + sum_{own spikes s<t} dV_n(t)/ds * ds/dw    <- MISSING

Test it numerically against the real simulator: compare the code's eps to a central finite
difference of the ACTUAL sub-threshold voltage, at places where a reset intervenes.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

C = np.array([[0, 1], [0, 2], [1, 2]], np.int32)
N, OUTS = 3, [2]
params = G.mkparams(520)
T = {n: G.sp(G.fsim(C, N, np.array([200., 1200., 700.], np.float32), params), n)
     for n in range(N)}


def probe(label, w, edge, tprobe):
    """edge = synapse index; tprobe = time at which we want dV_2/dw."""
    w = np.array(w, float)
    V = G.fsim(C, N, w, params); spall = {p: G.sp(V, p) for p in range(N)}
    eps, L, vsub, wreq = G.traces(C, N, w, spall, params.steps, {o: T[o] for o in OUTS}, V)
    pre, post = int(C[edge, 0]), int(C[edge, 1])
    code = float(eps[(pre, post)][tprobe])

    # TRUE derivative of the simulated sub-threshold voltage at tprobe, central difference
    h = max(1.0, abs(w[edge]) * 1e-3)
    def vat(x):
        ww = w.copy(); ww[edge] = x
        return float(G.fsim(C, N, ww, params)[tprobe, post]), G.sp(G.fsim(C, N, ww, params), post)
    vp, sp_p = vat(w[edge] + h)
    vm, sp_m = vat(w[edge] - h)
    fd = (vp - vm) / (2 * h)
    print(f"\n=== {label} ===")
    print(f"    w={w.tolist()}  edge N{pre}->N{post}  probe t={tprobe}")
    print(f"    N{pre} spikes {spall[pre]}   N{post} spikes {spall[post]}")
    print(f"    eps (code, direct only) = {code:.6e}")
    print(f"    finite diff (true dV/dw)= {fd:.6e}")
    print(f"    MISSING TERM            = {fd - code:.6e}")
    if abs(fd) > 1e-12 and code == 0.0:
        print(f"    ^^ code says the weight has NO effect here; the simulator says it does")
    print(f"    N{post} train at w+h {sp_p}")
    print(f"    N{post} train at w-h {sp_m}")


# (1) the WEAK case: g(1->2)=0 because N1@223 is masked out of the 293 epoch
probe("3n D seed0 WEAK: does w(1->2) really not affect V2(293)?",
      [243., 940., 379.], 2, 293)

# (2) a case where an EARLIER own spike gates a later target: probe N2 at a target that
#     sits just after one of N2's own spikes
probe("3n D seed0 WEAK: V2 at target 233 (own spike at 238 nearby)",
      [243., 940., 379.], 2, 233)

# (3) seed4 TWICE: spurious N2 spikes at 186/386; probe the 293 target
probe("3n D seed4 TWICE: V2(293) wrt w(1->2), spurious spikes present",
      [274., 1199., 1019.], 2, 293)
