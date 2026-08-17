"""At the chain trap, what does it take to make g(0->1) and g(1->2) BOTH positive?

Pushing N1 and N2 earlier means raising w0 and w1 -- and raising w1 past the critical 444.5
fixes the spike COUNT as a side effect, so the count never has to be reasoned about.
Both gradients must be POSITIVE (w0 456->500, w1 415->500).  Decompose which term supplies
or fights that sign.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

C = np.array([[0, 1], [1, 2], [2, 3]], np.int32)
N, TRUE, STUCK = 4, [500., 500., 500.], np.array([456., 415., 626.])
params = G.mkparams(520)
T = {n: G.sp(G.fsim(C, N, np.array(TRUE, np.float32), params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}


def probe(move, occl, act, req):
    G.MOVE_GAIN, G.OCCL_GAIN, G.MOVE_ACT, G.REQ_GAIN = move, occl, act, req
    V = G.fsim(C, N, STUCK, params); s = {p: G.sp(V, p) for p in range(N)}
    eps, L, vs, wr, ea, Lm = G.traces(C, N, STUCK, s, params.steps, {3: T[3]}, V, full=True)
    g = np.zeros(3)
    for n in range(N):
        for si in inc[n]:
            k = int(C[si, 0])
            if act and n == 3 and move > 0:
                g[si] = float(np.dot(L[n] - Lm[n], eps[(k, n)]) + np.dot(Lm[n], ea[(k, n)]))
            else:
                g[si] = float(np.dot(L[n], eps[(k, n)]))
    return g, L, s


print(f"stuck {STUCK.tolist()};  need w0 UP (456->500), w1 UP (415->500)")
print(f"N1 {G.sp(G.fsim(C,N,STUCK,params),1)}  N2 {G.sp(G.fsim(C,N,STUCK,params),2)}"
      f"  N3 {G.sp(G.fsim(C,N,STUCK,params),3)}   target N3 {T[3]}\n")
print(f"{'MOVE':>5} {'OCCL':>5} {'ACT':>4} {'REQ':>4}  {'g(0->1)':>11} {'g(1->2)':>11}"
      f" {'g(2->3)':>11}   {'L[2]@212':>10}  verdict")
for move in (0.0, 0.25):
    for occl in (0.0, 1.0):
        for act in (0, 1):
            g, L, s = probe(move, occl, act, 3.0)
            ok = "BOTH UP" if g[0] > 0 and g[1] > 0 else (
                "w1 up only" if g[1] > 0 else ("w0 up only" if g[0] > 0 else "neither"))
            print(f"{move:>5} {occl:>5} {act:>4} {3.0:>4}  {g[0]:>11.3e} {g[1]:>11.3e}"
                  f" {g[2]:>11.3e}   {L[2][212]:>10.3e}  {ok}")
