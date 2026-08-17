"""Which part is broken: the DEMAND, or the mapping from demand to weight direction?

A hidden neuron has (a) demands to fire/not fire at times, (b) its own voltage and spikes,
(c) a direction for its input weights, (d) requests to send upstream.
Sign agreement on the hidden edges is 71-72% with the current construction.  Is that
because the DEMAND it receives is wrong, or because dot(L, eps) cannot turn even a correct
demand into a correct direction?

ORACLE TEST: hand every hidden neuron its TRUE spike times as targets -- the same hinge the
output gets -- and re-measure sign agreement.  If the oracle is ~100%, (a)/(d) are the
broken parts and (c) is fine.  If it is still ~70%, (c) is broken and no amount of better
demand construction will help.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grad_trace as G

CASES = {
    "4n G": ([[0, 1], [1, 2], [0, 3], [2, 3]], 4, [3], [250., 500., 1200., 700.]),
    "4n F": ([[0, 1], [1, 2], [0, 3], [2, 3]], 4, [3], [240., 1200., 1200., 1100.]),
    "3n D": ([[0, 1], [0, 2], [1, 2]], 3, [2], [200., 1200., 700.]),
}
G.NEW_DEMAND = 0
params = G.mkparams(520)
rng = np.random.default_rng(0)

for name, (E, N, outs, Wl) in CASES.items():
    C = np.array(E, np.int32); TRUE = np.array(Wl)
    T = {n: G.sp(G.fsim(C, N, TRUE.astype(np.float32), params), n) for n in range(N)}
    inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
    hid = [n for n in range(N) if n not in outs and n != 0]
    hid_edges = [si for si in range(len(Wl)) if int(C[si, 1]) in hid]
    pts = [TRUE * rng.uniform(0.5, 1.5, len(Wl)) for _ in range(40)]

    def sign_acc(oracle):
        agree = live = zero = 0
        for w in pts:
            w = np.array(w, float)
            V = G.fsim(C, N, w, params); s = {p: G.sp(V, p) for p in range(N)}
            # oracle: every hidden neuron gets its TRUE spikes as targets, like an output
            tg = {o: T[o] for o in outs}
            if oracle:
                for n in hid:
                    tg[n] = T[n]
            eps, L, vs, wr = G.traces(C, N, w, s, params.steps, tg, V)
            g = np.zeros(len(w))
            for n in range(N):
                for si in inc[n]:
                    g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
            for si in hid_edges:
                want = TRUE[si] - w[si]
                if abs(want) < 1e-9:
                    continue
                if g[si] == 0.0:
                    zero += 1
                else:
                    live += 1
                    agree += int(np.sign(g[si]) == np.sign(want))
        return agree, live, zero

    a0, l0, z0 = sign_acc(False)
    a1, l1, z1 = sign_acc(True)
    print(f"=== {name}   hidden edges {[('w' + str(int(C[si,0])) + '->' + str(int(C[si,1]))) for si in hid_edges]} ===")
    print(f"    inferred demand : {a0}/{l0} = {100.0*a0/max(l0,1):5.1f}% correct sign,"
          f" {z0} zero")
    print(f"    ORACLE demand   : {a1}/{l1} = {100.0*a1/max(l1,1):5.1f}% correct sign,"
          f" {z1} zero")
