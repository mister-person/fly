"""4n G: does the creation request reach the INTERMEDIATE hidden neuron N2?

N3's missing marks should raise a request on N2 (fire here), which should raise one on N1.
If N2 is silent, g(1->2) = dot(L[2], eps[(1,2)]) is the only thing that can revive it.
Classify all 8 seeds and inspect the demand on N2.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import multiprocessing as mp
import grad_trace as G

E = [[0, 1], [1, 2], [0, 3], [2, 3]]
N, OUTS, TRUE = 4, [3], [250., 500., 1200., 700.]


def _job(seed):
    C = np.array(E, np.int32); params = G.mkparams(520)
    T = {n: G.sp(G.fsim(C, N, np.array(TRUE, np.float32), params), n) for n in range(N)}
    w0 = (np.array(TRUE, np.float32) *
          np.random.default_rng(seed).uniform(0.5, 1.5, 4)).astype(float)
    w = G.train(C, N, OUTS, w0.copy(), T, params, rounds=1600, lr=G.LR)
    V = G.fsim(C, N, w, params); s = {p: G.sp(V, p) for p in range(N)}
    inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
    out = {}
    for rg in (0.0, 3.0):
        G.REQ_GAIN = rg
        eps, L, vsub, wreq = G.traces(C, N, w, s, params.steps, {o: T[o] for o in OUTS}, V)
        g = np.zeros(4)
        for n in range(N):
            for si in inc[n]:
                g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
        out[rg] = (L[2].copy(), L[1].copy(), g, eps[(1, 2)].copy())
    return seed, np.round(w, 0).tolist(), s, T, out


if __name__ == "__main__":
    with mp.get_context("spawn").Pool(8) as p:
        res = p.map(_job, range(8))
    print(f"true w {TRUE}   N1 {res[0][3][1]}  N2 {res[0][3][2]}")
    print(f"target N3 {res[0][3][3]}\n")
    print(f"critical weight {G.TH/float(G.HK.max()):.1f}  "
          f"(w(1->2)=500 true is only just above it)\n")
    for seed, w, s, T, out in res:
        Lp, L1p, gp, e12 = out[3.0]
        Lz, L1z, gz, _ = out[0.0]
        req2 = Lp - Lz                      # the creation request part on N2
        state = ("N2 SILENT" if not s[2] else
                 f"N2 x{len(s[2])}" + ("" if len(s[2]) == 2 else " (want 2)"))
        print(f"seed{seed}: w={w}  N1={s[1]} N2={s[2]}  [{state}]")
        print(f"    N3={s[3]}")
        print(f"    g={np.array2string(gp, precision=2)}   g(1->2)={gp[1]:+.3e}")
        print(f"    L[2]: {len(np.nonzero(Lp)[0])} nonzero, max|L2|={np.abs(Lp).max():.3e}"
              f"   request part max={np.abs(req2).max():.3e}")
        print(f"    eps[(1->2)] sum={e12.sum():.3e}  "
              f"(zero => N1's spikes cannot drive N2 at all)")
