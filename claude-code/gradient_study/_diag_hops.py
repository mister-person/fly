"""Do gradients attenuate across the N1->N2->N3 hops?

4n F and 4n G have the SAME topology and the same depth ([[0,1],[1,2],[0,3],[2,3]], out N3)
but F is 8/8 and G is 3/8 -- so if the per-hop attenuation is comparable, depth is not the
cause.  Measure max|L| at each layer and the gradient on each edge, at truth and at each
seed's endpoint.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp

E = [[0, 1], [1, 2], [0, 3], [2, 3]]
CASES = {"4n F": [240., 1200., 1200., 1100.], "4n G": [250., 500., 1200., 700.]}
EDGE = ["w(0->1)", "w(1->2)", "w(0->3)", "w(2->3)"]


def _job(a):
    name, Wl, seed = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    C = np.array(E, np.int32); params = G.mkparams(520)
    W = np.array(Wl, np.float32)
    T = {n: G.sp(G.fsim(C, 4, W, params), n) for n in range(4)}
    inc = {n: np.where(C[:, 1] == n)[0] for n in range(4)}
    if seed is None:
        w = np.array(Wl, float)
    else:
        w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, 4)).astype(float)
        w = G.train(C, 4, [3], w0.copy(), T, params, rounds=3200, lr=G.LR)
    V = G.fsim(C, 4, w, params); s = {p: G.sp(V, p) for p in range(4)}
    eps, L, vs, wr = G.traces(C, 4, w, s, params.steps, {3: T[3]}, V)
    g = np.zeros(4)
    for n in range(4):
        for si in inc[n]:
            g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
    mx = [float(np.abs(L[n]).max()) for n in range(4)]
    return name, seed, np.round(w, 0).tolist(), mx, g, s[3] == T[3], \
        {n: len(s[n]) for n in range(4)}


def main():
    jobs = [(nm, W, sd) for nm, W in CASES.items() for sd in [None] + list(range(8))]
    with mp.get_context("spawn").Pool(16) as p:
        res = p.map(_job, jobs)
    for nm in CASES:
        print(f"===== {nm}  true {CASES[nm]} =====")
        print(f"{'seed':>6} {'ok':>4} {'counts':>14}  {'max|L3|':>9} {'max|L2|':>9}"
              f" {'max|L1|':>9}  {'L2/L3':>7} {'L1/L2':>7}   {'g(0->1)':>10} {'g(1->2)':>10}")
        for name, seed, w, mx, g, ok, cnt in [r for r in res if r[0] == nm]:
            lbl = "TRUE" if seed is None else str(seed)
            r32 = mx[2] / mx[3] if mx[3] > 0 else float('nan')
            r21 = mx[1] / mx[2] if mx[2] > 0 else float('nan')
            c = f"{cnt[1]}/{cnt[2]}/{cnt[3]}"
            print(f"{lbl:>6} {('OK' if ok else '--'):>4} {c:>14}  {mx[3]:9.2e} {mx[2]:9.2e}"
                  f" {mx[1]:9.2e}  {r32:7.3f} {r21:7.3f}   {g[0]:10.2e} {g[1]:10.2e}")
        print()


if __name__ == "__main__":
    main()
