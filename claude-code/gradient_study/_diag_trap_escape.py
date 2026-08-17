"""Can the trap escape purely by pushing N1 and N2 earlier?

OCCL_GAIN flips g(0->1) the wrong way here; MOVE_GAIN pushes both the right way but is too
weak to flip g(1->2).  Sweep MOVE_GAIN with OCCL off, find where BOTH gradients go positive,
then actually train from the stuck point with those settings and see if it escapes.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp

TRUE = [500., 500., 500.]
STUCK = [456., 415., 626.]
GAINS = [0.25, 0.5, 1.0, 2.0, 4.0]


def _grad(a):
    move, occl = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    G.MOVE_GAIN, G.OCCL_GAIN = move, occl
    C = np.array([[0, 1], [1, 2], [2, 3]], np.int32); params = G.mkparams(520)
    T = {n: G.sp(G.fsim(C, 4, np.array(TRUE, np.float32), params), n) for n in range(4)}
    w = np.array(STUCK)
    V = G.fsim(C, 4, w, params); s = {p: G.sp(V, p) for p in range(4)}
    eps, L, vs, wr = G.traces(C, 4, w, s, params.steps, {3: T[3]}, V)
    inc = {n: np.where(C[:, 1] == n)[0] for n in range(4)}
    g = np.zeros(3)
    for n in range(4):
        for si in inc[n]:
            g[si] = float(np.dot(L[n], eps[(int(C[si, 0]), n)]))
    # and train from the stuck point with these settings
    w2 = G.train(C, 4, [3], np.array(STUCK, float), T, params, rounds=3200, lr=G.LR)
    V2 = G.fsim(C, 4, w2, params)
    return move, occl, g, float(L[2][212]), np.round(w2, 0).tolist(), \
        G.sp(V2, 3), T[3], {n: G.sp(V2, n) for n in range(4)}


def main():
    jobs = [(m, o) for m in GAINS for o in (0.0, 1.0)]
    with mp.get_context("spawn").Pool(min(16, len(jobs))) as p:
        res = p.map(_grad, jobs)
    print(f"stuck {STUCK} -> true {TRUE};  need g(0->1) AND g(1->2) POSITIVE\n")
    print(f"{'MOVE':>5} {'OCCL':>5}  {'g(0->1)':>11} {'g(1->2)':>11} {'L[2]@212':>11}"
          f"  {'both up?':>9}   trained from stuck ->")
    for move, occl, g, l212, w2, out, tgt, s in res:
        both = "YES" if g[0] > 0 and g[1] > 0 else "no"
        ok = "ESCAPED" if out == tgt else f"stuck {out}"
        print(f"{move:>5} {occl:>5}  {g[0]:>11.3e} {g[1]:>11.3e} {l212:>11.3e}"
              f"  {both:>9}   {w2} {ok}")


if __name__ == "__main__":
    main()
