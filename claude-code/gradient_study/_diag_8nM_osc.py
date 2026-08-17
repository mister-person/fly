"""(1) Is the output still oscillating?  (2) Which seed gets every output COUNT right?

Path efficiency = |net displacement| / |total path travelled| per weight.  On 4n G the
hidden edges ran at 1.8-4.4% while the direct edge ran at 55.7%; that is what made the case
need 10k iterations for ~600 iterations of progress.
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp


def _job(seed):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    from _diag import CASES, steps_for
    E, N, outs, Wl = CASES["8n M"]
    C = np.array(E, np.int32); params = G.mkparams(steps_for("8n M"))
    W = np.array(Wl, np.float32)
    T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    H = []
    G.train(C, N, outs, w0.copy(), T, params, rounds=3200, lr=G.LR,
            cb=lambda it, w, *a: H.append(w.copy()))
    H = np.array(H)
    path = np.abs(np.diff(H, axis=0)).sum(axis=0)
    net = np.abs(H[-1] - H[0])
    eff = 100.0 * net / np.maximum(path, 1e-9)
    V = G.fsim(C, N, H[-1], params)
    F = {n: G.sp(V, n) for n in range(N)}
    counts = [len(F[o]) for o in outs]
    want = [len(T[o]) for o in outs]
    err = []
    for o in outs:
        f, t = F[o], T[o]
        err.append(99.0 if len(f) != len(t)
                   else float(np.mean([abs(a - b) for a, b in zip(f, t)])))
    return seed, counts, want, err, eff.tolist(), [float(x) for x in H[-1]]


def main():
    import numpy as np
    with mp.get_context("spawn").Pool(8) as p:
        res = p.map(_job, range(8))
    print("path EFFICIENCY = net displacement / total path travelled, per weight\n")
    print(f"{'seed':>4} {'out counts':>12} {'want':>9} {'mean|dt|':>9}  "
          f"{'eff fan-out (0->1..4)':>24}  {'eff fan-in (mean)':>18}")
    exact_count = []
    for seed, counts, want, err, eff, w in res:
        ok = counts == want
        if ok:
            exact_count.append(seed)
        e = "  ".join(f"{x:5.1f}" for x in err) if max(err) < 99 else "  CNT"
        print(f"{seed:>4} {str(counts):>12} {str(want):>9} {e:>9}  "
              f"{'  '.join(f'{x:4.1f}' for x in eff[:4]):>24}  "
              f"{np.mean(eff[4:]):17.1f}%"
              + ("   <- ALL COUNTS RIGHT" if ok else ""))
    print(f"\nseeds with every output count correct: {exact_count}")


if __name__ == "__main__":
    main()
