"""8n M vs KICK_DEAD, in parallel (1040 steps x 8 seeds is too slow serially)."""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp

GAINS = [0.0, 1.0, 10.0, 100.0]


def _job(a):
    kd, seed = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    from _diag import CASES, steps_for
    G.KICK_DEAD = kd
    E, N, outs, Wl = CASES["8n M"]
    C = np.array(E, np.int32); params = G.mkparams(steps_for("8n M"))
    W = np.array(Wl, np.float32)
    T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    w = G.train(C, N, outs, w0.copy(), T, params, rounds=3200, lr=G.LR)
    V = G.fsim(C, N, w, params)
    ok = all(G.sp(V, o) == T[o] for o in outs)
    alive = sum(1 for n in (3, 4) if G.sp(V, n))
    err = 0.0
    for o in outs:
        f, t = G.sp(V, o), T[o]
        err += 99.0 if len(f) != len(t) else float(np.mean([abs(a - b) for a, b in zip(f, t)]))
    return kd, seed, ok, alive, err / len(outs)


def main():
    jobs = [(kd, s) for kd in GAINS for s in range(8)]
    with mp.get_context("spawn").Pool(16) as p:
        res = p.map(_job, jobs)
    for kd in GAINS:
        sub = [r for r in res if r[0] == kd]
        import numpy as np
        print(f"  KICK_DEAD={kd:6}: {sum(r[2] for r in sub)}/8 exact"
              f"   N3/N4 alive {sum(r[3] for r in sub)}/16"
              f"   mean err {np.mean([r[4] for r in sub]):6.2f}")


if __name__ == "__main__":
    main()
