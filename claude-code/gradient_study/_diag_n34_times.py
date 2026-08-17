"""With the accumulators revived and roughly correctly weighted, WHAT are they firing?

Two questions:
 (a) do N3/N4 hit their own true times, or just settle on a shared period?
 (b) does the best lever config (BETA1 0.97) cost the suite, as it did before CREATE?
This file answers (a); the suite run answers (b).
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp

CFGS = [("default      ", dict(BETA1=0.9,  CREATE=0.3, LR=10.0)),
        ("BETA1=0.97   ", dict(BETA1=0.97, CREATE=0.3, LR=10.0))]


def _job(a):
    tag, cfg, seed = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    from _diag import CASES, steps_for
    for k, v in cfg.items():
        setattr(G, k, v)
    E, N, outs, Wl = CASES["8n M"]
    C = np.array(E, np.int32); params = G.mkparams(steps_for("8n M"))
    W = np.array(Wl, np.float32)
    T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    w = G.train(C, N, outs, w0.copy(), T, params, rounds=3200, lr=cfg["LR"])
    V = G.fsim(C, N, w, params)
    return tag, seed, {n: G.sp(V, n) for n in range(N)}, T, [float(x) for x in w]


def main():
    import numpy as np
    jobs = [(t, c, s) for t, c in CFGS for s in range(4)]
    with mp.get_context("spawn").Pool(8) as p:
        res = p.map(_job, jobs)
    T = res[0][3]
    print(f"TRUE  N1 {len(T[1])}x  N2 {len(T[2])}x  N3 {T[3]}  N4 {T[4]}")
    print(f"      N3 period {np.diff(T[3]).tolist()}   N4 period {np.diff(T[4]).tolist()}\n")
    for tag, _ in CFGS:
        print(f"=== {tag} ===")
        for t, seed, F, _, w in [r for r in res if r[0] == tag]:
            def per(x):
                d = np.diff(x)
                return f"{int(d[0])}" if len(d) and len(set(d.tolist())) == 1 else str(d.tolist()[:3])
            same = F[3] == F[4]
            print(f"  seed{seed}: N1 {len(F[1])}x  N2 {len(F[2])}x  "
                  f"N3 {len(F[3])}x {F[3][:4]} p={per(F[3])}  "
                  f"N4 {len(F[4])}x {F[4][:4]} p={per(F[4])}"
                  + ("   N3==N4 LOCKSTEP" if same else ""))
            # how close is each to ITS OWN true train?
            for n in (3, 4):
                if len(F[n]) == len(T[n]):
                    e = float(np.mean([abs(a - b) for a, b in zip(F[n], T[n])]))
                    print(f"        N{n}: right count, mean|dt| {e:.1f}")
                else:
                    print(f"        N{n}: count {len(F[n])} vs true {len(T[n])}")
        print()


if __name__ == "__main__":
    main()
