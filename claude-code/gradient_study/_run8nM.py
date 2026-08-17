"""8n M recovery at 520 vs 1040 sim steps.

At 520 the case gives 9 output spikes for 16 weights; at 1040 it gives 18, so doubling the
runtime roughly doubles the supervision without changing the structure (hidden rates stay
mixed: 11/10/5/3 instead of 5/5/2/1).
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp


def _job(a):
    steps, seed, rounds = a
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np, grad_trace as G
    from _diag import CASES
    E, N, outs, Wl = CASES["8n M"]
    C = np.array(E, np.int32); W = np.array(Wl, np.float32)
    params = G.mkparams(steps)
    T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    w = G.train(C, N, outs, w0.copy(), T, params, rounds=rounds, lr=G.LR)
    V = G.fsim(C, N, w, params)
    ok = all(G.sp(V, o) == T[o] for o in outs)
    err = []
    for o in outs:
        f, t = G.sp(V, o), T[o]
        err.append(99.0 if len(f) != len(t)
                   else float(np.mean([abs(a - b) for a, b in zip(f, t)])))
    wrel = float(np.abs(np.array(w) - np.array(Wl)).max() / max(np.abs(Wl)))
    return steps, seed, ok, err, wrel, [int(round(x)) for x in w]


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3200
    jobs = [(s, sd, rounds) for s in (520, 1040) for sd in range(8)]
    with mp.get_context("spawn").Pool(16) as p:
        res = p.map(_job, jobs)
    for steps in (520, 1040):
        sub = [r for r in res if r[0] == steps]
        n = sum(r[2] for r in sub)
        errs = [e for r in sub for e in r[3] if e < 99]
        cnt = sum(1 for r in sub for e in r[3] if e >= 99)
        print(f"steps={steps}: EXACT {n}/8"
              f"   outputs with wrong count {cnt}/{len(sub)*3}"
              f"   mean|dt| over count-ok outputs {np.mean(errs) if errs else float('nan'):.2f}")
        for r in sorted(sub, key=lambda x: x[1]):
            print(f"    seed{r[1]}: {'OK  ' if r[2] else 'fail'}  per-output |dt| "
                  f"{[round(e,1) if e < 99 else 'CNT' for e in r[3]]}")
    print()


if __name__ == "__main__":
    import numpy as np
    main()
