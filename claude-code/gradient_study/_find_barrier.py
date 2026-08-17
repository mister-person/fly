"""Smallest case showing the SUB-CRITICAL BARRIER.

Below the critical single-spike weight (444.5) a neuron cannot fire from one presynaptic
spike -- it must accumulate, so it fires RARELY and the output spike COUNT is wrong.  The
claim is that no gradient points back across that barrier.  Find the smallest net where
that alone causes the failure.

Candidates, simplest first:
  A  2 neurons, 1 edge:   N0 -> N1, output N1        (barrier on an OBSERVED neuron)
  B  3 neurons, 2 edges:  N0 -> N1 -> N2, output N2  (barrier on a HIDDEN edge)
"""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import multiprocessing as mp
import grad_trace as G

CRIT = 444.5
CAND = {
    "A 2n  N0->N1        (out N1, barrier OBSERVED)": ([[0, 1]], 2, [1], [500.]),
    "B 3n  N0->N1->N2    (out N2, barrier HIDDEN)":   ([[0, 1], [1, 2]], 3, [2], [500., 500.]),
}


def _job(a):
    tag, E, N, outs, Wl, seed = a
    C = np.array(E, np.int32); params = G.mkparams(520)
    W = np.array(Wl, np.float32)
    T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
    w = G.train(C, N, outs, w0.copy(), T, params, rounds=3200, lr=G.LR)
    V = G.fsim(C, N, w, params)
    ok = all(G.sp(V, o) == T[o] for o in outs)
    return tag, seed, np.round(w0, 0).tolist(), np.round(w, 0).tolist(), \
        {n: G.sp(V, n) for n in range(N)}, T, ok


if __name__ == "__main__":
    jobs = [(tag, E, N, o, W, s) for tag, (E, N, o, W) in CAND.items() for s in range(8)]
    with mp.get_context("spawn").Pool(16) as p:
        res = p.map(_job, jobs)
    print(f"critical single-spike weight {CRIT}\n")
    for tag, (E, N, outs, Wl) in CAND.items():
        sub = [r for r in res if r[0] == tag]
        o = outs[0]
        print(f"=== {tag}   true w {Wl} ===")
        print(f"    target N{o} = {sub[0][5][o]}")
        nok = sum(r[6] for r in sub)
        for tg, seed, w0, w, s, T, ok in sub:
            below = [i for i, v in enumerate(w0) if v < CRIT]
            print(f"    seed{seed}: start {w0}{'  [SUB-CRITICAL]' if below else ''}"
                  f" -> {w}  N{o}={s[o]}  {'OK' if ok else 'FAIL'}")
        print(f"    recovered {nok}/8\n")
