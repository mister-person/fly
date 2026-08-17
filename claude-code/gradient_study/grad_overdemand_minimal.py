"""Minimal example that STILL over-fires under partial credit (no veto).

3 neurons, feed-forward:
    N0 -> N1  (w250): N1 is a weak ACCUMULATOR -- it needs several input spikes to
                      reach threshold, so it truly fires only 2x: [173, 373]
    N1 -> N2  (w700): strong
    N0 -> N2  (w300): direct bypass
    N2 (output) fires 3x: [140, 220, 399]

The hidden neuron legitimately fires LESS OFTEN (2x) than the output it drives (3x):
some of N2's spikes come from the N0 bypass, not from N1.  But anchor_targets() emits
ONE hidden spike per downstream pattern-instance, so N1 is demanded 3 spikes
([58, 138, 317]).  Partial credit does NOT filter this out -- N1 (w700) really is the
majority contributor to N2, so it passes r >= 1/fanin and collects every demand.

Driving N1 to fire as early as t=58 needs a large incoming weight, which then makes it
fire EVERY input period -> N1 ends up firing 5x [62,162,262,362,462] vs true 2x, and the
output degrades to [181, 381] vs target [140, 220, 399].

ROOT CAUSE: the inference has no way to represent "this hidden neuron should fire less
often than its downstream".  One demanded spike per downstream firing is an upper bound
baked into the anchoring, and it is exactly the 50-neuron over-firing failure.
"""
import sys, os, types
sys.path.insert(0, "/workspace/project/gradient_study")
sys.path.insert(0, "/workspace/project")
os.environ.setdefault("MPLBACKEND", "Agg")
for _n, _a in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        for _k, _v in _a.items():
            setattr(_m, _k, _v)
        sys.modules[_n] = _m
import numpy as np
import grad_credit as G

C = np.array([[0, 1], [1, 2], [0, 2]], np.int32)
N = 3
OUTS = [2]
W = np.array([250., 700., 300.], np.float32)


def main():
    params = G.mkparams(520)
    tv = G.fsim(C, N, W, params)
    T = {n: G.sp(tv, n) for n in range(N)}
    print("true spikes:", {n: T[n] for n in range(N)})
    print(f"  hidden N1 fires {len(T[1])}x but output N2 fires {len(T[2])}x "
          f"-- N1 must fire LESS often than its downstream.\n")

    for seed in range(4):
        w = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(W))).astype(float)
        w, tgt = G.train(C, N, OUTS, w, T, params, rounds=30)
        V = G.fsim(C, N, w, params)
        n1, out = G.sp(V, 1), G.sp(V, 2)
        flag = "N1 OVER-FIRES" if len(n1) > len(T[1]) else ("N1 silenced" if not n1 else "")
        print(f"seed{seed}: demanded N1 target={tgt.get(1)} ({len(tgt.get(1) or [])} spikes, "
              f"true {len(T[1])})")
        print(f"         N1 fires {n1}   output={out}  (target {T[2]})  "
              f"{'OK' if out == T[2] else 'FAIL'} {flag}")


if __name__ == "__main__":
    main()
