"""How does layer-local voltage-target TP do when INTERMEDIATE neurons have no
target information?

Chain  N0(input) -> N1 -> N2 -> N3(output).  Each neuron reads only its upstream
one, so a wrong intermediate corrupts everything downstream — intermediate
neurons are genuinely necessary (unlike the tiny N=4 mixed net).

Only N3's target is "given".  Three ways to handle the hidden N1, N2:

  ORACLE    N1,N2 get their TRUE target times            (upper bound)
  NO-INFO   N1,N2 untrained, frozen at random init       (no information)
  INVERTED  N1,N2 targets INFERRED by latency inversion   (target-prop trick)
            N_i target = (downstream target) - latency,  latency from the model

The latency is a property of the LIF impulse response (crossing lag at a
reference weight), estimable without observing the hidden neurons.
"""

import sys, os
sys.path.insert(0, "/workspace/project/gradient_study")
import numpy as np
from grad_recurrent import (NETS, full_sim, spikes_of, incoming, local_inputs,
                            train_recurrent, count_err, timing_err, T, OUTPUT)
from grad_method import lif_tangent as _lt


def estimate_latency(w=500.0):
    """Crossing lag of a single edge at weight w: input suprathreshold at t=0."""
    ia = np.zeros((1, T), bool); ia[0, 0] = True
    _, sp, _ = _lt(np.array([w]), ia, T)
    return sp[0] if sp else 71


def main():
    net = NETS["chain"]          # N0->N1->N2->N3
    Vt = full_sim(net, net["w"])
    T_true = {n: spikes_of(Vt, n) for n in range(4)}
    print("Chain true spikes:", {n: T_true[n] for n in range(4)})

    lat = estimate_latency(500.0)
    print(f"Estimated per-edge latency (model impulse response) = {lat} steps")

    # inverted targets for hidden neurons: peel back the output target by latency
    t3 = T_true[OUTPUT]
    t2_inv = [t - lat for t in t3]
    t1_inv = [t - 2 * lat for t in t3]
    print(f"Output N3 target: {t3}")
    print(f"  inferred N2 target (t3-{lat}): {t2_inv}   (true {T_true[2]})")
    print(f"  inferred N1 target (t3-{2*lat}): {t1_inv}   (true {T_true[1]})")

    conditions = {
        "ORACLE  (hidden = true times)":  ({1: T_true[1], 2: T_true[2], 3: t3}, [1, 2, 3]),
        "NO-INFO (hidden untrained)":     ({3: t3},                             [3]),
        "INVERTED(hidden = t-latency)":   ({1: t1_inv, 2: t2_inv, 3: t3},       [1, 2, 3]),
    }

    print("\n" + "=" * 70)
    for name, (targets, train_ns) in conditions.items():
        succ = 0; best = None
        for seed in range(6):
            w = train_recurrent(net, targets, seed=seed, train_neurons=train_ns,
                                iters=60, inner=30)
            V = full_sim(net, w)
            out_sp = spikes_of(V, OUTPUT)
            ok = len(out_sp) == len(t3)
            succ += int(ok)
            score = abs(len(out_sp) - len(t3))
            if best is None or score < best[0]:
                best = (score, out_sp, V)
        te = timing_err(best[2], {OUTPUT: t3}).get(OUTPUT, float("nan"))
        print(f"  {name:32s}: output-count OK {succ}/6   best N3={best[1]}  "
              f"(true {t3})  timing_err={te:.0f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
