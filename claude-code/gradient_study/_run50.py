"""Run the 50-neuron RECURRENT_CASES through the current trace method."""
import os, sys, dataclasses
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, "/workspace/project/gradient_study")
sys.path.insert(0, "/workspace/project")
import multiprocessing as mp


def _job(a):
    ci, seed, rounds = a
    sys.path.insert(0, "/workspace/project/gradient_study")
    sys.path.insert(0, "/workspace/project")
    import numpy as np, dataclasses
    import grad_trace as G          # installs the brian2/neuron_model stubs
    import jax_spiking_model as sim
    from test_cases import RECURRENT_CASES, _make_recurrent_weights
    tc = RECURRENT_CASES[ci]
    conns, tw = _make_recurrent_weights(tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
                                        tc["num_neurons"], tc["output_neurons"])
    C = np.array(conns, np.int32); W = np.array(tw, np.float32)
    N = tc["num_neurons"]; outs = list(tc["output_neurons"])
    params = dataclasses.replace(sim.default_params, steps=1000)
    T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
    w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(W))).astype(float)
    w = G.train(C, N, outs, w0.copy(), T, params, rounds=rounds, lr=G.LR)
    V = G.fsim(C, N, w, params)
    res = []
    for o in outs:
        f, t = G.sp(V, o), T[o]
        if f == t:
            res.append((o, "EXACT", 0.0, len(t), len(f)))
        elif len(f) == len(t):
            res.append((o, "count-ok", float(np.mean([abs(a - b) for a, b in zip(f, t)])),
                        len(t), len(f)))
        else:
            res.append((o, "COUNT", 99.0, len(t), len(f)))
    return ci, seed, N, len(W), res


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    ncase = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    jobs = [(ci, s, rounds) for ci in range(ncase) for s in range(seeds)]
    with mp.get_context("spawn").Pool(min(16, len(jobs))) as p:
        res = p.map(_job, jobs)
    print(f"50-neuron RECURRENT_CASES, {rounds} rounds\n")
    tot = ex = 0
    for ci, seed, N, ne, out in sorted(res):
        for o, tag, err, nt, nf in out:
            tot += 1; ex += int(tag == "EXACT")
            print(f"  case{ci} seed{seed}  N={N} edges={ne}  out N{o}: {tag:9s}"
                  f" targets={nt} found={nf}"
                  + (f"  mean|dt|={err:.1f}" if tag == "count-ok" else ""))
    print(f"\n  EXACT {ex}/{tot}")


if __name__ == "__main__":
    main()
