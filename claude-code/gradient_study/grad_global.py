"""Global-gradient ("ML model") training instead of greedy local solves.

One loss over EVERY neuron (deep supervision to the true spike trains), one
gradient through the whole recurrent net via the differentiable soft-sim, Adam.
Unlike per-neuron target prop, this gradient accounts for coupling: changing an
upstream weight is credited for its downstream effect through backprop.

Fixed beta (no schedule).  Compare output-count recovery to the local target-prop
collapse (1-12/50) and to soft homotopy's best.
"""

import sys, os, dataclasses, types
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
import jax, jax.numpy as jnp
from homotopy_core import soft_sim, hard_sim as _hard_sim
import jax_spiking_model as sim
from test_cases import RECURRENT_CASES, _make_recurrent_weights

BETA = float(os.environ.get("BETA", "12"))
TAU = 20.0
ITERS = int(os.environ.get("ITERS", "400"))
SEEDS = int(os.environ.get("SEEDS", "2"))


def fwd_exp_conv(x, decay):
    def step(c, xt):
        c = c * decay + xt
        return c, c
    _, S = jax.lax.scan(step, jnp.zeros(x.shape[-1]), x)
    return S


def run_case(ci):
    tc = RECURRENT_CASES[ci]
    conns, tw = _make_recurrent_weights(tc["topo_seed"], tc["p_connect"], tc["trial_idx"],
                                        tc["num_neurons"], tc["output_neurons"])
    params = dataclasses.replace(sim.default_params, steps=1000)
    C = jnp.array(np.array(conns, np.int32)); N = tc["num_neurons"]; A = jnp.array([0])
    th = params.threshold; outs = tc["output_neurons"]
    w_true = jnp.array(tw, np.float32)
    lo, hi = w_true * 0.1, w_true * 5.0
    decay = jnp.float32(np.exp(-1.0 / TAU))

    target_v = np.array(_hard_sim(w_true, params, C, N, A))
    T_true = {n: np.where(target_v[:, n] >= th)[0].tolist() for n in range(N)}

    # deep-supervision target: soft van-Rossum trace of ALL neurons at true weights
    v_soft_true = soft_sim(w_true, jnp.float32(BETA), params, C, N, A)
    S_target = fwd_exp_conv(jax.nn.sigmoid(BETA * (v_soft_true / th - 1.0)), decay)

    @jax.jit
    def loss(w):
        v = soft_sim(w, jnp.float32(BETA), params, C, N, A)
        S = fwd_exp_conv(jax.nn.sigmoid(BETA * (v / th - 1.0)), decay)
        return jnp.sum((S - S_target) ** 2)                 # ALL neurons
    vg = jax.jit(jax.value_and_grad(loss))

    best = None
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        w = w_true * jnp.array(rng.uniform(0.5, 1.5, len(tw)), jnp.float32)
        m = jnp.zeros_like(w); v = jnp.zeros_like(w)
        for t in range(1, ITERS + 1):
            l, g = vg(w)
            g = jnp.nan_to_num(g)
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
            lr = 2.0 if t < ITERS * 0.7 else 0.5
            w = jnp.clip(w - lr * mh / (jnp.sqrt(vh) + 1e-12), lo, hi)
        Vh = np.array(_hard_sim(w, params, C, N, A))
        sp = {n: int(np.sum(Vh[:, n] >= th)) for n in outs}
        cnt = sum(sp[n] == len(T_true[n]) for n in outs)
        net = sum(1 for n in range(N) if int(np.sum(Vh[:, n] >= th)) == len(T_true[n]))
        outloss = float(sum(np.sum((target_v[:, n] - Vh[:, n]) ** 2) for n in outs))
        if best is None or cnt > best[0] or (cnt == best[0] and outloss < best[3]):
            best = (cnt, sp, net, outloss)
    tgt = {n: len(T_true[n]) for n in outs}
    return tc["name"], tgt, best


def main():
    print(f"Global-gradient deep supervision (BETA={BETA}, iters={ITERS}, seeds={SEEDS})")
    print(f"{'case':20s} {'target':>10s}  {'output':>12s} {'cnt':>4s} {'net':>6s} {'outloss':>9s}")
    for ci in range(3):
        name, tgt, (cnt, sp, net, outloss) = run_case(ci)
        ts = "/".join(str(tgt[n]) for n in tgt); ss = "/".join(str(sp[n]) for n in sp)
        print(f"{name:20s} {ts:>10s}  {ss:>12s} {cnt}/3 {net:>3}/50 {outloss:>9.2e}")
    print("\nref: soft homotopy case0 EXACT 7/2/1, case1 EXACT 4/2/7, case2 unsolved;  "
          "local TP net 1-12/50")


if __name__ == "__main__":
    main()
