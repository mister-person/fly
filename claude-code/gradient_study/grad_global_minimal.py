"""Minimal example that fails the GLOBAL gradient (deep supervision, no schedule).

Timing-fork topology: a slow path and a fast bypass to the same neuron create a
double-well; global gradient descent from random init can fall into the wrong
(fast) basin and stay.  Try to shrink it to the smallest failing net and show the
local minimum (loss along the true->found axis).
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

params = dataclasses.replace(sim.default_params, steps=400)
TH = params.threshold
A = jnp.array([0])
BETA = float(os.environ.get("BETA", "12"))
TAU = 20.0
ITERS = 500


def fwd_exp_conv(x, decay):
    def step(c, xt):
        c = c * decay + xt
        return c, c
    _, S = jax.lax.scan(step, jnp.zeros(x.shape[-1]), x)
    return S


def sp_of(V, n):
    return np.where(V[:, n] >= TH)[0].tolist()


def globalgrad(C_np, N, w_true_np, seeds=6, anneal=False):
    C = jnp.array(C_np, jnp.int32)
    w_true = jnp.array(w_true_np, jnp.float32)
    lo, hi = w_true * 0.1, w_true * 5.0
    decay = jnp.float32(np.exp(-1.0 / TAU))
    tv = np.array(_hard_sim(w_true, params, C, N, A))
    T_true = {n: sp_of(tv, n) for n in range(N)}

    def make_loss(beta):
        vst = soft_sim(w_true, jnp.float32(beta), params, C, N, A)
        Stg = fwd_exp_conv(jax.nn.sigmoid(beta * (vst / TH - 1.0)), decay)
        @jax.jit
        def loss(w):
            v = soft_sim(w, jnp.float32(beta), params, C, N, A)
            S = fwd_exp_conv(jax.nn.sigmoid(beta * (v / TH - 1.0)), decay)
            return jnp.sum((S - Stg) ** 2)          # deep supervision, all neurons
        return jax.jit(jax.value_and_grad(loss))

    betas = [BETA] if not anneal else [0.5, 1, 2, 3, 5, 8, 13, 21, 34]
    best = None
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        w = w_true * jnp.array(rng.uniform(0.5, 1.5, len(w_true_np)), jnp.float32)
        for beta in betas:
            vg = make_loss(beta)
            m = jnp.zeros_like(w); v = jnp.zeros_like(w)
            for t in range(1, (ITERS // len(betas)) + 1):
                l, g = vg(w); g = jnp.nan_to_num(g)
                m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
                mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
                w = jnp.clip(w - 1.0 * mh / (jnp.sqrt(vh) + 1e-12), lo, hi)
        Vh = np.array(_hard_sim(w, params, C, N, A))
        ol = float(sum(np.sum((tv[:, n] - Vh[:, n]) ** 2) for n in range(N)))
        match = all(sp_of(Vh, n) == T_true[n] for n in range(N))
        if best is None or ol < best[0]:
            best = (ol, np.array(w), {n: sp_of(Vh, n) for n in range(N)}, match)
    return T_true, best, tv


NETS = {
    "3-neuron fork  N0->N1->N2 + bypass N0->N2":
        (np.array([[0, 1], [1, 2], [0, 2]], np.int32), 3, np.array([500., 500., 50.], np.float32)),
    "4-neuron fork  N0->N1->N2->N3 + bypass N0->N3":
        (np.array([[0, 1], [1, 2], [2, 3], [0, 3]], np.int32), 4, np.array([500., 500., 500., 50.], np.float32)),
}


def main():
    for anneal in [False, True]:
        print(f"\n=== GLOBAL gradient, deep supervision, "
              f"{'ANNEALED beta' if anneal else f'fixed beta={BETA}'} ===")
        for name, (C, N, wt) in NETS.items():
            T_true, (ol, wf, spf, match), tv = globalgrad(C, N, wt, anneal=anneal)
            out = N - 1
            tag = "RECOVERED" if match else "FAILED"
            print(f"  {name}")
            print(f"     {tag}: out N{out} target {T_true[out]}  found {spf[out]}  "
                  f"(full-net loss {ol:.2e})")


if __name__ == "__main__":
    main()
