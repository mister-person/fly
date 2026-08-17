"""Fix the minimal global-gradient failure with a barrier-free GLOBAL loss.

The van-Rossum (spike) loss has a creation barrier -> the global gradient can't add
the missing spikes (N2 collapses, N3 dies).  Replace it with the VOLTAGE-TARGET
loss applied to EVERY neuron: at each target time push the (soft) voltage up to
threshold; elsewhere keep it below.  V is monotonic in the drive, so there is no
creation barrier.  Same global soft-sim gradient, deep supervision, no schedule.
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
MARGIN = 0.12
LAM = 1.0
WIN = 8


def sp_of(V, n):
    return np.where(V[:, n] >= TH)[0].tolist()


def globalgrad_vt(C_np, N, wt_np, seeds=6, iters=800):
    C = jnp.array(C_np, jnp.int32); w_true = jnp.array(wt_np, jnp.float32)
    lo, hi = w_true * 0.1, w_true * 5.0
    tv = np.array(_hard_sim(w_true, params, C, N, A))
    T_true = {n: sp_of(tv, n) for n in range(N)}
    Tsteps = params.steps
    # target mask (reach th) and suppression mask (stay below), per neuron
    tgt = np.zeros((Tsteps, N)); supp = np.ones((Tsteps, N))
    for n in range(N):
        for t in T_true[n]:
            tgt[t, n] = 1.0
        for t in T_true[n]:
            supp[max(0, t - WIN):t + WIN, n] = 0.0     # don't suppress near a target
    tgt = jnp.array(tgt, jnp.float32); supp = jnp.array(supp, jnp.float32)
    cap = (1.0 - MARGIN) * TH

    @jax.jit
    def loss(w):
        v = soft_sim(w, jnp.float32(BETA), params, C, N, A)     # soft voltages, all neurons
        create = jnp.sum(tgt * jax.nn.relu(TH - v) ** 2)         # reach threshold at targets
        suppress = jnp.sum(supp * jax.nn.relu(v - cap) ** 2)     # stay below elsewhere
        return create + LAM * suppress
    vg = jax.jit(jax.value_and_grad(loss))

    best = None
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        w = w_true * jnp.array(rng.uniform(0.5, 1.5, len(wt_np)), jnp.float32)
        m = jnp.zeros_like(w); v = jnp.zeros_like(w)
        for t in range(1, iters + 1):
            l, g = vg(w); g = jnp.nan_to_num(g)
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
            w = jnp.clip(w - 3.0 * mh / (jnp.sqrt(vh) + 1e-12), lo, hi)
        Vh = np.array(_hard_sim(w, params, C, N, A))
        match = all(sp_of(Vh, n) == T_true[n] for n in range(N))
        ol = float(sum(np.sum((tv[:, n] - Vh[:, n]) ** 2) for n in range(N)))
        if best is None or ol < best[0]:
            best = (ol, {n: sp_of(Vh, n) for n in range(N)}, match)
    return T_true, best


NETS = {
    "3-neuron fork": (np.array([[0, 1], [1, 2], [0, 2]], np.int32), 3, np.array([500., 500., 50.], np.float32)),
    "4-neuron fork": (np.array([[0, 1], [1, 2], [2, 3], [0, 3]], np.int32), 4, np.array([500., 500., 500., 50.], np.float32)),
    "4-neuron feedback": (np.array([[0, 1], [1, 2], [2, 1], [2, 3]], np.int32), 4, np.array([500., 500., 50., 500.], np.float32)),
}


def main():
    print(f"GLOBAL gradient with VOLTAGE-TARGET loss (deep supervision, beta={BETA}, no schedule)")
    for name, (C, N, wt) in NETS.items():
        T_true, (ol, spf, match) = globalgrad_vt(C, N, wt)
        out = N - 1
        print(f"  {name}: {'RECOVERED' if match else 'FAILED'}  "
              f"out N{out} target {T_true[out]} found {spf[out]}  (loss {ol:.2e})")


if __name__ == "__main__":
    main()
