"""A FAIR test of hidden credit assignment: a multi-input readout with an
OFF-PATTERN target, so (a) retiming is actually possible (several inputs firing
at different times) and (b) the output count is NOT just handed over by the
periodic input.

Net:  N0(input) drives a staggered hidden chain H1->H2->H3->H4 (each fires ~71
      steps after the previous, so within a window they fire at spread times),
      and output O reads ALL of H1..H4.
The output target is chosen OFF the natural pattern (a specific subset/among the
hidden-driven crossing times), so O must weight the right hidden neurons AND the
hidden neurons must fire at the right times.

Conditions for the hidden neurons H1..H4:
  ORACLE    hidden trained to their true times
  NO-INFO   hidden untrained (frozen random)     <- the question
  READOUT   only O trained; hidden frozen but O sees all of them (can it select?)
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
import jax.numpy as jnp
import jax_spiking_model as sim
from grad_method import lif_tangent, TH
from grad_multi_neuron import voltage_grad

params = dataclasses.replace(sim.default_params, steps=340)
T = params.steps

# neurons: 0=input, 1..4 = hidden chain, 5 = output reading all hidden
#   0->1->2->3->4  and 1->5,2->5,3->5,4->5
C = np.array([[0, 1], [1, 2], [2, 3], [3, 4],
              [1, 5], [2, 5], [3, 5], [4, 5]], np.int32)
N = 6
INPUT = [0]
OUTPUT = 5
# true weights: chain strong; readout picks HALF the hidden neurons (2 and 4)
w_true = np.array([500., 500., 500., 500.,    # chain 0-1-2-3-4
                   40., 500., 40., 500.], np.float32)  # readouts 1->5..4->5


def full_sim(w):
    V, _, _ = sim.run_sim(params, jnp.array(C), N,
                          jnp.array(np.asarray(w, np.float32)), jnp.array(INPUT))
    return np.array(V)


def spikes_of(V, n):
    return np.where(V[:, n] >= TH)[0].tolist()


def incoming(n):
    idx = np.where(C[:, 1] == n)[0]
    return idx, C[idx, 0]


def local_inputs(V, pres):
    ia = np.zeros((len(pres), T), bool)
    for k, p in enumerate(pres):
        ia[k] = V[:, p] >= TH
    return ia


def train(targets, train_ns, seed, iters=60, inner=30, step=4.0, alpha=0.5):
    rng = np.random.default_rng(seed)
    w = (w_true * rng.uniform(0.5, 1.5, len(w_true))).astype(np.float64)
    for _ in range(iters):
        V = full_sim(w)
        for n in train_ns:
            if n not in targets:
                continue
            syn, pres = incoming(n)
            ia = local_inputs(V, pres)
            wl = w[syn].copy()
            for _ in range(inner):
                g = voltage_grad(wl, ia, targets[n], T, suppress=True)[0]
                gn = np.linalg.norm(g)
                if gn > 1e-30:
                    wl = np.clip(wl - step * g / gn, 20, 3000)
            w[syn] = (1 - alpha) * w[syn] + alpha * wl
    return w


def main():
    Vt = full_sim(w_true)
    T_true = {n: spikes_of(Vt, n) for n in range(N)}
    for n in range(N):
        print(f"  true N{n}: {T_true[n]}")
    t_out = T_true[OUTPUT]
    print(f"Output target (OFF-pattern, set by which hidden fire): {t_out}")

    hidden = [1, 2, 3, 4]
    conds = {
        "ORACLE  (hidden = true times)": ({**{h: T_true[h] for h in hidden}, OUTPUT: t_out},
                                          hidden + [OUTPUT]),
        "NO-INFO (hidden frozen random)": ({OUTPUT: t_out}, [OUTPUT]),
        "READOUT (only O trained, sees all hidden)": ({OUTPUT: t_out}, [OUTPUT]),
    }
    print("\n" + "=" * 70)
    for name, (targets, train_ns) in conds.items():
        succ = 0; bestsp = None; bestscore = 1e9
        for seed in range(6):
            w = train(targets, train_ns, seed)
            V = full_sim(w)
            sp = spikes_of(V, OUTPUT)
            ok = len(sp) == len(t_out)
            succ += int(ok)
            sc = abs(len(sp) - len(t_out))
            if sc < bestscore:
                bestscore = sc; bestsp = sp
        print(f"  {name:42s}: count OK {succ}/6   best N5={bestsp} (true {t_out})")
    print("=" * 70)


if __name__ == "__main__":
    main()
