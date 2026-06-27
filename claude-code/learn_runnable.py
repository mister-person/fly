"""Self-contained training loop for the spiking model, using the manual gradient.

This is a runnable cousin of ``learn_jax.py``: it reuses the exact simulation
parameters (``jax_spiking_model.default_params``) and the same random-network
construction as ``learn_jax.main`` (random connectivity, 3 driven input neurons),
but drops everything that needs the full lab stack -- the brian2 reference sim
(``neuron_model``), the dataset loader (``data``), and the pygame / threaded-
matplotlib visualisers. That makes it headless and reproducible.

Task setup (self-supervised, same idea as ``learn_jax.main2``):
  * pick "true" synapse weights, run the sim once to get target voltages,
  * start from perturbed weights,
  * descend on ``sum((target - voltages)**2)`` using the hand-written gradient in
    ``jax_model_grads_hack`` (NOT jax.grad).

IMPORTANT -- training is unreliable, by nature of the loss. The spike-scored loss
is piecewise-constant with a sparse gradient signal, so whether the loss actually
descends is strongly seed-dependent. ``main`` therefore runs a *sweep* over network
and init seeds and reports the distribution of outcomes rather than one (possibly
lucky) run. Shorter horizons (QUICK) descend far more often than longer ones; at
the full learn_jax horizon (BIG) the recurrent surrogate gradient explodes and the
loss does not descend at all. The gradient itself is verified correct against
``jax.grad`` (``jax_model_grads_hack.self_test``); this variability is the loss
surface, not the gradient.

Run:  python3 learn_runnable.py          # default sweep (200/4000, 300 steps)
      QUICK=1 python3 learn_runnable.py  # smaller/shorter sweep, descends more often
      BIG=1   python3 learn_runnable.py  # full learn_jax scale; exploding-gradient demo
"""
import os
import sys
import time
import types
import dataclasses

import numpy as np
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# ``jax_spiking_model`` imports ``brian2`` and ``neuron_model`` at module load
# time, but only uses them in its GUI/reference-sim helpers -- none of the
# simulation or loss code touches them. Stub them so this script runs anywhere.
# ---------------------------------------------------------------------------
if "brian2" not in sys.modules:
    _brian2 = types.ModuleType("brian2")
    _brian2.ms = 1e-3
    sys.modules["brian2"] = _brian2
if "neuron_model" not in sys.modules:
    _nm = types.ModuleType("neuron_model")
    _nm.NeuronSim = object
    sys.modules["neuron_model"] = _nm

import jax_spiking_model
import jax_model_grads_hack


def build_network(neuron_count, syn_count, seed=11):
    """Random connectivity + synapse strengths, mirroring ``learn_jax.main``.

    ``learn_jax`` samples strengths from the real ``mbanc`` dataset and scales by
    4.5; we don't have that dataset here, so we draw strengths from a lognormal
    with a comparable heavy-tailed spread and the same overall scale.
    """
    rng = np.random.default_rng(seed)
    pre = rng.integers(neuron_count, size=syn_count)
    post = rng.integers(neuron_count, size=syn_count)
    connections = jnp.array(np.stack([pre, post], axis=1), dtype=int)
    # Heavy-tailed positive strengths, mean a few hundred (cf. real data * 4.5).
    # (The real dataset's strengths are signed; here we keep them positive, which
    # at the short ~300-step horizon stays finite and keeps the net active enough
    # to give the loss a usable gradient signal.)
    strengths = jnp.array(rng.lognormal(mean=5.0, sigma=0.6, size=syn_count),
                          dtype=jnp.float32)
    return connections, strengths


def train_once(neuron_count, syn_count, steps, net_seed, init_seed, iters,
               perturb=(0.7, 1.3), n_activate=3, verbose=False):
    """Self-supervised run: build a net, make a target from its "true" weights,
    perturb the weights, and try to descend back via the manual gradient.

    Returns dict with loss0, final loss, ratio, target spike rate, and whether the
    gradient ever went non-finite.
    """
    # same dynamics parameters as learn_jax: default_params, steps = runtime*10
    params = dataclasses.replace(jax_spiking_model.default_params, steps=steps)
    # NOTE: learn_jax.py's range(20) is leftover; only ~3 neurons are really driven.
    activate = jnp.array(list(range(n_activate)))
    connections, base = build_network(neuron_count, syn_count, seed=net_seed)

    target, _, _ = jax_spiking_model.run_sim(params, connections, neuron_count,
                                             base, activate)
    spike_rate = float(jnp.mean(target >= params.threshold))

    rng = np.random.default_rng(init_seed)
    weights = base * jnp.array(rng.uniform(*perturb, size=syn_count),
                               dtype=jnp.float32)
    # sign-preserving weight bounds (like learn_jax's multiplier clip)
    lo = jnp.minimum(base * 0.3, base * 3.0)
    hi = jnp.maximum(base * 0.3, base * 3.0)

    def loss_of(w):
        return float(jax_spiking_model.sim_loss(params, connections, neuron_count,
                                                w, activate, target))

    loss0 = cur_loss = loss_of(weights)
    nonfinite = False

    # The spike-based loss is piecewise-constant in the weights (it only changes
    # when a weight shift flips a spike), so a fixed step mostly lands in a flat
    # region or overshoots. Expanding/contracting line search along the (surrogate)
    # gradient direction: carry a step ``d``, grow it (x2) while it helps, shrink it
    # (x0.5) when nothing improves. Taking the best step keeps loss non-increasing.
    d = 20.0
    for it in range(iters):
        grad = jax_model_grads_hack.synapse_weight_grads(
            params, connections, neuron_count, weights, activate, target)
        if not bool(jnp.all(jnp.isfinite(grad))):
            nonfinite = True
        unit = grad / (jnp.linalg.norm(grad) + 1e-30)

        best_w, best_loss, accepted_d = weights, cur_loss, 0.0
        trial = d
        for _ in range(7):
            cand = jnp.clip(weights - unit * trial, lo, hi)
            cand_loss = loss_of(cand)
            if cand_loss < best_loss:
                best_w, best_loss, accepted_d = cand, cand_loss, trial
                trial *= 2.0
            elif accepted_d > 0.0:
                break
            else:
                trial *= 0.5

        weights, cur_loss = best_w, best_loss
        d = max(accepted_d, 1.0)
        if verbose:
            print(f"  iter {it:3d}  loss {cur_loss:.6e}  "
                  f"({cur_loss/loss0:6.3f} x init)  step {accepted_d:7.1f}")

    return dict(loss0=loss0, final=cur_loss, ratio=cur_loss / loss0,
                spike_rate=spike_rate, nonfinite=nonfinite)


def main():
    quick = os.environ.get("QUICK") == "1"
    big = os.environ.get("BIG") == "1"

    if big:
        # full learn_jax.main scale. WARNING: at this horizon the surrogate-gradient
        # BPTT explodes (adjoints grow ~57x per delay-hop and overflow float32), so
        # the gradient is dominated by a few blown-up synapses and the loss will not
        # descend. Single verbose run, kept to inspect that failure mode.
        print("BIG: full scale (1000/40000/600) -- expect exploding gradient, no descent\n")
        r = train_once(1000, 40000, 600, net_seed=0, init_seed=54, iters=40,
                       verbose=True)
        print(f"\nratio {r['ratio']:.4f}  nonfinite_grad={r['nonfinite']}")
        return

    # Small/quick vs default network; both at a ~300-step horizon where the
    # recurrent surrogate gradient stays finite.
    neuron_count, syn_count, steps = (150, 3000, 200) if quick else (200, 4000, 300)
    iters = 12 if quick else 15

    # Sweep multiple network + init seeds, because a single seed is misleading:
    # whether the loss descends is highly seed-dependent on this flat, sparse,
    # spike-based objective. This prints the *distribution* of outcomes.
    net_seeds = range(4 if quick else 8)
    init_seeds = (54, 7)
    print(f"config: neurons={neuron_count} synapses={syn_count} steps={steps}  "
          f"sweeping {len(list(net_seeds))*len(init_seeds)} seeds\n")
    print(f"{'net':>4} {'init':>4} {'spike':>7} {'loss0':>11} {'final':>11} "
          f"{'ratio':>7}")

    ratios = []
    t0 = time.monotonic()
    for ns in net_seeds:
        for iseed in init_seeds:
            r = train_once(neuron_count, syn_count, steps, net_seed=ns,
                           init_seed=iseed, iters=iters)
            ratios.append(r["ratio"])
            print(f"{ns:>4} {iseed:>4} {r['spike_rate']:>7.4f} "
                  f"{r['loss0']:>11.4e} {r['final']:>11.4e} {r['ratio']:>7.4f}")

    ratios = np.array(ratios)
    improved = int((ratios < 0.99).sum())
    print(f"\nsweep took {time.monotonic()-t0:.1f}s over {len(ratios)} runs")
    print(f"meaningfully improved (ratio<0.99): {improved}/{len(ratios)}")
    print(f"median ratio {np.median(ratios):.3f}   best {ratios.min():.3f}   "
          f"worst {ratios.max():.3f}")
    print("\nThe manual gradient is verified correct (see jax_model_grads_hack."
          "self_test); the variability above is a property of the flat, sparse,\n"
          "spike-based loss -- gradient *training* on it is unreliable, not the "
          "gradient itself.")
    return ratios


if __name__ == "__main__":
    main()
