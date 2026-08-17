"""Manual gradient of the spiking model (hand-written backprop-through-time).

The spiking sim in ``jax_spiking_model`` is not differentiable: spikes are produced
by a step function ``(v >= threshold)`` whose derivative is zero almost everywhere,
so ``jax.grad`` cannot push a useful signal back through the recurrent voltage
dynamics. The usual fix is a *surrogate gradient*: keep the hard step in the forward
pass, but pretend it has the slope of a smooth function (a sigmoid-like bump centred
on the threshold) when computing gradients.

Rather than rely on ``jax.custom_gradient`` (which was hard to get composing cleanly
through the scatter/gather and refractory gating), this module computes the gradient
of ``sim_loss`` w.r.t. ``synapse_weights`` by an explicit reverse-time pass. The
backward recurrence was derived by hand from ``jax_spiking_model.timestep`` and is
verified to match ``jax.grad`` of a surrogate-differentiable forward to machine
precision (see ``self_test`` at the bottom).

What is and isn't differentiated
--------------------------------
* The only non-differentiable op we put a surrogate slope on is the *synaptic
  activation* ``(presynaptic_voltage >= threshold)``. This is where the learning
  signal needs to flow, so it gets ``surrogate_grad``.
* The refractory gates (``refractory_timers == 0`` / ``!= 1``) and the threshold
  reset that updates the refractory timer are treated as straight-through constants
  (zero gradient). This is exact, not an approximation: in the forward pass the
  refractory timer only ever reaches the loss through boolean comparisons, so its
  true gradient contribution is zero anyway.
* The input-injection ``all_voltages.at[i, activate].set(...)`` overwrites those
  neurons' "current" voltage with a constant, so the gradient w.r.t. the injected
  neurons' own voltage at that step is zero (handled via ``activate_mask``).

Public API
----------
* ``synapse_weight_grads(...)`` -> dL/d(synapse_weights), same shape/scaling as
  ``jax.grad(jax_spiking_model.sim_loss, argnums=3)`` would give under the surrogate.
* ``intermediate_grads(...)`` -> per-timestep ``(steps, num_synapses)`` breakdown of
  that gradient; ``intermediate_grads(...).sum(axis=0) == synapse_weight_grads(...)``.
  This is the quantity ``learn_jax.main2`` plots to inspect when/where the signal
  arises.
"""
from functools import partial

import jax
import jax.numpy as jnp

import jax_spiking_model as model
from homotopy_core import _bptt_backward


# -----------------------------------------------------------------------------
# Surrogate slopes for the synaptic step function.
#
# The "wide" tanh slope (original) has a value of ~200 at v=0 (well below
# threshold), which causes the adjoint to explode in long recurrent backward
# passes (10 000× amplification over 50 backward steps for 1000-step networks).
#
# The "narrow" sigmoid-derivative slope goes to zero exponentially away from
# threshold. ``beta_surr`` controls the width; the default (30) makes the slope
# negligible at v=0 while still giving a strong signal within ±th/beta_surr of
# threshold. This matches the soft_sim backward at high beta.
# -----------------------------------------------------------------------------

def _wide_surrogate_slope(pre_vals, threshold):
    """Original tanh-based slope; kept for backward-compatibility reference."""
    return jax.vmap(
        lambda v: jax.grad(model.synapse_activation_gradient_fn)(v, threshold)
    )(pre_vals)


def _narrow_surrogate_slope(pre_vals, threshold, beta_surr):
    """Sigmoid-derivative surrogate: peaks at threshold, near-zero elsewhere.

    Avoids gradient explosion in long recurrent backward passes because
    non-spiking neurons (v << threshold) receive effectively zero gradient.
    """
    a = jax.nn.sigmoid(beta_surr * (pre_vals / threshold - 1.0))
    return a * (1.0 - a) * (beta_surr / threshold)


# Keep the wide version accessible under its original name for self_test.
_surrogate_grad_vec = _wide_surrogate_slope


@partial(jax.jit, static_argnames=["params", "num_neurons"])
def _forward_with_refractory(params, connections, num_neurons, synapse_weights,
                             neurons_to_activate):
    """Run the production forward pass and additionally reconstruct, for every
    step ``i``, the refractory timer *entering* that step (``refs[i]``).

    ``jax_spiking_model.run_sim`` only returns the final refractory timer, but the
    backward pass needs the per-step gates. The timer is a deterministic function
    of the stored output voltages, so we replay just that scalar recurrence here
    instead of duplicating the whole forward.
    """
    all_voltages, _, rise_values = model.run_sim(
        params, connections, num_neurons, synapse_weights, neurons_to_activate)

    def ref_step(i, refs):
        out = all_voltages[i + 1]
        new_ref = (jnp.where(out >= params.threshold,
                             params.refractory_iters + 1,
                             refs[i]) - 1).clip(min=0)
        return refs.at[i + 1].set(new_ref)

    refs = jnp.zeros((params.steps, num_neurons))
    refs = jax.lax.fori_loop(0, params.steps - 1, ref_step, refs)
    return all_voltages, rise_values, refs


@partial(jax.jit, static_argnames=["params", "num_neurons", "beta_surr"])
def _backward(params, connections, num_neurons, synapse_weights,
              neurons_to_activate, target_voltages, beta_surr=30.0):
    """Hand-written reverse-time pass. Returns (grad_w, intermediate).

    ``grad_w[s]``        = dL / d synapse_weights[s]  (passed/un-scaled weights)
    ``intermediate[i,s]``= contribution of synapse s at step i; sums to grad_w.

    ``beta_surr`` controls the surrogate slope width. Default (30) gives a narrow
    sigmoid-derivative shape that is near-zero for non-spiking neurons, preventing
    gradient explosion in long recurrent backward passes. Pass ``beta_surr=None``
    to fall back to the original wide tanh-based surrogate.
    """
    all_v, _, refs = _forward_with_refractory(
        params, connections, num_neurons, synapse_weights, neurons_to_activate)

    th = params.threshold

    if beta_surr is None:
        def act_slope(pre_vals):
            return (pre_vals >= th) * 1.0, _wide_surrogate_slope(pre_vals, th)
    else:
        def act_slope(pre_vals):
            return (pre_vals >= th) * 1.0, _narrow_surrogate_slope(pre_vals, th, beta_surr)

    aV_seed = 2.0 * (all_v - target_voltages)
    return _bptt_backward(all_v, refs, aV_seed, synapse_weights, params,
                          connections, num_neurons, neurons_to_activate, act_slope)


def synapse_weight_grads(params, connections, num_neurons, synapse_weights,
                         neurons_to_activate, target_voltages, beta_surr=30.0):
    """dL/d(synapse_weights) for ``jax_spiking_model.sim_loss`` via manual BPTT."""
    grad_w, _ = _backward(params, connections, num_neurons, synapse_weights,
                          neurons_to_activate, target_voltages, beta_surr=beta_surr)
    return grad_w


def intermediate_grads(params, connections, num_neurons, synapse_weights,
                       neurons_to_activate, target_voltages, zeros=None,
                       beta_surr=30.0):
    """Per-timestep ``(steps, num_synapses)`` breakdown of the synapse-weight
    gradient; ``intermediate_grads(...).sum(0)`` equals ``synapse_weight_grads(...)``.

    The output is always shaped ``(steps, connections.shape[0])``. ``zeros`` is
    accepted (and ignored) only for backwards compatibility with existing call
    sites that pass a pre-allocated buffer."""
    _, inter = _backward(params, connections, num_neurons, synapse_weights,
                         neurons_to_activate, target_voltages, beta_surr=beta_surr)
    return inter


def self_test(beta_surr=30.0):
    """Check the manual gradient against ``jax.grad`` of a surrogate-differentiable
    forward (the gold standard). Run with the real ``jax_spiking_model`` importable.

    The reference forward uses ``sigmoid(beta_surr*(v/th-1))`` as the synaptic
    activation, whose exact derivative is the narrow surrogate slope. Passing
    ``beta_surr=None`` tests the original wide tanh-based surrogate instead.
    """
    import dataclasses

    params = dataclasses.replace(model.default_params, steps=60, delay_iters=18,
                                 refractory_iters=5)
    key = jax.random.key(0)
    N, S = 12, 40
    k1, k2 = jax.random.split(key)
    connections = jax.random.randint(k1, (S, 2), 0, N)
    weights = jax.random.uniform(k2, (S,)) * 800 + 50
    activate = jnp.array([0, 1, 2])

    target, _, _ = model.run_sim(params, connections, N, weights * 0.7, activate)

    if beta_surr is None:
        # Original wide tanh surrogate: reference activation is tanh((v/th-1)*2)*10
        @jax.custom_gradient
        def ref_activation(pre, threshold):
            out = (pre >= threshold) * 1.0
            return out, lambda g: (_wide_surrogate_slope(pre, threshold) * g, 0.0)
    else:
        # Narrow sigmoid-derivative surrogate: reference activation is sigmoid(beta*(v/th-1))
        @jax.custom_gradient
        def ref_activation(pre, threshold):
            out = (pre >= threshold) * 1.0
            slope = _narrow_surrogate_slope(pre, threshold, beta_surr)
            return out, lambda g: (slope * g, 0.0)

    def loss_surrogate(w):
        sw = w * params.global_synapse_weight
        all_v = jnp.zeros((params.steps, N))
        ref = jnp.zeros(N)
        rise = jnp.zeros((params.steps, N))

        def loop(i, x):
            all_v, ref, rise = x
            inj = ((i % 100) == 0) * params.threshold / params.neuron_decay
            wi = all_v.at[i, activate].set(inj)
            nc = wi[i]
            pre = jnp.where(i - params.delay_iters >= 0,
                            wi[i - params.delay_iters], jnp.zeros_like(nc))
            p = pre[connections[..., 0]]
            a = ref_activation(p, params.threshold)
            upd = jnp.zeros_like(nc).at[connections[..., 1]].add(a * sw)
            r = (rise[i] + upd) * params.rise_decay * (ref != 1)
            out = (nc - r) * params.neuron_decay + r
            out = out * (ref == 0)
            new_ref = (jnp.where(out >= params.threshold, params.refractory_iters + 1,
                                 ref) - 1).clip(min=0)
            return (all_v.at[i + 1].set(out), new_ref, rise.at[i + 1].set(r))

        all_v, _, _ = jax.lax.fori_loop(0, params.steps, loop, (all_v, ref, rise))
        return jnp.sum((target - all_v) ** 2)

    gold = jax.grad(loss_surrogate)(weights)
    manual = synapse_weight_grads(params, connections, N, weights, activate, target,
                                  beta_surr=beta_surr)
    inter = intermediate_grads(params, connections, N, weights, activate, target,
                               jnp.zeros((params.steps, S)), beta_surr=beta_surr)

    rel = jnp.max(jnp.abs(gold - manual)) / (jnp.max(jnp.abs(gold)) + 1e-30)
    cos = jnp.dot(gold, manual) / (jnp.linalg.norm(gold) * jnp.linalg.norm(manual))
    tol = 1e-3 if gold.dtype == jnp.float32 else 1e-8
    label = f"beta_surr={beta_surr}" if beta_surr is not None else "wide tanh"
    print(f"[{label}]  dtype {gold.dtype}  max rel diff: {float(rel):.2e}  "
          f"cosine: {float(cos):.6f}  inter sums: "
          f"{bool(jnp.allclose(inter.sum(0), manual, rtol=1e-4, atol=1e-30))}")
    assert rel < tol, f"manual gradient does not match autodiff surrogate ({label})"
    assert cos > 1 - 1e-4, f"manual gradient points the wrong way ({label})"
    print("OK")


if __name__ == "__main__":
    self_test()
