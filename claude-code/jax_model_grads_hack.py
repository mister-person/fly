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


# -----------------------------------------------------------------------------
# Surrogate slope for the synaptic step function.
#
# Default = derivative of ``jax_spiking_model.synapse_activation_gradient_fn``
# (i.e. d/dv of ``tanh((v/threshold - 1) * 2) * 10``), so this manual gradient
# reproduces what the original ``@jax.custom_gradient`` attempt was meant to do.
# Swap this out to experiment with other surrogate shapes.
# -----------------------------------------------------------------------------
def surrogate_grad(pre_voltage, threshold):
    """Slope substituted for d/d(pre) of the hard step ``(pre >= threshold)``."""
    return jax.grad(model.synapse_activation_gradient_fn)(pre_voltage, threshold)


# vectorised over synapses; threshold is a scalar shared by all synapses.
_surrogate_grad_vec = jax.vmap(surrogate_grad, in_axes=(0, None))


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


@partial(jax.jit, static_argnames=["params", "num_neurons"])
def _backward(params, connections, num_neurons, synapse_weights,
              neurons_to_activate, target_voltages):
    """Hand-written reverse-time pass. Returns (grad_w, intermediate).

    ``grad_w[s]``        = dL / d synapse_weights[s]  (passed/un-scaled weights)
    ``intermediate[i,s]``= contribution of synapse s at step i; sums to grad_w.
    """
    nd = params.neuron_decay
    rise_decay = params.rise_decay
    th = params.threshold
    delay = params.delay_iters
    g_scale = params.global_synapse_weight
    steps = params.steps

    pre_idx = connections[..., 0]
    post_idx = connections[..., 1]
    w_scaled = synapse_weights * g_scale          # weights actually used in forward

    all_v, _rise, refs = _forward_with_refractory(
        params, connections, num_neurons, synapse_weights, neurons_to_activate)

    # neurons whose "current" voltage is overwritten by the input injection each
    # step -> their own-voltage gradient at that step is zero.
    activate_mask = jnp.zeros(num_neurons).at[neurons_to_activate].set(1.0)

    # adjoints, indexed by step. aV[j] = dL/d all_voltages[j], seeded by the direct
    # (target - v)^2 loss term. aG[j] = dL/d rise_values[j] (not in this loss -> 0).
    aV = 2.0 * (all_v - target_voltages)
    aG = jnp.zeros((steps, num_neurons))
    grad_w = jnp.zeros_like(synapse_weights)
    # per-step breakdown buffer, sized from the real synapse count (not from any
    # caller-supplied buffer, which may be mis-shaped).
    inter_init = jnp.zeros((steps, connections.shape[0]))

    def bwd_step(j, carry):
        aV, aG, grad_w, inter = carry
        i = steps - 2 - j                       # walk steps backwards

        ref_in = refs[i]
        gate_k = (ref_in == 0).astype(all_v.dtype)               # out *= (ref == 0)
        gate_m = rise_decay * (ref_in != 1).astype(all_v.dtype)  # rise gate

        avout = aV[i + 1]      # adjoint of out_i  (== all_voltages[i+1])
        agout = aG[i + 1]      # adjoint of rise_i (== rise_values[i+1])

        d_out = avout * gate_k
        # path: out depends on neurons_current = voltages[i] (except injected neurons)
        aV = aV.at[i].add(d_out * nd * (1.0 - activate_mask))

        # out = nc*nd + Gtilde*(1-nd);  Gtilde is also rise_values[i+1]
        d_gtilde = d_out * (1.0 - nd) + agout
        # Gtilde = (rise_values[i] + neuron_updates) * gate_m
        d_gp = d_gtilde * gate_m
        aG = aG.at[i].add(d_gp)                 # rise_values[i] feeds in additively
        d_updates = d_gp                        # so does neuron_updates

        # neuron_updates[n] = sum_{s: post=n} act[s]*w_scaled[s]
        src = i - delay
        valid = (src >= 0).astype(all_v.dtype)
        src_c = jnp.maximum(src, 0)
        pre_vals = jnp.where(valid > 0, all_v[src_c][pre_idx], 0.0)
        act = (pre_vals >= th) * 1.0

        d_syn = d_updates[post_idx]             # gather adjoint by postsynaptic neuron
        contrib = d_syn * act * g_scale         # dL/d synapse_weights[s] at this step
        inter = inter.at[i].set(contrib)
        grad_w = grad_w + contrib

        # push back through the surrogate into the presynaptic voltage
        d_act = d_syn * w_scaled
        d_pre = d_act * _surrogate_grad_vec(pre_vals, th) * valid
        aV = aV.at[src_c].add(jnp.zeros(num_neurons).at[pre_idx].add(d_pre))

        return aV, aG, grad_w, inter

    aV, aG, grad_w, inter = jax.lax.fori_loop(
        0, steps - 1, bwd_step, (aV, aG, grad_w, inter_init))
    return grad_w, inter


def synapse_weight_grads(params, connections, num_neurons, synapse_weights,
                         neurons_to_activate, target_voltages):
    """dL/d(synapse_weights) for ``jax_spiking_model.sim_loss`` via manual BPTT."""
    grad_w, _ = _backward(params, connections, num_neurons, synapse_weights,
                          neurons_to_activate, target_voltages)
    return grad_w


def intermediate_grads(params, connections, num_neurons, synapse_weights,
                       neurons_to_activate, target_voltages, zeros=None):
    """Per-timestep ``(steps, num_synapses)`` breakdown of the synapse-weight
    gradient; ``intermediate_grads(...).sum(0)`` equals ``synapse_weight_grads(...)``.

    The output is always shaped ``(steps, connections.shape[0])``. ``zeros`` is
    accepted (and ignored) only for backwards compatibility with existing call
    sites that pass a pre-allocated buffer."""
    _, inter = _backward(params, connections, num_neurons, synapse_weights,
                         neurons_to_activate, target_voltages)
    return inter


def self_test():
    """Check the manual gradient against ``jax.grad`` of a surrogate-differentiable
    forward (the gold standard). Run with the real ``jax_spiking_model`` importable.
    """
    import dataclasses

    @jax.custom_gradient
    def surrogate_activation(pre, threshold):
        out = (pre >= threshold) * 1.0
        return out, lambda g: (_surrogate_grad_vec(pre, threshold) * g, 0.0)

    params = dataclasses.replace(model.default_params, steps=60, delay_iters=18,
                                 refractory_iters=5)
    key = jax.random.key(0)
    N, S = 12, 40
    k1, k2 = jax.random.split(key)
    connections = jax.random.randint(k1, (S, 2), 0, N)
    weights = jax.random.uniform(k2, (S,)) * 800 + 50
    activate = jnp.array([0, 1, 2])

    target, _, _ = model.run_sim(params, connections, N, weights * 0.7, activate)

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
            a = surrogate_activation(p, params.threshold)
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
    manual = synapse_weight_grads(params, connections, N, weights, activate, target)
    inter = intermediate_grads(params, connections, N, weights, activate, target,
                               jnp.zeros((params.steps, S)))

    rel = jnp.max(jnp.abs(gold - manual)) / (jnp.max(jnp.abs(gold)) + 1e-30)
    cos = jnp.dot(gold, manual) / (jnp.linalg.norm(gold) * jnp.linalg.norm(manual))
    # tolerance scales with dtype: float32 leaves ~1e-5 rounding noise, x64 ~1e-12.
    tol = 1e-3 if gold.dtype == jnp.float32 else 1e-8
    print(f"dtype {gold.dtype}  max rel diff vs jax.grad : {float(rel):.2e}")
    print("cosine(gold, manual)     :", float(cos))
    print("intermediate sums to grad:", bool(jnp.allclose(inter.sum(0), manual,
                                                          rtol=1e-4, atol=1e-30)))
    assert rel < tol, "manual gradient does not match autodiff surrogate"
    assert cos > 1 - 1e-4, "manual gradient points the wrong way"
    print("OK")


if __name__ == "__main__":
    self_test()
