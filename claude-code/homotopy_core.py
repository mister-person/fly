"""Shared core for softness-homotopy training on spiking neural networks.

Provides the differentiable forward pass, the one-stage Adam optimizer, and the
LR schedule used by both learn_homotopy.py (batch sweep) and
visualize_training.py (live visualization).

All topology is passed explicitly so callers can fix it (visualize_training) or
vmap over it (learn_homotopy).
"""
import sys
import types
from functools import partial

import jax
import jax.numpy as jnp

# ── brian2 / neuron_model stubs ───────────────────────────────────────────────
# jax_spiking_model imports these at module load; stub them so this code runs
# without the full lab stack.
for _n, _attrs in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        for _k, _v in _attrs.items():
            setattr(_m, _k, _v)
        sys.modules[_n] = _m

import jax_spiking_model as _sim


# ── forward scan (stores per-step ref history for manual backward) ─────────────

def _soft_sim_scan(w, beta, params, connections, num_neurons, neurons_activate):
    """Like soft_sim but also returns per-step ref history ``refs[i]``.

    The ref history is needed by the manual backward pass: treating the
    refractory gates as straight-through constants requires knowing the ref
    value that was *active* at each step.
    """
    nd    = params.neuron_decay
    rd    = params.rise_decay
    th    = params.threshold
    delay = params.delay_iters
    sw    = w * params.global_synapse_weight

    V    = jnp.zeros((params.steps, num_neurons))
    ref  = jnp.zeros(num_neurons)
    rise = jnp.zeros((params.steps, num_neurons))
    refs = jnp.zeros((params.steps, num_neurons))

    def loop(i, x):
        V, ref, rise, refs = x
        refs = refs.at[i].set(ref)          # snapshot ref entering this step
        inj   = ((i % 100) == 0) * th / nd
        wi    = V.at[i, neurons_activate].set(inj)
        nc    = wi[i]
        pre_v = jnp.where(i - delay >= 0, wi[i - delay], jnp.zeros_like(nc))
        act   = jax.nn.sigmoid(beta * (pre_v[connections[..., 0]] / th - 1.0))
        upd   = jnp.zeros_like(nc).at[connections[..., 1]].add(act * sw)
        r     = (rise[i] + upd) * rd * (ref != 1)
        out   = (nc - r) * nd + r
        out   = out * (ref == 0)
        fire    = jax.nn.sigmoid(beta * (out / th - 1.0))
        new_ref = (ref + fire * (params.refractory_iters + 1 - ref) - 1).clip(min=0)
        return (V.at[i + 1].set(out), new_ref, rise.at[i + 1].set(r), refs)

    V, _, _, refs = jax.lax.fori_loop(0, params.steps, loop, (V, ref, rise, refs))
    return V, refs


# ── shared reverse-time sweep ─────────────────────────────────────────────────

def _bptt_backward(V, refs, aV_seed, w, params, connections, num_neurons,
                   neurons_activate, act_slope_fn):
    """Manual BPTT sweep shared by soft_sim (sigmoid slopes) and hard_sim
    (surrogate slopes in jax_model_grads_hack).

    Parameters
    ----------
    V            : (steps, num_neurons) voltage history from the forward pass
    refs         : (steps, num_neurons) refractory-timer history (ref entering each step)
    aV_seed      : (steps, num_neurons) initial adjoint dL/dV (e.g. 2*(V-target) for MSE)
    w            : synapse weights (unscaled)
    act_slope_fn : callable(pre_vals) -> (act, d_act_d_v), vectorised over synapses.
                   ``act`` is the activation used in the forward; ``d_act_d_v``
                   is its derivative with respect to the pre-synaptic voltage.

    Returns
    -------
    grad_w : (n_syn,) dL/dw
    inter  : (steps, n_syn) per-step contribution to grad_w; sums to grad_w
    """
    nd       = params.neuron_decay
    rd       = params.rise_decay
    th       = params.threshold
    delay    = params.delay_iters
    g_scale  = params.global_synapse_weight
    steps    = params.steps

    pre_idx  = connections[..., 0]
    post_idx = connections[..., 1]
    w_scaled = w * g_scale
    act_mask = jnp.zeros(num_neurons).at[neurons_activate].set(1.0)

    aV     = aV_seed
    aG     = jnp.zeros((steps, num_neurons))
    grad_w = jnp.zeros_like(w)
    inter  = jnp.zeros((steps, connections.shape[0]))

    def bwd_step(j, carry):
        aV, aG, grad_w, inter = carry
        i = steps - 2 - j                       # reverse-time index

        ref_in = refs[i]
        gate_k = (ref_in == 0).astype(V.dtype)               # out gate
        gate_m = rd * (ref_in != 1).astype(V.dtype)          # rise gate

        avout = aV[i + 1]                        # adjoint of out at step i
        agout = aG[i + 1]                        # adjoint of rise at step i+1

        # adjoint of out → voltage at step i (skipping injected neurons)
        d_out = avout * gate_k
        aV    = aV.at[i].add(d_out * nd * (1.0 - act_mask))

        # adjoint of rise at step i (via out and rise recurrence)
        d_gtilde  = d_out * (1.0 - nd) + agout
        d_gp      = d_gtilde * gate_m
        aG        = aG.at[i].add(d_gp)
        d_updates = d_gp                         # adjoint of synapse contribution

        # pre-synaptic voltages (delayed)
        src      = i - delay
        valid    = (src >= 0).astype(V.dtype)
        src_c    = jnp.maximum(src, 0)
        pre_vals = jnp.where(valid > 0, V[src_c, pre_idx], 0.0)

        act, d_slope = act_slope_fn(pre_vals)

        # weight gradient: sum over time of (adjoint of post-neuron) * act * g_scale
        d_syn   = d_updates[post_idx]
        contrib = d_syn * act * g_scale
        inter   = inter.at[i].set(contrib)
        grad_w  = grad_w + contrib

        # push adjoint back through activation into pre-synaptic voltage
        d_act = d_syn * w_scaled
        d_pre = d_act * d_slope * valid
        aV    = aV.at[src_c].add(
            jnp.zeros(num_neurons).at[pre_idx].add(d_pre))

        return aV, aG, grad_w, inter

    _, _, grad_w, inter = jax.lax.fori_loop(
        0, steps - 1, bwd_step, (aV, aG, grad_w, inter))
    return grad_w, inter


# ── differentiable forward pass (custom VJP uses manual BPTT) ─────────────────

@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def soft_sim(w, beta, params, connections, num_neurons, neurons_activate):
    """Soft-sigmoid forward pass with a fast manual VJP.

    Trick 1: sigmoid(beta*(x/threshold - 1)) replaces every hard step, making
    the whole trajectory smooth so gradients are meaningful.

    The VJP is implemented via a hand-written reverse-time sweep (_bptt_backward)
    rather than JAX autodiff through the scan.  This is ~6x faster because the
    manual sweep avoids storing the full O(steps) tape that autodiff requires.

    The slope used in the manual backward is d/dv of sigmoid(beta*(v/th-1)),
    i.e. act*(1-act)*beta/th, which is exact for this forward activation.
    """
    V, _ = _soft_sim_scan(w, beta, params, connections, num_neurons, neurons_activate)
    return V


def _soft_sim_fwd(w, beta, params, connections, num_neurons, neurons_activate):
    V, refs = _soft_sim_scan(w, beta, params, connections, num_neurons, neurons_activate)
    return V, (w, beta, V, refs)   # residuals: only JAX arrays


def _soft_sim_bwd(params, connections, num_neurons, neurons_activate, residuals, dL_dV):
    w, beta, V, refs = residuals
    th = params.threshold

    def act_slope(pre_vals):
        a = jax.nn.sigmoid(beta * (pre_vals / th - 1.0))
        return a, a * (1.0 - a) * (beta / th)

    grad_w, _ = _bptt_backward(V, refs, dL_dV, w, params, connections,
                                num_neurons, neurons_activate, act_slope)
    return grad_w, jnp.zeros_like(beta)


soft_sim.defvjp(_soft_sim_fwd, _soft_sim_bwd)


def hard_sim(w, params, connections, num_neurons, neurons_activate):
    """True hard (non-differentiable) forward, for measuring real loss."""
    return _sim.run_sim(params, connections, num_neurons, w, neurons_activate)[0]


# ── one beta stage: Adam + best-iterate ───────────────────────────────────────

def homotopy_stage(w0, base, lo, hi, beta, lr,
                   params, connections, num_neurons, neurons_activate,
                   *, nopt=300, observe_last=False, patience=50, rtol=1e-3):
    """Adam on soft_loss(w, soft_target(base, beta)) with best-iterate tracking.

    Trick 2: target = soft_sim(base, beta) — same softness as the model being
    optimised, so the global minimum is always at w = base regardless of beta.

    observe_last=True restricts the loss to the last neuron's trace only.
    The gradient still flows back through the full chain.

    Early stopping: every `patience` steps, if the best loss hasn't improved by
    more than `rtol` (relative) since the last check, the stage exits early.
    Set patience=0 to disable and always run all nopt steps.
    """
    target = soft_sim(base, beta, params, connections, num_neurons, neurons_activate)

    def soft_loss(w):
        v = soft_sim(w, beta, params, connections, num_neurons, neurons_activate)
        if observe_last:
            idx = connections[-1, 1]
            return jnp.sum((target[:, idx] - v[:, idx]) ** 2)
        return jnp.sum((target - v) ** 2)

    vg = jax.value_and_grad(soft_loss)

    if patience > 0:
        # carry: (w, m, v, bw, bl, t, l_check, done)
        # l_check = best loss at start of current patience window
        def cond(c):
            _, _, _, _, _, t, _, done = c
            return (t < nopt) & ~done

        def body(c):
            w, m, v, bw, bl, t, l_check, done = c
            l, g = vg(w)
            g    = jnp.nan_to_num(g)
            bw, bl = jax.lax.cond(l < bl, lambda: (w, l), lambda: (bw, bl))
            m  = 0.9   * m + 0.1   * g
            v  = 0.999 * v + 0.001 * g * g
            t1 = (t + 1).astype(jnp.float32)
            step = (m / (1 - 0.9 ** t1)) / (jnp.sqrt(v / (1 - 0.999 ** t1)) + 1e-12)
            w_new  = jnp.clip(w - lr * step, lo, hi)
            new_t  = t + 1
            at_end = (new_t % patience == 0)
            rel_imp = (l_check - bl) / (jnp.abs(l_check) + 1e-10)
            done_now    = done | (at_end & (rel_imp < rtol))
            l_check_new = jax.lax.cond(at_end, lambda: bl, lambda: l_check)
            return (w_new, m, v, bw, bl, new_t, l_check_new, done_now)

        l0 = vg(w0)[0]
        z   = jnp.zeros_like(w0)
        init = (w0, z, z, w0, l0, jnp.int32(0), l0, jnp.bool_(False))
        _, _, _, bw, _, _, _, _ = jax.lax.while_loop(cond, body, init)
    else:
        def body(t, c):
            w, m, v, bw, bl = c
            l, g = vg(w)
            g    = jnp.nan_to_num(g)
            bw, bl = jax.lax.cond(l < bl, lambda: (w, l), lambda: (bw, bl))
            m  = 0.9   * m + 0.1   * g
            v  = 0.999 * v + 0.001 * g * g
            t1 = (t + 1).astype(jnp.float32)
            step = (m / (1 - 0.9 ** t1)) / (jnp.sqrt(v / (1 - 0.999 ** t1)) + 1e-12)
            return (jnp.clip(w - lr * step, lo, hi), m, v, bw, bl)

        l0 = vg(w0)[0]
        z   = jnp.zeros_like(w0)
        _, _, _, bw, _ = jax.lax.fori_loop(0, nopt, body, (w0, z, z, w0, l0))

    return bw


# ── learning-rate schedule ────────────────────────────────────────────────────

def lr_for_beta(beta, lr_scale=1.0):
    """Default per-stage LR: 1.0 for low beta, tapering as the model sharpens."""
    return (1.0 if beta <= 2 else (0.5 if beta <= 8 else 0.2)) * lr_scale
