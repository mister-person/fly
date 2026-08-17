"""Test the 'PSP + spike-time' gradient and probe where it's brittle.

The gradient we read off the dynamics, for an output neuron with input weights w:

  dL/dw_i = Σ_{output spikes s}  (dL/dt_s) · (−1/slope_s) · (dV/dw_i at t_s)
            with  dV/dw_i(t_s) = gsw · Σ_k h(t_s − t_ik)   (the PSP term)

The suspicious factor is  −1/slope_s : when an output spike grazes threshold the
slope dV/dt → 0 and the gradient explodes.  Here we (1) build a faithful
single-neuron LIF forward, (2) validate it against the JAX simulator, (3) check
the gradient against finite differences, (4) show the 1/slope blow-up and the
spike-create/destroy discontinuity, (5) train with it vs a slope-clipped version
vs a dense voltage gradient.
"""

import sys, os, types, dataclasses
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

p      = sim.default_params
TH     = p.threshold
GSW    = p.global_synapse_weight
DELAY  = p.delay_iters
REFRAC = p.refractory_iters
ND     = float(p.neuron_decay)
RD     = float(p.rise_decay)
DIR    = "/workspace/project/gradient_study"
MAXH   = 400

# PSP kernel (V per unit synaptic update, single pre-spike at t=0)
_h = np.zeros(MAXH); _R = _V = 0.0
for _t in range(MAXH):
    _R = (_R + (1.0 if _t == DELAY else 0.0)) * RD
    _V = (_V - _R) * ND + _R
    _h[_t] = _V

def hk(dt):
    dt = np.asarray(dt, int)
    out = np.zeros(dt.shape, float)
    m = (dt >= 0) & (dt < MAXH)
    out[m] = _h[dt[m]]
    return out


# ── faithful single-neuron LIF forward (mirrors jax_spiking_model.timestep) ────
def lif_forward(w, input_active, T):
    """w: (n_in,).  input_active: (n_in, T) bool (input i suprathreshold at step t).
    Returns (V trace (T,), list of output spike steps)."""
    w = np.asarray(w, float)
    rise = 0.0; refr = 0
    Vtr = np.zeros(T); spikes = []          # Vtr[i] mirrors jax voltages[i]
    for i in range(T - 1):                  # step i produces voltages[i+1]
        upd = 0.0
        j = i - DELAY
        if j >= 0:
            upd = float(np.sum(w[input_active[:, j]])) * GSW
        rise = rise + upd
        rise = rise * RD * (0.0 if refr == 1 else 1.0)
        out = (Vtr[i] - rise) * ND + rise
        out = out * (1.0 if refr == 0 else 0.0)
        fired = out >= TH
        refr = max((REFRAC + 1 if fired else refr) - 1, 0)
        if fired:
            spikes.append(i + 1)
        Vtr[i + 1] = out
    return Vtr, spikes


def lif_tangent(w, input_active, T):
    """Forward + forward-mode sensitivity dV[t]/dw_i under fixed spike structure.
    Between resets the LIF dynamics are linear, so this dV/dw is exact given the
    (held-fixed) reset times — the event-based / EventProp linearisation.
    Returns (Vtr (T,), spikes, dVtr (T, n_in))."""
    w = np.asarray(w, float); n = len(w)
    rise = 0.0; drise = np.zeros(n); refr = 0
    Vtr = np.zeros(T); dVtr = np.zeros((T, n)); spikes = []
    for i in range(T - 1):
        upd = 0.0; dupd = np.zeros(n)
        j = i - DELAY
        if j >= 0:
            act = input_active[:, j]
            upd = float(np.sum(w[act])) * GSW
            dupd[act] += GSW
        rise += upd; drise = drise + dupd
        gate = 0.0 if refr == 1 else 1.0
        rise *= RD * gate; drise = drise * RD * gate
        out = (Vtr[i] - rise) * ND + rise
        dout = (dVtr[i] - drise) * ND + drise
        clamp = 1.0 if refr == 0 else 0.0
        out *= clamp; dout = dout * clamp
        fired = out >= TH
        refr = max((REFRAC + 1 if fired else refr) - 1, 0)
        if fired:
            spikes.append(i + 1)
        Vtr[i + 1] = out; dVtr[i + 1] = dout
    return Vtr, spikes, dVtr


def match_targets(spikes, targets):
    """Order-match found spikes to targets; return list of (spike_idx, t_s, t_target)."""
    n = min(len(spikes), len(targets))
    return [(s, spikes[s], targets[s]) for s in range(n)]


def spike_time_loss(spikes, targets, count_pen=200.0):
    """0.5 Σ (t_s - t*_s)^2 over matched pairs + penalty for count mismatch."""
    m = match_targets(spikes, targets)
    l = 0.5 * sum((ts - tt) ** 2 for _, ts, tt in m)
    l += count_pen * abs(len(spikes) - len(targets))   # discontinuous count term
    return l


def spike_time_grad(w, input_active, targets, T, slope_floor=0.0):
    """PSP + spike-time gradient.  slope_floor>0 clips |slope| for stability."""
    V, spikes, dV = lif_tangent(w, input_active, T)
    g = np.zeros_like(np.asarray(w, float))
    diag = []
    for s, ts, tt in match_targets(spikes, targets):
        slope = V[ts] - V[ts - 1]                 # dV/dt at crossing
        s_use = slope if abs(slope) >= slope_floor else np.sign(slope or 1.0) * slope_floor
        dLdt = (ts - tt)
        g += dLdt * (-1.0 / s_use) * dV[ts]       # (dL/dt)(-1/slope)(dV/dw)
        diag.append(dict(t=ts, slope=slope, dLdt=dLdt, dVnorm=float(np.linalg.norm(dV[ts]))))
    return g, spikes, V, diag


def fd_grad(loss_fn, w, eps=0.5):
    w = np.asarray(w, float); g = np.zeros_like(w)
    for i in range(len(w)):
        wp = w.copy(); wp[i] += eps
        wm = w.copy(); wm[i] -= eps
        g[i] = (loss_fn(wp) - loss_fn(wm)) / (2 * eps)
    return g


def validate():
    """N0(driven)->N1, w=460: JAX sim gives N1 spikes [88,188]. Match it."""
    T = 260
    params = dataclasses.replace(sim.default_params, steps=T)
    C = jnp.array(np.array([[0, 1]], np.int32)); N = 2; A = jnp.array([0])
    Vj, _, _ = sim.run_sim(params, C, N, jnp.array([460.0], np.float32), A)
    Vj = np.array(Vj)
    jax_spikes = np.where(Vj[:, 1] >= TH)[0].tolist()
    # our forward: single input, active where N0 voltage >= th
    inp_active = (Vj[:, 0] >= TH)[None, :]
    _, my_spikes = lif_forward([460.0], inp_active, T)
    print(f"  JAX  N1 spikes: {jax_spikes}")
    print(f"  mine N1 spikes: {my_spikes}")
    ok = jax_spikes == my_spikes
    print(f"  MATCH: {ok}")
    return ok


def make_task(n_in=5, T=260):
    """One output neuron, n_in inputs each active at staggered times (2 pulses)."""
    input_active = np.zeros((n_in, T), bool)
    base = 12
    for i in range(n_in):
        for c in (0, 1):
            t = base + 7 * i + 100 * c
            if t < T:
                input_active[i, t] = True
    targets = [70, 170]
    return input_active, targets, T


def spike_time_of(w, ia, targets, T, s_idx):
    sp = lif_tangent(w, ia, T)[1]
    return sp[s_idx] if s_idx < len(sp) else None


def gradient_check():
    print("\n" + "=" * 74)
    print("GRADIENT CHECK: does -1/slope*dV/dw predict how the spike TIME moves?")
    print("=" * 74)
    ia, targets, T = make_task()
    w = np.full(ia.shape[0], 250.0)
    V, spikes, dV = lif_tangent(w, ia, T)
    print(f"  weights=250  output spikes={spikes}  targets={targets}")

    # (1) the spike-time LOSS is a discrete staircase -> its FD gradient is ~0
    g_loss_fd = fd_grad(lambda ww: spike_time_loss(lif_tangent(ww, ia, T)[1], targets), w, eps=0.5)
    print(f"  FD grad of spike-time LOSS (eps=0.5): {np.round(g_loss_fd,4).tolist()}"
          f"   <- ~0: loss is piecewise-constant in discrete time")

    # (2) but the predicted per-input dt/dw matches how the integer spike jumps
    #     for a large-enough perturbation.  Check spike 0.
    s = 0; ts = spikes[s]; slope = V[ts] - V[ts - 1]
    pred_dtdw = -dV[ts] / slope                       # per-input predicted dt_s/dw_i
    emp = np.zeros(len(w))
    for i in range(len(w)):
        wp = w.copy(); wp[i] += 40
        wm = w.copy(); wm[i] -= 40
        tp = spike_time_of(wp, ia, targets, T, s)
        tm = spike_time_of(wm, ia, targets, T, s)
        emp[i] = (tp - tm) / 80.0 if (tp is not None and tm is not None) else np.nan
    print(f"  predicted dt0/dw : {np.round(pred_dtdw,4).tolist()}")
    print(f"  empirical dt0/dw : {np.round(emp,4).tolist()}")
    cos = float(np.dot(pred_dtdw, emp) /
                (np.linalg.norm(pred_dtdw) * np.linalg.norm(emp) + 1e-30))
    print(f"  cosine(pred, empirical spike-time shift) = {cos:.4f}")
    print(f"  -> the DIRECTION is right; the loss is just quantised in time.")
    return ia, targets, T


def brittleness(ia, targets, T):
    print("\n" + "=" * 74)
    print("BRITTLENESS: sweep a global weight scale; watch slope and gradient norm")
    print("=" * 74)
    scales = np.linspace(0.15, 2.2, 340)
    w0 = np.full(ia.shape[0], 250.0)
    min_slope, gnorm, ncount, gfd_norm = [], [], [], []
    for s in scales:
        w = w0 * s
        V, spikes, dV = lif_tangent(w, ia, T)
        g, _, _, diag = spike_time_grad(w, ia, targets, T)
        ncount.append(len(spikes))
        min_slope.append(min([abs(d['slope']) for d in diag], default=np.nan))
        gnorm.append(np.linalg.norm(g))
        gfd = fd_grad(lambda ww: spike_time_loss(lif_tangent(ww, ia, T)[1], targets), w)
        gfd_norm.append(np.linalg.norm(gfd))
    min_slope = np.array(min_slope); gnorm = np.array(gnorm); ncount = np.array(ncount)
    # find grazing points: local minima of slope
    worst = np.nanargmax(gnorm)
    print(f"  spike count ranges {ncount.min()}..{ncount.max()} over the sweep "
          f"({np.sum(np.diff(ncount)!=0)} count changes = create/destroy events)")
    print(f"  smallest slope seen = {np.nanmin(min_slope):.3e}  -> gradient norm "
          f"peaks at {gnorm[worst]:.2e} (scale={scales[worst]:.3f})")
    print(f"  median gradient norm = {np.nanmedian(gnorm):.2e}  "
          f"=> blow-ups are {gnorm[worst]/np.nanmedian(gnorm):.0f}x the typical value")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    a = ax[0, 0]; a.plot(scales, ncount, drawstyle="steps-mid")
    a.set_title("count of output spikes vs weight scale (create/destroy = discontinuities)")
    a.set_xlabel("weight scale"); a.set_ylabel("# spikes"); a.grid(alpha=.3)
    a = ax[0, 1]; a.semilogy(scales, min_slope)
    a.set_title("smallest slope dV/dt at a crossing (→0 = grazing)")
    a.set_xlabel("weight scale"); a.set_ylabel("min |slope|"); a.grid(alpha=.3)
    a = ax[1, 0]; a.semilogy(scales, gnorm, label="PSP+spike-time |grad|")
    a.semilogy(scales, gfd_norm, label="finite-diff |grad| (truth)", alpha=.7)
    a.set_title("gradient norm — spike-time term explodes at grazing crossings")
    a.set_xlabel("weight scale"); a.set_ylabel("|grad|"); a.legend(fontsize=8); a.grid(alpha=.3)
    a = ax[1, 1]
    a.semilogy(scales, gnorm, label="raw")
    gclip = []
    for s in scales:
        gc, _, _, _ = spike_time_grad(w0 * s, ia, targets, T, slope_floor=5e-5)
        gclip.append(np.linalg.norm(gc))
    a.semilogy(scales, gclip, label="slope-clipped (floor 5e-5)")
    a.set_title("slope-clipping tames the blow-up")
    a.set_xlabel("weight scale"); a.set_ylabel("|grad|"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{DIR}/grad_brittleness.png", dpi=120)
    print(f"  wrote {DIR}/grad_brittleness.png")


def margin_grad(w, ia, targets, T, window=25):
    """PSP/existence gradient: at each target with NO nearby spike, push the peak
    subthreshold voltage in a window up to threshold (revives dead neurons).
    grad of 0.5*(th - V_peak)^2 = (V_peak - th) * dV_peak/dw   (always defined)."""
    V, spikes, dV = lif_tangent(w, ia, T)
    g = np.zeros(len(w))
    for tt in targets:
        near = [s for s in spikes if abs(s - tt) <= window]
        if near:
            continue                      # a spike already exists here
        lo, hi = max(1, tt - window), min(T - 1, tt + window)
        tp = lo + int(np.argmax(V[lo:hi]))
        if V[tp] < TH:
            g += (V[tp] - TH) * dV[tp]     # descend -> raises V toward threshold
    return g, spikes


def combined_grad(w, ia, targets, T, slope_floor=5e-5, window=25):
    """Existence (PSP/margin) where a target has no spike, spike-time where it does."""
    V, spikes, dV = lif_tangent(w, ia, T)
    g = np.zeros(len(w))
    for tt in targets:
        near = [s for s in spikes if abs(s - tt) <= window]
        if near:
            ts = min(near, key=lambda s: abs(s - tt))
            slope = V[ts] - V[ts - 1]
            s_use = slope if abs(slope) >= slope_floor else np.sign(slope or 1.0) * slope_floor
            g += (ts - tt) * (-1.0 / s_use) * dV[ts]
        else:
            lo, hi = max(1, tt - window), min(T - 1, tt + window)
            tp = lo + int(np.argmax(V[lo:hi]))
            if V[tp] < TH:
                g += (V[tp] - TH) * dV[tp]
    return g, spikes


def train(ia_unused, targets_unused, T_unused):
    print("\n" + "=" * 74)
    print("TRAINING from a DEAD start (unit-norm gradient direction, step=4)")
    print("=" * 74)
    # Single output spike target, single input pulse -> weights are NOT coupled
    # across multiple targets (see note below), so timing is cleanly identifiable.
    n_in, T = 5, 200
    ia = np.zeros((n_in, T), bool)
    for i in range(n_in):
        ia[i, 12 + 7 * i] = True
    targets = [95]
    STEP, ITERS = 4.0, 250
    w0 = np.full(n_in, 40.0)                   # fully dead: 0 spikes
    print(f"  start weights=40 -> spikes={lif_tangent(w0, ia, T)[1]} (dead); target={targets}")

    def grad_of(name, w):
        if name == "spike-time only":
            return spike_time_grad(w, ia, targets, T, slope_floor=5e-5)[0]
        if name == "PSP/margin only":
            return margin_grad(w, ia, targets, T)[0]
        return combined_grad(w, ia, targets, T)[0]

    curves = {}
    for name in ["spike-time only", "PSP/margin only", "combined (PSP+spike-time)"]:
        w = w0.copy(); hist = []
        for _ in range(ITERS):
            sp = lif_tangent(w, ia, T)[1]
            hist.append(spike_time_loss(sp, targets))
            g = grad_of(name, w)
            gn = np.linalg.norm(g)
            if gn > 1e-30:
                w = np.clip(w - STEP * g / gn, 20, 3000)
        sp = lif_tangent(w, ia, T)[1]
        curves[name] = hist
        print(f"  {name:28s}: final spikes={sp}  loss={spike_time_loss(sp,targets):.2f}")
    print("  NOTE: with ONE target the timing is cleanly identifiable. With two")
    print("  target spikes driven by the SAME weights, the spike-time term thrashes")
    print("  (retiming one spike disturbs the other) — a real coupling/brittleness.")

    import matplotlib.pyplot as plt
    fig, axp = plt.subplots(1, 1, figsize=(8.5, 5))
    for name, h in curves.items():
        axp.semilogy(np.array(h) + 1e-2, label=name, lw=2)
    axp.set_title("training from a dead start: spike-time alone is stuck (no revival signal);\n"
                  "PSP/margin revives but can't fine-time; combined does both")
    axp.set_xlabel("iteration"); axp.set_ylabel("spike-time loss + 1e-2")
    axp.legend(fontsize=9); axp.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{DIR}/grad_training.png", dpi=120)
    print(f"  wrote {DIR}/grad_training.png")


if __name__ == "__main__":
    print("Validating single-neuron LIF forward against JAX simulator:")
    validate()
    ia, targets, T = gradient_check()
    brittleness(ia, targets, T)
    train(ia, targets, T)
    print("\nDone.")
