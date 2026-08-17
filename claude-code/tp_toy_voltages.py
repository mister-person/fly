"""Voltage traces in the feed-forward toy (tp_anticorr circuit).

Same circuit: 6 drifting correlated inputs -> Nh -> 2-hop readout chain -> out.
Recover Nh's split + chain weights from an observation window, then DEPLOY on the
eval window and capture the full membrane V(t). Overlay:
    true weights   (black)   -- target
    b=th recovered (red)     -- biased one-hop RHS (aims V=theta exactly)
    b=Vact recovered (blue)  -- unbiased RHS (aims for the true crossing value)

What to look for:
  * true trace OVERSHOOTS theta at each spike step (h-integration crosses between
    steps) -> b=theta targets too low -> red weights slightly low -> crosses LATE.
  * blue (Vact) sits on top of black; red lags a step or two, and the lag COMPOUNDS
    down the readout chain (bottom panel).
Saves tp_toy_voltages.png.
"""

import sys, os, types
os.environ.setdefault("LOSS", "st")
os.environ.setdefault("MPLBACKEND", "Agg")
for _n, _attrs in [("brian2", {"ms": 1e-3}), ("neuron_model", {"NeuronSim": object})]:
    if _n not in sys.modules:
        _m = types.ModuleType(_n); [setattr(_m, k, v) for k, v in _attrs.items()]; sys.modules[_n] = _m
sys.path.insert(0, "/workspace/project")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import jax_spiking_model as sim

p = sim.default_params
th, gsw = p.threshold, p.global_synapse_weight
delay, refr = p.delay_iters, p.refractory_iters
nd, rd = float(p.neuron_decay), float(p.rise_decay)

MAX_H = 600
h = np.zeros(MAX_H); _R = _V = 0.0
for t in range(MAX_H):
    _R = (_R + (1.0 if t == delay else 0.0)) * rd
    _V = (_V - _R) * nd + _R
    h[t] = _V

def forward_trace(input_trains, weights, steps):
    """Isolated LIF returning (spikes, full V(t) array). V is the post-reset,
    post-refractory-mask voltage actually used for thresholding each step."""
    events = []
    for sp, w in zip(input_trains, weights):
        for tk in sp:
            ta = tk + delay
            if 0 <= ta < steps:
                events.append((ta, w * gsw))
    events.sort(); ev = iter(events); nxt = next(ev, None)
    V = R = 0.0; ref = 0; out = []; trace = np.zeros(steps)
    for t in range(steps):
        upd = 0.0
        while nxt and nxt[0] == t:
            upd += nxt[1]; nxt = next(ev, None)
        R = (R + upd) * rd * (ref != 1)
        V = (V - R) * nd + R
        V = V * (ref == 0)
        trace[t] = V
        if V >= th and ref == 0:
            out.append(t); ref = refr + 1
        elif ref > 0:
            ref -= 1
    return out, trace

def contrib_rows(input_trains, spikes):
    rows = []; reset_end = 0
    for Tj in spikes:
        rows.append([sum(h[Tj - tk] for tk in sp
                     if tk + delay > reset_end and 0 < Tj - tk < MAX_H)
                     for sp in input_trains])
        reset_end = Tj + refr
    return np.array(rows)

def ridge_nnls(A, b, rf=1e-3):
    lam = rf * np.trace(A.T @ A) / max(A.shape[1], 1)
    return nnls(np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])]),
                np.concatenate([b, np.zeros(A.shape[1])]))[0]

def recover(input_trains, spikes, trace, bmode):
    A = contrib_rows(input_trains, spikes)
    keep = A.max(axis=1) > 1e-12
    A = A[keep]; sp = [s for s, k in zip(spikes, keep) if k]
    if A.shape[0] == 0:
        return np.zeros(len(input_trains))
    b = np.array([trace[s] / gsw if bmode == "vact" else th / gsw for s in sp])
    return ridge_nnls(A, b)

# ── toy circuit (matches tp_anticorr.py) ───────────────────────────────────
NIN = 6
IN = [[15 + 2 * i + (130 + i) * k for k in range(1200)] for i in range(NIN)]
W_SPLIT_TRUE = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
CHAIN_DEPTH, W_CHAIN_TRUE = 2, 520.0
EVAL = 560
W_OBS = 2800     # observation window used for recovery (well-determined)

def build_layers(steps, w_split, w_chain):
    """Return list of (spikes, trace) for Nh then each chain layer."""
    layers = [forward_trace(IN, w_split, steps)]
    for d in range(CHAIN_DEPTH):
        layers.append(forward_trace([layers[-1][0]], [w_chain[d]], steps))
    return layers

# ground truth on the OBSERVATION window (source of targets) and EVAL window
obs_true = build_layers(W_OBS, W_SPLIT_TRUE, [W_CHAIN_TRUE] * CHAIN_DEPTH)
eval_true = build_layers(EVAL,  W_SPLIT_TRUE, [W_CHAIN_TRUE] * CHAIN_DEPTH)

# recover Nh split + each chain weight from the observation window, per b-mode
rec = {}
for bmode in ["th", "vact"]:
    w_split = recover(IN, obs_true[0][0], obs_true[0][1], bmode)
    w_chain = []
    for d in range(CHAIN_DEPTH):
        pre_sp = obs_true[d][0]
        post_sp, post_tr = obs_true[d + 1]
        w_chain.append(float(recover([pre_sp], post_sp, post_tr, bmode)[0]))
    rec[bmode] = build_layers(EVAL, w_split, w_chain)
    we = 100 * np.mean(np.abs(w_split - W_SPLIT_TRUE) / W_SPLIT_TRUE)
    print(f"b={bmode:>4}: split wErr={we:5.1f}%  chain={[round(x) for x in w_chain]}"
          f"  Nh spikes={rec[bmode][0][0]}  out spikes={rec[bmode][-1][0]}")
print(f"true : Nh spikes={eval_true[0][0]}  out spikes={eval_true[-1][0]}")

# ── plot: Nh (top) and output (bottom) membrane traces over the eval window ─
tt = np.arange(EVAL)
styles = {"true": ("black", "true weights", 1.8, 1.0),
          "th":   ("#c0392b", r"recovered  $b=\theta$", 1.3, 0.9),
          "vact": ("#2980b9", r"recovered  $b=V_{\rm actual}$", 1.3, 0.9)}
series = {"true": eval_true, "th": rec["th"], "vact": rec["vact"]}

fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
for ax, layer_i, name in [(axes[0], 0, "Nh (6-input hidden neuron)"),
                          (axes[1], CHAIN_DEPTH, "output neuron")]:
    for key, lay in series.items():
        c, lbl, lw, al = styles[key]
        ax.plot(tt, lay[layer_i][1], color=c, lw=lw, alpha=al, label=lbl)
        for s in lay[layer_i][0]:
            ax.plot(s, lay[layer_i][1][s], "o", color=c, ms=4, alpha=al)
    ax.axhline(th, ls="--", lw=1.0, color="0.5")
    ax.text(EVAL * 0.995, th, r" $\theta$", va="bottom", ha="right",
            color="0.4", fontsize=9)
    ax.set_ylabel("membrane V")
    ax.set_title(name, fontsize=10, loc="left")
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
axes[1].set_xlabel("time (steps)  —  eval window")
fig.suptitle(f"Toy voltage traces (recovered from W={W_OBS}, deployed on eval): "
             r"$b=\theta$ crosses late & compounds down the chain", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("tp_toy_voltages.png", dpi=140)
print("\nSaved tp_toy_voltages.png")

# ── zoom figure: first Nh spike, showing the threshold OVERSHOOT ────────────
s0 = eval_true[0][0][0]
lo_z, hi_z = max(0, s0 - 12), min(EVAL, s0 + 6)
figz, axz = plt.subplots(figsize=(6, 4))
for key, lay in series.items():
    c, lbl, lw, al = styles[key]
    axz.plot(np.arange(lo_z, hi_z), lay[0][1][lo_z:hi_z], "o-", color=c, ms=3,
             lw=lw, alpha=al, label=lbl)
axz.axhline(th, ls="--", lw=1.0, color="0.5"); axz.text(lo_z, th, r" $\theta$",
            va="bottom", color="0.4", fontsize=9)
axz.set_title(f"Zoom: Nh first crossing (true overshoots $\\theta$ by "
              f"{eval_true[0][1][s0]/th:.3f}x)", fontsize=10)
axz.set_xlabel("time (steps)"); axz.set_ylabel("membrane V")
axz.legend(fontsize=8, frameon=False); axz.grid(alpha=0.25)
figz.tight_layout(); figz.savefig("tp_toy_voltages_zoom.png", dpi=140)
print("Saved tp_toy_voltages_zoom.png")
