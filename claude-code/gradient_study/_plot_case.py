"""Output-voltage figures for any registered case: true weights, and a stuck point.

    python3 _plot_case.py "8n M" [seed] [steps]

Small multiples, one panel per output, shared x -- each output has its own crossings and
stacking keeps the geometry readable.  One series per panel, so the title carries identity
and no legend box is needed.  Palette = categorical slots 1-3, validated all-pairs light
(worst CVD dE 9.2, normal-vision 24.0); aqua is under 3:1 on the surface, so every panel is
directly labelled.  Hollow rings on the threshold line are TARGETS, filled markers are
FOUND spikes, so a ring with nothing under it is a miss.
"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import grad_trace as G
from _diag import CASES, steps_for

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9992"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

name = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else steps_for(name)
E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
params = G.mkparams(STEPS)
W = np.array(Wl, np.float32)
T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
t = np.arange(params.steps)
slug = name.replace(" ", "") + ("" if STEPS == steps_for(name) else f"_{STEPS}")


def figure(V, F, w, tag, subtitle, fname):
    fig, axes = plt.subplots(len(outs), 1, figsize=(11 if STEPS <= 520 else 15, 2.4 * len(outs) + 1.6), sharex=True,
                             facecolor=SURFACE, gridspec_kw=dict(hspace=0.34))
    axes = np.atleast_1d(axes)
    for ax, o, col in zip(axes, outs, SERIES):
        ax.set_facecolor(SURFACE)
        v = np.asarray(V[:, o], float)
        ax.plot(t, v, color=col, lw=1.8, solid_joinstyle="round", zorder=3)
        ax.axhline(G.TH, color=MUTED, lw=1.0, ls=(0, (5, 4)), zorder=2)
        ax.plot(T[o], [G.TH] * len(T[o]), "o", ms=9, mfc="none", mec=MUTED, mew=1.6, zorder=4)
        ax.plot(F[o], [v[s] for s in F[o]], "o", ms=8, mfc=col, mec=SURFACE, mew=2, zorder=5)
        hit = sum(1 for x in T[o] if any(abs(x - y) <= 2 for y in F[o]))
        head = (f"N{o}   {len(F[o])} spikes {F[o]}" if tag == "true" else
                f"N{o}   found {F[o]}   ·   target {T[o]}   ·   {hit}/{len(T[o])} on target")
        ax.set_title(head, loc="left", fontsize=10, color=INK, pad=9)
        ax.set_ylabel("V", fontsize=9, color=INK2)
        ax.set_ylim(-0.0005, max(float(v.max()) * 1.18, G.TH * 1.35))
        ax.grid(True, axis="y", color="#e8e8e4", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#dcdcd6")
        ax.tick_params(colors=INK2, labelsize=8.5, length=3)
    axes[0].annotate("hollow ring = target", xy=(params.steps - 4, G.TH), xytext=(0, 7),
                     textcoords="offset points", ha="right", fontsize=8.5, color=INK2)
    axes[-1].set_xlabel("timestep", fontsize=9, color=INK2)
    fig.suptitle(subtitle, x=0.008, y=0.985, ha="left", fontsize=12, color=INK)
    fig.text(0.008, 0.93 if len(outs) > 2 else 0.90,
             "w = " + str([int(round(x)) for x in w]), ha="left", fontsize=8.5,
             color=INK2, family="monospace")
    fig.subplots_adjust(top=0.86, left=0.07, right=0.985, bottom=0.09)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


Vt = G.fsim(C, N, W, params)
figure(Vt, T, list(Wl), "true",
       f"{name} output voltages at the TRUE weights — hidden layer fires at mixed rates",
       f"{slug}_true.png")

w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
live = {}
G.train(C, N, outs, w0.copy(), T, params, rounds=3200, lr=G.LR,
        cb=lambda it, w, *a: live.update(w=w.copy()))
w = live["w"]
V = G.fsim(C, N, w, params)
F = {n: G.sp(V, n) for n in range(N)}
figure(V, F, w, "stuck", f"{name} at a stuck point (seed {seed}) — output voltages vs targets",
       f"{slug}_stuck_seed{seed}.png")
print("\ntrue  w", [int(x) for x in Wl])
print("found w", [int(round(x)) for x in w])
for n in range(N):
    print(f"  N{n}: found {F[n]}   true {T[n]}")
