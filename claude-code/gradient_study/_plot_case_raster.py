"""Output voltages with a colour-coded input raster underneath each panel.

    python3 _plot_case_raster.py "8n M" [seed] [steps]

The voltage trace is now NEUTRAL ink, not a categorical colour: the panel title identifies
the output, and the categorical hues are spent on the four hidden SOURCES instead, which is
the thing being compared.  Each panel carries a four-row raster of the ARRIVAL times
(spike + DELAY_ITERS) of N1..N4, so a coincidence reads as four ticks lining up under a
threshold crossing.  Rows are direct-labelled, which also discharges the aqua contrast WARN.

Palette validated all-pairs light: worst CVD dE 9.2 (deutan), normal-vision 16.3.
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
TRACE = "#3f3f3c"
SRC = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

name = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else steps_for(name)
E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
params = G.mkparams(STEPS)
W = np.array(Wl, np.float32)
T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
hidden = [n for n in range(N) if n not in outs and n != 0]
t = np.arange(params.steps)
slug = name.replace(" ", "") + ("" if STEPS == steps_for(name) else f"_{STEPS}")


def figure(V, F, w, tag, subtitle, fname):
    top = max(float(np.asarray(V[:, o], float).max()) for o in outs)
    top = max(top * 1.15, G.TH * 1.3)
    band = top * 0.42                       # raster occupies the strip below zero
    rowy = [-band * (0.22 + 0.20 * i) for i in range(len(hidden))]
    fig, axes = plt.subplots(len(outs), 1, figsize=(11 if STEPS <= 520 else 15,
                                                    2.9 * len(outs) + 1.5),
                             sharex=True, facecolor=SURFACE,
                             gridspec_kw=dict(hspace=0.30))
    axes = np.atleast_1d(axes)
    for ax, o in zip(axes, outs):
        ax.set_facecolor(SURFACE)
        v = np.asarray(V[:, o], float)
        ax.axhline(G.TH, color=MUTED, lw=1.0, ls=(0, (5, 4)), zorder=2)
        ax.plot(t, v, color=TRACE, lw=1.6, solid_joinstyle="round", zorder=3)
        ax.plot(T[o], [G.TH] * len(T[o]), "o", ms=9, mfc="none", mec=MUTED, mew=1.6, zorder=4)
        ax.plot(F[o], [v[s] for s in F[o]], "o", ms=7.5, mfc=TRACE, mec=SURFACE,
                mew=1.8, zorder=5)
        ax.axhline(0.0, color="#e2e2dd", lw=0.9, zorder=1)
        # input raster: arrival times of each hidden source
        for i, (h, col) in enumerate(zip(hidden, SRC)):
            arr = [q + G.DELAY_ITERS for q in F[h] if q + G.DELAY_ITERS < params.steps]
            ax.vlines(arr, rowy[i] - band * 0.075, rowy[i] + band * 0.075,
                      color=col, lw=2.0, zorder=4)
            ax.annotate(f"N{h}", xy=(0, rowy[i]), xytext=(-6, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=8, color=col, annotation_clip=False)
        hit = sum(1 for x in T[o] if any(abs(x - y) <= 2 for y in F[o]))
        head = (f"N{o}   {len(F[o])} spikes {F[o]}" if tag == "true" else
                f"N{o}   found {F[o]}   ·   target {T[o]}   ·   {hit}/{len(T[o])} on target")
        ax.set_title(head, loc="left", fontsize=9.5, color=INK, pad=8)
        ax.set_ylabel("V", fontsize=9, color=INK2)
        ax.set_ylim(-band, top)
        ax.set_yticks([0, G.TH])
        ax.set_yticklabels(["0", "th"])
        ax.grid(False)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#dcdcd6")
        ax.tick_params(colors=INK2, labelsize=8.5, length=3)
    axes[0].annotate("hollow ring = target spike", xy=(params.steps - 4, G.TH),
                     xytext=(0, 7), textcoords="offset points", ha="right",
                     fontsize=8.5, color=INK2)
    axes[-1].set_xlabel("timestep", fontsize=9, color=INK2)
    fig.suptitle(subtitle, x=0.008, y=0.986, ha="left", fontsize=12, color=INK)
    fig.text(0.008, 0.947, "coloured ticks = ARRIVAL of each hidden source "
             "(spike + delay); a crossing needs four of them together",
             ha="left", fontsize=8.5, color=INK2)
    fig.text(0.008, 0.917, "w = " + str([int(round(x)) for x in w]),
             ha="left", fontsize=8, color=INK2, family="monospace")
    fig.subplots_adjust(top=0.868, left=0.075, right=0.985, bottom=0.085)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


figure(G.fsim(C, N, W, params), T, list(Wl), "true",
       f"{name} — output voltages and the hidden spikes that drive them (TRUE weights)",
       f"{slug}_raster_true.png")

w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
live = {}
G.train(C, N, outs, w0.copy(), T, params, rounds=3200, lr=G.LR,
        cb=lambda it, w, *a: live.update(w=w.copy()))
w = live["w"]
V = G.fsim(C, N, w, params)
F = {n: G.sp(V, n) for n in range(N)}
figure(V, F, w, "stuck",
       f"{name} at a stuck point (seed {seed}, LIVE weights) — inputs vs outputs",
       f"{slug}_raster_stuck{seed}.png")
for n in range(N):
    print(f"  N{n}: {F[n]}   true {T[n]}")
