"""Voltage traces for 8n K's three output neurons.

Small multiples (one panel per output, shared x) rather than an overlay: each output has
its own spike times and its own threshold crossings, and stacking them keeps the crossing
geometry readable.  One series per panel, so the panel title carries identity and no legend
box is needed.  Palette = categorical slots 1-3, validated all-pairs light mode
(worst CVD dE 9.2, normal-vision 24.0); aqua sits below 3:1 on the surface, so every panel
is directly labelled.
"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import grad_trace as G
from _diag import CASES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9992"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

E, N, outs, Wl = CASES["8n K"]
C = np.array(E, np.int32)
params = G.mkparams(520)
V = G.fsim(C, N, np.array(Wl, np.float32), params)
T = {n: G.sp(V, n) for n in range(N)}
t = np.arange(params.steps)

fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True,
                         facecolor=SURFACE, gridspec_kw=dict(hspace=0.32))
for ax, o, col in zip(axes, outs, SERIES):
    ax.set_facecolor(SURFACE)
    v = np.asarray(V[:, o], float)
    ax.plot(t, v, color=col, lw=1.8, solid_joinstyle="round", zorder=3)
    # threshold: a reference line, recessive, not a series
    ax.axhline(G.TH, color=MUTED, lw=1.0, ls=(0, (5, 4)), zorder=2)
    ax.annotate("threshold", xy=(params.steps - 4, G.TH), xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=8.5, color=INK2)
    # spikes: marker on the crossing, >=8px, 2px surface ring so overlaps stay legible
    ax.plot(T[o], [v[s] for s in T[o]], "o", ms=8, mfc=col, mec=SURFACE, mew=2, zorder=4)
    for s in T[o]:
        ax.annotate(str(s), xy=(s, v[s]), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK2)
    ax.set_title(f"N{o}   {len(T[o])} spikes at {T[o]}",
                 loc="left", fontsize=10.5, color=INK, pad=9)
    ax.set_ylabel("V", fontsize=9, color=INK2)
    ax.set_ylim(-0.0005, max(float(v.max()) * 1.18, G.TH * 1.35))
    ax.grid(True, axis="y", color="#e8e8e4", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#dcdcd6")
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)

axes[-1].set_xlabel("timestep", fontsize=9, color=INK2)
fig.suptitle("8n K output voltages — every fan-in edge is 0.22–0.36× threshold, "
             "so each spike needs ~4 coincident arrivals",
             x=0.008, y=0.985, ha="left", fontsize=12, color=INK)
fig.text(0.008, 0.925, "input N0 fires every 100 steps; hidden N1–N4 fire at 39/48/62/88 "
         "(+100k); dashed line = firing threshold 7.0e-03",
         ha="left", fontsize=9, color=INK2)
fig.subplots_adjust(top=0.86, left=0.07, right=0.985, bottom=0.09)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "8nK_output_voltages.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
for o in outs:
    v = np.asarray(V[:, o], float)
    print(f"  N{o}: spikes {T[o]}  peak V {v.max():.4e}  th {G.TH:.4e}")
