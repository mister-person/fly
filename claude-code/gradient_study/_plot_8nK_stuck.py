"""8n K output voltages at a STUCK point, drawn to match the true-weight figure.

Same small-multiples form and palette as _plot_8nK.py so the two can be read side by side.
Found spikes are the filled markers on the trace; TARGET times are hollow rings on the
threshold line, so a miss shows as a ring with no marker under it.
"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import grad_trace as G
from _diag import CASES

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9992"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

E, N, outs, Wl = CASES["8n K"]
C = np.array(E, np.int32)
params = G.mkparams(520)
W = np.array(Wl, np.float32)
T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
w0 = (W * np.random.default_rng(SEED).uniform(0.5, 1.5, len(Wl))).astype(float)
live = {}
G.train(C, N, outs, w0.copy(), T, params, rounds=3200, lr=G.LR,
        cb=lambda it, w, *a: live.update(w=w.copy()))
w = live["w"]
V = G.fsim(C, N, w, params)
F = {n: G.sp(V, n) for n in range(N)}
t = np.arange(params.steps)

fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True,
                         facecolor=SURFACE, gridspec_kw=dict(hspace=0.34))
for ax, o, col in zip(axes, outs, SERIES):
    ax.set_facecolor(SURFACE)
    v = np.asarray(V[:, o], float)
    ax.plot(t, v, color=col, lw=1.8, solid_joinstyle="round", zorder=3)
    ax.axhline(G.TH, color=MUTED, lw=1.0, ls=(0, (5, 4)), zorder=2)
    # targets as hollow rings ON the threshold line; a ring with no filled marker = missed
    ax.plot(T[o], [G.TH] * len(T[o]), "o", ms=9, mfc="none", mec=MUTED, mew=1.6, zorder=4)
    ax.plot(F[o], [v[s] for s in F[o]], "o", ms=8, mfc=col, mec=SURFACE, mew=2, zorder=5)
    hit = sum(1 for x in T[o] if any(abs(x - y) <= 2 for y in F[o]))
    ax.set_title(f"N{o}   found {len(F[o])} spikes {F[o]}   ·   target {T[o]}   "
                 f"·   {hit}/{len(T[o])} on target",
                 loc="left", fontsize=10, color=INK, pad=9)
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
fig.suptitle(f"8n K at a stuck point (seed {SEED}) — output voltages vs targets",
             x=0.008, y=0.985, ha="left", fontsize=12, color=INK)
fig.text(0.008, 0.925,
         "found w = " + str([int(round(x)) for x in w]) + "\n"
         "true  w = " + str([int(x) for x in Wl]),
         ha="left", fontsize=8.5, color=INK2, family="monospace")
fig.subplots_adjust(top=0.855, left=0.07, right=0.985, bottom=0.09)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"8nK_stuck_seed{SEED}.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
print("found w", [int(round(x)) for x in w])
print("true  w", [int(x) for x in Wl])
for n in range(N):
    print(f"  N{n}: found {F[n]}   true {T[n]}")
