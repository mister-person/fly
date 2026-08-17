"""Weight trajectories through the stuck point: where is the gradient actually taking us?

    python3 _plot_wtraj.py "4n F" [seed] [settle] [steps]

Trains `settle` rounds to reach the stuck point, then plots every weight over the NEXT
`steps` iterations.  One panel per weight rather than four lines on shared axes: each weight
has its own true value, and what matters is the distance from THAT, so every panel carries
its own dashed reference and the y-range is set around the pair.  A shared axis would make
the small weights unreadable and hide exactly the movement in question.

The run is a SINGLE train() call so the Adam state is continuous across the boundary --
restarting the optimiser at the stuck point would reset the moment estimates and show a
transient that is an artefact of the measurement.  w is taken from the callback, never from
the return value, which is the best-ever iterate rather than the live one.
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
C_W, C_TRUE, C_CNT = "#2a78d6", "#9a9992", "#eb6834"

name = sys.argv[1] if len(sys.argv) > 1 else "4n F"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SETTLE = int(sys.argv[3]) if len(sys.argv) > 3 else 3200
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 30

E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
params = G.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)

rec = []
def cb(it, w, upd, g, spall, vsub, L):
    if it >= SETTLE:
        rec.append((it, np.asarray(w, float).copy(),
                    [len(spall[n]) for n in range(N)]))

G.train(C, N, outs, w0.copy(), T, params, rounds=SETTLE + STEPS, lr=G.LR, cb=cb)
rec = rec[:STEPS + 1]
it = np.array([r[0] for r in rec]) - SETTLE
Wt = np.array([r[1] for r in rec])          # (steps, nweights)
Cnt = np.array([r[2] for r in rec])          # (steps, N)

import field_trace as _FT
lab = [f"{nm}   N{int(C[k,0])}->N{int(C[k,1])}"
       for k, nm in enumerate(_FT.wlabels(C))]
nw = len(Wl)
fig, axes = plt.subplots(nw + 1, 1, figsize=(10.5, 1.55 * (nw + 1) + 2.0), sharex=True,
                         facecolor=SURFACE, gridspec_kw=dict(hspace=0.45))
for k in range(nw):
    ax = axes[k]
    ax.set_facecolor(SURFACE)
    tv = float(Wl[k])
    ax.axhline(tv, color=C_TRUE, lw=1.3, ls=(0, (5, 4)), zorder=2)
    ax.plot(it, Wt[:, k], color=C_W, lw=2.0, zorder=4)
    ax.plot(it[:1], Wt[:1, k], "o", ms=6, mfc=C_W, mec=SURFACE, mew=1.5, zorder=5)
    lo = min(Wt[:, k].min(), tv); hi = max(Wt[:, k].max(), tv)
    pad = max((hi - lo) * 0.28, abs(tv) * 0.06, 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    drift = Wt[-1, k] - Wt[0, k]
    ax.set_title(f"{lab[k]}   ·   {Wt[0,k]:.0f} → {Wt[-1,k]:.0f} over {STEPS} steps "
                 f"({drift:+.1f})   ·   true {tv:.0f}  (dashed)",
                 loc="left", fontsize=9, color=INK, pad=4)
    ax.set_ylabel("weight", fontsize=8.5, color=INK2)
ax = axes[-1]
ax.set_facecolor(SURFACE)
for n in range(1, N):
    ax.plot(it, Cnt[:, n], lw=2.0, zorder=4,
            color=[C_W, C_CNT, "#1baf7a", "#4a3aa7"][(n - 1) % 4])
    ax.annotate(f"N{n}", (it[-1], Cnt[-1, n]), xytext=(5, 0),
                textcoords="offset points", fontsize=8.5, color=INK2, va="center")
for n in range(1, N):
    ax.axhline(len(T[n]), color=C_TRUE, lw=1.0, ls=(0, (2, 3)), zorder=1)
ax.set_ylabel("spikes", fontsize=8.5, color=INK2)
ax.set_title("spike COUNT per neuron   ·   dotted = true count",
             loc="left", fontsize=9, color=INK, pad=4)
ax.set_xlabel(f"iterations past the stuck point (iteration {SETTLE})",
              fontsize=9, color=INK2)
for a in axes:
    for sp_ in ("top", "right"):
        a.spines[sp_].set_visible(False)
    for sp_ in ("left", "bottom"):
        a.spines[sp_].set_color("#dcdcd6")
    a.tick_params(colors=INK2, labelsize=8, length=3)
fig.suptitle(f"{name} seed {seed} — weights over {STEPS} gradient steps from the stuck point",
             x=0.008, y=0.99, ha="left", fontsize=12, color=INK)
fig.text(0.008, 0.958, f"single train() call, Adam state continuous   ·   "
         f"true {_FT.wstr(C, Wl)}", ha="left", fontsize=8.5, color=INK2)
fig.subplots_adjust(top=0.905, left=0.10, right=0.955, bottom=0.075)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"wtraj_{name.replace(' ','_')}_s{seed}.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
print(f"  start w {[round(float(x),1) for x in Wt[0]]}   true {[float(x) for x in Wl]}")
print(f"  end   w {[round(float(x),1) for x in Wt[-1]]}")
print(f"  per-step |dw| mean {np.abs(np.diff(Wt, axis=0)).mean(axis=0).round(3).tolist()}")
print(f"  counts start {Cnt[0].tolist()}  end {Cnt[-1].tolist()}  true "
      f"{[len(T[n]) for n in range(N)]}")
