"""One positive run = one requested spike, however wide the run is.

    python3 _plot_runs.py "14n Q" [seed] [neurons...]

Zooms on the neurons whose field never returns to zero.  The shaded span is a single positive
RUN; bumps_of() puts exactly one request at its centroid, so the count handed downstream is the
number of runs, not the number of spikes the run stands for.  Rings mark where the neuron
really has to fire inside that same span.
"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import field_trace as F
from _diag import CASES, steps_for

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9992"
C_V, C_F, C_L = "#3f3f3c", "#2a78d6", "#eb6834"

name = sys.argv[1] if len(sys.argv) > 1 else "14n Q"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
want = [int(x) for x in sys.argv[3:]] or [9, 7]
ROUNDS = 800

E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
p = F.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
w0 = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
w = np.asarray(F.train(C, N, outs, w0.copy(), T, p, rounds=ROUNDS, lr=F.LR), float)
V = F.fsim(C, N, np.asarray(w, np.float32), p)
spall = {n: F.sp(V, n) for n in range(N)}
g, Fl, L, ep = F.gradient(C, N, w, spall, p.steps, {o: T[o] for o in outs})
t = np.arange(p.steps)

fig, axes = plt.subplots(len(want), 1, figsize=(11.5, 3.1 * len(want) + 1.6), sharex=True,
                         facecolor=SURFACE, gridspec_kw=dict(hspace=0.32))
axes = np.atleast_1d(axes)
for ax, n in zip(axes, want):
    ax.set_facecolor(SURFACE)
    bumps = F.bumps_of(Fl[n])
    pk = max(float(np.abs(Fl[n]).max()), 1e-30)
    for q, _h, r in bumps:
        ax.axvspan(r[0], r[-1], color="#dbe8f8", alpha=0.75, lw=0, zorder=0)
    ax.axhline(0.0, color="#e0e0da", lw=1.0, zorder=1)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (5, 4)), zorder=1)
    ax.plot(t, np.asarray(V[:, n], float) / F.TH, color=C_V, lw=1.3,
            label="V / threshold", zorder=3)
    ax.plot(t, Fl[n] / pk, color=C_F, lw=1.9, label=f"field / {pk:.1e}", zorder=4)
    ax.plot([b[0] for b in bumps], [Fl[n][b[0]] / pk for b in bumps], "o", ms=11,
            mfc=C_F, mec=SURFACE, mew=2.0, zorder=8,
            label=f"REQUESTS ({len(bumps)})")
    ax.plot(T[n], [-0.62] * len(T[n]), "o", ms=8, mfc="none", mec=C_L, mew=1.8,
            clip_on=False, zorder=9, label=f"true spikes ({len(T[n])})")
    ax.plot(spall[n], [-0.62] * len(spall[n]), "|", ms=10, color=INK2, mew=2.0,
            clip_on=False, zorder=10, label=f"fires ({len(spall[n])})")
    widest = max(bumps, key=lambda b: b[2][-1] - b[2][0])
    k = sum(1 for s in T[n] if widest[2][0] <= s <= widest[2][-1])
    ax.set_title(f"N{n} — widest run spans {int(widest[2][0])}..{int(widest[2][-1])} "
                 f"({int(widest[2][-1]-widest[2][0])} steps, {k} true spikes inside) "
                 f"and yields ONE request at {widest[0]}",
                 loc="left", fontsize=9.6, color=INK, pad=6)
    ax.set_ylim(-0.85, 1.6)
    ax.set_ylabel("indexed", fontsize=8.6, color=INK2)
    ax.legend(frameon=False, fontsize=7.9, loc="upper right", ncol=5, labelcolor=INK2,
              handlelength=1.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color("#dcdcd6")
    ax.tick_params(colors=INK2, labelsize=8.2, length=3)
axes[-1].set_xlabel("timestep", fontsize=9, color=INK2)
fig.suptitle(f"{name} seed {seed} — a positive field run yields one request no matter how "
             f"many spikes it covers", x=0.007, y=0.985, ha="left", fontsize=12, color=INK)
fig.text(0.007, 0.952, "shaded = one positive run of the field · REFRAC=22, so these spans "
                       "could hold dozens of spikes",
         ha="left", fontsize=8.7, color=INK2)
fig.subplots_adjust(top=0.90, left=0.065, right=0.99, bottom=0.09)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"runs_{name.replace(' ', '_')}_s{seed}.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
