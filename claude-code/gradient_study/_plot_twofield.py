"""Graph the two field variables per hidden neuron: urgency and implied_w.

    python3 _plot_twofield.py "8n M" [seed]

Two panels per hidden neuron rather than twin axes -- they are different quantities in
different units (urgency in volts, implied_w in weight), and a dual-axis chart is never the
right answer.  Each panel is a single series, so the title carries identity and no legend
box is needed.  On the implied_w panel the neuron's TRUE outgoing weights are drawn as
reference rules, so "is it in the right range" is readable directly.
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
C_URG, C_IW = "#2a78d6", "#eb6834"

name = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
params = G.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: G.sp(G.fsim(C, N, W, params), n) for n in range(N)}
inc = {n: np.where(C[:, 1] == n)[0] for n in range(N)}
hidden = [n for n in range(N) if n not in outs and n != 0 and len(inc[n])]

w = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)
V = G.fsim(C, N, np.asarray(w, np.float32), params)
s = {n: G.sp(V, n) for n in range(N)}
eps, L, vsub, wr = G.traces(C, N, w, s, params.steps, {o: T[o] for o in outs}, V)
U, IW = G.demand_field(C, N, w, s, params.steps, {o: T[o] for o in outs}, vsub, eps)
t = np.arange(params.steps)

rows = [(n, k) for n in hidden for k in ("urgency", "implied_w")]
fig, axes = plt.subplots(len(rows), 1, figsize=(11, 1.85 * len(rows) + 2.4), sharex=True,
                         facecolor=SURFACE, gridspec_kw=dict(hspace=0.55))
axes = np.atleast_1d(axes)
for ax, (n, kind) in zip(axes, rows):
    ax.set_facecolor(SURFACE)
    trueout = [(int(C[si, 1]), float(Wl[si])) for si in np.where(C[:, 0] == n)[0]]
    if kind == "urgency":
        y = U[n]
        ax.axhline(0, color="#dcdcd6", lw=1.0, zorder=1)
        ax.plot(t, y, color=C_URG, lw=1.8, zorder=3)
        span = max(float(np.abs(y).max()), 1e-30)
        ax.set_ylim(-1.35 * span, 1.35 * span)
        mark = -1.18 * span
        head = (f"N{n}  urgency — how much a spike is wanted here"
                f"   ·   {int((y != 0).sum())} pts, peak {span:.2e}")
        ax.set_ylabel("volts", fontsize=8.5, color=INK2)
    else:
        y = IW[n]
        fin = np.isfinite(y)
        ax.plot(t[fin], y[fin], color=C_IW, lw=1.8, zorder=3)
        for d, wv in trueout:
            ax.axhline(wv, color=MUTED, lw=1.0, ls=(0, (5, 4)), zorder=2)
        lab = ", ".join(f"N{d}:{wv:.0f}" for d, wv in trueout)
        hi = float(np.nanmax(y)) if fin.any() else 1.0
        top = max(hi * 1.1, max(wv for _, wv in trueout) * 2.0)
        ax.set_ylim(-0.16 * top, top)
        mark = -0.10 * top
        head = (f"N{n}  implied_w — outgoing weight that would make firing here useful"
                f"   ·   dashed = true ({lab})")
        ax.set_ylabel("weight", fontsize=8.5, color=INK2)
    ax.plot(T[n], [mark] * len(T[n]), "o", ms=8, mfc="none", mec=MUTED, mew=1.5,
            clip_on=False, zorder=6)
    ax.plot(s[n], [mark] * len(s[n]), "o", ms=6, mfc=INK2, mec=SURFACE, mew=1.3,
            clip_on=False, zorder=7)
    ax.set_title(head, loc="left", fontsize=9.5, color=INK, pad=6)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 4))
    ax.yaxis.get_offset_text().set(color=INK2, fontsize=7.5)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    for sp_ in ("left", "bottom"):
        ax.spines[sp_].set_color("#dcdcd6")
    ax.tick_params(colors=INK2, labelsize=8, length=3)
axes[-1].set_xlabel("timestep", fontsize=9, color=INK2)
fig.suptitle(f"{name} — the two field variables at each hidden neuron (seed {seed}, start weights)",
             x=0.008, y=0.985, ha="left", fontsize=12, color=INK)
fig.text(0.008, 0.940, "hollow ring = TRUE spike time,  filled = current spike "
         "(on the lower rule of each panel)", ha="left", fontsize=8.5, color=INK2)
fig.subplots_adjust(top=0.885, left=0.10, right=0.985, bottom=0.065)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"twofield_{name.replace(' ', '')}.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
for n in hidden:
    fin = np.isfinite(IW[n])
    tw = [float(Wl[si]) for si in np.where(C[:, 0] == n)[0]]
    print(f"  N{n}: urgency peak {np.abs(U[n]).max():.2e}   implied_w range "
          f"{np.nanmin(IW[n]) if fin.any() else float('nan'):.0f}.."
          f"{np.nanmax(IW[n]) if fin.any() else float('nan'):.0f}"
          f"   true outgoing {tw}")
