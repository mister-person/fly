"""Error surface over a 2-D slice of weight space: is the local subproblem solvable?

    python3 _plot_wslice.py "over-demand" 2 1 2   # freeze w[0], scan w[1] and w[2]

Freezing every weight but two makes the remaining problem enumerable, which answers a
question the gradient cannot: does an improvement EXIST nearby, and does the field point at
it?  The three marked points are the run's stuck state, the best point in the slice, and the
true weights.

FORM.  Continuous magnitude over two continuous axes -> heatmap.  But the quantity is not
uniformly continuous: where the output has the WRONG NUMBER of spikes there is no meaningful
timing error at all, and folding that into the same ramp would put a categorical state
("wrong count") on a magnitude scale and swamp its range.  Those cells get a neutral, and
the ramp is spent entirely on the cells that are comparable.

COLOR.  Sequential = one hue, light -> dark (blue ramp, steps 100..700 of the reference
palette), lightest = smallest error.  No rainbow, no hue at either end.  The marker colors
are text ink, not series colors -- identity comes from the direct labels beside them.
"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import field_trace as F
from _diag import CASES, steps_for

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9992"
NOCOUNT = "#e4e3de"        # neutral: "wrong spike count", not a magnitude
# sequential blue, reference palette steps 100 -> 700
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)

name = sys.argv[1] if len(sys.argv) > 1 else "over-demand"
FREEZE = int(sys.argv[2]) if len(sys.argv) > 2 else 2      # neuron whose inputs are scanned
IX = int(sys.argv[3]) if len(sys.argv) > 3 else 1
IY = int(sys.argv[4]) if len(sys.argv) > 4 else 2

E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
p = F.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
lab = F.wlabels(C)
STUCK = np.array([252., 832., 250.]) if name == "over-demand" else np.array(Wl, float)
tgt = T[outs[0]]

xs = np.arange(100., 1501., 10.)
ys = np.arange(100., 601., 5.)
err = np.full((len(ys), len(xs)), np.nan)
bad = np.zeros((len(ys), len(xs)), bool)
for j, b in enumerate(ys):
    for i, a in enumerate(xs):
        w = STUCK.copy(); w[IX] = a; w[IY] = b
        s2 = F.sp(F.fsim(C, N, np.asarray(w, np.float32), p), outs[0])
        if len(s2) != len(tgt):
            bad[j, i] = True
        else:
            err[j, i] = float(np.mean([abs(u - v) for u, v in zip(s2, tgt)]))

jb, ib = np.unravel_index(np.nanargmin(np.where(bad, np.nan, err)), err.shape)
best = (xs[ib], ys[jb], err[jb, ib])

fig, ax = plt.subplots(figsize=(9.6, 6.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ext = [xs[0], xs[-1], ys[0], ys[-1]]
ax.imshow(np.where(bad, 1.0, np.nan), origin="lower", extent=ext, aspect="auto",
          cmap=LinearSegmentedColormap.from_list("n", [NOCOUNT, NOCOUNT]), zorder=1)
im = ax.imshow(err, origin="lower", extent=ext, aspect="auto", cmap=CMAP, zorder=2,
               vmin=float(np.nanmin(err)), vmax=float(np.nanpercentile(err, 97)))

# distinct offset per label: the three points sit close together, so a shared offset
# collides (measured: "best in slice" overprinted "true" at the first render)
pts = [(STUCK[IX], STUCK[IY], "stuck", (12, -14)),
       (best[0], best[1], f"best in slice ({best[2]:.2f})", (-8, 20)),
       (float(Wl[IX]), float(Wl[IY]), "true", (10, -16))]
for x, y, tag, off in pts:
    ax.plot([x], [y], "o", ms=10, mfc=INK, mec=SURFACE, mew=2.0, zorder=6)
    ax.annotate(tag, (x, y), xytext=off, textcoords="offset points",
                fontsize=9.5, color=INK, zorder=7,
                bbox=dict(boxstyle="round,pad=0.22", fc=SURFACE, ec="none", alpha=0.85))
ax.annotate("", xy=(best[0], best[1]), xytext=(STUCK[IX], STUCK[IY]),
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1.6,
                            shrinkA=9, shrinkB=9), zorder=5)

cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.045)
cb.set_label("mean |Δt| at the output  (timesteps)", fontsize=9, color=INK2)
cb.ax.tick_params(colors=INK2, labelsize=8, length=3)
cb.outline.set_visible(False)
ax.plot([], [], "s", ms=10, mfc=NOCOUNT, mec="#d5d4cf", label="wrong spike count")
ax.legend(frameon=False, fontsize=8.8, loc="upper right", labelcolor=INK2)
ax.set_xlabel(f"{lab[IX]}   (N{int(C[IX,0])} → N{int(C[IX,1])})", fontsize=9.5, color=INK2)
ax.set_ylabel(f"{lab[IY]}   (N{int(C[IY,0])} → N{int(C[IY,1])})", fontsize=9.5, color=INK2)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
for s_ in ("left", "bottom"):
    ax.spines[s_].set_color("#dcdcd6")
ax.tick_params(colors=INK2, labelsize=8.5, length=3)
fig.suptitle(f"{name} — output timing error over N{FREEZE}'s two input weights, "
             f"{lab[0]} frozen at {STUCK[0]:.0f}",
             x=0.008, y=0.985, ha="left", fontsize=12, color=INK)
fig.text(0.008, 0.938, f"target {tgt}   ·   true {F.wstr(C, Wl)}   ·   "
         f"no exact solution exists in this slice; best is {best[2]:.2f} steps",
         ha="left", fontsize=8.8, color=INK2)
fig.subplots_adjust(top=0.875, left=0.075, right=0.945, bottom=0.095)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"wslice_{name.replace(' ','_')}.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
print(f"  stuck {lab[IX]}={STUCK[IX]:.0f} {lab[IY]}={STUCK[IY]:.0f}")
print(f"  best  {lab[IX]}={best[0]:.0f} {lab[IY]}={best[1]:.0f}  err {best[2]:.2f}")
print(f"  true  {lab[IX]}={Wl[IX]:.0f} {lab[IY]}={Wl[IY]:.0f}")
print(f"  wrong-count cells: {bad.mean()*100:.1f}%")
