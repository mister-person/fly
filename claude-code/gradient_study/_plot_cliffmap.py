"""Does a CORRECT COUNT mean anything, around 4n G's post-cliff state?

    python3 _plot_cliffmap.py

The count metric is satisfiable by configurations that are structurally wrong: a net can
produce the right NUMBER of output spikes through a completely different firing pattern, one
whose timing can never be fixed.  If that is common, count-ok is a bad proxy and the suite has
been optimising against it.

So: hold w03/w23 at the post-cliff values and scan the two hidden weights over a range
containing the true point, the stuck point and the post-cliff point.  Every cell is scored the
way the suite scores -- count first, then timing only where the count matches.

FORM.  Two continuous axes, one magnitude -> heatmap.  But the magnitude is undefined where
the count is wrong, and folding "wrong count" into the same ramp would put a categorical state
on a continuous scale and swamp its range.  Those cells take a neutral; the ramp is spent
entirely on the cells that are comparable to each other.  Same convention and palette as
_plot_wslice.py.
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
NOCOUNT = "#e4e3de"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)

name = "4n G"
E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
p = F.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
tgt = list(T[outs[0]])
lab = F.wlabels(C)

POST = np.array([790.9546, 845.9796, 984.7412, 506.9758])   # deepest descent, past the cliff
# which point supplies the FROZEN w03/w23 -- the post-cliff state, or truth
_HOLD = os.environ.get("HOLD", "post")
STUCK = np.array([1035., 1202., 1124., 741.])
TRUE = np.array(Wl, float)

xs = np.arange(100., 1501., 10.)
ys = np.arange(100., 1401., 10.)
err = np.full((len(ys), len(xs)), np.nan)
bad = np.zeros((len(ys), len(xs)), bool)
for j, b in enumerate(ys):
    for i, a in enumerate(xs):
        w = (POST.copy() if _HOLD == "post" else np.array(Wl, float))
        w[0] = a; w[1] = b
        s2 = list(F.sp(F.fsim(C, N, np.asarray(w, np.float32), p), outs[0]))
        if len(s2) != len(tgt):
            bad[j, i] = True
        else:
            err[j, i] = float(np.mean([abs(u - v) for u, v in zip(s2, tgt)]))

ok = ~bad
_HW = POST if _HOLD == "post" else np.array(Wl, float)
print(f"{name}: w03/w23 held at {_HOLD.upper()} values {_HW[2]:.0f}/{_HW[3]:.0f}")
print(f"  grid {len(xs)}x{len(ys)} = {ok.size} cells")
print(f"  COUNT-CORRECT cells: {int(ok.sum())} ({100*ok.mean():.1f}%)")
if ok.any():
    e = err[ok]
    print(f"  |dt| over those:  min {np.nanmin(e):.1f}   median {np.nanmedian(e):.1f}   "
          f"max {np.nanmax(e):.1f}")
    print(f"  count-correct AND |dt| <= 2: {int((err[ok] <= 2).sum())} cells "
          f"({100*(err[ok] <= 2).sum()/ok.size:.2f}% of the grid)")

fig, ax = plt.subplots(figsize=(9.8, 6.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ext = [xs[0], xs[-1], ys[0], ys[-1]]
ax.imshow(np.where(bad, 1.0, np.nan), origin="lower", extent=ext, aspect="auto",
          cmap=LinearSegmentedColormap.from_list("n", [NOCOUNT, NOCOUNT]), zorder=1)
im = ax.imshow(err, origin="lower", extent=ext, aspect="auto", cmap=CMAP, zorder=2)
cb = fig.colorbar(im, ax=ax, pad=0.015)
cb.set_label("|Δt| per output spike, where the COUNT is correct", fontsize=9, color=INK2)
cb.ax.tick_params(colors=INK2, labelsize=8.2)
cb.outline.set_visible(False)

for pt, txt, mk in ((TRUE, "true weights", "*"), (STUCK, "stuck point", "o"),
                    (POST, "post-cliff", "D")):
    ax.plot([pt[0]], [pt[1]], mk, ms=15 if mk == "*" else 9, mfc="none", mec=INK, mew=2.0,
            zorder=5)
    ax.annotate(txt, (pt[0], pt[1]), textcoords="offset points", xytext=(12, 9),
                fontsize=9, color=INK, zorder=6)
ax.set_xlabel(f"{lab[0]}  (true {TRUE[0]:.0f})", fontsize=9.5, color=INK2)
ax.set_ylabel(f"{lab[1]}  (true {TRUE[1]:.0f})", fontsize=9.5, color=INK2)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
for s_ in ("bottom", "left"):
    ax.spines[s_].set_color("#dcdcd6")
ax.tick_params(colors=INK2, labelsize=8.4, length=3)
fig.suptitle(f"{name} — is a correct spike COUNT worth anything here?",
             x=0.007, y=0.985, ha="left", fontsize=12.5, color=INK)
fig.text(0.007, 0.938,
         f"w03/w23 frozen at the {_HOLD} values ({_HW[2]:.0f}/{_HW[3]:.0f}) · "
         f"grey = wrong number of output spikes (no timing error is defined there) · "
         f"{int(ok.sum())} of {ok.size} cells ({100*ok.mean():.2f}%) have the right count",
         ha="left", fontsize=8.8, color=INK2)
fig.subplots_adjust(top=0.86, left=0.075, right=1.0, bottom=0.10)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"cliffmap_4n_G_{_HOLD}.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
