"""Sweep the straight line from a stuck point to the true weights: what is in the way?

    python3 _plot_path.py "over-demand" [lo] [hi]

Two panels, never one with two y-axes -- an error in timesteps and a directional derivative
in demand units are different quantities:

  TOP     output timing error along the path.  Where the output has the wrong spike COUNT
          there is no timing error to plot, so those alpha are marked on the baseline rather
          than folded into the curve.
  BOTTOM  the method's own gradient projected onto the path direction, g . dhat.  The update
          is w += step * g, so g . dhat > 0 means "this gradient moves toward the answer".
          Any alpha where the top panel descends and the bottom panel is negative is a point
          where the landscape is fine and the METHOD is pointing the wrong way.

alpha = 0 is the stuck point, alpha = 1 the true weights; the sweep runs past both ends so
the shape either side is visible.
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
C_ERR, C_G, C_G2, BAD = "#2a78d6", "#eb6834", "#1baf7a", "#e4e3de"

name = sys.argv[1] if len(sys.argv) > 1 else "over-demand"
LO = float(sys.argv[2]) if len(sys.argv) > 2 else -0.15
HI = float(sys.argv[3]) if len(sys.argv) > 3 else 1.15

E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
p = F.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
STUCK = {"over-demand": np.array([247., 867., 318.])}.get(name, np.array(Wl, float) * 1.2)
TRUE = np.array(Wl, float)
d = TRUE - STUCK
dhat = d / max(np.linalg.norm(d), 1e-12)
lab = F.wlabels(C)

al = np.linspace(LO, HI, 241)
err = np.full(len(al), np.nan)
bad = np.zeros(len(al), bool)
gd = np.zeros(len(al))
gd2 = np.zeros(len(al))
for i, a in enumerate(al):
    w = STUCK + a * d
    V = F.fsim(C, N, np.asarray(w, np.float32), p)
    s = {n: F.sp(V, n) for n in range(N)}
    tot, nb = 0.0, 0
    for o in outs:
        if len(s[o]) != len(T[o]):
            nb += 1
        else:
            tot += float(np.mean([abs(x - y) for x, y in zip(s[o], T[o])]))
    if nb:
        bad[i] = True
    else:
        err[i] = tot / len(outs)
    g, _, _, _ = F.gradient(C, N, w, s, p.steps, {o: T[o] for o in outs})
    gd[i] = float(np.dot(g, dhat))
    sf = {n: F.sp_frac(V, n) for n in range(N)}
    g2, _, _, _ = F.gradient(C, N, w, s, p.steps, {o: T[o] for o in outs}, sf)
    gd2[i] = float(np.dot(g2, dhat))

fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.6), sharex=True, facecolor=SURFACE,
                         gridspec_kw=dict(hspace=0.28, height_ratios=[1.15, 1.0]))
for ax in axes:
    ax.set_facecolor(SURFACE)
# shade the wrong-count stretches on both panels
edges = np.flatnonzero(np.diff(bad.astype(int)) != 0)
segs, st = [], (0 if bad[0] else None)
for e in edges:
    if bad[e] and not bad[e + 1]:
        segs.append((al[st], al[e])); st = None
    elif not bad[e] and bad[e + 1]:
        st = e + 1
if st is not None and bad[-1]:
    segs.append((al[st], al[-1]))
for ax in axes:
    for a0, a1 in segs:
        ax.axvspan(a0, a1, color=BAD, zorder=0, lw=0)

ax = axes[0]
ax.plot(al, err, color=C_ERR, lw=2.2, zorder=3)
ax.set_ylabel("mean |Δt|  (timesteps)", fontsize=9, color=INK2)
ax.set_title("output timing error along the path   ·   shaded = wrong spike count",
             loc="left", fontsize=9.5, color=INK, pad=5)

ax = axes[1]
ax.axhline(0, color=MUTED, lw=1.1, zorder=2)
ax.plot(al, gd, color=C_G, lw=2.2, zorder=3, label="integer spike times")
ax.plot(al, gd2, color=C_G2, lw=2.2, zorder=4, label="sub-sample crossing times")
ax.fill_between(al, 0, gd, where=(gd < 0), color=C_G, alpha=0.13, zorder=1, lw=0)
ax.legend(frameon=False, fontsize=8.6, loc="lower right", labelcolor=INK2,
          handlelength=1.8, ncol=2)
ax.set_ylabel("g · d̂   (toward truth if > 0)", fontsize=9, color=INK2)
ax.set_title("the method's gradient, projected onto the path direction",
             loc="left", fontsize=9.5, color=INK, pad=5)
ax.set_xlabel("α along stuck → true      (0 = stuck, 1 = true weights)",
              fontsize=9.5, color=INK2)

for ax in axes:
    for a, tag in ((0.0, "stuck"), (1.0, "true")):
        ax.axvline(a, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=2)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color("#dcdcd6")
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)
for a, tag in ((0.0, "stuck"), (1.0, "true")):
    axes[0].annotate(tag, (a, axes[0].get_ylim()[1]), xytext=(4, -12),
                     textcoords="offset points", fontsize=9, color=INK2)

msk = (al >= 0) & (al <= 1)
frac = float(np.mean(gd[msk] > 0))
frac2 = float(np.mean(gd2[msk] > 0))
fig.suptitle(f"{name} — the straight line from the stuck point to the answer",
             x=0.008, y=0.985, ha="left", fontsize=12, color=INK)
fig.text(0.008, 0.938,
         f"stuck {F.wstr(C, STUCK)}   →   true {F.wstr(C, TRUE)}   ·   "
         f"points toward truth on {frac*100:.0f}% of the path "
         f"(integer) / {frac2*100:.0f}% (sub-sample)",
         ha="left", fontsize=8.8, color=INK2)
fig.subplots_adjust(top=0.875, left=0.095, right=0.985, bottom=0.10)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"path_{name.replace(' ','_')}.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
ok = (al >= 0) & (al <= 1)
print(f"  err at alpha=0: {'wrong count' if bad[np.argmin(abs(al))] else err[np.argmin(abs(al))]:.3f}"
      f"   at alpha=1: {err[np.argmin(abs(al-1))]:.3f}")
print(f"  wrong-count stretches: {segs}")
print(f"  g.dhat > 0 on {frac*100:.1f}% (integer) / {frac2*100:.1f}% (sub-sample)")
