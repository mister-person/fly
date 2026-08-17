"""The same stuck→true sweep under each smoothing variant, as small multiples.

    python3 _plot_smooth_compare.py "over-demand"

FORM.  Five versions of ONE quantity (g · dhat along the path) -> small multiples, not five
lines on shared axes: the traces cross constantly and a five-colour legend would be doing
work the panel titles do better.  A single series per panel needs no legend at all.

SCALE.  All gradient panels share one y-range, deliberately.  The variants differ in
AMPLITUDE as well as shape -- fractional eps is 4x smaller than sub-sampled L -- and
per-panel autoscaling would hide exactly that.  The reference error curve on top has its own
scale because it is a different quantity (timesteps, not demand).

The point to read: which panels spend their time BELOW zero.  g . dhat > 0 means the
gradient moves toward the answer, and the top panel shows that answer is monotonically
downhill the whole way.
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
C_ERR, C_G = "#2a78d6", "#eb6834"

name = sys.argv[1] if len(sys.argv) > 1 else "over-demand"
E, N, outs, Wl = CASES[name]
C = np.array(E, np.int32)
p = F.mkparams(steps_for(name))
W = np.array(Wl, np.float32)
T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
STUCK = {"over-demand": np.array([247., 867., 318.])}.get(name, np.array(Wl, float) * 1.2)
TRUE = np.array(Wl, float)
d = TRUE - STUCK
dhat = d / max(np.linalg.norm(d), 1e-12)

# (label, SUBSAMPLE, SUB_EPS, CENTROID, SPREAD, suite score or None)
CFG = [("integer everything  (baseline)",              0, 0, 0, 0, "59/104"),
       ("centroid bump position   [DEFAULT]",          0, 0, 1, 0, "65/104"),
       ("centroid + spread requests over their runs",  0, 0, 1, 1, "62/104"),
       ("sub-sample the demand L",                     1, 0, 0, 0, "57/104"),
       ("sub-sample L + fractional eps + centroid",    1, 1, 1, 0, "2/104")]

al = np.linspace(0.0, 1.0, 241)
err = np.full(len(al), np.nan)
G = np.zeros((len(CFG), len(al)))
sv = (F.SUBSAMPLE, F.SUB_EPS, F.CENTROID, F.SPREAD)
for i, a in enumerate(al):
    w = STUCK + a * d
    V = F.fsim(C, N, np.asarray(w, np.float32), p)
    s = {n: F.sp(V, n) for n in range(N)}
    sf = {n: F.sp_frac(V, n) for n in range(N)}
    tot, nb = 0.0, 0
    for o in outs:
        if len(s[o]) != len(T[o]):
            nb += 1
        else:
            tot += float(np.mean([abs(x - y) for x, y in zip(s[o], T[o])]))
    if not nb:
        err[i] = tot / len(outs)
    for c, (_lab, sub, se, ce, spd, _sc) in enumerate(CFG):
        F.SUBSAMPLE, F.SUB_EPS, F.CENTROID, F.SPREAD = sub, se, ce, spd
        g, _, _, _ = F.gradient(C, N, w, s, p.steps, {o: T[o] for o in outs},
                                sf if sub else None)
        G[c, i] = float(np.dot(g, dhat))
F.SUBSAMPLE, F.SUB_EPS, F.CENTROID, F.SPREAD = sv

nrow = len(CFG) + 1
fig, axes = plt.subplots(nrow, 1, figsize=(10.2, 1.55 * nrow + 1.9), sharex=True,
                         facecolor=SURFACE, gridspec_kw=dict(hspace=0.42))
ax = axes[0]
ax.set_facecolor(SURFACE)
ax.plot(al, err, color=C_ERR, lw=2.2, zorder=3)
ax.set_ylabel("mean |Δt|", fontsize=8.5, color=INK2)
ax.set_title("output timing error along the path — monotonically downhill, no barrier",
             loc="left", fontsize=9.5, color=INK, pad=4)

lim = float(np.abs(G).max()) * 1.12
for c, (lb, _s, _e, _ce, _sp, sc) in enumerate(CFG):
    ax = axes[c + 1]
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=MUTED, lw=1.0, zorder=2)
    ax.plot(al, G[c], color=C_G, lw=1.9, zorder=3)
    ax.fill_between(al, 0, G[c], where=(G[c] < 0), color=C_G, alpha=0.16, zorder=1, lw=0)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("g · d̂", fontsize=8.5, color=INK2)
    frac = 100.0 * float(np.mean(G[c] > 0))
    jm = float(np.abs(np.diff(G[c])).max())
    ax.set_title(f"{lb}   ·   toward truth {frac:.0f}% of the path   ·   "
                 f"largest jump {jm:.1e}"
                 + (f"   ·   suite {sc}" if sc else "   ·   suite not run"),
                 loc="left", fontsize=9, color=INK, pad=4)
axes[-1].set_xlabel("α along stuck → true      (0 = stuck, 1 = true weights)",
                    fontsize=9.5, color=INK2)
for ax in axes:
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color("#dcdcd6")
    ax.tick_params(colors=INK2, labelsize=8, length=3)
fig.suptitle(f"{name} — the same sweep under each smoothing variant",
             x=0.008, y=0.99, ha="left", fontsize=12, color=INK)
fig.text(0.008, 0.962, f"stuck {F.wstr(C, STUCK)}   →   true {F.wstr(C, TRUE)}"
         "   ·   gradient panels share one y-scale, so amplitude is comparable",
         ha="left", fontsize=8.6, color=INK2)
fig.subplots_adjust(top=0.925, left=0.088, right=0.99, bottom=0.068)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"smooth_{name.replace(' ','_')}.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
for c, (lb, *_r) in enumerate(CFG):
    print(f"  {lb:34} aligned {100*np.mean(G[c]>0):5.1f}%   "
          f"max|g| {np.abs(G[c]).max():.2e}   largest jump {np.abs(np.diff(G[c])).max():.2e}")
