"""What do the field's positive runs actually look like, in cells the peak-move fixes vs breaks?

    python3 _plot_runshapes.py

The request a neuron receives is ONE time extracted from a whole positive run of the field.
Two estimators disagree -- the amplitude-weighted centroid (default) and the argmax -- and
switching between them flips 8 cells of the suite, 3 one way and 5 the other.  This asks
whether the runs in those two groups look different.

FORM.  Small multiples: many instances of one quantity (a run profile), grouped.  Left
column = cells the peak-move FIXES, right = cells it BREAKS, so the comparison is spatial
and no colour is spent on it.  A bottom row overlays every run in each group, normalised in
both axes, to show the shape DISTRIBUTION rather than four hand-picked examples.

Per panel the two estimators are marked, plus the neuron's true spike times where they fall
inside the run -- that is the thing either estimator is trying to hit.
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
C_FIX, C_BRK, C_CEN, C_AMX = "#2a78d6", "#eb6834", "#4a3aa7", "#1baf7a"

FIXED = [("3-cycle", 0), ("over-demand", 3), ("3n A", 1)]
BROKEN = [("3-cycle", 3), ("3-cycle", 5), ("2-cycle", 5),
          ("over-demand", 2), ("over-demand", 4)]
NSHOW = 4


def collect(cells, iters=150):
    out = []
    for name, seed in cells:
        E, N, outs, Wl = CASES[name]
        C = np.array(E, np.int32)
        p = F.mkparams(steps_for(name))
        W = np.array(Wl, np.float32)
        T = {n: F.sp(F.fsim(C, N, W, p), n) for n in range(N)}
        w = (W * np.random.default_rng(seed).uniform(0.5, 1.5, len(Wl))).astype(float)

        def cb(it, ww, upd, g, spall, Fl, L):
            if it % 25:
                return
            for n in range(N):
                if n in outs:
                    continue
                pos = np.nonzero(Fl[n] > 0)[0]
                if not len(pos):
                    continue
                for r in np.split(pos, np.nonzero(np.diff(pos) > 1)[0] + 1):
                    if len(r) < 8:
                        continue
                    v = Fl[n][r]
                    out.append(dict(case=name, seed=seed, it=it, n=n, idx=r, val=v,
                                    true=list(T[n]),
                                    ratio=float(v.max() / max(np.median(v), 1e-30))))
        F.train(C, N, outs, w.copy(), T, p, rounds=iters, lr=F.LR, cb=cb)
    return out


def pick(rs, k):
    """Span the peak/median range rather than take the k largest."""
    rs = sorted(rs, key=lambda d: d["ratio"])
    if len(rs) <= k:
        return rs
    return [rs[int(round(q * (len(rs) - 1)))] for q in np.linspace(0.1, 0.9, k)]


fx, bk = collect(FIXED), collect(BROKEN)
sel = {"FIXED by moving the request to the peak": (pick(fx, NSHOW), C_FIX, fx),
       "BROKEN by it": (pick(bk, NSHOW), C_BRK, bk)}

fig, axes = plt.subplots(NSHOW + 1, 2, figsize=(11.4, 1.62 * (NSHOW + 1) + 2.1),
                         facecolor=SURFACE,
                         gridspec_kw=dict(hspace=0.62, wspace=0.16))
for col, (title, (rows, colr, allr)) in enumerate(sel.items()):
    for i in range(NSHOW):
        ax = axes[i][col]
        ax.set_facecolor(SURFACE)
        if i >= len(rows):
            ax.axis("off"); continue
        d = rows[i]
        r, v = d["idx"], d["val"]
        ax.plot(r, v, color=colr, lw=2.0, zorder=3)
        ax.fill_between(r, 0, v, color=colr, alpha=0.13, zorder=1, lw=0)
        cen = float(np.sum(r * v) / v.sum())
        amx = float(r[np.argmax(v)])
        ax.axvline(cen, color=C_CEN, lw=1.7, ls=(0, (4, 3)), zorder=4)
        ax.axvline(amx, color=C_AMX, lw=1.7, ls=(0, (1, 2)), zorder=4)
        for t in d["true"]:
            if r[0] <= t <= r[-1]:
                ax.plot([t], [0], "o", ms=8, mfc="none", mec=INK, mew=1.7,
                        clip_on=False, zorder=6)
        ax.set_title(f"{d['case']} seed{d['seed']}  N{d['n']}  it{d['it']}   ·   "
                     f"len {len(r)}   peak/med {d['ratio']:.2f}   "
                     f"|cen−argmax| {abs(cen-amx):.0f}",
                     loc="left", fontsize=8.4, color=INK, pad=3)
        ax.set_yticks([])
    ax = axes[NSHOW][col]
    ax.set_facecolor(SURFACE)
    for d in allr:
        v = d["val"]
        x = np.linspace(0, 1, len(v))
        ax.plot(x, v / max(v.max(), 1e-30), color=colr, lw=0.8, alpha=0.16, zorder=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("position within the run (normalised)", fontsize=8.6, color=INK2)
    ax.set_title(f"all {len(allr)} runs, normalised — the shape DISTRIBUTION",
                 loc="left", fontsize=8.6, color=INK, pad=3)
    axes[0][col].annotate(title, xy=(0.5, 1.62), xycoords="axes fraction",
                          ha="center", fontsize=10.5, color=INK)
for row in axes:
    for ax in row:
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#dcdcd6")
        ax.tick_params(colors=INK2, labelsize=7.6, length=3)
fig.suptitle("field positive runs — the cells the peak-move fixes vs the ones it breaks",
             x=0.008, y=0.99, ha="left", fontsize=12, color=INK)
fig.text(0.008, 0.962,
         "dashed = centroid (default)   ·   dotted = argmax   ·   hollow ring = a TRUE spike "
         "time inside the run   ·   x is the timestep",
         ha="left", fontsize=8.4, color=INK2)
fig.subplots_adjust(top=0.885, left=0.045, right=0.99, bottom=0.075)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runshapes.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
for tag, (_r, _c, allr) in sel.items():
    A = np.array([[len(d["idx"]), d["ratio"]] for d in allr])
    print(f"  {tag:44} {len(allr):>4} runs   median len {np.median(A[:,0]):6.1f}   "
          f"median peak/med {np.median(A[:,1]):.2f}")
